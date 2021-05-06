import os
import PIL
import cv2
import PIL.Image
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed
from tensorflow.keras.utils import plot_model
#from attention import Attention
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import utility files
from models import *
import config
import utils
import argparse
import matplotlib.pyplot as plt


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())


# init model
MODEL_ARCH = args["model"]
model = final_model(MODEL_ARCH)
for layer in model.layers:
    print(layer,layer.input_shape, layer.output_shape)

# plot model architecture
plot_model(model, to_file=config.MODEL_PATH+MODEL_ARCH+".png", show_shapes=True, show_layer_names=True)

# init optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=config.INIT_LR, decay=config.INIT_LR/config.EPOCHS)

# init loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# prepare metrics
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

# init tensorboard writers
train_write_path = os.path.join(config.TENSORBOARD_TRAIN_WRITER, MODEL_ARCH)
val_write_path = os.path.join(config.TENSORBOARD_VAL_WRITER, MODEL_ARCH)
writer_ls = [train_write_path, val_write_path]
for writer in writer_ls:
    if not os.path.exists(writer):
        os.makedirs(writer)
train_writer = tf.summary.create_file_writer(train_write_path)
test_writer  = tf.summary.create_file_writer(val_write_path)

# init callbacks
checkpoint_dir = config.CHKPT_PATH
checkpoint_prefix = os.path.join(checkpoint_dir, MODEL_ARCH, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
best_val_acc = 0.0


def early_stop(loss_list, min_delta, patience):
    # no early stopping for 2*patience epochs 
    if len(loss_list)//patience < 2 :
        return False
    # mean loss for last patience epochs and second-last patience epochs
    mean_prev = np.mean(loss_list[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(loss_list[::-1][:patience]) #last
    # can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_prev) #abs change
    delta_abs = np.abs(delta_abs / mean_prev)  # relative change
    if delta_abs < min_delta:
        print("[INFO] Loss didn't change much from last %d epochs"%(patience))
        print("[INFO] Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False


#'@tf.function' decorator compiles a function into a callable tensorflow graph
@tf.function
def train_batch(step, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True) # forward pass
        loss_value = loss_fn(y, logits) # compute loss
    # compute gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # update metrics
    train_acc_metric.update_state(y, logits)
    # write training loss and accuracy to tensorboard
    with train_writer.as_default():
        tf.summary.scalar('train loss', loss_value, step=step)
        tf.summary.scalar('train accuracy', train_acc_metric.result(), step=step)
    return loss_value


@tf.function
def test_batch(step, x, y):
    # forward pass, no backprop, inference mode
    val_logits = model(x, training=False)
    # compute loss
    loss_value = loss_fn(y, val_logits)
    # update metrics
    val_acc_metric.update_state(y, val_logits)
    # write test loss and accuracy to tensorboard
    with test_writer.as_default():
        tf.summary.scalar('val loss', loss_value, step=step)
        tf.summary.scalar('val accuracy', val_acc_metric.result(), step=step)
    return loss_value


#################################
#          TRAINING LOOP        #
#################################
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

print("[INFO] Training model...")
epochs = config.EPOCHS
for epoch in range(1,epochs+1):
    print("\nStart of epoch %d" % (epoch,))
    t = time.time()
    print("[INFO] Preparing training & validation generators...")
    val_dataset = tf.data.Dataset.from_generator(generator=utils.val_image_gen, 
                                            output_types=(tf.float32, tf.int32),) 

    train_dataset = tf.data.Dataset.from_generator(generator=utils.train_image_gen, 
                                            output_types=(tf.float32, tf.int32),)
    

    # iterate over the batches of the dataset
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        train_loss_value = train_batch(step, x_batch_train, y_batch_train)

        # log every _ batches
        if step % 50 == 0:
            print("Training loss (1 batch) at step %d: %.4f" % (step, float(train_loss_value)))
            print("%d samples seen" % ((step + 1) * config.BS))

    # run a validation loop at the end of each epoch
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        val_loss_value = test_batch(step, x_batch_val, y_batch_val)

    template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'

    print(template.format(
        round((time.time() - t)/60, 2), epoch + 1,
        train_loss_value, float(train_acc_metric.result()),
        val_loss_value, float(val_acc_metric.result())
    ))
    
    # save model checkpoints
    if float(val_acc_metric.result()) > best_val_acc:
        checkpoint.save(file_prefix = checkpoint_prefix)
        print('[INFO] prev val acc: {} new best val accuracy: {}'.format(best_val_acc, 
                float(val_acc_metric.result())))
        best_val_acc = float(val_acc_metric.result())

    train_loss_list.append(train_loss_value)
    val_loss_list.append(val_loss_value)
    train_acc_list.append(float(train_acc_metric.result()))
    val_acc_list.append(float(val_acc_metric.result()))

    # reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

    # early stopping (if necessary)
    FINAL_EPOCH = epoch
    earlyStop = early_stop(val_loss_list, min_delta=0.01, patience=10)
    if earlyStop:
        print("[INFO] Early stop at epoch= %d/%d"%(epoch,epochs))
        print("[INFO] Terminating training...")
        break

# plot the training loss and accuracy
N = np.arange(0, FINAL_EPOCH)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, train_loss_list, label="train_loss")
plt.plot(N, val_loss_list, label="val_loss")
plt.plot(N, train_acc_list, label="train_acc")
plt.plot(N, val_acc_list, label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["model"]+"_plot.png")

#################################
#          MODEL SAVING         #
#################################
# save the model
if not os.path.exists(os.path.join(config.MODEL_PATH, args["model"])):
    os.makedirs(os.path.join(config.MODEL_PATH, args["model"]))
model.save(os.path.join(config.MODEL_PATH, args["model"]))

with open("val_acc_results_"+args["model"]+".txt", "w") as f:
    for i in val_acc_list:
        f.write(str(i)+"\n")
