import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from models import *
import time
import numpy as np
import seaborn as sns
import pandas as pd

batch_size = 64
epochs = 10
raw_input = (32,32,3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

x_train = x_train[1000:,:,:,:]
y_train = y_train[1000:,:]
x_val = x_train[:1000,:,:,:]
y_val = y_train[:1000,:]
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape) 
# print(x_test.shape, y_test.shape)   

# scale datasets
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# target / class name
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# convert the class vectors - both train & val to the binary class matrix 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# Prepare the training dataset (separate elements of the input tensor for efficient input pipelines)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


# 1. Iterate over the number of epochs
# 2. For each epoch, iterate over the datasets, in batches (x, y)
# 3. For each batch, open GradientTape() scope
# 4. Inside this scope, call the model, the forward pass, and compute the loss
# 5. Outside this scope, retrieve the gradients of the weights of the model with regard to the loss
# 6. Next, use the optimizer to update the weights of the model based on the gradients

# Initialize model object
model = CustomConvNet(num_classes=len(class_names))
model.build_graph(raw_input).summary()

# Instantiate an optimizer to train the model
optimizer = tf.keras.optimizers.Adam()

# Instantiate a loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Prepare the metrics
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

# tensorboard writer 
train_writer = tf.summary.create_file_writer('output/logs/train/')
test_writer  = tf.summary.create_file_writer('output/logs/val/')

@tf.function
def train_step(step, x, y):
    '''
    input: x, y <- typically batches
    input: step <- batch step 
    return: loss value
    '''
    # start the scope of gradient 
    with tf.GradientTape() as tape:
       logits = model(x, training=True) # forward pass
       train_loss_value = loss_fn(y, logits) # compute loss 
       
    # compute gradient 
    grads = tape.gradient(train_loss_value, model.trainable_weights)

    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metrics
    train_acc_metric.update_state(y, logits)

    # write training loss and accuracy to the tensorboard
    with train_writer.as_default():
        tf.summary.scalar('loss', train_loss_value, step=step)
        tf.summary.scalar('accuracy', train_acc_metric.result(), step=step)
    
    return train_loss_value


@tf.function
def test_step(step, x, y):
    '''
    input: x, y <- typically batches 
    input: step <- batch step
    return: loss value
    '''
    # forward pass, no backprop, inference mode 
    val_logits = model(x, training=False) 

    # Compute the loss value 
    val_loss_value = loss_fn(y, val_logits)

    # Update val metrics
    val_acc_metric.update_state(y, val_logits)

    # write test loss and accuracy to the tensorboard
    with test_writer.as_default():
        tf.summary.scalar('val loss', val_loss_value, step=step)
        tf.summary.scalar('val accuracy', val_acc_metric.result(), step=step) 

    return val_loss_value


for epoch in range(epochs):
    
    t = time.time()

    # Iterate over the batches of the train dataset
    for train_batch_step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
        train_loss_value = train_step(train_batch_step, x_batch_train, y_batch_train)

    # evaluation on validation set 
    # Run a validation loop at the end of each epoch
    for test_batch_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        test_batch_step = tf.convert_to_tensor(test_batch_step, dtype=tf.int64)
        val_loss_value = test_step(test_batch_step, x_batch_val, y_batch_val)

    template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'
    print(template.format(
        round((time.time() - t)/60, 2), epoch + 1,
        train_loss_value, float(train_acc_metric.result()),
        val_loss_value, float(val_acc_metric.result())
    ))
        
    # Reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()


######################################
#         EVALUATING MODEL           #
######################################

# Multiclass ROC AUC score #
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# Classification report #
print("evaluating network...")
preds = model.predict(x_test, verbose=1)
print(classification_report(y_test.argmax(axis=1), 
                            preds.argmax(axis=1), 
                            target_names=class_names))

# ROC AUC score #
print("ROC AUC score: ", multiclass_roc_auc_score(y_test, preds.argmax(axis=1)))

# # Confusion matrix #
# preds = model.predict(x_test, verbose=2)
# preds = np.argmax(preds, axis=1)
# cm = confusion_matrix(np.argmax(y_test, axis=1), preds)
# cm = pd.DataFrame(cm, range(10),range(10))
# plt.figure(figsize = (10,10))
# sns.heatmap(cm, annot=True, annot_kws={"size": 12})
# plt.show()

# Testing on new images
image_file_list = ["./kitten.jpg", "./puppy.jpg"]  
for img in image_file_list:
    test_image  = image.load_img(img, target_size=(32,32))
    test_image  = image.img_to_array(test_image) 
    test_image  = np.expand_dims(test_image, axis=0) 
    prediction  = model.predict(test_image)[0] 
    label_index = np.argmax(prediction)
    print("Prediction for {} : {}".format(img, str(class_names[label_index])))


######################################
#         SAVING WEIGHTS             #
######################################

# The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, 
# while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and 
# custom layers without requiring the orginal code
model.save('net', save_format='tf')
# A new folder 'net' will be created in the working directory: contains 'assets', 'saved_model.pb', 'variables'
# The model architecture and training configuration, including the optimizer, losses, and metrics are stored in saved_model.pb
# The weights are saved in the variables directory

# OR

# save only the trained weights
#model.save_weights('net.h5')

######################################
#         LOADING WEIGHTS            #
######################################

# When saving the model and its layers, the SavedModel format stores the class name, call function, losses, 
# and weights (and the config, if implemented). The call function defines the computation graph of the model/layer. 
# In the absence of the model/layer config, the call function is used to create a model that exists like the original model 
# which can be trained, evaluated, and used for inference.
#new_model = tf.keras.models.load_model("net", compile=False)

# OR

# call the build method
#new_model = CustomConvNet() 
#new_model.build((x_train.shape))
# reload the weights 
#new_model.load_weights('net.h5')
