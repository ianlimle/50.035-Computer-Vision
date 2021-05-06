import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import classification_report
from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from modules import config
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imutils
import random
import cv2
import os


def plot_training(H, N, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


totalTrain = len(list(imutils.paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(imutils.paths.list_images(config.VAL_PATH)))
totalTest = len(list(imutils.paths.list_images(config.TEST_PATH)))

# construct the image generator for data augmentation
trainAug = ImageDataGenerator(rotation_range=18, 
						 zoom_range=0.15, 
						 width_shift_range=0.2, 
                         height_shift_range=0.2, 
						 shear_range=0.15, 
						 horizontal_flip=True, 
                         fill_mode="nearest")

# we'll be adding mean subtraction to
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BS)


# construct our base model
print("[INFO] compiling base model...")
base_model = ResNet50(input_tensor=Input(shape=(config.WIDTH,config.HEIGHT,config.DEPTH)),
                      include_top=False,
                      weights='imagenet')

# construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(5, 5))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(256, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(config.CLASSES), activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)
print(model.summary())
print(base_model.summary())

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
base_model.trainable = False

# compile the model
print("[INFO] compiling model...")
opt = Adam(lr=config.INIT_LR, 
		   decay=config.INIT_LR / config.WARMUP_EPOCHS)
model.compile(loss="categorical_crossentropy", 
              optimizer=opt, 
			  metrics=['accuracy'])

# train the head_model (top layers added on top of the MobileNetV2 base model) 
print("[INFO] training head model...")
H = model.fit(trainGen,
			  validation_data=valGen,
			  validation_steps=totalVal // config.BS,
			  steps_per_epoch=totalTrain // config.BS,
	          epochs=config.WARMUP_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict(testGen,
	                     steps=(totalTest // config.BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


# show a classification report
print(classification_report(testGen.classes, 
                            predIdxs,
	                        target_names=testGen.class_indices.keys()))

# plot the training history
plot_training(model, config.WARMUP_EPOCHS, config.WARMUP_PLOT_PATH)

# reset the data generators
trainGen.reset()
valGen.reset()

print("No. of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards
train_from_layer = 100
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[train_from_layer:]:
  layer.trainable = True
# show which layers are trainable
for layer in base_model.layers:
	print("{}: {}".format(layer, layer.trainable))

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = Adam(lr=config.INIT_LR, 
		   decay=config.INIT_LR / config.FINETUNE_EPOCHS)
model.compile(loss="categorical_crossentropy", 
			  optimizer=opt,
	          metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
H = model.fit(
	trainGen,
	steps_per_epoch=totalTrain // config.BS,
	validation_data=valGen,
	validation_steps=totalVal // config.BS,
	epochs=config.FINETUNE_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict(testGen,
	                     steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, 
						    predIdxs,
	                        target_names=testGen.class_indices.keys()))

plot_training(H, config.FINETUNE_EPOCHS, config.UNFROZEN_PLOT_PATH)

# serialize the model to disk
print("[INFO] serializing network...")
model.save(config.MODEL_PATH, save_format="h5")