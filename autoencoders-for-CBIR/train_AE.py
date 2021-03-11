# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from modules.conv_AE import ConvAutoencoder
from modules import config
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import random
import pickle
import cv2
import os

#########################################################################################
# print("[INFO] loading images...")
# data, labels = [], []
# imagePaths = sorted(list(imutils.paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)

# for imagePath in imagePaths:
# 	img = cv2.imread(imagePath)
# 	img = imutils.resize(img, width=28)
# 	img = img_to_array(img)
# 	data.append(img)

# 	label = imagePath.split(os.path.sep)[-2]
# 	labels.append(label)

# # scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# num_classes = len(set(labels))
# labels = np.array(labels)

# # partition the data into training and testing splits
# (trainX, testX, _, _) = train_test_split(data, labels, test_size=0.15, random_state=42)

# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2, 
#                          height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, 
#                          fill_mode="nearest")

# # construct our convolutional autoencoder
# print("[INFO] compiling autoencoder...")
# autoencoder = ConvAutoencoder.build(28, 28, 3)
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# autoencoder.compile(loss="mse", optimizer=opt, metrics=['accuracy'])

# # train the convolutional autoencoder
# print("[INFO] training autoencoder...")
# H = autoencoder.fit(
# 	aug.flow(trainX, trainX, batch_size=BS),
# 	validation_data=(testX, testX),
# 	steps_per_epoch=trainX.shape[0] // BS,
# 	epochs=EPOCHS,
# 	verbose=1)
########################################################################################

def visualize_predictions(decoded, 
						  ground_truth, 
						  samples=10):

	# initialize our list of output images
	outputs = None

	# loop over our number of output samples
	for i in range(0, samples):
		# grab the original image and reconstructed image
		original = (ground_truth[i] * 255).astype("uint8")
		recon = (decoded[i] * 255).astype("uint8")
		
		# stack the original and reconstructed image side-by-side
		output = np.hstack([original, recon])
		
		# if the outputs array is empty, initialize it as the current
		# side-by-side image display
		if outputs is None:
			outputs = output
		# otherwise, vertically stack the outputs
		else:
			outputs = np.vstack([outputs, output])
	# return the output images
	return outputs


def train_autoencoder(width, 
                      height, 
                      depth, 
					  epochs, 
					  init_lr, 
					  batch_size,
					  model_path,
					  vis_path,
					  plot_path):

	# load the MNIST dataset
	print("[INFO] loading MNIST dataset...")
	((trainX, _), (testX, _)) = mnist.load_data()

	# add a channel dimension to every image in the dataset, 
	# then scale pixel intensities to the range [0, 1]
	trainX = np.expand_dims(trainX, axis=-1)
	testX = np.expand_dims(testX, axis=-1)
	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0

	# construct our convolutional autoencoder
	print("[INFO] building autoencoder...")
	autoencoder = ConvAutoencoder.build(width, 
										height, 
										depth)
	opt = Adam(lr=init_lr, decay=init_lr / epochs)
	autoencoder.compile(loss="mse", optimizer=opt)

	# train the convolutional autoencoder
	H = autoencoder.fit(
		trainX, trainX,
		validation_data=(testX, testX),
		epochs=epochs,
		batch_size=batch_size)

	# use the convolutional autoencoder to make predictions on the
	# testing images, construct the visualization, and then save to disk
	print("[INFO] making predictions...")
	decoded = autoencoder.predict(testX)
	vis = visualize_predictions(decoded, testX)
	cv2.imwrite(vis_path, vis)

	# construct a plot that plots and saves the training history
	N = np.arange(0, epochs)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plot_path)

	# serialize the autoencoder model to disk
	print("[INFO] saving autoencoder...")
	autoencoder.save(model_path, save_format="h5")


def generate_feature_vector_index(model_path, 
								  index_path):
	
	# load the MNIST dataset
	print("[INFO] loading MNIST training split...")
	((trainX, _), (testX, _)) = mnist.load_data()
	
	# PREPROCESS: add a channel dimension to every image in the training split
	# scale the pixel intensities to the range [0, 1]
	trainX = np.expand_dims(trainX, axis=-1)
	trainX = trainX.astype("float32") / 255.0

	# load our autoencoder from disk
	print("[INFO] loading autoencoder model...")
	autoencoder = load_model(model_path)

	# create the encoder model which consists of *just* the encoder
	# portion of the autoencoder
	encoder = Model(inputs=autoencoder.input,
					outputs=autoencoder.get_layer("encoded").output)

	# quantify the contents of our input images using the encoder
	print("[INFO] encoding images...")
	features = encoder.predict(trainX)
	print("[INFO] features shape: ", features.shape)

	# construct index dictionary that maps the index of the MNIST training
	# image to its corresponding latent-space representation
	indexes = list(range(0, trainX.shape[0]))
	data = {"indexes": indexes, "features": features}

	# write the data dictionary to disk
	print("[INFO] saving index...")
	f = open(index_path, "wb")
	f.write(pickle.dumps(data))
	f.close()


if __name__ == "__main__":

	train_autoencoder(width=config.WIDTH, 
					  height=config.HEIGHT, 
					  depth=config.DEPTH, 
					  epochs=config.EPOCHS, 
					  init_lr=config.INIT_LR, 
					  batch_size=config.BS,
					  model_path=config.MODEL_PATH,
				      vis_path=config.VIS_PATH,
					  plot_path=config.PLOT_PATH)

	generate_feature_vector_index(model_path=config.MODEL_PATH,
						          index_path=config.INDEX_PATH)