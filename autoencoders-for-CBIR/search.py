# import the necessary packages
from modules import config
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import os

def euclidean(a, b):
	# compute and return the euclidean distance between two vectors
	return np.linalg.norm(a - b)


def perform_search(queryFeatures, index, maxResults=64):
	# initialize our list of results
	results = []

	# loop over our index
	for i in range(0, len(index["features"])):
		# compute the euclidean distance between our query features
		# and the features for the current image in our index, then
		# update our results list with a 2-tuple consisting of the
		# computed distance and the index of the image
		d = euclidean(queryFeatures, index["features"][i])
		results.append((d, i))
	
    # sort the results and grab the top ones
	results = sorted(results)[:maxResults]
    # return the list of results
	return results


def query(model_path,
		  index_path,
		  samples,
		  top_n):

	# load the MNIST dataset
	((trainX, _), (testX, _)) = mnist.load_data()

	# add a channel dimension to every image in the dataset, then scale
	# the pixel intensities to the range [0, 1]
	trainX = np.expand_dims(trainX, axis=-1)
	testX = np.expand_dims(testX, axis=-1)
	trainX = trainX.astype("float32") / 255.0
	testX = testX.astype("float32") / 255.0

	# load the autoencoder model and index from disk
	print("[INFO] loading autoencoder and index...")
	autoencoder = load_model(model_path)
	index = pickle.loads(open(index_path, "rb").read())

	# create the encoder model which consists of *just* the encoder
	# portion of the autoencoder
	encoder = Model(inputs=autoencoder.input,
					outputs=autoencoder.get_layer("encoded").output)

	# quantify the contents of our input testing images using the encoder
	print("[INFO] encoding testing images...")
	features = encoder.predict(testX)

	# randomly sample a set of testing query image indexes
	queryIdxs = list(range(0, testX.shape[0]))
	queryIdxs = np.random.choice(queryIdxs, size=samples, replace=False)

	# loop over the testing indexes
	for i in queryIdxs:
		# take the features for the current image, find all similar
		# images in our dataset, and then initialize our list of result images
		queryFeatures = features[i]
		results = perform_search(queryFeatures, index, maxResults=top_n)
		images = []
		
		# loop over the results
		for (d, j) in results:
			# grab the result image, convert it back to the range
			# [0, 255], and then update the images list
			image = (trainX[j] * 255).astype("uint8")
			image = np.dstack([image] * 3)
			images.append(image)
		
		# display the query image
		query = (testX[i] * 255).astype("uint8")
		if not os.path.exists("queries"):
			os.makedirs("queries")
		cv2.imwrite("queries/"+str(i)+".png", query)
		
		# build a montage from the results and display it
		montage = build_montages(images, (28, 28), (15, 15))[0]
		if not os.path.exists("results"):
			os.makedirs("results")
		cv2.imwrite("results/"+str(i)+".png", montage)


if __name__ == "__main__":

	query(model_path=config.MODEL_PATH,
		  index_path=config.INDEX_PATH,
		  samples=config.SAMPLES,
		  top_n=config.TOP_N)