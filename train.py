import os
import tensorflow as tf
import numpy as np
import math
import timeit

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


class CustomConvNet(tf.keras.Model):
    def __init__(self, num_classes, channel_dim=-1):
        super(CustomConvNet, self).__init__()
        ############################################################################
        # TODO: Construct a model that performs well on CIFAR-10                   #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # initialize the first CONV
        self.conv_init = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")
        
        # initialize CONV_1x1_@ where @: no. of filters
        self.conv_1x1_32 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding="same")
        self.conv_1x1_96 = tf.keras.layers.Conv2D(96, (1, 1), strides=(1, 1), padding="same")
        self.conv_1x1_80 = tf.keras.layers.Conv2D(80, (1, 1), strides=(1, 1), padding="same")
        self.conv_1x1_48 = tf.keras.layers.Conv2D(48, (1, 1), strides=(1, 1), padding="same")
        self.conv_1x1_112 = tf.keras.layers.Conv2D(112, (1, 1), strides=(1, 1), padding="same")
        self.conv_1x1_176 = tf.keras.layers.Conv2D(176, (1, 1), strides=(1, 1), padding="same")
        
        # initialize CONV_3x3_@ where @: no. of filters
        self.conv_3x3_32 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")
        self.conv_3x3_48 = tf.keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding="same")
        self.conv_3x3_64 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")
        self.conv_3x3_80 = tf.keras.layers.Conv2D(80, (3, 3), strides=(1, 1), padding="same")
        self.conv_3x3_96 = tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding="same")
        self.conv_3x3_160 = tf.keras.layers.Conv2D(160, (3, 3), strides=(1, 1), padding="same")
        
        # initialize downsampling CONV_3x3 
        self.conv_down_80 = tf.keras.layers.Conv2D(80, (3, 3), strides=(2, 2), padding="valid")
        self.conv_down_96 = tf.keras.layers.Conv2D(96, (3, 3), strides=(2, 2), padding="valid")
        
        # initialize the downsampling POOL
        self.maxpool = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))
        
        self.bn = tf.keras.layers.BatchNormalization(axis=channel_dim)
        
        self.relu = tf.keras.layers.Activation("relu")
        
        self.avepool = tf.keras.layers.AveragePooling2D((7,7))
        
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense = tf.keras.layers.Dense(num_classes)
        
        self.softmax = tf.keras.layers.Activation("softmax")
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                            END OF YOUR CODE                              #
        ############################################################################
    
    def call(self, input_tensor, channel_dim=-1, training=True):
        ############################################################################
        # TODO: Construct a model that performs well on CIFAR-10                   #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        x = self.conv_init(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
       
        # build 2 Inception modules followed by a downsample module
        # 1st Inception module
        x1 = self.conv_1x1_32(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_32(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # 2nd Inception module
        x1 = self.conv_1x1_32(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_48(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # downsample module
        x1 = self.conv_down_80(x)
        x2 = self.maxpool(x)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        
        
        # build 4 Inception modules followed by a downsample module
        # 1st Inception module
        x1 = self.conv_1x1_112(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_48(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # 2nd Inception module
        x1 = self.conv_1x1_96(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_64(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # 3rd Inception module
        x1 = self.conv_1x1_80(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_80(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # 4th Inception module
        x1 = self.conv_1x1_48(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_96(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # downsample module
        x1 = self.conv_down_96(x)
        x2 = self.maxpool(x)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        
        
        # build 2 Inception modules followed by global POOL and dropout
        # 1st Inception module
        x1 = self.conv_1x1_176(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_160(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # 2nd Inception module
        x1 = self.conv_1x1_176(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv_3x3_160(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x = tf.keras.layers.concatenate([x1, x2], axis=channel_dim)
        # global pool
        x = self.avepool(x)
        # dropout
        x = self.dropout(x)
        
        
        # build the softmax classifier
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                            END OF YOUR CODE                              #
        ############################################################################
        
        return x


def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test



num_epochs = 10
init_lr = 1e-3
batch_size = 128

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
              "dog", "frog", "horse", "ship", "truck"]

# load the CIFAR-10 dataset
print("loading CIFAR-10 dataset...")
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

# convert the labels from integers to vectors
print("converting labels from integers to vectors...")
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)
y_test = lb.transform(y_test)

# construct the image generator for data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=18, 
                                                      zoom_range=0.15, 
                                                      width_shift_range=0.2, 
                                                      height_shift_range=0.2, 
                                                      shear_range=0.15, 
                                                      horizontal_flip=True, 
                                                      fill_mode="nearest")

# initialize the optimizer and compile the model
print("compiling model...")
#model = CustomConvNet(len(labelNames))
model = CustomConvNet(len(labelNames))

opt = tf.keras.optimizers.Adam(lr=init_lr, decay=init_lr/num_epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("training network...")
H = model.fit(aug.flow(X_train, y_train, batch_size=batch_size),
              validation_data=(X_val, y_val),
              steps_per_epoch=X_train.shape[0] // batch_size,
              epochs=num_epochs)

