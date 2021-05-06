import os
import PIL
import cv2
import PIL.Image
import glob
import pickle
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed
#from attention import Attention
# import utility files
import config

# get list of subdirectories for training and validation datasets ie. [signer1_sample1, signer1_sample2, ...]
print("[INFO] Retrieving lists of subdirectory names for training & validation...")
if "val_subdir_10_preprocess.txt" in os.listdir():
    with open("val_subdir_10_preprocess.txt", "rb") as fp:   # Unpickling
        val_subdir_ls = pickle.load(fp)
    with open("train_subdir_10_preprocess.txt", "rb") as fp2:   # Unpickling
        train_subdir_ls = pickle.load(fp2)
else:
    val_subdir_ls = [x[0] for x in os.walk(config.VAL_IMGS_PATH) if x[0] != config.VAL_IMGS_PATH]
    train_subdir_ls = [x[0] for x in os.walk(config.TRAIN_IMGS_PATH) if x[0] != config.TRAIN_IMGS_PATH]
    with open("val_subdir_10_preprocess.txt", "wb") as fp:   #Pickling
        pickle.dump(val_subdir_ls, fp)
    with open("train_subdir_10_preprocess.txt", "wb") as fp2:   #Pickling
        pickle.dump(train_subdir_ls, fp2)
print("[INFO] No. of train samples: ", len(train_subdir_ls), 
      "No. of val samples: ", len(val_subdir_ls))


def parse_image(filepath):
    # retrieve label and image data
    label = int(filepath.split(os.path.sep)[-1].split('_')[-2])
    image = cv2.imread(filepath)
    return image, label


def preprocess(img, height, width, augment=False):
    # apply resizing & rescaling
    resize_and_rescale = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(height, width),
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    img = resize_and_rescale(img)
    # apply data augmentation only to training set
    if augment:
        img = tf.image.adjust_brightness(img, 0.4)
        img = tf.image.adjust_contrast(img, 0.2)
        img = tf.image.adjust_hue(img, 0.2)
        img = tf.image.adjust_saturation(img, 2)
    return img


def train_image_gen():
    # select a no. of subfolders (no. of video samples) for the batch
    batch_subdir_ls = np.random.choice(a=train_subdir_ls, size=len(train_subdir_ls))
    #batch_subdir_ls = np.random.choice(a=train_subdir_ls, size=config.TRAIN_SAMPLES)

    # loop thru each subfolder (video sample)
    for _, subdir in enumerate(batch_subdir_ls):
        # retrieve list of image filepaths for a video sample 
        imgPaths = [f for f in glob.glob(subdir + "**/*")]

        x_vid, y_vid = [], []

        try:
            # loop thru the filepaths to obtain the img data and label
            for path in range(0, len(imgPaths), int(len(imgPaths)//10)):
            #for path in range(0, len(imgPaths)):
                img, label = parse_image(imgPaths[path])
                x_vid.append(img)
                
            ###################################
            #     Data Augmentation START     #
            ###################################
            x_vid_arr = np.array(x_vid, dtype='float32')
            x_vid_arr = preprocess(x_vid_arr,  
                                height=config.HEIGHT, 
                                width=config.WIDTH,
                                augment=False)
            y_vid.append(label)
            y_vid_arr = np.array([y_vid])
            #################################
            #     Data Augmentation END     #
            #################################


            #################################
            #          WITH Padding         #
            #################################
            # pad the array to follow a predefined no. of frames per video
            # image tensors of shape (1, 30, 256, 256, 3) means
            # batch size/no. of videos: 1
            # no. of frames per video AFTER padding: 30 
            # frame height: 256
            # frame width: 256
            # no. of channels: 3
            '''
            res_x = np.zeros((config.FRAMES_PADDED, 
                            x_vid_arr.shape[1], 
                            x_vid_arr.shape[2], 
                            x_vid_arr.shape[3]))

            res_x[:x_vid_arr.shape[0], 
                :x_vid_arr.shape[1], 
                :x_vid_arr.shape[2], 
                :x_vid_arr.shape[3]] = x_vid_arr[:min(config.FRAMES_PADDED, x_vid_arr.shape[0]),:,:,:]        

            res_y = np.zeros((1,config.FRAMES_PADDED))

            res_y[:,:y_vid_arr.shape[1]] = y_vid_arr[:,:min(config.FRAMES_PADDED, y_vid_arr.shape[1])]

            # create the batch img array and batch labels array 
            batch_x = res_x                       
            batch_x = np.expand_dims(batch_x, axis=0)  #for shape: (1, N, H, W, C) where N is no. of frames
            batch_y = res_y                            #for shape: (1, N, 226) where N is no. of frames
            '''
            #################################
            #           WITH Padding        #
            #################################


            #################################
            #            NO Padding         #
            #################################
            batch_x = x_vid_arr                        
            batch_x = np.expand_dims(batch_x, axis=0)  #for shape: (1, N, H, W, C) where N is no. of frames
            batch_y = np.array(y_vid)            
            batch_y = np.expand_dims(batch_y, axis=0)  #for shape: (1, N, 226) where N is no. of frames
            #################################
            #            NO Padding         #
            #################################

            # convert class vectors (integers from 0 to num_classes) into one-hot encoded class matrix 
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=config.NUM_CLASSES) 

        except:
            continue
        
        #return (batch_x, batch_y)
        yield (batch_x, batch_y)


def val_image_gen():
    # select a no. of subfolders (no. of video samples) for the batch
    batch_subdir_ls = np.random.choice(a=val_subdir_ls, size=len(val_subdir_ls))
    #batch_subdir_ls = np.random.choice(a=val_subdir_ls, size=config.VAL_SAMPLES)

    # loop thru each subfolder (video sample)
    for _, subdir in enumerate(batch_subdir_ls):
        # retrieve list of image filepaths for a video sample 
        imgPaths = [f for f in glob.glob(subdir + "**/*")]

        x_vid, y_vid = [], []

        try:
            # loop thru the filepaths to obtain the img data and label
            for path in range(0, len(imgPaths), int(len(imgPaths)//10)):
            #for path in range(0, len(imgPaths)):
                img, label = parse_image(imgPaths[path])
                x_vid.append(img)
                
            ###################################
            #     Data Augmentation START     #
            ###################################
            x_vid_arr = np.array(x_vid, dtype='float32')
            x_vid_arr = preprocess(x_vid_arr,  
                                height=config.HEIGHT, 
                                width=config.WIDTH,
                                augment=False)
            y_vid.append(label)
            y_vid_arr = np.array([y_vid])
            #################################
            #     Data Augmentation END     #
            #################################


            #################################
            #          WITH Padding         #
            #################################
            # pad the array to follow a predefined no. of frames per video
            # image tensors of shape (1, 30, 256, 256, 3) means
            # batch size/no. of videos: 1
            # no. of frames per video AFTER padding: 30 
            # frame height: 256
            # frame width: 256
            # no. of channels: 3
            '''
            res_x = np.zeros((config.FRAMES_PADDED, 
                            x_vid_arr.shape[1], 
                            x_vid_arr.shape[2], 
                            x_vid_arr.shape[3]))

            res_x[:x_vid_arr.shape[0], 
                :x_vid_arr.shape[1], 
                :x_vid_arr.shape[2], 
                :x_vid_arr.shape[3]] = x_vid_arr[:min(config.FRAMES_PADDED, x_vid_arr.shape[0]),:,:,:]        

            res_y = np.zeros((1,config.FRAMES_PADDED))

            res_y[:,:y_vid_arr.shape[1]] = y_vid_arr[:,:min(config.FRAMES_PADDED, y_vid_arr.shape[1])]

            # create the batch img array and batch labels array 
            batch_x = res_x                      
            batch_x = np.expand_dims(batch_x, axis=0)  #for shape: (1, N, H, W, C) where N is no. of frames
            batch_y = res_y                            #for shape: (1, N, 226) where N is no. of frames
            '''
            #################################
            #           WITH Padding        #
            #################################


            #################################
            #            NO Padding         #
            #################################
            batch_x = x_vid_arr                        
            batch_x = np.expand_dims(batch_x, axis=0)  #for shape: (1, N, H, W, C) where N is no. of frames
            batch_y = np.array(y_vid)            
            batch_y = np.expand_dims(batch_y, axis=0)  #for shape: (1, N, 226) where N is no. of frames
            #################################
            #            NO Padding         #
            #################################

            # convert class vectors (integers from 0 to num_classes) into one-hot encoded class matrix 
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=config.NUM_CLASSES)

        except:
            continue

        #return (batch_x, batch_y)
        yield (batch_x, batch_y)
