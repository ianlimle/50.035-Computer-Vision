import os
import glob
import cv2
import pandas as pd

# import config file for params
import config


label_dict = {'22': '0',
              '23': '1',
              '24': '2',
              '25': '3',
              '26': '4', 
              '27': '5', 
              '28': '6',
              '29': '7',
              '30': '8',
              '31': '9'}

# CHANGE parameters here
INPUT_DATASETS = [config.TRAIN_IMGS_PATH, config.VAL_IMGS_PATH]
OUTPUT_DATASETS = [config.TRAIN_NEWIMGS_PATH, config.VAL_NEWIMGS_PATH]

for vid_dir, img_dir in zip(INPUT_DATASETS, OUTPUT_DATASETS):

    # list all video paths in videos dataset directory
    vidPaths = [f for f in glob.glob(vid_dir + "**/*/*")]

    for p in vidPaths:
        img = cv2.imread(p)
        #print(p)
        signer = p.split(os.path.sep)[-1].split("_")[0]
        sample = p.split(os.path.sep)[-1].split("_")[1]
        label = p.split(os.path.sep)[-1].split("_")[-2]
        idx = p.split(os.path.sep)[-1].split("_")[-1]

        for k,v in label_dict.items():
            if str(label) == k:
                label = v

        subfolder = str(signer)+'_'+str(sample)

        # create the output dataset directory if it doesn't exist
        if not os.path.exists(os.path.sep.join([img_dir, subfolder])):
            os.makedirs(os.path.sep.join([img_dir, subfolder]))
        
        filename = str(signer)+'_'+str(sample)+'_'+str(label)+'_'+str(idx) 
        newpath = os.path.sep.join([img_dir, subfolder, filename])
        #print(newpath)
        cv2.imwrite(newpath, img)
