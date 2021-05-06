import os
import glob
import pandas as pd

# import config file for params
import config

# CHANGE parameters here
INPUT_DATASETS = [config.TRAIN_VIDS_PATH, config.VAL_VIDS_PATH]
OUTPUT_DATASETS = [config.TRAIN_IMGS_PATH, config.VAL_IMGS_PATH]
LABELS_DATASETS = [config.TRAIN_LABELS, config.VAL_LABELS]
FPS = config.FPS
NUM_PADDED = config.VID_NUM_PADDED
OUTPUT_FILE_TYPE = config.OUTPUT_FILE_TYPE


for vid_dir, img_dir, labels_file in zip(INPUT_DATASETS, OUTPUT_DATASETS, LABELS_DATASETS):

    # create the output dataset directory if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # read labels file to extract labels
    df = pd.read_csv(labels_file, usecols=[0,1], names=['filename','label'])

    # list all video paths in videos dataset directory
    vidPaths = [f for f in glob.glob(vid_dir + "**/*")]

    for p in vidPaths:

        file_name = p.split(os.path.sep)[-1].split("_color")[-2]
        #print(file_name)

        label = df.loc[df['filename'] == file_name, 'label'].iloc[0]
        #print(label)

        # for every video, make a new directory to store all frames for that video
        vid_frames_dir = os.path.sep.join([img_dir, str(label), str(file_name)])
        if not os.path.exists(vid_frames_dir):
            os.makedirs(vid_frames_dir)

        new_filename = str(file_name)+"_"+str(label)
        #print(new_filename)

        # perform system call to extract frames from each video
        os.system("ffmpeg -i "+str(p)+" -vf fps="+str(FPS)+" "+str(vid_frames_dir)+"/"+str(new_filename)+"_%0"+str(NUM_PADDED)+"d.png")
        # eg. DATA/images/train2/136/signer1_sample1/signer1_sample1_136_001.png
