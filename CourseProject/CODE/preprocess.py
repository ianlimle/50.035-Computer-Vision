import os
import glob
import cv2
import av
import pandas as pd
import numpy as np

# import config file for params
import config

# CHANGE parameters here
INPUT_DATASETS = [(config.TRAIN_VID_COLOR_PATH, config.TRAIN_VID_DEPTH_PATH), 
                 (config.VAL_VID_COLOR_PATH, config.VAL_VID_DEPTH_PATH),
                 (config.TEST_VID_COLOR_PATH, config.TEST_VID_DEPTH_PATH)]
OUTPUT_DATASETS = [config.TRAIN_IMGS_PATH, config.VAL_IMGS_PATH, config.TEST_IMGS_PATH]
LABELS_DATASETS = [config.TRAIN_LABELS, config.VAL_LABELS, config.TEST_LABELS]
FPS = config.FPS
NUM_PADDED = config.VID_NUM_PADDED
OUTPUT_FILE_TYPE = config.OUTPUT_FILE_TYPE


for (vid_dir_color,vid_dir_depth), img_dir, labels_file in zip(INPUT_DATASETS, OUTPUT_DATASETS, LABELS_DATASETS):

    # create the output dataset directory if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # read labels file to extract labels
    df = pd.read_csv(labels_file, usecols=[0,1], names=['filename','label'])

    # list all video paths in videos dataset directory
    vidPaths_color = [f for f in glob.glob(vid_dir_color + "**/*")]
    vidPaths_depth = [f for f in glob.glob(vid_dir_depth + "**/*")]

    for color_vid, depth_vid in zip(sorted(vidPaths_color), sorted(vidPaths_depth)):
        basename_color = os.path.basename(color_vid)
        basename_depth = os.path.basename(depth_vid)
        #print(basename_color, basename_depth)

        file_name = color_vid.split(os.path.sep)[-1].split("_color")[-2]
        #print(file_name)
        label = df.loc[df['filename'] == file_name, 'label'].iloc[0]
        #print(label)
        new_filename = str(file_name)+"_"+str(label)
        #print(new_filename)

        # for every video, make a new directory to store all frames for that video
        vid_frames_dir = os.path.sep.join([img_dir, str(label), str(file_name)])
        if not os.path.exists(vid_frames_dir):
            os.makedirs(vid_frames_dir)
        
        try:
            # extract to array
            rgb_arr = []
            container_rgb = av.open(color_vid)
            for packet in container_rgb.demux():
                for frame in packet.decode():
                    rgb_arr.append(np.array(frame.to_image()))
                                    
            depth_arr = []
            container_depth = av.open(depth_vid)
            for packet in container_depth.demux():
                for frame in packet.decode():
                    depth_arr.append(np.array(frame.to_image()))

            count = 1
            # display - correct color orientation
            for i in range(len(rgb_arr)):
                c = cv2.cvtColor(rgb_arr[i], cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(depth_arr[i], cv2.COLOR_BGR2GRAY)
                overlay = cv2.bitwise_and(c,c, mask=gray)
                cv2.imwrite(os.path.join(vid_frames_dir, new_filename+'_'+str(count)+'.png'), overlay)
                print(os.path.join(vid_frames_dir, new_filename+'_'+str(count)+'.'+config.OUTPUT_FILE_TYPE))
                count += 1
        except:
            pass
