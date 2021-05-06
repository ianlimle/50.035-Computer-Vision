import os
import cv2
import math
import importlib
from imutils import paths

import config
importlib.reload(config)

frame_rate = 3 #save every X frames

attempt = "cv2_3"
dataset = "val_set_SUBSET"
INPUT_DATASET = os.path.sep.join([config.DATA_PATH, dataset])
print(INPUT_DATASET)
paths = list(paths.list_files(INPUT_DATASET))
vidPaths = [f for f in paths if f.endswith(".mp4")]
vidPaths.sort()
print(vidPaths)

for path in vidPaths:

    count = 0
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )

    label = path.split(os.path.sep)[-1].strip(".mp4")
    output_path = os.path.sep.join([config.DATA_PATH, dataset, attempt, label])
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # frameRate = cap.get(frame_rate)
    frameRate = frame_rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        # print(frameRate, math.floor(frameRate))
        # print(frameId)
        if (frameId % math.floor(frameRate) == 0):
            filename ="frame%d.jpg" % count;count+=1
            cv2.imwrite(os.path.sep.join([output_path, filename]), frame)
    cap.release()
    print ("Done!")
