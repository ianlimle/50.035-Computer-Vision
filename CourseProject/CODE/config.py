import os
from pathlib import Path

# init the base path to the *new* directory 
BASE_PATH = ""
dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = str(Path(dir_path).parents[0])

# init DATA directory
DATA_PATH = os.path.sep.join([BASE_PATH, "DATA"])

# derive the raw videos directory
TRAIN_VID_COLOR_PATH = os.path.sep.join([DATA_PATH, "videos/train_color"])
TRAIN_VID_DEPTH_PATH = os.path.sep.join([DATA_PATH, "videos/train_depth"])

VAL_VID_COLOR_PATH = os.path.sep.join([DATA_PATH, "videos/val_color"])
VAL_VID_DEPTH_PATH = os.path.sep.join([DATA_PATH, "videos/val_depth"])

TEST_VID_COLOR_PATH = os.path.sep.join([DATA_PATH, "videos/test_color"])
TEST_VID_DEPTH_PATH = os.path.sep.join([DATA_PATH, "videos/test_depth"])

# derive the images directory
TRAIN_IMGS_PATH = os.path.sep.join([DATA_PATH, "images/train_10_preprocess"])
VAL_IMGS_PATH = os.path.sep.join([DATA_PATH, "images/val_10_preprocess"])
TEST_IMGS_PATH = os.path.sep.join([DATA_PATH, "images/test_10_preprocess"])

# # derive the new images directory after renaming
# TRAIN_NEWIMGS_PATH = os.path.sep.join([DATA_PATH, "images/train_10_bal_new"])
# VAL_NEWIMGS_PATH = os.path.sep.join([DATA_PATH, "images/val_10_bal_new"])

# derive the labels directory
TRAIN_LABELS = os.path.sep.join([DATA_PATH, "labels/train_labels.csv"])
VAL_LABELS = os.path.sep.join([DATA_PATH, "labels/val_labels.csv"])
TEST_LABELS = os.path.sep.join([DATA_PATH, "labels/test_labels.csv"])

CLASSES = [0,1,2,3,4,5,6,7,8,9]

# num of classes
NUM_CLASSES = len(CLASSES)

# frames per video after padding
FRAMES_PADDED = 10 # was 30 earlier

FPS = 30
VID_NUM_PADDED = 3
OUTPUT_FILE_TYPE = "png"

# initialize the width, height and no. of channels
WIDTH = 256
HEIGHT = 256
DEPTH = 3

# initialize the number of epochs to train for
# initial learning rate, batch size, finetuning epochs
BS = 1
EPOCHS = 30
INIT_LR = 1e-4

# path to tensorboard logs
TENSORBOARD_TRAIN_WRITER = 'output/logs/train'
TENSORBOARD_VAL_WRITER = 'output/logs/val'

MODEL_PATH = "output/model/"
CHKPT_PATH = "output/checkpoints/"
