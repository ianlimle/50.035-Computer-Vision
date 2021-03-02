import os

# initilaize the path to the original input dataset
ORIG_INPUT_DATASET = "" 

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = ""

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8
# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1
# define the names of the classes
CLASSES = [""]

# initialize the with, height and no. of channels
WIDTH = 224
HEIGHT = 224
DEPTH = 3

# initialize the number of epochs to train for
# initial learning rate and batch size
WARMUP_EPOCHS = 50
FINETUNE_EPOCHS = 20
INIT_LR = 1e-3
BS = 32

# path to output trained autoencoder
MODEL_PATH = "outputs/model.h5"
# path to output plot file
WARMUP_PLOT_PATH = "outputs/head_training.png"
UNFROZEN_PLOT_PATH = "outputs/fine_tuned.png"

# path to test image
IMAGE = ""
