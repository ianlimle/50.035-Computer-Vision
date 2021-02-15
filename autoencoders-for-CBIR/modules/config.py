# initialize the with, height and no. of channels
WIDTH = 28
HEIGHT = 28
DEPTH = 1

# initialize the number of epochs to train for
# initial learning rate and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32

# path to output trained autoencoder
MODEL_PATH = "model_outputs/autoencoder.h5"
# path to output reconstructed visualization file
VIS_PATH = "model_outputs/reconstruction.png"
# path to output plot file
PLOT_PATH = "model_outputs/plot.png"
# path to output features index file
INDEX_PATH = "model_outputs/index.pickle"

# no. of test queries to perform
SAMPLES = 10
# top-N results for a query
TOP_N = 64