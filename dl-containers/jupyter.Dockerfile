# NVIDIA CUDA image as a base, runtime flavor because it includes CuDNN
# mark this image as "jupyter-base" so we could use it by name
FROM nvidia/cuda:10.2-runtime AS jupyter-base
WORKDIR /
# install python and its tools
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools
RUN pip3 -q install pip --upgrade
# install all basic packages
RUN pip3 install \
    # Jupyter itself
    jupyter \
    # Numpy and Pandas are required a-priori
    numpy pandas \
    # PyTorch with CUDA 10.2 support and Torchvision
    torch torchvision \
    # Upgraded version of Tensorboard with more features
    tensorboardX \
    # Tensorflow 2.x
    tensorflow

# use the base image by its name - "jupyter-base"
FROM jupyter-base
# install additional packages
RUN pip3 install \
    # Hugging Face Transformers
    transformers \
    # Progress bar to track experiments
    barbar