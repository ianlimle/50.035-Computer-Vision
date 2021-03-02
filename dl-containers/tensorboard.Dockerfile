# Tensorboard for experiments tracking
# To build: sudo docker build -t dl-tensorboard:latest -f tensorboard.Dockerfile .
# To run interactively: sudo docker run --rm -it dl-tensorboard:latest /bin/bash

FROM python:3.8-slim-buster
RUN pip3 install tensorboard