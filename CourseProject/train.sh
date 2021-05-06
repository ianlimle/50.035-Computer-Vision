#!/bin/sh
cd CODE
python pipeline.py --model vgg16_lstm
python pipeline.py --model vgg16_fpm_lstm
python pipeline.py --model vgg16_lstm_attention
python pipeline.py --model vgg16_fpm_lstm_attention
python pipeline.py --model vgg16_fpm_blstm_attention
