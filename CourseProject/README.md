# CV2021
This README documents work for the 50.035 Computer Vision course taken in Spring 2021. Specifically, the course project requires the team to propose, design and build a meaningful solution using Computer Vision concepts. The team has selected the research track (general) where we are looking to study the following paper, *AUTSL: A Large Scale Multi-Model Turkish Sign Language Dataset and Baseline Methods (Sincan & Keles, 2020)*, implement some algorithms, perform experiments, and propose improvements on the algorithms e.g., training algorithm, network architecture. The expectation is that by the end of the project, the team has gained reasonable understanding of the selected paper(s) and performed some related experiments.   

```
├── CODE
|   ├── __init__.py
|   ├── config.py
|   ├── extract_frames.py
│   ├── models.py
│   ├── pipeline.py
|   ├── preprocess.py
|   ├── rename_labels.py
|   ├── utils.py
│   └── video_to_frames.py
|
├── DATA (will be ignored)
│   ├── images
|   |   ├── val
|   |   |   ├── signer1_sample1
|   |   |   |   ├── signer1_sample1_136_01.png
|   |   |   |   └── ...
|   |   |   └── ...
|   |   └── train
|   |       ├── signer1_sample2 
|   |       |   ├── signer1_sample2_100_01.png
|   |       |   └── ...
|   |       └── ...
│   ├── labels
|   |   ├── train_labels.csv
|   |   └── val_labels.csv
|   └── videos
|       ├── val
|       |   ├── signer1_sample1_color.mp4
|       |   └── ...
|       └── train
|           ├── signer1_sample2_color.mp4
|           └── ...
├── .gitignore
└── README.md
```

* Note that the DATA folder will be ignored, and must be created locally

## Pre-requisites
- Python >= 3.6
- Pillow==6.2.1
- attention==4.0
- av==8.0.3
- glob2==0.7
- matplotlib==3.2.1
- opencv-contrib-python==4.2.0
- pandas==0.25.3
- tensorflow==2.2.0 / tensorflow-gpu==2.2.0
```
$ pip install -r requirements.txt
```

## Download AUTSL dataset
```
$ wget -i download.txt
```

## Decompress the data
All files are encrypted. To decompress the data, use the associated keys:
- Train data: MdG3z6Eh1t
- Validation data: bhRY5B9zS2
- Validation labels: zYX5W7fZ
- Test data: ds6Kvdus3o
```
$ sudo apt-get update
$ sudo apt-get install p7zip-full
$ 7z x train_set_vfbha39.zip.001
OR
$ 7z x val_set_bjhfy68.zip.001
OR 
$ 7z x test_set_xsaft57.zip.001
```
**Ensure data folders adopt the same naming and file structure as that of README.md. Make changes in ./CODE/config.py if necessary.**

## Extract input frames from videos
```
(WITHOUT Background Masking) 
$ cd CODE
$ python extract_frames.py

(WITH Background Masking)
$ cd CODE
$ python preprocess.py
```

## Training
``` 
$ cd CODE
$ python pipeline.py --model [MODEL_NAME]
```
MODEL_NAME must be one of the following:
- vgg16_lstm
- vgg19_lstm
- resnet_lstm
- vgg16_fpm_lstm
- vgg19_fpm_lstm
- resnet_fpm_lstm
- vgg16_lstm_attention
- vgg19_lstm_attention
- resnet_lstm_attention
- vgg16_fpm_lstm_attention
- vgg19_fpm_lstm_attention
- resnet_fpm_lstm_attention
- vgg16_fpm_blstm_attention
- vgg19_fpm_blstm_attention
- resnet_fpm_blstm_attention
``` 
$ ./train.sh
```
We have provided a convenience script to train for a single backbone variant i.e. VGG16 / VGG19 / ResNet50

## Acknowledgements
```
@ARTICLE{9210578,  
author={Sincan, Ozge Mercanoglu and Keles, Hacer Yalim},  
journal={IEEE Access},   
title={AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset and Baseline Methods},   
year={2020},  
volume={8},  
number={},  
pages={181340-181355},  
doi={10.1109/ACCESS.2020.3028072}
}
```
