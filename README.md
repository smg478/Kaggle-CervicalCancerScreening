[//]: # (Image References)
[image1]: ./ref/cervix.PNG


# Cervical cancer screening

Warning: This data contains graphic contents that some may find disturbing.

![alt text][image1]

The goal of this challenge was to identify woman's cervix type based on images. More specifically, three types of cervix need to be classified here which can lead to real-time determinations about patients’ treatment eligibility based on cervix type.
This repository contains the keras solution files of the competition. Results produced from the pipeline was ranked 7th in the competition.

# The problem

- Imbalanced class.
- Too few training images. External data can not be found easily.
- Area of interest was smaller than the image provided.


The problem was approached in 2 stages:

1. Train an object detector ([R-FCN](https://github.com/YuwenXiong/py-R-FCN) - Caffe) and crop out ROIs autometically.
2. Classify the cropped ROIs using a seperate model (VGG16, VGG19, InceptionV3, Xception, Res50) ([Keras](https://keras.io/applications/)).

This repository only contains the 2nd part of the solution. Therefore, only keras implementation is available here. Thanks to [ZFTurbo](https://github.com/ZFTurbo) for his [keras implementation](https://www.kaggle.com/zfturbo/fishy-keras-lb-1-25267) on Fisheries Monitoring classification challange.

# Requirements

- tensorflow
- Keras
- scipy
- numpy
- pandas
- opencv-python
- scikit-image


# Data

Place 'train'and 'test' data folders in the 'input' folder. Input folder also contains the manual annotation of the ROI's for R-CNN training.

# Train

Run train_finetune_5fld.py (loads all training images to memory)

or 

Run train_finetune_from_directory.py (reads images from hard drive in mini-batches) to train on imagenet pretrained models.

# Test and submit

Run submit.py to predict and generate submissions
