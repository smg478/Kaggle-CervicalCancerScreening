## Kaggle cervical cancer screening competition solution based on keras.

This repository contains the keras solution files of the competition. Results produced from the pipeline was ranked 7th in the competition.

The problem was solved in 2 stages:

1. Train an object detector ([R-FCN](https://github.com/YuwenXiong/py-R-FCN) - Caffe) and crop out ROIs autometically.
2. Classify the cropped ROIs using a seperate model (VGG16, VGG19, InceptionV3, Xception, Res50) ([Keras](https://keras.io/applications/)).

This implementation only contains the 2nd part of the solution. Therefore, only keras models are available here. Thanks to [ZFTurbo](https://github.com/ZFTurbo) for his [keras implementation](https://www.kaggle.com/zfturbo/fishy-keras-lb-1-25267) of Fisheries Monitoring classification challange.

## Requirements

Keras 1.0 w/ TF backend

sklearn

cv2


## Usage

### Data

Place 'train'and 'test' data folders in the 'input' folder.

### Train

Run train_finetune_5fld.py (loads all training images to memory)

or 

Run train_finetune_from_directory.py (reads images from hard drive at every mini-batch) to train on imagenet pretrained models.

### Test and submit

Run prediction_submission.py to predict and generate submissions
