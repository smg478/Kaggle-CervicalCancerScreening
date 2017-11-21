# Solution
Kaggle cervical cancer screening competition solution based on keras.

1. Put training images in input folder(input images for keras models are cropped ROIs from full scale images)

2. Finetune imagenet pretrained models using train_finetune_5fld.py (loading all training images to memory)
or train_finetune_from_directory.py (reading images from hard drive at every mini-batch)

3. Predict and generate submissions using prediction_submission.py.

 
