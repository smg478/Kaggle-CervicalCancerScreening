
from keras.models import Model, load_model
import os
import six
import numpy as np
import pandas as pd
import cv2
import glob
# ==============================================================================
#file_name = 'final_folder/vgg16_final'
#file_name = 'final_folder/vgg19_final'
#file_name = 'final_folder/res50_final'
#file_name = 'final_folder/inception_final'
file_name = 'final_folder/xception_final'

model = load_model('%s.h5'%file_name)

sample_subm = pd.read_csv("../sample_submission_stg2.csv")
ids = sample_subm['image_name'].values

for id in ids:
    print('Predict for image {}'.format(id))
    files = glob.glob("../input/test/" + id)
    image_list = []
    for f in files:
        image = cv2.imread(f)
        image = cv2.resize(image, (299,299)) # xception, inception = 299, resnet,vgg = 224
        image = image.astype('float32')
        image = image / 255
        image_list.append(image)

    image_list = np.array(image_list)
    #image_list = np.expand_dims(image_list, axis=0)

    predictions = model.predict(image_list, verbose=1, batch_size=1)

    #Denominator = sum(predictions)
    #predictions[0,0] = predictions[0,0] / Denominator
    #predictions[0,1] = predictions[0,1] / Denominator
    #predictions[0,2] = predictions[0,2] / Denominator

    #np.clip(predictions, 0.10, 0.90, out=predictions)

    sample_subm.loc[sample_subm['image_name'] == id, 'Type_1'] = predictions[0, 0]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_2'] = predictions[0, 1]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_3'] = predictions[0, 2]



sample_subm.to_csv("%s_noclip_stg2.csv"%file_name, index=False)
