
from __future__ import division

import os.path
import densenet
#import densenet_fcn


import os
import six
import numpy as np
import pandas as pd
import cv2
import glob
import random
import datetime
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

np.random.seed(2016)
random.seed(2016)

from keras import applications
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.models import Model, load_model
from keras.models import Sequential
from keras import layers

from keras.layers import Input, Activation, merge, Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

#from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adagrad, Adam, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator

#from keras import backend as K
#K.set_image_dim_ordering('th')


from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#------------------------------------------------------------------------------------------------


# ------------------------------------ From Fish Keras ------------------------------------------


def get_im_cv2(path):
    img = cv2.imread(path)

    #print img.shape
    #print type(img)
    #resized = np.resize(img, (224, 224,3))
    #plt.imshoe(img)

    resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train_cropped', fld, '*.jpg')        # input folder
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    #print('Reshape...')
    #train_data = train_data.transpose((0,3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 3)   # Train class = 3

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    #test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

# ------------------------------- create model ---------------

#--------------------- From scratch -------------------------
'''
def create_model():
    input_shape = (224, 224, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    return model



'''
#-------------------------- Pretrained ---------------------------------------------------------------------------------


def create_model():

    input_tensor = Input(shape=(224,224, 3))

    # create the base pre-trained model
    #base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)    #299x299
    base_model = applications.VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)            #224x224
    #base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)       #224x224
    #base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)        #299x299

    # DenseNet-121
    #depth = 22  # 40
    #nb_dense_block = 3 #3
    #growth_rate = 8  # 12
    #nb_filter = 16  # 16
    #dropout_rate = 0.0  # 0.0 for data augmentation
    #bottleneck = True
    #reduction = 0.0
    #base_model = densenet.DenseNet((224,224,3), classes=3, depth=depth, nb_dense_block=nb_dense_block,
    #                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
    #                          bottleneck=bottleneck, reduction=reduction, include_top=False, weights=None,
    #                          input_tensor=input_tensor)    # output: globalAveragePooling2D#

    #base_model = densenet_fcn.DenseNetFCN((128,128,3), nb_dense_block=5, growth_rate=16, nb_layers_per_block=4,
    #            reduction=0.0, dropout_rate=0.0, weight_decay=1E-4, init_conv_filters=48,
    #            include_top=False, weights=None, input_tensor=input_tensor, classes=3, activation='softmax',
    #            upsampling_conv=128, upsampling_type='subpixel')

    # add a global spatial average pooling layer
    y = base_model.output
    y = Flatten()(y)  # Res50, vgg16, vgg19
    # y = GlobalAveragePooling2D()(y)    # xception,inception
    y = layers.noise.GaussianNoise(0.5)(y)

    y = Dense(2048)(y)
    #y = BatchNormalization()(x)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    #y = Dropout(0.5)(y)


    predictions = Dense(3, activation='softmax')(y)

    #x = layers.Dense(1024)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.advanced_activations.LeakyReLU()(x)
    #x = layers.Dropout(0.25)(x)

    # this is the model we will train
    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    #model.load_weights('densenet/densenet_1500Crop_rot90_best_1.h5',by_name=False)
    #model.load_weights('xception/xception_1500Crop_1.h5',by_name=False)
    #model.load_weights('models/finetune_fc_vgg19.h5',by_name=True)
    #print('Weights loaded.')

    #train_top_only = False

    #if train_top_only:
    #for layer in base_model.layers:                     # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
    #        layer.trainable = False
    #else:
    #for layer in model.layers[:85]:
    #        layer.trainable = False
    #for layer in model.layers[85:]:
    #        layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   #Fine-tune fc layer
    #model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    #model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    #model.summary()

    return model



#-----------------------------------------------------------------------------------------------------------------------

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=5):
    # input image dimensions
    batch_size = 64    # vgg=64, resnet50, inception, xception = 8
    nb_epoch = 200
    random_state = 51
    first_rl = 96
    data_augmentation = True


    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()


        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        weights_file = 'vgg16/vgg16_1500Crop_rot90_best_%s.h5' % num_fold
        #if os.path.exists(weights_file):
        #   model.load_weights(weights_file)
        #   print("weights loaded.")

        out_dir = "xception/"


        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.5e-6)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50)
        model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True, mode='auto')
        csv_logger = CSVLogger('cervical_log_%s.csv'%num_fold)

        if not data_augmentation:
            print('Not using data augmentation.')

            class_weight = {0: 3.1,
                            1: 1.,
                            2: 1.7}

            history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(X_valid, Y_valid),
                      shuffle=True,
                      class_weight=class_weight,
                      callbacks=[lr_reducer, csv_logger,early_stopper])

            # list all data in history
            print(history.history.keys())
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            model.save(
                'xception/xception_1500Crop_%s.h5' % num_fold)

        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:

            datagen = ImageDataGenerator(#rescale=1. /255,
                featurewise_center=False,  # set input mean to 0 over the dataset
                #samplewise_center=True,  # set each sample mean to 0
                #featurewise_std_normalization=True,  # divide inputs by std of the dataset
                #samplewise_std_normalization=True,  # divide each input by its std
                #zca_whitening=True,  # apply ZCA whitening
                rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,
                #shear_range=0.2,
                zoom_range=0.0,
                fill_mode='nearest')  # randomly flip images

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(X_train)

            class_weight = {0: 3.2,
                            1: 1.,
                            2: 1.8}
            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                validation_data=(X_valid, Y_valid),
                                epochs=nb_epoch, verbose=1, max_q_size=100,
                                class_weight=class_weight,
                                callbacks=[csv_logger,lr_reducer,model_checkpoint,early_stopper])
            '''
            # list all data in history
            print(history.history.keys())
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            '''
            # Save classification model weight
            model.save('vgg16/vgg16_1500Crop_rot90_%s.h5'%num_fold)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)


    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = '_' + str(np.round(score, 3)) + '_flds_' + str(nfolds) + '_eps_' + str(nb_epoch) + '_fl_' + str(
        first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)



if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5
    info_string, models = run_cross_validation_create_models(num_folds)

    run_cross_validation_process_test(info_string, models)