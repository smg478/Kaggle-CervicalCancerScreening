# https://www.kaggle.com/zfturbo/fishy-keras-lb-1-25267

from __future__ import division

import datetime
import glob
import os
import os.path
import random
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from keras import __version__ as keras_version
from keras import applications
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
np.random.seed(2016)
random.seed(2016)


def get_im_cv2(path):
    img = cv2.imread(path)
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
        path = os.path.join('..', 'input', 'train_cropped', fld, '*.jpg')
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
    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 3)  # Train class = 3
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.astype('float32')
    test_data = test_data / 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y = layers.noise.GaussianNoise(0.5)(y)

    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)

    predictions = Dense(3, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    # model.load_weights('densenet/densenet_1500Crop_rot90_best_1.h5',by_name=False)
    # print('Weights loaded.')

    # for layer in model.layers[:85]:
    #        layer.trainable = False
    # for layer in model.layers[85:]:
    #        layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    model.summary()

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=5):
    batch_size = 64
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
        # if os.path.exists(weights_file):
        #   model.load_weights(weights_file)
        #   print("weights loaded.")

        out_dir = "inceptionv3/"

        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.5e-6)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50)
        model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True, mode='auto')
        csv_logger = CSVLogger('cervical_log_%s.csv' % num_fold)

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
                                callbacks=[lr_reducer, csv_logger, early_stopper])

            model.save('inceptionv3/inceptionv3_%s.h5' % num_fold)

        else:
            print('Using real-time data augmentation.')
            datagen = ImageDataGenerator(featurewise_center=False,
                                         rotation_range=90,
                                         width_shift_range=0.0,
                                         height_shift_range=0.0,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                         # shear_range=0.2,
                                         zoom_range=0.0,
                                         fill_mode='nearest')

            datagen.fit(X_train)
            class_weight = {0: 3.2,
                            1: 1.,
                            2: 1.8}
            history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                          samples_per_epoch=X_train.shape[0],
                                          validation_data=(X_valid, Y_valid),
                                          epochs=nb_epoch, verbose=1, max_q_size=100,
                                          class_weight=class_weight,
                                          callbacks=[csv_logger, lr_reducer, model_checkpoint, early_stopper])

            model.save('vgg16/vgg16_1500Crop_rot90_%s.h5' % num_fold)

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
