
import os.path
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras import backend as K

from keras import applications
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception


from keras.models import Model
from keras.models import Sequential
from keras import layers

from keras.layers import Input, Activation, merge, Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

#from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm

print np.__version__
#print theano.__version__

#---------------------------------------- Ground truth -----------------------------------------------------------------

# dimensions of our images.
img_width, img_height = 224,224

# train - val directory
train_data_dir = '../Full_data/train'
validation_data_dir = '../cropped_data/val'


nb_train_samples = 1500
nb_validation_samples = 1500
nb_epoch = 100
batch_size = 64

input_tensor = Input(shape=(img_width, img_height, 3))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# ======================================= Model Definitions ============================================================

# create the base pre-trained model

# base_model = InceptionV3(weights='imagenet', include_top=False)
# 299x299 Both th and tf and dim ordering
# base_model = applications.VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
# 224x224 Both th and tf and dim ordering
base_model = applications.VGG19(weights='imagenet', include_top=False,input_tensor=input_tensor)
# 224x224 Both th and tf and dim ordering
#base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling='avg')
# 224x224 both th and tf and dim ordering
#base_model = Xception(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling='avg')
# 299x299 only tf


# add a global spatial average pooling layer
x = base_model.output

x = Flatten()(x)  # Res50, vgg16
#x = Dropout(0.75)(x)
#x = GlobalAveragePooling2D()(x)             # xception
x = layers.noise.GaussianNoise(.5)(x)
x = Dense(2048, kernel_constraint= maxnorm(2.), activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)


x = Dense(3, activation='softmax')(x)

# this is the model we will train
inputs = input_tensor

model = Model(inputs, x)

#model.load_weights(trained_weights_path)
#model.load_weights(top_model_weights_path)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

#for layer in base_model.layers:                     # vgg19 11(16:last) blk) layer freeze, Res50 91 layer (140) freeze, xception 85
#    layer.trainable = False
for layer in model.layers[:18]:
    layer.trainable = False
for layer in model.layers[18:]:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   #Fine-tune fc layer
#model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy","mae"])
print("Finished compiling")
print("Building model...")

print('Model loaded.')

for i, layer in enumerate(model.layers):
    print(i, layer.name)




# ----------------------- pre-trained keras end -----------------------------------------------------------------------


# ============================================ model definition end ====================================================

# ============================================ Data Process ============================================================



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_center=False,
                                    samplewise_std_normalization=False,
                                    rescale=1. /255,        # 1. /255
                                    rotation_range=30,
                                    shear_range=0.0,
                                    zoom_range=0.6,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip= True,
                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(samplewise_center=False,
                                  rescale=1. /255)

train_generator = train_datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
                                                    validation_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# ====================================== Data process end ==============================================================

# ====================================== Training ======================================================================

# Parameters for training
# Load model
weights_file="vgg19/vgg19_1500_Fullimg_halfRes_outFlatten_Drop50_Noise50_Dense2048_hflip_zoom20_rot30_sft10_batch64.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("weights loaded.")

out_dir="vgg19/"

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=10, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
model_checkpoint = ModelCheckpoint(weights_file, monitor="acc", save_best_only=True, mode='auto')
csv_logger = CSVLogger('training_log.csv')

callbacks=[ lr_reducer, model_checkpoint, csv_logger]

class_weight = {0: 3.2,                     #3.2=main set
                1: 1.0,
                2: 1.8}

history = model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=nb_epoch,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size,
                verbose=1,
                class_weight=class_weight,
                callbacks=callbacks)


val_acc = history.history['val_acc'][-1]

# save model
#model.save('flow_xception__rot30_zoom70_sh20_batch08_drop50_val_acc_%s.h5'%val_acc)                #save trained model
#model.save('finetune_fc67_Res50.h5')                                  #save fc layer for pre-trained model

# =================================== Plot training performance =======================================================

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

#==================================  end plot training performance ====================================================