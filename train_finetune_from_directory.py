import os.path

import numpy as np
from keras import backend as K
from keras import layers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.constraints import maxnorm
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------- Params ---------------------------------------------------------------
img_width, img_height = 224, 224
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

base_model = InceptionV3(weights='imagenet', include_top=False)
# base_model = applications.VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
# base_model = applications.VGG19(weights='imagenet', include_top=False,input_tensor=input_tensor)
# base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling='avg')
# base_model = Xception(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling='avg')

x = base_model.output
x = Flatten()(x)  # Res50, vgg16
# x = Dropout(0.75)(x)
# x = GlobalAveragePooling2D()(x)             # xception
x = layers.noise.GaussianNoise(.5)(x)
x = Dense(2048, kernel_constraint=maxnorm(2.), activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

inputs = input_tensor
model = Model(inputs, x)

# model.load_weights(trained_weights_path)
# model.load_weights(top_model_weights_path)

# for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:18]:
    layer.trainable = False
for layer in model.layers[18:]:
    layer.trainable = True

print("Model created")

model.summary()
optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", "mae"])
print("Finished compiling")
print("Building model...")

print('Model loaded.')

for i, layer in enumerate(model.layers):
    print(i, layer.name)

# ============================================ Data Process ============================================================

train_datagen = ImageDataGenerator(featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_center=False,
                                   samplewise_std_normalization=False,
                                   rescale=1. / 255,
                                   rotation_range=30,
                                   shear_range=0.0,
                                   zoom_range=0.6,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# ====================================== Training ======================================================================
weights_file = "incepv3/incepv3_0.85.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("weights loaded.")

out_dir = "incepv3/"

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=10, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
model_checkpoint = ModelCheckpoint(weights_file, monitor="acc", save_best_only=True, mode='auto')
csv_logger = CSVLogger('training_log.csv')

callbacks = [lr_reducer, model_checkpoint, csv_logger]
class_weight = {0: 3.2,
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
model.save('flow_incepv3_%s.h5' % val_acc)
