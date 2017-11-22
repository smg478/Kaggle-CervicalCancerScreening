from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                    samplewise_center=False,
                                    samplewise_std_normalization=False,
                                    rescale=1. /255,        # 1. /255
                                    rotation_range=90,
                                    #shear_range=0.1,
                                    zoom_range=0.2,
                                    width_shift_range=0.0,
                                    height_shift_range=0.0,
                                    horizontal_flip= True,
                                    #channel_shift_range=.1,
                                    fill_mode='nearest',
                                    #cval=0.,
                             )

img = load_img('/home/galib/intel/scripts/0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='', save_format='jpeg'):
    i += 1
    if i > 40:
        break  # otherwise the generator would loop indefinitely
