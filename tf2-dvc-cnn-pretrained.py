# # Dogs-vs-cats classification with CNNs
#
# In this notebook, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of dogs from images of cats using
# TensorFlow 2.0 / Keras. This notebook is largely based on the blog
# post [Building powerful image classification models using very
# little data]
# (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
#
# **Note that using a GPU with this notebook is highly recommended.**
#
# First, the needed imports.

import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import random
import pathlib

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Activation, Dropout, Conv2D,
                                    Flatten, MaxPooling2D, InputLayer)
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  array_to_img, 
                                                  img_to_array, load_img)
from tensorflow.keras import applications, optimizers

from tensorflow.keras.callbacks import TensorBoard

import numpy as np

print('Using Tensorflow version:', tf.__version__,
      'Keras version:', tf.keras.__version__,
      'backend:', tf.keras.backend.backend())


# ## Data
# 
# The training dataset consists of 2000 images of dogs and cats, split
# in half.  In addition, the validation set consists of 1000 images.

if 'DATADIR' in os.environ:
    DATADIR = os.environ['DATADIR']
else:
    DATADIR = "/scratch/project_2000859/demo/extracted/"

datapath = os.path.join(DATADIR, "dogs-vs-cats/train-2000/")

nimages = dict()
nimages['train'] = 2000
nimages['validation'] = 1000

# ### Image paths and labels

def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    assert image_count == nimages[dataset], "Found {} images, expected {}".format(image_count, nimages[dataset])
    return image_paths

image_paths = dict()
image_paths['train'] = get_paths('train')
image_paths['validation'] = get_paths('validation')

label_names = sorted(item.name for item in pathlib.Path(datapath+'train').glob('*/')
                     if item.is_dir())
label_to_index = dict((name, index) for index,name in enumerate(label_names))

def get_labels(dataset):
    return [label_to_index[pathlib.Path(path).parent.name]
            for path in image_paths[dataset]]
    
image_labels = dict()
image_labels['train'] = get_labels('train')
image_labels['validation'] = get_labels('validation')

# ### Data augmentation
# 
# We need to resize all training and validation images to a fixed
# size. Here we'll use 160x160 pixels.
# 
# Then, to make the most of our limited number of training examples,
# we'll apply random transformations (crop and horizontal flip) to
# them each time we are looping over them. This way, we "augment" our
# training dataset to contain more data. There are various
# transformations readily available in TensorFlow, see tf.image
# (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image)
# for more information.

INPUT_IMAGE_SIZE = [160, 160, 3]


def preprocess_image(image, augment):
    image = tf.image.decode_jpeg(image, channels=3)
    if augment:
        image = tf.image.resize(image, [256, 256])
        image = tf.image.random_crop(image, INPUT_IMAGE_SIZE)
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, True), label


def load_and_not_augment_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image, False), label


# ### TF Datasets
# 
# Let's now define our TF Datasets
# (https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#class_dataset)
# for training and validation data. First the Datasets contain the
# filenames of the images and the corresponding labels.

train_dataset = tf.data.Dataset.from_tensor_slices((image_paths['train'],
                                                    image_labels['train']))
validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths['validation'],
                                                         image_labels['validation']))

# We then map() the filenames to the actual image data and decode the images.
# Note that we shuffle and augment only the training data.

BATCH_SIZE = 32

train_dataset = train_dataset.map(load_and_augment_image,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.map(load_and_not_augment_image,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ## Reuse a pre-trained CNN
#
# We now reuse a pretrained network.  Here we'll use one of the
# pre-trained networks available from Keras
# (https://keras.io/applications/).  We remove the top layers and
# freeze the pre-trained weights.

# ### Initialization

model = Sequential()

model.add(InputLayer(input_shape=INPUT_IMAGE_SIZE))  # possibly needed due to a bug in Keras

pt_model = applications.VGG16(weights='imagenet', include_top=False,
                              input_shape=INPUT_IMAGE_SIZE)
pretrained_first_trainable_layer = 15

print('Using {} pre-trained model'.format(pt_model.name))

for layer in pt_model.layers:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

print(model.summary())


# We then stack our own, randomly initialized layers on top of the pre-trained network.

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# ### Learning 1: New layers

logdir = os.path.join(os.getcwd(), "logs", "dvc-{}-reuse-{}".format(
    pt_model.name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 10

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc-" + pt_model.name + "-reuse.h5"
print('Saving model to', fname)
model.save(fname)


# ### Learning 2: Fine-tuning
#
# Once the top layers have learned some reasonable weights, we can
# continue training by unfreezing the last blocks of the pre-trained
# network so that it may adapt to our data. The learning rate should
# be smaller than usual.

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

for layer in model.layers[pretrained_first_trainable_layer:]:
    layer.trainable = True
    print(layer.name, "now trainable")

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

logdir = os.path.join(os.getcwd(), "logs", "dvc-{}-finetune-{}".format(
    pt_model.name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
print('TensorBoard log directory:', logdir)
os.makedirs(logdir)
callbacks = [TensorBoard(log_dir=logdir)]

epochs = 20

history = model.fit(train_dataset, epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=callbacks, verbose=2)

fname = "dvc-" + pt_model.name + "-finetune.h5"
print('Saving model to', fname)
model.save(fname)
print('All done')
