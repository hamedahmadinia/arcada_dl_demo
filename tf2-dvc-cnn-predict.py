# Predict with trained Dogs-vs-cats model

import os
# Try to disable CUDA when running inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# TF_CPP_MIN_LOG_LEVEL:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# We need to resize all test images to a fixed size. Here we'll use
# 160x160 pixels.
#
# Unlike the training images, we do not apply any random
# transformations to the test images.
INPUT_IMAGE_SIZE = [160, 160, 3]

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, INPUT_IMAGE_SIZE[:2])
    image /= 255.0  # normalize to [0,1] range
    return image

# Class labels:
classes = ['cat', 'dog']

# Initialization:
if len(sys.argv)<3:
    print('ERROR: model file or image path missing')
    sys.exit()

# It might be faster to use CPU for inference,
# since it takes a while to initialize the GPU.
with tf.device('/cpu:0'):
    model = load_model(sys.argv[1])
image = load_image(sys.argv[2])

# Inference:
prediction = model.predict(tf.expand_dims(image, 0))
print('Predicted label for "' + sys.argv[2] + '" is: ' + classes[int(np.round(prediction).item())])
print('All done')
