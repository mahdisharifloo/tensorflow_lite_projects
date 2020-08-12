# -*- coding: utf-8 -*-

import cv2
from tensorflow.keras import applications
import numpy as np 
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import optimizers
import argparse
import tensorflow as tf 

#Default dimensions we found online
img_width, img_height = 224, 224 
len_data = 800
#Create a bottleneck file
top_model_weights_path = 'bayesian_conv_model.h5'
num_classes = 2
vgg16 = applications.VGG16(include_top=False, weights='imagenet')

kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (len_data *1.0)

model_vi = Sequential()
# model_vi.add(Flatten(input_shape=train_data.shape[1:])) 
model_vi.add(tf.keras.layers.InputLayer(input_shape=(7,7,512), name="input"))
model_vi.add(tfp.layers.Convolution2DFlipout(8,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn,input_shape=(32,32,3)))
model_vi.add(tfp.layers.Convolution2DFlipout(8,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
model_vi.add(tfp.layers.Convolution2DFlipout(16,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.Convolution2DFlipout(16,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
model_vi.add(tf.keras.layers.Flatten())
model_vi.add(tfp.layers.DenseFlipout(100, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.DenseFlipout(100, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.DenseFlipout(num_classes, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))

model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
model_vi.summary()

model_vi.load_weights(top_model_weights_path)

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="image path")
args = parser.parse_args()

image_path = args.image_path
image_ = cv2.imread(image_path)
image_ = cv2.resize(image_,(224,224))
input_image = np.expand_dims(image_,axis=0)
features = vgg16.predict(input_image)
result = model_vi.predict(features)
print(np.argmax(result))
