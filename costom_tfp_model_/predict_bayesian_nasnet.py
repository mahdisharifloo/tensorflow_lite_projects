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

#Default dimensions we found online
img_width, img_height = 224, 224 
len_data = 800
#Create a bottleneck file
top_model_weights_path = 'bayesian_model_nasnet.h5'
num_classes = 2
nasnet = applications.NASNetMobile(include_top=False, weights='imagenet')

kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (len_data *1.0)

model_vi = Sequential()
model_vi.add(Flatten(input_shape=(7,7,1056))) 
model_vi.add(tfp.layers.DenseFlipout(100, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.DenseFlipout(50, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
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
features = nasnet.predict(input_image)
result = model_vi.predict(features)
print(np.argmax(result))
