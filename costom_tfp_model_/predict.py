# -*- coding: utf-8 -*-
import cv2
from tensorflow.keras import applications
import numpy as np 
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from tensorflow import keras
#Default dimensions we found online
img_width, img_height = 224, 224 
 
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'

vgg16 = applications.VGG16(include_top=False, weights='imagenet')
model = Sequential() 
model.add(Flatten(input_shape=(7,7,512))) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(2, activation='softmax'))
model.load_weights(top_model_weights_path)


image_path = '/home/mahdi/Pictures/images.jpeg'
image_ = cv2.imread(image_path)
image_ = cv2.resize(image_,(224,224))
input_image = np.expand_dims(image_,axis=0)
features = vgg16.predict(input_image)
model.predict(features)
