# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import  lite
from tensorflow.keras.applications import nasnet
import time 
import cv2 
import numpy as np 

def NASNetMobile(image_bytes):
    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = nasnet.preprocess_input(image_batch)
    nasnet_features =nasnet_extractor.predict(processed_imgs)
    flattened_features = nasnet_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return flattened_features

nasnet_extractor = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

print('[STATUS] Normal nasnetMobile start computing ...')
print(time.perf_counter())
print(time.ctime())

img_path = "/home/mahdi/Pictures/images.jpeg"
Image = cv2.imread(img_path)

image_size =tuple((224, 224))
Image = Image[:,:,:3]
Image = cv2.resize(Image,image_size)
pred = NASNetMobile(Image)
print(time.perf_counter())
print(time.ctime())

interpreter = tf.lite.Interpreter(model_path="tf_lite_model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#Load input image
input_image = cv2.imread(img_path)
input_shape = input_details[0]['shape']
#Reshape input_image 
input_image = cv2.resize(input_image,(input_shape[1],input_shape[2]))
input_image = np.expand_dims(input_image,axis=0)
input_image = np.array(input_image,dtype=np.float32)
print('[STATUS] TFlite nasnetMobile start conputing ...')
print(time.perf_counter())
print(time.ctime())

#Set the value of Input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
#prediction for input data
output_data = interpreter.get_tensor(output_details[0]['index'])
probability = output_data[0][0] * 100 #prediction probability
probability = probability.flatten()
print(time.perf_counter())
print(time.ctime())

