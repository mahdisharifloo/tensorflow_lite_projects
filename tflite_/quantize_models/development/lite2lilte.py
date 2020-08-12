# -*- coding: utf-8 -*-

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import cv2
import os 
from tqdm import tqdm

keras_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',include_top=False)



file = '/home/mahdi/Pictures/images.jpeg'
def im_lite_loader(im_path):
    input_shape = (1,224,224,3)
    input_image = cv2.imread(im_path)
    #Reshape input_image 
    input_image = cv2.resize(input_image,(input_shape[1],input_shape[2]))
    input_image = np.expand_dims(input_image,axis=0)
    input_image = np.array(input_image,dtype=np.uint8)
    return input_image

def rep_data_gen():
    input_shape = (1,224,224,3)
    img_dir = '/home/mahdi/projects/dgkala/data/watch_glasses/'
    BATCH_SIZE=32
    a = []
    img_list = os.listdir(img_dir)
    for file_name in tqdm(img_list):
        img = cv2.imread(img_dir + file_name)
        img = cv2.resize(img, (input_shape[1],input_shape[2]))
        img = img / 255.0
        img = img.astype(np.float32)
        a.append(img)
    a = np.array(a)
    print(a.shape) # a is np array of 160 3D images
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(BATCH_SIZE):
        print(i.shape)
        yield [i]
        
        
def model_converter_quant_int(keras_model,save=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # tflite_model = converter.convert()
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = rep_data_gen
    tflite_model_quant = converter.convert()
    if save:
        lite_name = keras_model.name +'_quant_int'+'.tflite'
        print(lite_name)
        open(lite_name,'wb').write(tflite_model_quant)
    return tflite_model_quant
    

tflite_model_quant = model_converter_quant_int(keras_model)


model_path = 'mobilenetv2_1.00_224_quant_int.tflite'
# interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
# Get input and output tensors
interpreter = tf.lite.Interpreter(model_path=model_path)

input_details = interpreter.get_input_details()

# Your network currently has an input shape (1, 128, 80 , 1),
# but suppose you need the input size to be (2, 128, 200, 1).
interpreter.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

input_image = im_lite_loader(file)


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_image)

interpreter.invoke()
#prediction for input data
output_data = interpreter.get_tensor(output_details[0]['index'])
