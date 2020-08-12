# -*- coding: utf-8 -*-


import tensorflow as tf 
import cv2  
import os
# %matplotlib inline
import pickle as pkl
import progressbar
from time import sleep
from shutil import copyfile
import argparse
import time
import pandas as pd
import numpy as np 
import itertools
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras import applications 
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime


# from dotenv import load_dotenv
# from pathlib import Path  # python3 only
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)

def im_lite_loader(im_path):
    input_image = cv2.imread(im_path)
    #Reshape input_image 
    input_image = cv2.resize(input_image,(input_shape[1],input_shape[2]))
    input_image = np.expand_dims(input_image,axis=0)
    input_image = np.array(input_image,dtype=np.float32)
    return input_image

def lite_feature_ext(input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    #prediction for input data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    features = output_data[0]
    features = features.flatten()
    features = np.reshape(features,(1,-1))
    return features


def prediction_compute(pred):
    max_probability_label = np.argmax(pred)

    if max_probability_label==0:
        label = 'glasses'
        print('glasses')
    else:
        label = 'watch'
        print('watch')
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    parser.add_argument("image_path",type=str, help="image path", default="tf_lite_model.tflite")

    args = parser.parse_args()
    
    
    #Create a bottleneck file
    # top_model_weights_path = 'bayesian_conv_model.h5'
    # loading up our datasets
    img_width, img_height = 224, 224 

    interpreter = tf.lite.Interpreter(model_path='mobilenetV2_quant.tflite')
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    with open('mlp_classifier.pkl' ,'rb') as f:
        classifier = pkl.load(f)
        
    image_path = args.image_path
    Image = im_lite_loader(image_path)
    feature_arr = lite_feature_ext(Image)
    pred = classifier.predict(feature_arr)
    label = prediction_compute(pred)
    
    
    
