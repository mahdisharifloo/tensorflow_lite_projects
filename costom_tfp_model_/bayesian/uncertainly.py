# -*- coding: utf-8 -*-


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
from tqdm import  tqdm


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


# def prediction_compute(pred):
#     max_probability_label,max_probability = np.argmax(pred),np.max(pred)
#     treshold = 0.6
#     print(pred)
#     if max_probability>=treshold:
#         if max_probability_label==0:
#             label = 'glasses'
#             print('glasses')
#         else:
#             label = 'watch'
#             print('watch')
#     else:
#         label = 'unknown'
#         print('unknown')
        
#     return label

def model_config():
    train_data_shape = (800,1,1000)
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (train_data_shape[0] *1.0)
    
    model_vi = Sequential()
    model_vi.add(tf.keras.layers.InputLayer(train_data_shape[1:], name="input"))
    model_vi.add(tf.keras.layers.Flatten())
    model_vi.add(tfp.layers.DenseFlipout(500, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(100, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(50, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(6, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))
    
    model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    model_vi.summary()
    return model_vi
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    # parser.add_argument("image_path",type=str, help="image path", default="tf_lite_model.tflite")

    # args = parser.parse_args()
    
    img_width, img_height = 224, 224 
    batch_size = 32
    
    interpreter = tf.lite.Interpreter(model_path='mobilenetV2_quant.tflite')
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    model_vi = model_config()
    
    model_vi.load_weights('bayesian_conv_model.h5')
    
    # ####################################################
    test_path = 'data/head_wear/'
    datagen = ImageDataGenerator(rescale=1. / 255) 
    generator = datagen.flow_from_directory( 
        test_path, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode=None, 
        shuffle=False) 
    nb_test_samples = len(generator.filenames) 
    num_classes = len(generator.class_indices) 
    test_features = {'img':[],'lite_features':[],'pred':[]}
    # extracting test data 
    print('[STATUS] features extraction process on test dataset running ...')
    print('[TIME]: ', time.perf_counter())
    bar = progressbar.ProgressBar(maxval=nb_test_samples, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i,file_path in tqdm(enumerate(generator.filenames)):
        img_path = test_path+ file_path
        Image = im_lite_loader(img_path)
        single_feature_table = lite_feature_ext(Image)
        test_features['img'].append(file_path)
        test_features['lite_features'].append(single_feature_table)
        bar.update(i+1)
    print('[TIME]: ', time.perf_counter())
    print('[STATUS] feature extraction process done')
    test_data = np.array(test_features['lite_features'])
    test_labels = generator.classes
    test_labels = to_categorical(test_labels, num_classes=num_classes)
    ####################################################
    image_path = '/home/mahdi/Pictures/images.jpeg'
    # image_path = args.image_path
    Image = im_lite_loader(image_path)
    feature_arr = lite_feature_ext(Image)
    pred = model_vi.predict(feature_arr)
    # print(list(generator.class_indices.keys()))
    for i in range(0,19):
        pred = model_vi.predict(feature_arr)
        max_pred = np.argmax(pred)
        print(pred,max_pred)

    pred_vi=np.zeros((len(test_data),6))
    pred_max_p_vi=np.zeros((len(test_data)))
    pred_std_vi=np.zeros((len(test_data)))
    entropy_vi = np.zeros((len(test_data)))
    
    for i in tqdm(range(0,len(test_data))):
      multi_img=np.tile(test_data[i],(50,1,1,1))
      preds=model_vi.predict(multi_img)
      pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
      pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
      pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
      entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
    pred_labels_vi=np.array([test_labels[np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)
    
    
    image_known = 'data/train/watch/101441.jpg'
    image_unknown = 'data/head_wear/head_wear/105516502.jpg'
    sample_known = cv2.imread(image_known)
    sample_unknown = cv2.imread(image_unknown)
    input_image_known = im_lite_loader(image_known)
    input_image_unknown = im_lite_loader(image_unknown)
    input_ready_known = lite_feature_ext(input_image_known)
    input_ready_unknown = lite_feature_ext(input_image_unknown)
    labels = ['glasses', 'men_closthing', 'ring', 'shoes', 'watch', 'women_clothing']
    
    plt.figure(figsize=(20,15))
    plt.subplot(4,3,1)
    plt.axis('off')
    plt.text(0.5,0.5, "Input image",fontsize=22,horizontalalignment='center')
    plt.subplot(4,3,2)
    plt.imshow(sample_known)
    plt.title("known class",fontsize=25)
    plt.subplot(4,3,3)
    plt.imshow(sample_unknown)
    plt.title("unknown class",fontsize=25)
    
    
    plt.subplot(4,3,7)
    plt.axis('off')
    plt.subplot(4,3,8)
    for i in range(0,50):
      plt.scatter(range(0,6),model_vi.predict(input_ready_known),c="blue",alpha=0.2)
    plt.xticks(range(0,6),labels=labels)
    plt.subplot(4,3,6)
    for i in range(0,50):
      plt.scatter(range(0,6),model_vi.predict(input_ready_unknown),c="blue",alpha=0.2)
    plt.xticks(range(0,6),labels=labels)
    plt.show()
