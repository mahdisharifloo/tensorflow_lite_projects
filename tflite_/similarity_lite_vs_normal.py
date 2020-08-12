# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import  lite
from tensorflow.keras.applications import nasnet
from sklearn.metrics.pairwise import cosine_similarity
import time 
import cv2 
import numpy as np 
import argparse


def NASNetMobile(image_bytes):
    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = nasnet.preprocess_input(image_batch)
    nasnet_features =nasnet_extractor.predict(processed_imgs)
    flattened_features = nasnet_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return flattened_features

def im_normal_loader(im_path):
    Image = cv2.imread(im_path)
    image_size =tuple((224, 224))
    Image = Image[:,:,:3]
    Image = cv2.resize(Image,image_size)
    return Image

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
    features = output_data[0][0] * 100 #prediction probability
    features = features.flatten()
    features = np.reshape(features,(1,-1))
    return features

def cosin(table1,table2):
    global feat_extractor
    similarity_table = cosine_similarity(table1, table2)
    return  np.mean(similarity_table)

if __name__ == "__main__":
    nasnet_extractor = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path1", type=str, help="input first image for compare")
    parser.add_argument("img_path2", type=str, help="input second image for compare with first")
    parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    args = parser.parse_args()
    
    img_path1 = args.img_path1
    img_path2 = args.img_path2
    
    print('[STATUS] Normal nasnetMobile start computing ...')
    print(time.perf_counter())
    
    Image1 = im_normal_loader(img_path1)
    Image2 = im_normal_loader(img_path2)
    normal_features1 = NASNetMobile(Image1)
    normal_features2 = NASNetMobile(Image2)
    normal_sim_result = cosin(normal_features1,normal_features2)
    print(time.perf_counter())
    print('normal similarity percent is : ',normal_sim_result)
    
    
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    #Load input image
    input_image1 = im_lite_loader(img_path1)
    input_image2 = im_lite_loader(img_path2)
    
    print('[STATUS] TFlite nasnetMobile start conputing ...')
    print(time.perf_counter())
    
    
    lite_features1 = lite_feature_ext(input_image1)
    lite_features2 = lite_feature_ext(input_image2)
    lite_sim_result = cosin(lite_features1,lite_features2)
    print(time.perf_counter())
    print('lite similarity percent is : ',lite_sim_result)


