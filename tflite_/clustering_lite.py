import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
from tensorflow.keras.applications import nasnet
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
import pickle as pkl
import progressbar
import time
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_dir_path", type=str, help="input directory path that has images")
    parser.add_argument("n_clusters",type=int, help="number of clusters that you need", default=2)
    parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    args = parser.parse_args()
    
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    img_dir_path = args.image_dir_path
    features = {'img':[],'nasnet_lite':[],'cluster':[]}
    pics_num = os.listdir(img_dir_path)
    print('{} image founded .'.format(len(pics_num)))
    print('start features extraction process ...')
    print('[TIME]: ', time.perf_counter())
    bar = progressbar.ProgressBar(maxval=len(pics_num), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i,img_path in enumerate(pics_num):
        img_path = img_dir_path +'/'+ img_path
        try:
            input_image = im_lite_loader(img_path)
        except :
            continue
        lite_features = lite_feature_ext(input_image)
        features['img'].append(img_path)
        features['nasnet_lite'].append(lite_features)
        bar.update(i+1)
        
    pkl_file_name = img_dir_path.split('/')[-2]+'.pkl'
    with open(pkl_file_name,'wb') as f:
        pkl.dump(features,f)
    print('feature extraction process DONE.\nfeatures saved on : ',pkl_file_name)
    n_classes = args.n_clusters
    res_arr = np.array(features['nasnet_lite'])
    if len(res_arr.shape)==3:
        res_arr = np.reshape(res_arr,(res_arr.shape[0],res_arr.shape[2]))
    print('start clustering in {} clusters '.format(n_classes))
    res_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)
    
    for (name,cluster) in zip(features['img'],res_kmeans.labels_):
        features['cluster'].append(cluster)
        # print(name,cluster)
        
    kmeans_object_pkl_name = 'kmeans_lite.pkl'
    with open(kmeans_object_pkl_name,'wb') as f:
        pkl.dump(res_kmeans,f)
    print('clustering process DONE.\nyou can find kmeans object in : ',kmeans_object_pkl_name)

