# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import numpy as np 
import pickle as pkl
from tensorflow.keras.applications import nasnet
import pandas as pd
import time
from shutil import copyfile
import progressbar
from scipy import sparse
import tensorflow as tf
# make global variable for optimizing Ram space.
global nasNet

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


def make_dirs(data,unique_name):
    
    cluster_dir = 'results/'+unique_name+'/'
    os.makedirs(os.path.join((cluster_dir)),exist_ok=True)
    # path_cluster = os.path.join(cluster_dir)
    # path_vgg = os.path.join(vgg_dir)
    for i,row in data.iterrows():
        image_name = row[0].split('/')[-1]
        cluster_label_path = os.path.join(cluster_dir+str(row[1]))
        if not os.path.exists(cluster_label_path):
            os.mkdir(cluster_label_path)
        dst_cluster = os.path.join(cluster_label_path+'/'+image_name)
        copyfile(row[0], dst_cluster)
        

if __name__ == "__main__":
    model_path = 'vgg16_quant.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    pkl_kmeans_name = 'db/kmeans_dataset.pkl'
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
    # save into this path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="image directory path")
    args = parser.parse_args()
    data_dir = args.data_dir
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'
    data_names = os.listdir(data_dir)
    len_data = len(data_names)
    print('[STATUS] ',len_data,'data founded .\n')
    # dictionary for prediction
    prediction = {'image':[],'cluster_label':[]}
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,file_name in enumerate(data_names):
        file = data_dir+file_name
        Image = im_lite_loader(file)
        feature_arr = lite_feature_ext(Image)
        cluster_pred = kmeans.predict(feature_arr)[0]
        # cluster_label = cluster_pred_decoder(cluster_pred)
        # add result to dictionary 
        prediction['image'].append(file)
        prediction['cluster_label'].append(cluster_pred)

        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    unique_time = str(time.time())
    csv_name = unique_time+'.csv'
    df.to_csv(csv_name)
    make_dirs(df,unique_time)
