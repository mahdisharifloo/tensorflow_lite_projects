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
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers

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
        # print('glasses')
    else:
        label = 'watch'
        # print('watch')
    return label


def make_dirs(data,unique_name):
    
    result_dir = 'results/'+unique_name+'/'
    os.makedirs(os.path.join((result_dir)),exist_ok=True)
    # path_cluster = os.path.join(cluster_dir)
    # path_vgg = os.path.join(vgg_dir)
    for i,row in data.iterrows():
        image_name = row[0].split('/')[-1]
        result_label_path = os.path.join(result_dir+str(row[1]))
        if not os.path.exists(result_label_path):
            os.mkdir(result_label_path)
        dst = os.path.join(result_label_path+'/'+image_name)
        copyfile(row[0], dst)
        

if __name__ == "__main__":
    model_path = 'mobilenetV2_quant.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    
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
    prediction = {'image':[],'label':[]}
    

    with open('mlp_classifier.pkl' ,'rb') as f:
        classifier = pkl.load(f)
    
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,file_name in enumerate(data_names):
        file = data_dir+file_name
        Image = im_lite_loader(file)
        feature_arr = lite_feature_ext(Image)
        pred = classifier.predict(feature_arr)
        label  = prediction_compute(pred)
        # add result to dictionary 
        prediction['image'].append(file)
        prediction['label'].append(label)

        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    unique_time = str(time.time())
    csv_name = unique_time+'.csv'
    df.to_csv(csv_name)
    make_dirs(df,unique_time)
