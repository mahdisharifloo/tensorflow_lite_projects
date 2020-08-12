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
    max_probability_label,max_probability = np.argmax(pred),np.max(pred)
    treshold = 0.55
    # print(pred)
    if max_probability>=treshold:
        if max_probability_label==0:
            label = 'glasses'
            # print('watch')
        else:
            label = 'watch'
            # print('glasses')
    else:
        label = 'unknown'
        # print('unknown')
        
    return label,max_probability

def model_config():
    train_data_shape = (800,1,1000)
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (train_data_shape[0] *1.0)
    
    model_vi = Sequential()
    # model_vi.add(Flatten(input_shape=train_data.shape)) 
    model_vi.add(tf.keras.layers.InputLayer(train_data_shape[1:], name="input"))
    # model_vi.add(tfp.layers.Convolution2DFlipout(8,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    # model_vi.add(tfp.layers.Convolution2DFlipout(8,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    # model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
    # model_vi.add(tfp.layers.Convolution2DFlipout(16,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    # model_vi.add(tfp.layers.Convolution2DFlipout(16,kernel_size=(3,3),padding="same", activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    # model_vi.add(tf.keras.layers.MaxPooling2D((2,2)))
    model_vi.add(tf.keras.layers.Flatten())
    model_vi.add(tfp.layers.DenseFlipout(500, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(100, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(50, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(2, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))
    
    model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    # model_vi.summary()
    return model_vi

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
        dst = os.path.join(result_label_path+'/'+'_'+str(int(row[2]*100))+'_'+image_name)
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
    prediction = {'image':[],'label':[],'prob_percent':[]}
    
    model_vi = model_config()
    model_vi.load_weights('bayesian_conv_model.h5')
    
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,file_name in enumerate(data_names):
        file = data_dir+file_name
        Image = im_lite_loader(file)
        feature_arr = lite_feature_ext(Image)
        pred = model_vi.predict(feature_arr)
        label,prob_percent  = prediction_compute(pred)
        # add result to dictionary 
        prediction['image'].append(file)
        prediction['prob_percent'].append(prob_percent)
        prediction['label'].append(label)

        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    unique_time = str(time.time())
    csv_name = unique_time+'.csv'
    df.to_csv(csv_name)
    make_dirs(df,unique_time)
