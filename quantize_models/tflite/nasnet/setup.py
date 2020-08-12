
import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
from sklearn.cluster import KMeans
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications import nasnet
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans
import pickle as pkl
import progressbar
from time import sleep
from shutil import copyfile
import argparse
import time
from shutil import copyfile


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="input pictures directory that you wants cluster it.")
    parser.add_argument("n_clusters",type=int, help="number of clusters that you need", default=2)
    parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    args = parser.parse_args()
  
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    dir_path = args.dir_path
    features = {'img':[],'nasnet_lite':[],'cluster':[]}
    # pics_num = os.listdir(img_dir_path)
    data = os.listdir(dir_path)
    pics_num = len(data)
    print(pics_num,'data found .')
    print('[STATUS] features extraction process running ...')
    print('[TIME]: ', time.perf_counter())
    bar = progressbar.ProgressBar(maxval=pics_num, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i,file_path in enumerate(data):
        img_path = dir_path+ file_path
        
        try:
            Image = im_lite_loader(img_path)
        except:
            continue
        single_feature_table = lite_feature_ext(Image)
        features['img'].append(img_path)
        features['nasnet_lite'].append(single_feature_table)
        bar.update(i+1)
        
    print('[TIME]: ', time.perf_counter())
    print('[STATUS] feature extraction process done')
    if not os.path.exists('db/'):
        os.mkdir('db/')

    print('[TIME]: ', time.perf_counter())
    n_classes = args.n_clusters
    res_arr = np.array(features['nasnet_lite'])
    if len(res_arr.shape)==3:
        res_arr = np.reshape(res_arr,(res_arr.shape[0],res_arr.shape[2]))
    print('[STATUS] start kmeans clustering process.it may take few minutes ... ')
    print('[TIME]: ', time.perf_counter())
    res_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)
    
    for (name,cluster) in zip(features['img'],res_kmeans.labels_):
        features['cluster'].append(cluster)
    pkl_file_name = 'db/dataset.pkl'
    with open(pkl_file_name,'wb') as f:
        pkl.dump(features,f)
    # df = pd.DataFrame(features)
    print('[STATUS] done')
    kmeans_object_pkl_name = 'db/kmeans_dataset.pkl'
    with open(kmeans_object_pkl_name,'wb') as f:
        pkl.dump(res_kmeans,f)
    print('[STATUS] features saved on : {}'.format(pkl_file_name))
    print('[TIME]: ', time.perf_counter())
    print('[STATUS] clustering object saved on : {}'.format(kmeans_object_pkl_name))
    print('[STATUS] DONE')
    df = pd.DataFrame(features)
    df = df.drop(['nasnet_lite'],axis=1)
    df = df.sort_values(by=['cluster'])
    
    for row in df.iterrows():
        if not os.path.exists('cluster/'):
            os.mkdir('cluster')
        path = os.path.join("cluster/{}".format(row[1][1]))
        # print(path)
        if not os.path.exists(path):
            os.mkdir(path)
            print(path,'created')
        
        src = row[1][0]
        dst = path+'/'+row[1][0].split('/')[-1]
        
        copyfile(src, dst)
    
        
        
    
    
    
    
    
    
