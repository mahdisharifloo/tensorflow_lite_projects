# -*- coding: utf-8 -*-

from functools import wraps
import os
import tensorflow as tf 
import cv2  
import numpy as np 
from tensorflow.keras.applications import nasnet
import time
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import sparse
from shutil import copyfile
import argparse

# load_dotenv(verbose=True)

global data
# tf.config.gpu_options.allow_growth = True

def estaminetor_decorator(func):
    # Without the use of this decorator factory, the name of the example function would have been 'wrapper',
    # https://docs.python.org/3/library/functools.html#functools.wraps
    wraps(func)
    def wrapper(*args, **kwargs):
        # perf_counter has more accuracy
        # https://docs.python.org/3/library/time.html#time.perf_counter
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print("function %s takes %s" % (func.__name__, end-start))
        return result
    return wrapper

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

def cluster_sim(cluster_pred,single_nasnet_feature_arr,clusters):
    # checking similarity between single image and images that in clusters.
    cluster_images = clusters.loc[clusters['cluster'] == cluster_pred]
    compare = {'img':[],'clu_percent':[]}
    for name,feature in zip(cluster_images['img'],cluster_images['nasnet_lite']):
        percent = cosin(single_nasnet_feature_arr,feature )
        compare['img'].append(name)
        compare['clu_percent'].append(percent)

    df = pd.DataFrame(compare,index=compare['img'])
    df = df.drop(columns=['img'])
    df = df.sort_values(by='clu_percent',ascending=False, na_position='first')
    return df

def similar_dump(head_results):  
    for path in head_results.index:
        if not os.path.exists('similar_results/'):
            os.mkdir('similar_results')

        dst = 'similar_results/'+path.split('/')[-1]
        
        copyfile(path,dst)
    


parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="image file path")
args = parser.parse_args()

cluster_pkl_path = 'db/dataset.pkl'
pkl_kmeans_name = './db/kmeans_dataset.pkl'
model_path = "tf_lite_model.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

@estaminetor_decorator
def loader():
    pkl_kmeans_name = './db/kmeans_dataset.pkl'
    with open(cluster_pkl_path ,'rb') as f:
        clusters = pkl.load(f)
    clusters = pd.DataFrame(clusters)
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)

    return clusters, kmeans

@estaminetor_decorator
def main():
    number_of_item = 10
    clusters, kmeans = loader()
    image_path = args.image_path
    Image = im_lite_loader(image_path)
    feature_arr = lite_feature_ext(Image)
    cluster_pred = kmeans.predict(feature_arr)[0]
    print('cluster number predicted : ',cluster_pred)
    cluster_results = cluster_sim(cluster_pred,feature_arr,clusters)
    head_df =cluster_results.head(number_of_item) 
    print(head_df)
    similar_dump(head_df)
    cluster_results.to_csv('prediction_results.csv')

main()

