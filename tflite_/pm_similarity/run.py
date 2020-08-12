from functools import wraps
import os
import tensorflow as tf 
import cv2  
import numpy as np 
from tensorflow.keras.applications.vgg19 import preprocess_input 
from tensorflow.keras.applications import nasnet
import time
import mahotas
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import sparse
from shutil import copyfile
from dotenv import load_dotenv
import argparse

load_dotenv(verbose=True)

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


parser = argparse.ArgumentParser()
parser.add_argument("cluster_pkl_path", type=str, help="input pkl file that you wants cluster it.",default='db/dataset.pkl')
parser.add_argument("image_path", type=str, help="image file path")
parser.add_argument("pkl_kmeans_name", help="kmeans object that you train it before", default='./db/kmeans_dataset.pkl')
parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

@estaminetor_decorator
def loader():
    pkl_kmeans_name = './db/kmeans_dataset.pkl'
    cluster_pkl_path = os.getenv("CSV_PATH")
    if cluster_pkl_path == None:
        cluster_pkl_path = args.cluster_pkl_path
    with open(cluster_pkl_path ,'rb') as f:
        clusters = pkl.load(f)
    clusters = pd.DataFrame(clusters)
    with open(args.pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)

    return clusters, kmeans

@estaminetor_decorator
def main():
    number_of_item = 10
    clusters, kmeans = loader()
    image_path = os.getenv("IMAGE_PATH")
    if image_path == None:
        image_path = args.image_path
    Image = im_lite_loader(image_path)
    feature_arr = lite_feature_ext(Image)
    cluster_pred = kmeans.predict(feature_arr)[0]
    cluster_results = cluster_sim(cluster_pred,feature_arr,clusters)
    print(cluster_results.head(number_of_item))
    cluster_results.to_csv('prediction_results.csv')

main()

