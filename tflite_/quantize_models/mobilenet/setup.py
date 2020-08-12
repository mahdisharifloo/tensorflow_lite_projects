
import tensorflow as tf 
import pandas as pd
import os
import numpy as np 
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()  # for plot styling
import pickle as pkl
import progressbar
from shutil import copyfile
import argparse
import time
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras 
from tensorflow.keras.models import Model


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def feature_ext(input_image):
    pred = model.predict(input_image)
    pred = pred[0]
    pred = pred.flatten()
    return pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="input pictures directory that you wants cluster it.")
    parser.add_argument("n_clusters",type=int, help="number of clusters that you need", default=2)
    parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="vgg19_quantized.h5")
    args = parser.parse_args()
  
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = tf.keras.models.load_model(args.model_path)
    layer_name = loaded_model.layers[-1].name
    model = Model(loaded_model.input,loaded_model.get_layer(layer_name).output)
    
    dir_path = args.dir_path
    features = {'img':[],'features':[],'cluster':[]}
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
            input_image = prepare_image(img_path)
        except:
            continue
        single_feature_table =  feature_ext(input_image)
        features['img'].append(img_path)
        features['features'].append(single_feature_table)
        bar.update(i+1)
        
    print('[TIME]: ', time.perf_counter())
    print('[STATUS] feature extraction process done')
    if not os.path.exists('db/'):
        os.mkdir('db/')

    print('[TIME]: ', time.perf_counter())
    n_classes = args.n_clusters
    res_arr = np.array(features['features'])
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
    df = df.drop(['features'],axis=1)
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
    
        
        
    
    
    
    
    
    