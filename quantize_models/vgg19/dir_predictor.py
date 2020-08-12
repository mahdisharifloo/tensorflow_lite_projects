
import tensorflow as tf 
import pandas as pd
import os
import numpy as np 
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
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = tf.keras.models.load_model('vgg19_quantized.h5')
    layer_name = loaded_model.layers[-1].name
    model = Model(loaded_model.input,loaded_model.get_layer(layer_name).output)
    
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
        Image = prepare_image(file)
        feature_arr = np.expand_dims(feature_ext(Image),axis=0)
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