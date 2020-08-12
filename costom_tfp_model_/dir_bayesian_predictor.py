
import cv2
from tensorflow.keras import applications
import numpy as np 
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import optimizers
import argparse
import os 
import progressbar
import pandas as pd
import time
from shutil import copyfile


# make global variable for optimizing Ram space.

def load_and_extract(image_path):
    image_ = cv2.imread(image_path)
    image_ = cv2.resize(image_,(224,224))
    input_image = np.expand_dims(image_,axis=0)
    features = vgg16.predict(input_image)
    result = model_vi.predict(features)
    return result



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
    top_model_weights_path = 'bayesian_model.h5'
    len_data = 800
    num_classes = 2
    vgg16 = applications.VGG16(include_top=False, weights='imagenet')

    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (len_data *1.0)
    
    model_vi = Sequential()
    model_vi.add(Flatten(input_shape=(7,7,512))) 
    model_vi.add(tfp.layers.DenseFlipout(100, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(50, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
    model_vi.add(tfp.layers.DenseFlipout(num_classes, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))
    
    model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    model_vi.summary()
    
    model_vi.load_weights(top_model_weights_path)
    

    
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
    prediction = {'image':[],'pred':[]}
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,file_name in enumerate(data_names):
        file = data_dir+file_name
        pred = np.argmax(load_and_extract(file))
        # cluster_label = cluster_pred_decoder(cluster_pred)
        # add result to dictionary 
        prediction['image'].append(file)
        prediction['pred'].append(pred)
        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    unique_time = str(time.time())
    csv_name = unique_time+'.csv'
    df.to_csv(csv_name)
    make_dirs(df,unique_time)