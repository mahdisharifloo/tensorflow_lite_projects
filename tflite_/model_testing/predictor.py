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
import glob 
from sklearn.cluster import KMeans


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="image directory path")
    parser.add_argument("n_clusters",type=int, help="number of clusters that you need", default=2)
    args = parser.parse_args()
    model_paths_ = glob.glob('*.tflite')
        
    data_dir = args.data_dir
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'
    data_names = os.listdir(data_dir)
    len_data = len(data_names)

    print('[STATUS] ',len_data,'data founded .\n')
    for i,model_path in enumerate(model_paths_):
        model_name = model_path.split('.')[-2]
        prediction = {'image':[],'features':[],'cluster':[]}

        print('number: {}     model: {}'.format(i,model_path))
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
    
    
        bar = progressbar.ProgressBar(maxval=len_data, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        # for loop on data
        for i,file_name in enumerate(data_names):
            file = data_dir+file_name
            Image = im_lite_loader(file)
            feature_arr = lite_feature_ext(Image)
            prediction['image'].append(file)
            prediction['features'].append(feature_arr)
    
            bar.update(i+1)
        
        
        print('[TIME]: ', time.perf_counter())
        print('[STATUS] feature extraction process done')
        if not os.path.exists('db/'):
            os.mkdir('db/')    
        res_arr = np.array(prediction['features'])
        if len(res_arr.shape)==3:
            res_arr = np.reshape(res_arr,(res_arr.shape[0],res_arr.shape[2]))
        print('[STATUS] start kmeans clustering process.it may take few minutes ... ')
        print('[TIME]: ', time.perf_counter())
        res_kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(res_arr)

        for (name,cluster) in zip(prediction['image'],res_kmeans.labels_):
            prediction['cluster'].append(cluster)
        df = pd.DataFrame(prediction)

    
        df_pkl_name = 'db/'+model_name+'_'+str(time.time())+'.pkl'
        print('[TIME]: ', time.perf_counter())
        print('[STATUS] dataframe saved on : {}'.format(df_pkl_name))
        with open(df_pkl_name,'wb') as f:
            pkl.dump(df,f)
        kmeans_object_pkl_name = 'db/'+'kmeans_' +model_name +'_'+str(time.time()) +'.pkl'
        with open(kmeans_object_pkl_name,'wb') as f:
            pkl.dump(res_kmeans,f)
        print('[STATUS] cluster object saved on : {}'.format(kmeans_object_pkl_name))
        print('[STATUS] DONE')
        if not os.path.exists('results/'):
            os.mkdir('results')
        model_result = 'results/'+model_name
        if not os.path.exists(model_result):
            os.mkdir(model_result)
        for row in df.iterrows():

            path = os.path.join("results/{0}/{1}".format(model_name,row[1][2]))
            # print(path)
            if not os.path.exists(path):
                os.mkdir(path)
                print(path,'created')
                
            src = row[1][0]
            dst = path+'/'+row[1][0].split('/')[-1]
            
            copyfile(src, dst)
    
        