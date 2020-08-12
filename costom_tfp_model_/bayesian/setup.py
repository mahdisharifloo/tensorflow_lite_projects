
import tensorflow as tf 
import cv2  
import os
# %matplotlib inline
import pickle as pkl
import progressbar
from time import sleep
from shutil import copyfile
import argparse
import time
import pandas as pd
import numpy as np 
import itertools
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense 
from tensorflow.keras import applications 
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

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

def model_config(train_data_shape):
        
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
    model_vi.add(tfp.layers.DenseFlipout(num_classes, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))
    
    model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
    model_vi.summary()
    return model_vi
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str, help="input train directory")
    parser.add_argument("test_path", type=str, help="input test directory")
    parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
    args = parser.parse_args()
    
    disable_eager_execution()
    #Create a bottleneck file
    top_model_weights_path = 'bayesian_conv_model.h5'
    # loading up our datasets

    img_width, img_height = 224, 224 

    # number of epochs to train top model 
    epochs = 30 #this has been changed after multiple model run 
    # batch size used by flow_from_directory and predict_generator 
    batch_size = 32
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    # train_path = 'data/train/'
    # test_path = 'data/test/'
    train_path = args.train_path
    test_path = args.test_path
    train_features = {'img':[],'lite_features':[],'pred':[]}
    test_features = {'img':[],'lite_features':[],'pred':[]}

    # pics_num = os.listdir(img_dir_path)

    datagen = ImageDataGenerator(rescale=1. / 255) 
    #needed to create the bottleneck .npy files
    
    #__this can take an hour and half to run so only run it once. 
    #once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
    start = datetime.datetime.now()
     
    generator = datagen.flow_from_directory( 
        train_path, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode=None, 
        shuffle=False) 
    val_generator = datagen.flow_from_directory( 
        test_path, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode=None, 
        shuffle=False) 
    nb_train_samples = len(generator.filenames) 
    nb_test_samples = len(val_generator.filenames) 
    num_classes = len(generator.class_indices) 
    


    # # extracting train data 
    # print('[STATUS] features extraction process on train dataset running ...')
    # print('[TIME]: ', time.perf_counter())
    # bar = progressbar.ProgressBar(maxval=nb_train_samples, \
    # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # for i,file_path in tqdm(enumerate(generator.filenames)):
    #     img_path = train_path+ file_path
    #     Image = im_lite_loader(img_path)
    #     single_feature_table = lite_feature_ext(Image)
    #     train_features['img'].append(file_path)
    #     train_features['lite_features'].append(single_feature_table)
    #     bar.update(i+1)
        
    # print('[TIME]: ', time.perf_counter())
    # print('[STATUS] feature extraction process done')
    # if not os.path.exists('db/'):
    #     os.mkdir('db/')
    # print('[TIME]: ', time.perf_counter())
    # pkl_file_name = 'db/train_dataset.pkl'
    # with open(pkl_file_name,'wb') as f:
    #     pkl.dump(train_features,f)
        
    # # extracting test data 
    # print('[STATUS] features extraction process on test dataset running ...')
    # print('[TIME]: ', time.perf_counter())
    # bar = progressbar.ProgressBar(maxval=nb_test_samples, \
    # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # for i,file_path in tqdm(enumerate(val_generator.filenames)):
    #     img_path = test_path+ file_path

    #     Image = im_lite_loader(img_path)

    #     single_feature_table = lite_feature_ext(Image)
    #     test_features['img'].append(file_path)
    #     test_features['lite_features'].append(single_feature_table)
    #     bar.update(i+1)
        
    # print('[TIME]: ', time.perf_counter())
    # print('[STATUS] feature extraction process done')
    # if not os.path.exists('db/'):
    #     os.mkdir('db/')

    # print('[TIME]: ', time.perf_counter())
    # pkl_file_name = 'db/test_dataset.pkl'
    # with open(pkl_file_name,'wb') as f:
    #     pkl.dump(test_features,f)
    
    with open('db/train_dataset.pkl','rb') as f : 
        train_features = pkl.load(f)
    with open('db/test_dataset.pkl','rb') as f : 
        test_features = pkl.load(f) 
    print('model training process running ...')
    
    
    # load the bottleneck features saved earlier 
    train_data = np.array(train_features['lite_features'])
    test_data = np.array(test_features['lite_features'])
    # get the class labels for the training data, in the original order 
    train_labels = generator.classes 
    test_labels = val_generator.classes
    # convert the training labels to categorical vectors 
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)
    
    start = datetime.datetime.now()
    
    model_vi = model_config(train_data.shape)
    
    history = model_vi.fit(train_data, train_labels, 
       epochs=epochs,
       batch_size=batch_size, 
       validation_data=(test_data, test_labels))
    
    (eval_loss, eval_accuracy) = model_vi.evaluate( test_data, test_labels, batch_size=batch_size,verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
    print("[INFO] Loss: {}".format(eval_loss)) 
    end= datetime.datetime.now()
    elapsed= end-start
    print ("Time: ", elapsed)
    
    model_vi.save_weights(top_model_weights_path)
    
    #Graphing our training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy') 
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss') 
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
        
    
    
    
    
