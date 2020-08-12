# -*- coding: utf-8 -*-

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
import time

#Default dimensions we found online
img_width, img_height = 224, 224 
 
#Create a bottleneck file
top_model_weights_path = 'bayesian_model.h5'
# loading up our datasets
train_data_dir = 'data/train'
test_data_dir = 'data/test'
 
# number of epochs to train top model 
epochs = 5 #this has been changed after multiple model run 
# batch size used by flow_from_directory and predict_generator 
batch_size = 50 
#Loading vgc16 model

vgg16 = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255) 
#needed to create the bottleneck .npy files

#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
# start = datetime.datetime.now()
 
# generator = datagen.flow_from_directory( 
#     train_data_dir, 
#     target_size=(img_width, img_height), 
#     batch_size=batch_size, 
#     class_mode=None, 
#     shuffle=False) 
# val_generator = datagen.flow_from_directory( 
#    test_data_dir, 
#    target_size=(img_width, img_height), 
#    batch_size=batch_size, 
#    class_mode=None, 
#    shuffle=False) 

# nb_train_samples = len(generator.filenames) 
# nb_test_samples = len(val_generator.filenames) 

# num_classes = len(generator.class_indices) 


# predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
# predict_size_test =  int(math.ceil(nb_test_samples / batch_size)) 

# bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train) 
# bottleneck_features_test = vgg16.predict_generator(val_generator, predict_size_train) 

# np.save('bottleneck_features_train.npy', bottleneck_features_train)
# np.save('bottleneck_features_test.npy', bottleneck_features_test)

# end= datetime.datetime.now()
# elapsed= end-start
# print ('Time: ', elapsed)

#training data
generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
#test data
val_generator_top = datagen.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 

nb_train_samples = len(generator_top.filenames) 
nb_test_samples = len(val_generator_top.filenames) 
print('found {} train data'.format(nb_train_samples))
print('found {} test data'.format(nb_test_samples))
num_classes = len(generator_top.class_indices) 
print('nuber of classes : ',num_classes)
 
# load the bottleneck features saved earlier 
train_data = np.load('bottleneck_features_train.npy') 
test_data = np.load('bottleneck_features_test.npy') 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
test_labels = val_generator_top.classes
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

start = datetime.datetime.now()

kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (train_data.shape[0] *1.0)

model_vi = Sequential()

model_vi.add(Flatten(input_shape=train_data.shape[1:])) 
model_vi.add(tfp.layers.DenseFlipout(100, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.DenseFlipout(50, activation = keras.layers.LeakyReLU(alpha=0.3), kernel_divergence_fn=kernel_divergence_fn))
model_vi.add(tfp.layers.DenseFlipout(num_classes, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))

model_vi.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
model_vi.summary()

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

