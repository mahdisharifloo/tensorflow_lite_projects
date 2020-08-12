# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import  lite
from tensorflow.keras.applications import nasnet
from tensorflow.keras.applications import vgg16 , vgg19,resnet50,inception_v3,inception_resnet_v2
from tensorflow.keras.applications import mobilenet,xception,densenet,mobilenet_v2

def model_converter(model):
    converter = lite.TFLiteConverter.from_keras_model(model)
    tf_lite_model =  converter.convert()
    lite_name = model.name +'.tflite'
    print(lite_name)
    open(lite_name,'wb').write(tf_lite_model)
    
    
    
nasnet_mobile = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
# nasnet_large = nasnet.NASNetLarge(weights='imagenet', include_top=False,input_shape=(331, 331, 3))
vgg_16 = vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
vgg_19 = vgg19.VGG19(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
resnet_50 = resnet50.ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
inception_v3_ = inception_v3.InceptionV3(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
inception_resnet_v2_ = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
mobile_net = mobilenet.MobileNet(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
xception_ = xception.Xception(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
dense_net = densenet.DenseNet201(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
mobilenet_v2_ = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

models_list = [nasnet_mobile,vgg_16,vgg_19,resnet_50,inception_resnet_v2_,
               inception_v3_,mobile_net,xception_,dense_net,mobilenet_v2_]

for model in models_list:
    model_converter(model)


