#test.py
# Load tflite model and allocate tensorsimport tflite_runtime.interpreter as tflite
import numpy as np
#Used for reading image from path or using live camera
import cv2
import tensorflow as tf 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img_path", type=str, help="input image for testing model")
parser.add_argument("model_path",type=str, help="tflite model for make prediction", default="tf_lite_model.tflite")
args = parser.parse_args()

    
interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#Load input image
input_image = cv2.imread(args.img_path)
input_shape = input_details[0]['shape']
#Reshape input_image 
input_image = cv2.resize(input_image,(input_shape[1],input_shape[2]))
input_image = np.expand_dims(input_image,axis=0)
input_image = np.array(input_image,dtype=np.float32)
#Set the value of Input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)


interpreter.invoke()
#prediction for input data
output_data = interpreter.get_tensor(output_details[0]['index'])
probability = output_data[0]

probability.max()
preds = [x.mean() for x in probability]
pred = np.argmax(preds)
