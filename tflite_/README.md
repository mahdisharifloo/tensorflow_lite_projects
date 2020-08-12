# TfLite
this service work with tensorflow lite models for features extraction.
our goal is using small features and models for making our process faster. 

## how to make tensorflow lite models and using them

1. **make your normal keras or tensorflow model or import it from keras applications**   
    in this process we used nasnetMobile
2. **convert h5 model to tf lite model by tf_converter.py**
```python
# this lines convert models to tflite model
converter = lite.TFLiteConverter.from_keras_model(normall_model)
tf_lite_model =  converter.convert()
open('tf_lite_model.tflite','wb').write(tf_lite_model)
```
    after run this file you can find *tf_lite_model.tflite* you need this model for next process.  
3. **loading tflite models and make prediction with it.**
    - in this part we have to define interpreter for loading tflite model 
    - giving input and output shape and tensors
    - fiting image to input tensors 
    - giving results from last layer of model as output by giving output tensors  

## comparing similarity service that useing normal nesnet by service that useing lite nasnet.

if you look at this file : **similarity_lite_vs_normal.py**  
we have 2 image file that we want to compare similarity between them by 2 method
first method is normal method that we used befour in evoke services and second method is tflite method.  

we have 3 process:
1. input images and make it ready for neural network
    ```python
    def im_normal_loader(im_path):
    ...

    def im_lite_loader(im_path):
    ...
    ```
    this functions do this job for normal and lite models.
2. feature extraction process.
    ```python
    def NASNetMobile(image_bytes):
    ...
    def lite_feature_ext(input_image):
    ...
    ```
    this functions extract features from images.
3. comparing process. 
#### how to run similarity_lite_vs_normal.py
```bash
python <first image path> <second image path> tf_lite_model.tflite
```
## clustering process 
### pm runner service 
in this service you can make database of features and cluster them by kmeans clustering method.  
this process need setup befour running. 
in setup process we want to make database and kmeans clustering object that save them in db/ directory.
in running process you can set single image and giving the 10 top of the similar images that we have in database.
in running process we are using clustering method to making our searching space smaller.

### how to run setup process
```bash
cd pm_runner
python setup.py <csv file that crawlers make them befour that has image addresses and product id> <input root directory of data> <input number of clusters that you need to make> ../tf_lite_model.tflite
```
for example
```bash
python setup.py /home/mahdi/projects/dgkala/data/category-men-shoes/category-men-shoes.csv /home/mahdi/projects/dgkala/ 20 ../tf_lite_model.tflite
```
### how to run running process
```bash
python run.py <input your database> <single image path> <kmeans_object> <tflite model>
```
for example:
```bash
python run.py db/dataset.pkl /home/mahdi/Pictures/images.jpeg db/kmeans_dataset.pkl ../tf_lite_model.tflite
```
