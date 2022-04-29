# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:27:38 2020

@author: Cucu
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 01:16:23 2020

@author: Cucu
"""
#IMPORTING LIBRARIES
import tensorflow as tf
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator



#IMAGE DATA PREPROCESSING

#preprocessing the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        directory = r"C:\..\..\..\dataset\training_set",
        target_size=(64 , 64),
        batch_size=32,
        class_mode='binary')

#preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        directory = r"C:\..\..\..\dataset\test_set",
        target_size=(64 , 64),
        batch_size=32,
        class_mode='binary')


 
#BULDING THE CNN
#
#
#initialising the cnn
cnn = tf.keras.models.Sequential()


#convolution
cnn.add(tf.keras.Input(shape=(64, 64, 3)))
cnn.add(tf.keras.layers.Conv2D(filters = 32 , kernel_size = 3 , activation = 'relu' ))


#pooling
cnn.add(tf.keras.layers.MaxPool2D( pool_size = 2 , strides = 2))


#adding a SECOND CONVOLUTIONAL LAYER
cnn.add(tf.keras.layers.Conv2D(filters = 32 , kernel_size = 3 , activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D( pool_size = 2 , strides = 2))


#flattening
cnn.add(tf.keras.layers.Flatten())


#full connection
cnn.add(tf.keras.layers.Dense(units = 128 , activation = 'relu'))


#adding the output layer
cnn.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))




#TRAINING THE CNN
#
#
#compiling the cnn
cnn.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )


#training the cnn on the training_set & test_set
cnn.fit(x = training_set , validation_data = test_set , epochs = 25)



#MAKING A SINGLE PREDICTION
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r"C:\..\..\..\dataset\single_prediction\cat_or_dog_1.jpg" , target_size = (64 , 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis = 0)
result = cnn.predict(test_image)
training_set.class_indices


if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)