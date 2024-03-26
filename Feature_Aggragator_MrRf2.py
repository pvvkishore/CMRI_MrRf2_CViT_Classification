#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:20:38 2024

@author: user
"""
import os
import cv2
import numpy as np
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
def load_images_from_floder(folder_path,width,height):
    images = []
    labels = []
    class_labels = []
    class_label_counter = 0
    
    for class_label, class_name in enumerate(os.listdir(folder_path)):
        #print(class_label, class_name)
        class_path = os.path.join(folder_path,class_name)
        if os.path.isdir(class_path):
            for subfolder_name in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path)[:70]:
                        img_path = os.path.join(subfolder_path, filename)
                        if os.path.isfile(img_path):
                            img = cv2.imread(img_path)
                            img = cv2.resize(img, [width,height])
                            channels = cv2.split(img)
                            contrast_channels = [cv2.equalizeHist(channel) for channel in channels]
                            blurr_channels = [cv2.GaussianBlur(channel,[5,5],0) for channel in contrast_channels]
                            img_en_channels = [clahe.apply(channel) for channel in blurr_channels]
                            img_contrast_enhanced = cv2.merge(img_en_channels)
                            images.append(img_contrast_enhanced)
                            labels.append(class_label_counter)
                            class_labels.append(class_name)
            class_label_counter += 1
    return np.array(images), np.array(labels), class_labels
#%%
""" Test Module for the above Function"""
folder_path = "/home/user/Desktop/2023MRI_Image_DL/Dataset_Training_Tiny"
images_135, labels, class_labels = load_images_from_floder(folder_path,135, 135)
images_270, labels, class_labels = load_images_from_floder(folder_path,270, 270)
images_540, labels, class_labels  = load_images_from_floder(folder_path,540, 540)
#%%
print('images_135:',images_135.shape)
print('images_270:',images_270.shape)
print('images_540:',images_540.shape)
print('Labels:',labels.shape)
#%%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3,3, figsize = (10,10))
# for i in range(3):
#     for j in range(3):
#         axes[i,j].imshow(images_135[i+j])
#         axes[i,j].axis('off')
#         print(i)
#%%
""" Load Required Libraries"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Multiply, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
#%%
""" Prepare Data for the network"""
number_of_classes = 5
input_shape_135 = np.array([135,135,3])
input_shape_270 = np.array([270, 270, 3])
input_shape_540 = np.array([540, 540, 3])
X_train_135, X_test_135, y_train_135, y_test_135 = train_test_split(images_135, labels, test_size=0.1, random_state=42)
X_train_270, X_test_270, y_train_270, y_test_270 = train_test_split(images_270, labels, test_size=0.1, random_state=42)
X_train_540, X_test_540, y_train_540, y_test_540 = train_test_split(images_540, labels, test_size=0.1, random_state=42)
print(f"X_train_135 SHAPE:{X_train_135.shape} - y_train_135 SHAPE: {y_train_135.shape}")
print(f"X_test_135 SHAPE:{X_test_135.shape} - y_test_135 SHAPE: {y_test_135.shape}")
print(f"X_train_270 SHAPE:{X_train_270.shape} - y_train_270 SHAPE: {y_train_270.shape}")
print(f"X_test_270 SHAPE:{X_test_270.shape} - y_test_270 SHAPE: {y_test_270.shape}")
print(f"X_train_540 SHAPE:{X_train_540.shape} - y_train_540 SHAPE: {y_train_540.shape}")
print(f"X_test_540 SHAPE:{X_test_540.shape} - y_test_540 SHAPE: {y_test_540.shape}")
#%%
"""Build the multi resolution residual feature fusion - MrRf2"""
def build_MrRf2_model(input_shape = [input_shape_135, input_shape_270,input_shape_540], number_of_classes = number_of_classes):
    input_tensor_135 = Input(shape = input_shape[0])
    input_tensor_270 = Input(shape = input_shape[1])
    input_tensor_540 = Input(shape = input_shape[2])
    
    x = Conv2D(128, (3,3),activation='relu')(input_tensor_135)
    x = Conv2D(128,(3,3),activation = 'relu')(x)
    x_131 = x
    x = MaxPooling2D((2,2),strides = (2,2))(x)
    x = Conv2D(256,(3,3),activation = 'relu')(x)
    x = Conv2D(256,(3,3),activation = 'relu')(x)
    
    x = MaxPooling2D((2,2),strides = (2,2))(x)
    x_30 = x
    x = Conv2D(512,(3,3),activation = 'relu')(x)
    x = Conv2D(512,(3,3),activation = 'relu')(x)
    
    x = MaxPooling2D((2,2),strides = (2,2))(x)
    x_13 = x
    x = Conv2D(1024,(3,3),activation = 'relu')(x)
    x = Conv2D(1024,(3,3),activation = 'relu')(x)
    
    x = MaxPooling2D((2,2),strides = (2,2))(x)
    
    z = Conv2D(64, (3,3),activation='relu')(input_tensor_540)
    z = Conv2D(64,(3,3),activation = 'relu')(z)
    
    z = MaxPooling2D((2,2),strides = (2,2))(z)
    z = Conv2D(128,(3,3),activation = 'relu')(z)
    z1 = Conv2D(64, (1,1), padding= 'same', activation='relu')(z)
    z = Conv2D(128,(3,3),activation = 'relu')(z)
    
    z = MaxPooling2D((2,2),strides = (2,2))(z)
    z = Conv2D(256,(3,3),activation = 'relu')(z)
    z = Conv2D(256,(3,3),activation = 'relu')(z)
    
    z = MaxPooling2D((2,2),strides = (2,2))(z)
    z = Conv2D(512,(3,3),activation = 'relu')(z)
    z = Conv2D(512,(3,3),activation = 'relu')(z)
    z2 = Conv2D(256, (1,1),padding='same',activation='relu')(z)
    z = MaxPooling2D((2,2),strides = (2,2))(z)
    z = Conv2D(1024,(3,3),activation = 'relu')(z)
    z = Conv2D(1024,(3,3),activation = 'relu')(z)
    
    z = MaxPooling2D((2,2),strides = (2,2))(z)
    z = Conv2D(1024,(3,3),activation = 'relu')(z)
    z3 = z
    z = Conv2D(1024,(3,3),activation = 'relu')(z)
    
    z = MaxPooling2D((2,2),strides = (2,2))(z)
                
    y = Conv2D(64, (3,3),activation='relu')(input_tensor_270)
    y = Conv2D(64,(3,3),activation = 'relu')(y)
    y = Multiply()([z1,y])
    y = MaxPooling2D((2,2),strides = (2,2))(y)
    y = Conv2D(128,(3,3),activation = 'relu')(y)
    y = Multiply()([y,x_131])
    y = Conv2D(128,(3,3),activation = 'relu')(y)
    
    y = MaxPooling2D((2,2),strides = (2,2))(y)
    y = Conv2D(256,(3,3),activation = 'relu')(y)
    y = Conv2D(256,(3,3),activation = 'relu')(y)
    y = Multiply()([z2,y])
    y = MaxPooling2D((2,2),strides = (2,2))(y)
    y = Multiply()([y,x_30])
    y = Conv2D(512,(3,3),activation = 'relu')(y)
    y = Conv2D(512,(3,3),activation = 'relu')(y)
    
    y = MaxPooling2D((2,2),strides = (2,2))(y)
    y = Multiply()([y,x_13])
    y = Conv2D(1024,(3,3),activation = 'relu')(y)
    y = Multiply()([z3,y])
    y = Conv2D(1024,(3,3),activation = 'relu')(y)
    
    y = MaxPooling2D((2,2),strides = (2,2))(y)
    #concat = Concatenate(axis = -1)([x,y,z])
    y = Flatten()(y)
    
    y = Dense(512, activation = 'relu')(y)
    y = Dense(number_of_classes,activation = 'softmax')(y)
    
    model = Model(inputs = [input_tensor_135, input_tensor_270,input_tensor_540], outputs = y)
    return model
Model_MrRf2 = build_MrRf2_model(input_shape= [input_shape_135, input_shape_270,input_shape_540], number_of_classes=number_of_classes)
Model_MrRf2.summary()
#%%

# Compile the combined model for training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
Model_MrRf2.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#%%
from tensorflow.keras.callbacks import ModelCheckpoint

# Specify the path where you want to save the best model
checkpoint_path = 'best_model.h5'

# Create a ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='accuracy',  # Choose the metric to monitor (e.g., val_accuracy, val_loss)
    save_best_only=True,
    mode='max',  # 'max' if monitoring accuracy, 'min' if monitoring loss
    verbose=1
)
#%%
# Train the combined model
epochs = 20
batch_size = 5
# Replace the following with your actual training data and parameters
history = Model_MrRf2.fit(
    [X_train_135, X_train_270, X_train_540],
    y_train_135,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=([X_test_135, X_test_270, X_test_540],y_test_135),
    callbacks = [model_checkpoint]
    )
#%%

# # Extract features using the trained CNN model
# feature_model = tf.keras.models.Model(inputs=Model_MrRf2.inputs, outputs=Model_MrRf2.layers[-2].output)
# train_features = feature_model.predict([X_train_135,X_train_270,X_train_540])
# test_features = feature_model.predict([X_test_135,X_test_270, X_test_540])

# # Save the features and labels for downstream classification tasks
# np.save('train_features.npy', train_features)
# np.save('test_features.npy', test_features)
# np.save('train_labels.npy', y_train_135)
# np.save('test_labels.npy', y_test_135)
