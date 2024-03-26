#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:27:47 2024

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
                    for filename in os.listdir(subfolder_path)[:40]:
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
""" Load Required Libraries"""
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Multiply, Concatenate
from keras.layers import LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
#%%
""" Prepare Data for the network"""
number_of_classes = 5
input_shape_135 = (135,135,3)
input_shape_270 = (270, 270, 3)
input_shape_540 = (540, 540, 3)
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
""" Construct a Multi Resolution Feature Fusion Network for extracting features"""
input_tensor_135 = Input(shape = input_shape_135)
input_tensor_270 = Input(shape = input_shape_270)
input_tensor_540 = Input(shape = input_shape_540)

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

y_features = Dense(512, activation = 'relu')(y)
y = Dense(number_of_classes,activation = 'softmax')(y_features)

feature_aggragator_model = Model(inputs = [input_tensor_135, input_tensor_270,input_tensor_540], outputs = y, name = "MrRf2_CNN")
#c
#dot_image_file = '/home/user/Desktop/2023MRI_Image_DL/feature_aggragator.png'
#tf.keras.utils.plot_model(feature_aggragator_model, to_file = dot_image_file, show_shapes = True)
transformer_input = feature_aggragator_model.get_layer('dense').output
# Transformer blocks
def transformer_encoder(inputs, embedding_dim, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout)(inputs, inputs, inputs)
    x = Dropout(dropout)(x)
    res = LayerNormalization(epsilon=1e-6)(inputs + x)

    # Feed Forward Part
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(embedding_dim)(x)
    return LayerNormalization(epsilon=1e-6)(res + x)

# Assuming you have defined embedding_dim, num_heads, etc.
embedding_dim = 512
num_heads = 8
ff_dim = 4
dropout = 0.1
num_transformer_blocks = 2
x = transformer_input
print(x)
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

# Global Average Pooling
x = GlobalAveragePooling1D(data_format='channels_last')(x)

# MLP Head
mlp_units = [128]
mlp_dropout = 0.4
for dim in mlp_units:
    x = Dense(dim, activation='relu')(x)
    x = Dropout(mlp_dropout)(x)

# Output layer
outputs = Dense(number_of_classes, activation='softmax')(x)

# Build the transformer classifier model
transformer_classifier_model = Model(inputs=feature_aggragator_model.input, outputs=outputs)
transformer_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

transformer_classifier_model.summary()


