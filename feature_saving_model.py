#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:51:14 2024

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
                    for filename in os.listdir(subfolder_path)[:60]:
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
# num_samples, height, width, channels = images_135.shape
# sequence_len = 9
# if num_samples % sequence_len != 0:
#     raise ValueError('Number of samples should divide by sequence_length')
# new_shape = (num_samples // sequence_len, sequence_len, height, width, channels)
# reshape_images_135 = images_135.reshape(new_shape)

#%%
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow as tf
loaded_model = load_model('best_model.h5')
model = Model(inputs = loaded_model.input, outputs = loaded_model.layers[-2].output)
def extract_features(img_array135,img_array270,img_array540):
    img_array135 = tf.expand_dims(img_array135,0)
    img_array270 = tf.expand_dims(img_array270,0)
    img_array540 = tf.expand_dims(img_array540,0)
    features = model.predict([img_array135,img_array270,img_array540])
    return features.flatten()
#features_d = extract_features(images_135[0], images_270[0], images_540[0])
#print(features)
# Example usage for all images in each array
num_images = images_135.shape[0]

# Assuming images_135, images_270, and images_540 are lists containing pre-processed images
images_features = np.array([extract_features(images_135[i], images_270[i], images_540[i]) for i in range(num_images)])
#%%
#model.summary()
"""Reshape Features as a sequence of 9 images"""
num_samples, num_features = images_features.shape
num_sequences = num_samples // 9
feature_sequences = images_features[:num_sequences*9].reshape((num_sequences,9,num_features))
#%%
seq_labels = []

for i in range(num_sequences):
    group_labels = labels[i * 9: (i + 1) * 9]
    if len(set(group_labels)) == 1:
        seq_labels.append(group_labels[0])
#%%
"""SAVE the Grouped Sequences for traing the downstream transformer network"""
for i, (seq,label) in enumerate(zip(feature_sequences, seq_labels)):
    np.save(f'training_sample_{i+1}.npy', {'features':seq, 'label':label})
#%%
