#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:52:48 2024

@author: user
"""
import os
folder_path = 'MrRf2_Train_Features_for_Transformer'
files = os.listdir(folder_path)
num_features = len(files)
import numpy as np
features_for_classification = []
labels_for_classification = []
for i in range(1, num_features+1):
    file_path = os.path.join(folder_path, f'training_sample_{i}.npy')
    
    loaded_data = np.load(file_path, allow_pickle=True).item()
    
    # Access Features and Labels
    features = loaded_data['features']
    label = loaded_data['label']
    
    features_for_classification.append(features)
    labels_for_classification.append(label)
#features_for_classification = np.concatenate(features_for_classification, axis=0)
#labels_for_classification = np.array(labels_for_classification)
#%%
"""The final Transformer classifer"""
import tensorflow as tf
from tensorflow.keras import layers

input_shape = (9,512)
num_classes = 5

""" Design the transformer Netwrok"""
def transformer_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape = input_shape, name = "input_features_ACDC_CMRI")
    
    # Self Attention (SA) Network with MHA
    x = layers.MultiHeadAttention(num_heads = 12, key_dim = 512)(inputs, inputs)
    x = layers.Dropout(0.1)(x)
    res = layers.Add()([inputs,x])
    
    # Feed_Forward Network
    x = layers.Conv1D(filters = 512, kernel_size = 1, activation = 'relu')(res)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters = 512, kernel_size = 1, activation = 'linear')(x)
    x = layers.Dropout(0.1)(x)
    res = layers.Add()([res,x])
    
    # GAP layer
    x = layers.GlobalAveragePooling1D()(res)
    
    # Fully connected Dense layers
    x = layers.Dense(256, activation = 'relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation = 'relu')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation = 'softmax', name = 'output_class')(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'transformer_classifier_CMRI')
    return model

model = transformer_model(input_shape, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000001) 
model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()
#%%
features_for_classification = np.array(features_for_classification)
labels_for_classification = np.array(labels_for_classification)
labels_for_classification = labels_for_classification.astype(np.int32)
# Example of feature normalization
# features_for_classification = (features_for_classification - 
#                                features_for_classification.mean(axis=0)) / features_for_classification.std(axis=0)

#%%
"""Train the classifier"""
history = model.fit(features_for_classification, labels_for_classification,
          epochs = 200, batch_size = 128, validation_split = 0.3, verbose = 1)
#%%
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    