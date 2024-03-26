#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:38:16 2023

@author: root
"""

import cv2
import os
import numpy as np
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
def load_images_from_folder(folder_path):
    images = []
    labels = []
    class_labels = {}
    class_label_counter = 0
    
    for class_label, class_name in enumerate(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            class_labels[class_label] = class_name
            
            for subfolder_name in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path)[:50]:
                        img_path = os.path.join(subfolder_path, filename)
                        if os.path.isfile(img_path):
                            img = cv2.imread(img_path)
                            img = cv2.resize(img,(264,264))
                            channels = cv2.split(img)
                            contrast_channels = [cv2.equalizeHist(channel) for channel in channels]
                            img_CE = cv2.merge(contrast_channels)
                            img_blur = cv2.split(img_CE)
                            blurr_mri_channels = [cv2.GaussianBlur(channel,[5,5],0) for channel in img_blur]
                            mri_blurred = cv2.merge(blurr_mri_channels)
                            final_channels = cv2.split(mri_blurred)
                            img_en_channels = [clahe.apply(channel) for channel in final_channels]
                            final_image_merge = cv2.merge(img_en_channels)
                            images.append(final_image_merge/255.0)
                            labels.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images), np.array(labels),class_labels

folder_path = "/home/user/Desktop/2023MRI_Image_DL/Dataset_Training"
images,labels,class_labels = load_images_from_folder(folder_path)
# image_label_pairs = list(zip(images,labels))
# import random
# random.seed(42)
# random.shuffle(image_label_pairs)
# select_image_label_pairs = random.sample(image_label_pairs,50)
# images_50 = []
# labels_50 = []
# for idx, (image,label) in enumerate(select_image_label_pairs):
#     images_50.append(image)
#     labels_50.append(label)
    
    
#%%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3,3,figsize=(10,10))
# for i in range(3):
#     for j in range(3):
#         axes[i,j].imshow(images[j+i])
#         axes[i,j].axis('off')
        #print(i)
#cv2.imshow('Enhanced',images[10])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#%%
# Load necessary Libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
# %%
# Data preparation
num_classes = 5
input_shape = (264, 264, 3)
X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.2,random_state=42)
# %%
# f in print is to tell interpreter to print it as a string
# Include evaluations in {}
print(f"X_train Shape: {X_train.shape} - y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape} - y_test shape: {y_test.shape}")
# %%
# HyperParameter Configuration
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 5
image_size = 264
patch_size = 22
num_patches = (image_size // patch_size) ** 2
# Why number of patchesa are squared
projection_dim = 64
# What is projection dimension????
num_heads = 4
transformer_units = [projection_dim*2, projection_dim]
# Size of transformer layers -- Means what??
transformer_layers = 8
mlp_head_units = [2048, 1024]
# Size of dense layers of the final classifier
# mlp -- multi layer perceptron
# %%
# Implimentation of MLP
# Multi Layer Perceptron


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
# How this mlp model is created.
# hidden_units --> number of hidden layers
# gelu - Gaussian Error Linear Unit Activation Function
# %%
# Break the image into patches and embedded it as a layer
class Patches(layers.Layer):
    def __init__(self, pathch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size,
                                                  self.patch_size, 1],
                                           strides=[1, self.patch_size,
                                                    self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID",
                                           )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

""" Patches is written as a sub-class of layers.Layer class.
__init__ is a constructor to initialize the its attributes with an
argument patch_size. super().__init__() is used to call the 
parent class layers.Layer constructor. WHY? To ensure necessary
initializations from parent class
CALL Method is crucial for custom tensorflow layers. Defines forward pass
of the layer processess input data
batch_size of input in images tensor
tf.image.extract_patches is a tf function to extract patches from images
using patch_size and strides.
patch_dims computes number of dimensions in patches"""
# %%
# Check the above code block
plt.figure(figsize=(4, 4))
image = X_train[np.random.choice(range(X_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size))
patches = Patches(patch_size)(resized_image)
print(f"Image Size: {image_size} X {image_size}")
print(f"Patch_size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per Patch: {patches.shape[-1]}")
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i+1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
#%%
""" Impliment a patch encoding layer where it projects a patch on
to a vector of size projection_dim along with a learnable positional 
embedding layer"""
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units = projection_dim)
        self.positional_embedding = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim)
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.positional_embedding(positions)
        return encoded
#%%
# Test the PatchEncoder model
""""""
sample_patch = patches[0]
patch_encoder = PatchEncoder(num_patches = num_patches, projection_dim = projection_dim)
encoded_patch = patch_encoder(sample_patch)
print(encoded_patch.shape)
#%%
"""Build the Vision Transformer Model"""
def create_vit_classifier():
    inputs = layers.Input(shape = input_shape)
    # Augument data
    #augmented = data_agumentation(inputs)
    # Create Patches
    patches = Patches(patch_size)(inputs)
    # Encode Patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Cretate multiple layers of transformer blocks
    for _ in range(transformer_layers):
        # Normalization layer 1
        x1 = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        # Multi-Head Attention Layer
        attention_output = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = projection_dim, dropout = 0.1
            )(x1,x1)
        # skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Normalization layer 2
        x3 = layers.LayerNormalization(epsilon = 1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate = 0.1)
        # Skip Connection
        encoded_patches = layers.Add()([x3,x2])
    # Create a batch_size projection_dim tensor
    representation = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # ADD MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify
    logits = layers.Dense(num_classes)(features)
    # Create Keras Model
    model = keras.Model(inputs=inputs, outputs = logits)
    return model
#%%
""" Compile Train and Evaluate Model""" 
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay = weight_decay)
    model.compile(optimizer=optimizer, 
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = [
                      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                      keras.metrics.SparseTopKCategoricalAccuracy(5, name = "top-5-accuracy"),
                      ],
                  )
    checkpoint_filepath = "/home/user/Desktop/2023MRI_Image_DL/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    model.save('saved_model.h5')
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    
    predictions = model.predict(X_test)
    predictions_max = tf.argmax(predictions, axis=-1)
    
    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
#%%
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
#%%
from keras.models import load_model
loaded_model = load_model('/home/user/Desktop/2023MRI_Image_DL/tmp/checkpont')