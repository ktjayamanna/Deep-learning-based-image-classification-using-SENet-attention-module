# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 19:40:44 2021
This code derives the sailency map for a given image.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import matplotlib.cm as cm
plt.close('all')
#%%
def saliency(img, model):
    #preprocess
    img_original = np.copy(img)
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape((1, *img.shape))
    y_prediction = model.predict(img)
    #%% Calculate gradient to find the most significant pixels
    images = tf.Variable(img, dtype=float)
    with tf.GradientTape() as tape:
        prediction = model(images, training=False)
        classIdxSorted = np.argsort(prediction.numpy().flatten())[::-1]
        loss = prediction[0][classIdxSorted[0]]    
    grads = tape.gradient(loss, images)
    #%% Get the absolute value
    grad_abs = tf.math.abs(grads)
    #%% Find the max absolute gradient values along RGB channels.
    grad_abs__max = np.max(grad_abs, axis=3)[0]
    #%% Normalize between 0 - 1
    channel_min, channel_max  = np.min(grad_abs__max), np.max(grad_abs__max)
    heatmap = (grad_abs__max - channel_min) / (channel_max - channel_min + 1e-18)
    return heatmap


