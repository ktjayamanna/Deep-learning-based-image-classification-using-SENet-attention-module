# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:35:23 2021

@author: kjayamanna
"""
import os
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot
import Saliency as sm
from keras import backend as K
import matplotlib.cm as cm
#%%
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    return array
#%%
img_path = r"C:\Users\keven\Desktop\my\trilock_bernadette_xray_x-1.31_y-9.21_z16.59_att0.08.png"
img_size = (224, 224)
path = r"C:\Users\keven\OneDrive - University of Nebraska at Omaha\Fall 2021\CSCI 8300\Project B\se_after"
my_models = os.listdir(path)
super_imposed = []
for i in my_models:
    temp_path = os.path.join(path,i)
    model = keras.models.load_model(temp_path);
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    try:
        heatmap = sm.saliency(img_array, model)
    except ValueError:
        heatmap = sm.saliency(img_array, model)
    super_imposed.append(heatmap)

#%% Before
fig = pyplot.figure(figsize=(20,15))
position = 1
for i in range(1,len(super_imposed)+1):
    pyplot.subplot(1,7,i)
    pyplot.imshow(super_imposed[i-1]/255,cmap="jet",alpha=0.8)   
    pyplot.title('Position ' + f'{position}' )
    position += 1
pyplot.show()
pyplot.suptitle("Before CBAM which is located between pooling and dense")





