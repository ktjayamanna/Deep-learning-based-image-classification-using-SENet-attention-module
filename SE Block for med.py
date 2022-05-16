# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:30:22 2021

@author: kjayamanna
Ref: https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
"""
#%% Import Packages
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#%% You only need this cell if you want to use the SENet.
#swap is the layer index of the layer before SENet. (index starts at 0)
swap = 6
#Number of Layers
numSELayers = 5
#Ratio of the SENet compression. Found the Value from the paper.
rat = 1
#%% 
# model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
#%% Define the SENet.
def se_block(input_feature, ratio=rat):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    # Get the channel shape
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    #Average the color channels
    se_feature = layers.GlobalAveragePooling2D()(input_feature)
    se_feature = layers.Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    #Calculate the number of hidden units.
    denseShape = channel // ratio
    #Define the 1st dense layer.
    se_feature = layers.Dense(denseShape,
                       # activation=None,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)

    
    assert se_feature.shape[1:] == (1,1,denseShape)
    #Define the 2nd dense layer.
    se_feature = layers.Dense(channel,
                       # activation= None,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)


    assert se_feature.shape[1:] == (1,1,channel)
    #Multiply the input channels with the calculated scaler per channel (the vector).
    se_feature = layers.Multiply()([input_feature, se_feature])
    return se_feature

#%%
#Define Input Shape
input_shape = (224,224,3)
#Define Model Input
img_input = layers.Input(shape=input_shape)
#Load VGG16 with ImageNet Weights
trained_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
#%% Save the weights seperately
trainedWeights = []
for i in trained_model.layers:
  trainedWeights.append(i.get_weights())

#%%
'''
You may use SENets one at a time. So, uncomment the ones that should be included.
'''

# x = se_block(img_input)
x = img_input
# Block ,1
x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(x)



x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#
# x = se_block(x)
#



# Block 2
x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(x)
x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# # #
#Currently, we placed the SEblock between block2_conv2 and block3_conv1.
x = se_block(x)
# # #

# Block 3
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(x)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# # #
# x = se_block(x)
# # #

# Block 4
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# # #
# x = se_block(x)
# # #

# Block 5
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv1')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv2')(x)
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# #
# x = se_block(x)
# #

# Classification block
x = layers.Flatten(name='flatten1')(x)
x = layers.Dense(4096, activation='relu', name='fc1')(x)
x = layers.Dense(4096, activation='relu', name='fc2')(x)
#Define the number of classes
classes = 9
#Define the Softmax layer.
x = layers.Dense(classes, activation='softmax', name='Output')(x)
#Swap the old softmax with the new.
model = Model(img_input, x, name='vgg16')
#%% Take care of predecessors of SeNet
## Uncomment this for loop and comment below two for loops along with the SEBlock call, if you need the base model.
# for k in range(len(trainedWeights)-1):
#     model.layers[k].set_weights(trainedWeights[k])
#     model.layers[k].trainable = False

#Uncomment these two and comment the above for loop if you don't need the base model.
for k in range(0, swap + 1):
    model.layers[k].set_weights(trainedWeights[k])
    model.layers[k].trainable = False
#%  -1 in the loop to exclude the last layer swap
for i in range(swap + 1 + numSELayers,len(trainedWeights) + numSELayers - 1):
  model.layers[i].set_weights(trainedWeights[i-numSELayers])
  model.layers[i].trainable = False
#%% Make the last layer Trainable just in case if it is not already.
model.layers[-1].trainable = True

#%%
#Num of images the model run through before updating the model weights.
batch_size = 16
# This is the augmentation configuration we will use for training.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

#%%Define the image generators.
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\training',  # this is the target directory
        target_size=(224, 224), 
        batch_size=batch_size,
        class_mode='sparse'
        )  
#%
validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse'
        )
#%%Define the optimizer.
optimizer = keras.optimizers.Adam(lr=3e-4)
#%%  Compile the model.
model.compile(loss= keras.losses.SparseCategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'])
#%% Start training the model for 20 epochs.
hist = model.fit(
        train_generator,
        validation_data=validation_generator, verbose=1,epochs=20)

#%% Get the Training metrics
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs=range(len(acc)) # Get number of epochs
#%%
# =============================================================================
# Plot training and validation accuracy per epoch
# =============================================================================
plt.figure()
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training & Validation accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.title('Training & Validation loss Using Keras Early Stopping')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
#%% Define test image generator.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\testing',
        target_size=(224, 224),
        batch_size=1,
        class_mode='sparse',
        shuffle = False,
        )
#%% Make predictions with test generator.
preds = model.predict(test_generator)
#%%
#Get the predictions
y_preds = preds.argmax(axis=1)
#get the actual labels.
y_true = test_generator.classes 
#%%This function calculates the four regions of the confusion matrix.
def my_confusion(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)
#
TP, FP, TN, FN = my_confusion(y_true, y_preds)
#%% Calculate confusion matrix metrics.
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)















