# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:34:42 2021
Reference: https://github.com/ktjayamanna/Coursera-Courses/blob/main/Cats%20Vs%20Dogs%20Classifer%20Using%20TF%20and%20Keras.ipynb
@author: kjayamanna
@Description: This code can be used to split a dataset into Training, validation and test sets.
"""

import os
import random
from shutil import copyfile

#%%
#Folder that contains all the classes of images
pathDir = r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4\Generated'
#Destination you desire to copy the files.
targetPath = r'C:\Users\keven\OneDrive - University of Nebraska at Omaha\Dr.Khazanchi\Week 1\Medical Imaging\hw 4'
#%%
try:
    #Sub Directories
    allocation = ['training','validation', 'testing']
    
    #Sub Categories
    classes = os.listdir(pathDir)
    numImages = [len(os.listdir(pathDir + '\\' + i)) for i in classes]
    
    #create path
    path = os.path.join(targetPath, allocation[0]) 
    #Create Directory
    os.mkdir(path)
    # Create Path 
    path = os.path.join(targetPath, allocation[1]) 
    #Create Directory
    os.mkdir(path)
    # Create Path 
    path = os.path.join(targetPath, allocation[2]) 
    #Create Directory
    os.mkdir(path)
    
    #Create Folders
    for j in allocation:
        accessDir = os.path.join(targetPath, j)
        for i in classes:    
            # Create Path 
            path = os.path.join(accessDir, i) 
            #Create Directory
            os.mkdir(path)
except OSError:
    pass
#%%
SOURCE = pathDir
AllFiles = [];
Ignored = 0;
SPLIT_SIZE1 = 0.7
SPLIT_SIZE2 = 0.15

#put all the images and their weak label in a single list.
for className in classes:
    for FileName in os.listdir(os.path.join(SOURCE,className)):
        FilePath = os.path.join(SOURCE,className)
        if os.path.getsize(FilePath):
            AllFiles.append((FileName,className))
        else:
            Ignored += 1;
#Determine the amount of data each train, dev and test set get.            
BreakPoint1 = int(len(AllFiles)*SPLIT_SIZE1);
BreakPoint2 = int(len(AllFiles)*SPLIT_SIZE2);
#Shuffle the whole thing
Shuffled = random.sample(AllFiles,len(AllFiles));
#Distribute the filenames across three groups.
trainSet = Shuffled[:BreakPoint1];
devSet  = Shuffled[BreakPoint1:BreakPoint1 + BreakPoint2];
testSet = Shuffled[BreakPoint1 + BreakPoint2:];

# =============================================================================
# Copy Files
# =============================================================================
#%% Training set
for FileName in trainSet:
    copyfile(os.path.join(SOURCE,FileName[1], FileName[0]), os.path.join(targetPath,allocation[0], FileName[1],FileName[0]));
#%% Development Set
for FileName in devSet:
    copyfile(os.path.join(SOURCE,FileName[1], FileName[0]), os.path.join(targetPath,allocation[1], FileName[1],FileName[0]));
#%% Test Set
for FileName in testSet:
    copyfile(os.path.join(SOURCE,FileName[1], FileName[0]), os.path.join(targetPath,allocation[2], FileName[1],FileName[0]));




