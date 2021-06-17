# -*- coding: utf-8 -*-
"""
Created on Sun May 30 00:55:19 2021

@author: shen
"""
import random
import cv2
import glob
import os
from audiomentations import SpecCompose, SpecFrequencyMask
# In[] 
class MyTrainGenerator():
    def __init__(self, directory, imgs_files, y_onehot, index, batch_size, audio_aug_range, aug_bool=True, audio_aug_bool=True):
        self.directory = directory
        self.y_onehot = y_onehot
        self.index = index
        self.batch_size = batch_size
        self.imgs_files = imgs_files
        self.aug_bool = aug_bool
        self.audio_aug_bool = audio_aug_bool
        self.audio_aug_range = audio_aug_range
        self.augmenter = SpecCompose([SpecFrequencyMask(fill_mode="constant",fill_constant=0.0,min_mask_fraction=0.03,max_mask_fraction=0.20,p=0.3)])

    def generate(self):
        while True:
            random.shuffle(self.index) #打亂順序
            image_batch = []
            y_batch = []
            
            for i in self.index:
                if self.audio_aug_bool:
                    aug_index = str(random.randint(0,self.audio_aug_range)) #隨機取一個編號
                    file = os.path.join(self.directory, self.imgs_files[i], f'{self.imgs_files[i]}_{aug_index}.png')
                else:
                    file = os.path.join(self.directory, self.imgs_files[i]+'.png')
                
                image = cv2.imread(file)
                if self.aug_bool:
                    image = self.augmenter(magnitude_spectrogram=image)
                    
                image = image/255.0
                image_batch.append( image )
                
                y = self.y_onehot[i]
                y_batch.append( y )
                
                if len(image_batch) == self.batch_size:
                    x_train = np.array(image_batch)
                    y_train = np.array(y_batch)
    
                    yield(x_train, y_train)
                    image_batch = []
                    y_batch = []

