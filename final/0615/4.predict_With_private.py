# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:55:13 2021

@author: shen
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# In[]
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import keras

#model_folder = 'models_0610_fmax4_backnoise'

# In[function]
import cv2

def Predictgenerate(directory, batch_size):
    imgs = glob.glob(f"{directory}*.*")
    while True:
        image_batch = []
        for i, img in enumerate(imgs):
            i += 1
            im = cv2.imread(img)
            im = im/255.0
            image_batch.append( im )
            
            if len(image_batch) == batch_size:
                x_train = np.array(image_batch)
                yield (x_train)
                image_batch = []
            
# In[read]
# Parse audio files and extract training samples
public_dir = '.\\predict\\melspectrogram_dataset\\'

private_dir = '.\\predict\\melspectrogram_private_dataset\\'

# In[]
model_dir = f'.\\'
h5_models = glob.glob(f"{model_dir}*.h5")
batch_size = 50

preds = []
for h5_model in h5_models:
    print(h5_model)
    model = load_model(h5_model, compile=False)
    # public_dir
    gen_predict = Predictgenerate(directory = public_dir, batch_size = batch_size)
    pred_public = model.predict_generator(gen_predict, steps=10000//batch_size, verbose=1)
    # private_dir
    gen_predict = Predictgenerate(directory = private_dir, batch_size = batch_size)
    pred_private = model.predict_generator(gen_predict, steps=20000//batch_size, verbose=1)

    pred = np.concatenate((pred_public, pred_private), axis=0)
    preds.append( pred )
    
preds = np.array(preds)

# In[]
pred_mean = np.mean(preds,axis=0)

submission = pd.read_csv('sample_submission.csv')
submission.iloc[:,1:7] = pred_mean[:,:]
submission.to_csv(f'.\\predict\\{model_folder}\\result_total_{model_folder}.csv', index=False)
