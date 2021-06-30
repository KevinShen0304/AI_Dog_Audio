# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:38:00 2021

@author: shen
"""
# In[1]
import os

import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import librosa
import numpy as np

from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam

# In[]
import glob
model_dir = '.\\predict\\0623model_8000_aug\\'
h5_models = glob.glob(f"{model_dir}*.h5")

# In[4]
import cv2
img_dir = '.\\train\\melspectrogram_dataset\\'
img_paths = glob.glob(f'{img_dir}**\\*.png')

X = []
y = []
with tqdm(total=len(img_paths)) as pbar:
    for img_path in img_paths:
        pbar.update(1)
        label = img_path.rsplit(os.sep, 2)[-2]
        y.append( label )
        im = cv2.imread(img_path, 0) # 0=gray
        im = im/255.0
        im = im.reshape(im.shape[0],im.shape[1],1)
        X.append( im )

X = np.array(X)
y = np.array(y)

# In[4]
from pathlib import Path
import glob
# Parse audio files and extract training samples
imgs_files = [Path(img).stem for img in img_paths]

aug_dir = '.\\train\\melspectrogram_dataset_aug\\'
audio_aug_range = 9
# In[7]para
SPEC_SHAPE = [X.shape[1], X.shape[2], X.shape[3]]
batch_size = 50

# In[7]
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import keras

y_onehot = to_categorical(y)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4500)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):
    if fold_<0: continue #從開始
    trn_epoch = int(len(trn_idx)/batch_size)

    #if fold_>0: break
    print(fold_)
    x_val = X[val_idx]
    y_val = y_onehot[val_idx]

    train_Mygen = MyTrainGenerator(directory = aug_dir,
                                   imgs_files = imgs_files, 
                                    y_onehot = y_onehot, 
                                    y = y,
                                    index = trn_idx, 
                                    batch_size = batch_size,
                                    aug_bool=True, audio_aug_bool=True,
                                    audio_aug_range = audio_aug_range)
    train_gen = train_Mygen.generate()

    # call
    log_dir = '..\\log\\'
    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, 
                                  factor=0.75, min_lr=0.000001)
    checkpoint = ModelCheckpoint(log_dir + 'fold' + str(fold_)+  
                                 'ep{epoch:02d}-loss{loss:.3f}-accuracy{accuracy:.3f}-auc{auc:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}-val_auc{val_auc:.3f}.h5',
                                 monitor='auc', save_best_only=False, period=10)  #monitor='loss'

    early_stopping = EarlyStopping(monitor='loss', min_delta=-10, patience=1000, verbose=1)
    
    # model
    #model = get_model(SPEC_SHAPE)
    model = load_model(h5_models[fold_], compile=False)
    
    model.compile(loss='categorical_crossentropy', #binary_crossentropy #categorical_crossentropy
              optimizer=Adam(lr=0.0005), #0.001 #0.0001
              metrics=['accuracy',keras.metrics.AUC(name='auc')])
    
    model.fit_generator(generator = train_gen,
                        steps_per_epoch= trn_epoch,
                        validation_data =(x_val,y_val),
                        epochs = 300,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    #loss, acc, au = model.evaluate(x_val, y_val, verbose=2)
    # 儲存訓練好的模型# 模型輸出儲存的檔案
    model.save( f'model_{fold_}.h5')
    del model


