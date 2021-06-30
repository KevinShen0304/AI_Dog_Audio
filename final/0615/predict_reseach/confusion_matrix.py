# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:38:00 2021

@author: shen
"""
# In[1]
import os

import warnings
warnings.filterwarnings(action='ignore')

import glob
import pandas as pd
import librosa
import numpy as np

from tqdm import tqdm

from keras.models import load_model

# In[read img]
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

# In[7]model
SPEC_SHAPE = [X.shape[1], X.shape[2], X.shape[3]]

# In[7]model
model_folder = '0623model_8000'
model_dir = f'.\\predict\\{model_folder}\\'
h5_models = glob.glob(f"{model_dir}*.h5")

# In[7]
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
y_onehot = to_categorical(y)

y_predict = np.zeros((len(X), 10))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4500)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):
    print(fold_)
    x_val = X[val_idx]
    
    model = load_model(h5_models[fold_], compile=False)
    pred = model.predict(x_val)
    
    y_predict[val_idx] = pred

# In[]
import sklearn.metrics as metrics
matrix = metrics.confusion_matrix(y_onehot.argmax(axis=1), y_predict.argmax(axis=1))
matrix

predict = pd.DataFrame(y_predict, columns = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking','Other','Doorbell','Bird','Music_Instrument','Laugh_Shout_Scream'])
predict['name'] = img_paths
predict.to_excel('.\\predict_reseach\\confusion_result.xlsx', index=False)

