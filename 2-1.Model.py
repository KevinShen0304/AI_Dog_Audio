# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:53:34 2021
@author: shen
"""
# In[]
from keras.layers import *
from keras.models import *
from keras import layers
from keras.optimizers import *

import tensorflow as tf
from keras import backend as K

# In[]
from sklearn.metrics import roc_auc_score


# In[]
def get_model(SPEC_SHAPE):

    # model 
    img_input = Input(shape=(SPEC_SHAPE[0], SPEC_SHAPE[1], 1))
    x = img_input
    
    # 主線
    x = Conv2D(32, (8, 8), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = SpatialDropout2D(0.2)(x) #丟整個2D層
    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = SpatialDropout2D(0.2)(x) #丟整個2D層
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SpatialDropout2D(0.4)(x) #丟整個2D層
    x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(56)(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)
    # Create model.
    model = Model(img_input, x, name='model')
    #model.summary()

    return(model)

