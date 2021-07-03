# Concatenate & AveragePooling2D

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
    x = AveragePooling2D()(x)
    
    x = SpatialDropout2D(0.2)(x) #丟整個2D層
    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    out3 = x
    out3 = AveragePooling2D()(out3)
    
    x = SpatialDropout2D(0.2)(x) #丟整個2D層
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    out2 = x
    
    x = SpatialDropout2D(0.4)(x) #丟整個2D層
    x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out1 = x
    out = Concatenate()([out1, out2, out3])
    
    x = GlobalAveragePooling2D()(out)
    x = layers.Dropout(0.5)(x)
    x = Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(56)(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x, name='model') # Create model.

    return(model)

# In[]
model = get_model((128,128*4))
model.summary()


