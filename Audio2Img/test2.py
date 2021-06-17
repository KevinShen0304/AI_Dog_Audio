# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:39:15 2021
https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
@author: shen
"""

# In[1]
# Pick a file
audio_path = 'D:/AItrain/Tomofun/train/train/0/train_00046.wav' #train_00420
audio_path = 'D:/AItrain/dataset download/特別的/102842-3-1-6.wav' # 102842-3-1-6 # 101281-3-0-14
audio_path = 'D:/AItrain/dataset download/特別的/101281-3-0-14.wav' # 102842-3-1-6 # 101281-3-0-14

# In[2]
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

# In[]
SAMPLE_RATE = 8000 #8000
SIGNAL_LENGTH = 5 # seconds
SPEC_SHAPE = (100, 500) # height x width
FMIN = 0 #500
#Hz = 4000
FMAX = SAMPLE_RATE//2 #Hz//2 #SAMPLE_RATE//2
hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1)) #SAMPLE_RATE要除以幾份
# In[]
sig, rate = librosa.load(audio_path, sr = SAMPLE_RATE, offset=None, duration=5) #sr = None 保留音频的原始采样率
#sig2, rate = librosa.load(audio_path, sr = 32000, offset=None, duration=5) #sr = None 保留音频的原始采样率

plt.figure(figsize=(15, 5))
librosa.display.waveplot(sig, sr=rate)

# In[3]
mel_spec = librosa.feature.melspectrogram(y=sig, 
                                          sr=SAMPLE_RATE, 
                                          n_fft=SAMPLE_RATE//25, #1024  
                                          hop_length=hop_length, 
                                          n_mels=SPEC_SHAPE[0], 
                                          fmin=FMIN, 
                                          fmax=FMAX)

mel_spec = librosa.power_to_db(mel_spec, ref=np.min)  #np.max #ref ：参考值，振幅abs(S)相对于ref进行缩放

print(mel_spec.min())
print(mel_spec.max())
    
plt.figure(figsize=(15, 5))
#librosa.display.specshow(mel_spec, sr=rate)
plt.imshow(mel_spec, cmap='gray')
plt.show()

# In[3]
mfcc = librosa.feature.mfcc(y=sig, sr=SAMPLE_RATE, S=None, n_mfcc=SPEC_SHAPE[0], dct_type=2, norm='ortho')
# 获取特征值的维度
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfcc, sr=SAMPLE_RATE, x_axis='time')
# 对MFCC的数据进行处理

# In[]
S = librosa.feature.melspectrogram(y=sig, 
                                    sr=SAMPLE_RATE, 
                                    n_fft=SAMPLE_RATE//25, #1024  
                                    hop_length=hop_length, 
                                    n_mels=SPEC_SHAPE[0], 
                                    fmin=FMIN, 
                                    fmax=FMAX)
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfcc, sr=rate)
# 对MFCC的数据进行处理
