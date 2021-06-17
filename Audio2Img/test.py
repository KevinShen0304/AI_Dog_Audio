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
#audio_path = 'D:/AItrain/dataset download/sound_classification_50/train/0/1-30226-A-0.wav'
#audio_path = 'D:/AItrain/dataset download/sound_classification_50/train/10/1-26222-A-10.wav'
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
Hz = 4000
FMAX = SAMPLE_RATE//2 #Hz//2 #SAMPLE_RATE//2
hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1)) #SAMPLE_RATE要除以幾份
hop_length = 40
# In[]
sig, rate = librosa.load(audio_path, sr = SAMPLE_RATE, offset=None, duration=5) #sr = None 保留音频的原始采样率
#sig2, rate = librosa.load(audio_path, sr = 32000, offset=None, duration=5) #sr = None 保留音频的原始采样率

plt.figure(figsize=(15, 5))
librosa.display.waveplot(sig, sr=rate)

# =============================================================================
# # In[stft]
# n_fft = 2048
# D = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length))
# plt.figure(figsize=(15, 5))
# plt.plot(D)
# DB = librosa.amplitude_to_db(D, ref=np.max)
# librosa.display.specshow(DB, sr=rate, hop_length=hop_length, x_axis='time', y_axis='log');
# plt.colorbar(format='%+2.0f dB');
# =============================================================================
# In[3]
mel_spec = librosa.feature.melspectrogram(y=sig, 
                                          sr=SAMPLE_RATE, 
                                          n_fft=SAMPLE_RATE//25, #1024  
                                          hop_length=hop_length, 
                                          n_mels=SPEC_SHAPE[0], 
                                          fmin=FMIN, 
                                          fmax=FMAX)

mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 

plt.figure(figsize=(15, 5))
librosa.display.specshow(mel_spec, sr=rate)

# In[3]audiomentations
from audiomentations import *
# help(LoudnessNormalization)
augmenter = Compose([
    #AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1.0), #將高斯噪聲添加到樣本
    #AddGaussianSNR(min_SNR=0.001, max_SNR=0.1, p=0.3), # 將高斯噪聲添加到具有隨機信噪比（SNR）的樣本中
    #ClippingDistortion(min_percentile_threshold=5, max_percentile_threshold=5, p=1.0), # 從兩個輸入參數min_percentile_threshold和max_percentile_threshold之間的均勻分佈中得出將被裁剪的點的百分比。例如，如果抽取了30％，則樣本在第15個百分點以下或第85個百分點以上時將被裁剪。
    #FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.3, p=1.0), #屏蔽頻譜圖上的某些頻帶
    #Gain(min_gain_in_db=-12, max_gain_in_db=-12, p=1.0), # 將音頻乘以隨機幅度因子以減小或增大音量，警告：此轉換可能會返回[-1，1]範圍之外的樣本
    #LoudnessNormalization(min_lufs_in_db=-31, max_lufs_in_db=-13, p=1.0),
    Normalize(p=1.0),
    #PitchShift(min_semitones=-6, max_semitones=6, p=1.0), # 音調在不改變速度的情況下向上或向下移動聲音
    #PolarityInversion(p=0.5), # 翻轉音頻樣本
    #Resample(min_sample_rate=8000, max_sample_rate=44100, p=1) #不知道效果
    #Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0), #平移
    #TimeMask(min_band_part=0.0, max_band_part=0.5, fade=False, p=1.0), # 使音頻的隨機選擇部分靜音
    #TimeStretch(min_rate=0.0.75, max_rate=1.25, p=1.0), # 在不改變音高的情況下時間拉伸信號
    #Trim(top_db=60, p=1.0), #沒效果? 使用以下命令修剪音頻信號的前導和尾隨靜音 
])

sig = augmenter(samples=sig, sample_rate=rate)