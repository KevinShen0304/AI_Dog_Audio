# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:06:56 2021

@author: shen
"""

import os
import librosa
import numpy as np
from tqdm import tqdm
import cv2
import glob
from tqdm import tqdm

from audiomentations import *
import soundfile as sf

# Global vars
SAMPLE_RATE = 8000
SIGNAL_LENGTH = 5 # seconds
SPEC_SHAPE = (128, 128*4) # height x width
FMIN = 0
FMAX = 4000
LENGTH = SAMPLE_RATE*SIGNAL_LENGTH

aug_start_num = 0
aug_end_num = 10

#help(AddShortNoises)
sounds_path = '.\\train\\noise_foraug\\'
import warnings
warnings.filterwarnings("ignore")
# In[1]
# function
def sig_aug(filepath):
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None, duration=5) #  duration=5(s)

    zeros = np.zeros(LENGTH - len(sig)) #補0數量
    sig = np.append(sig, zeros)
    
    augmenter = Compose([
        Normalize(p=1.0),
        #PitchShift(min_semitones=-4.0, max_semitones=4.0, p=1.0),
        AddBackgroundNoise(sounds_path=sounds_path, min_snr_in_db=8, max_snr_in_db=30, p=1.0), #min_snr_in_db:比例關係，越小表示噪音越大
        #PolarityInversion(p=0.5), # 翻轉音頻樣本
        Shift(min_fraction=0, max_fraction=1.0, p=1.0), #平移
        #FrequencyMask(min_frequency_band=0.02, max_frequency_band=0.10, p=0.30), #屏蔽頻譜圖上的某些頻帶
        #TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5), # 在不改變音高的情況下時間拉伸信號
        ClippingDistortion(min_percentile_threshold=2, max_percentile_threshold=10, p=0.4), #之間的均勻分佈中得出將被裁剪的點的百分比
        TimeMask(min_band_part=0.02, max_band_part=0.05, fade=False, p=0.4), # 使音頻的隨機選擇部分靜音
    ])
    
    for i in range(aug_start_num,aug_end_num):
        if i == 0: #第一個保持原樣
            sample = sig
        else:
            sample = augmenter(samples=sig, sample_rate=rate)
        
        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=sample, 
                                                  sr=rate, 
                                                  n_fft=SAMPLE_RATE//25,
                                                  hop_length=hop_length, 
                                                  n_mels=SPEC_SHAPE[0], 
                                                  fmin=FMIN, 
                                                  fmax=FMAX)
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        mel_spec = 255 * mel_spec #轉為圖片
        mel_img = mel_spec.astype(np.uint8) 
        mel_img = cv2.resize(mel_img, (SPEC_SHAPE[1],SPEC_SHAPE[0]))
        #mel_img = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2BGR) #轉為BGR # cv2讀取默認BGR
        
        # Save as image file
        file_name = filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0]
        label = filepath.rsplit(os.sep, 2)[-2]
        save_dir = f'{output_dir}\\{label}\\{file_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file_name+'_'+str(i)+'.png')
        cv2.imwrite(save_path, mel_img)

# =============================================================================
#         # Save as audio file
#         save_audio_dir = f'{output_audio_dir}\\{label}\\{file_name}'
#         if not os.path.exists(save_audio_dir):
#             os.makedirs(save_audio_dir)
#         save_audio_path = os.path.join(save_audio_dir, file_name+'_'+str(i)+'.wav')
#         sf.write(save_audio_path, sample, rate)
# =============================================================================
        
# In[2]
audio_dir = '.\\train\\train\\'
output_dir = '.\\train\\melspectrogram_dataset_aug\\'
output_audio_dir = '.\\train\\audio_aug\\'
audio_paths = glob.glob(f'{audio_dir}**\\*.wav')

with tqdm(total=len(audio_paths)) as pbar:
    for i, audio_path in enumerate(audio_paths):
        #if i <1426: continue
        pbar.update(1)
        sig_aug(audio_path)