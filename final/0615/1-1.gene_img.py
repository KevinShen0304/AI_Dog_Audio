# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:01:06 2021

@author: shen
"""
import os
import librosa
import numpy as np
from tqdm import tqdm
import cv2
import glob

# Global vars
SAMPLE_RATE = 8000
SIGNAL_LENGTH = 5 # seconds
SPEC_SHAPE = (128, 128*4) # height x width
FMIN = 0
FMAX = 4000
LENGTH = SAMPLE_RATE*SIGNAL_LENGTH

# In[2]
def get_spectrograms(filepath, output_dir):
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None, duration=5) #  duration=5(s)

    zeros = np.zeros(LENGTH - len(sig)) #補0數量
    sig = np.append(sig, zeros)
    
    hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
    mel_spec = librosa.feature.melspectrogram(y=sig, 
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
    mel_img = mel_spec.astype(np.uint8) #
    mel_img = cv2.resize(mel_img, (SPEC_SHAPE[1],SPEC_SHAPE[0]))
    #mel_img = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2BGR) #轉為BGR # cv2讀取默認BGR
    
    # Save as image file
    save_path = os.path.join(output_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] +'.png')
    cv2.imwrite(save_path, mel_img)

# In[]train data
audio_dir = '.\\train\\train\\'
audio_paths = glob.glob(f'{audio_dir}**\\*.wav')

output_dir = '.\\train\\melspectrogram_dataset\\'

with tqdm(total=len(audio_paths)) as pbar:
    for audio_path in audio_paths:
        pbar.update(1)
        save_dir = os.path.join(output_dir, audio_path.rsplit(os.sep, 2)[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        get_spectrograms(audio_path, save_dir)
        
