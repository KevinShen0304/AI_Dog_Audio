
# In[1]
# Pick a file
audio_path = 'D:/AItrain/Tomofun/train/train/1/train_00208.wav' #train_00420

# In[2]
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf


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

# In[]function
def show_sig(sig):
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(sig, sr=rate)

def show_spec(sig):
    mel_spec = librosa.feature.melspectrogram(y=sig, 
                                              sr=SAMPLE_RATE, 
                                              n_fft=SAMPLE_RATE//25, #1024  
                                              hop_length=hop_length, 
                                              n_mels=SPEC_SHAPE[0], 
                                              fmin=FMIN, 
                                              fmax=FMAX)
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.min)  #np.max #ref ：参考值，振幅abs(S)相对于ref进行缩放
    # Normalize
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(mel_spec, sr=rate)

# In[3]
show_sig(sig)
show_spec(sig)

# In[3]
from audiomentations import *

for change in [3, 1, 0,-1,-3]:
    print(change)
    augmenter = Compose([
        PitchShift(min_semitones=change, max_semitones=change, p=1.0), #semitones半音
        #AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.01, p=1.0), #將高斯噪聲添加到樣本
    ])
    sig_gain = augmenter(samples=sig, sample_rate=rate)
    show_sig(sig_gain)
    show_spec(sig_gain)

    sf.write(f'change_{change}.wav', sig_gain, rate)