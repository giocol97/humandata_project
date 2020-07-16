import tensorflow as tf
import pyaudio
import wave

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa

model = load_model('modelCNN_raw.hdf5')
labels=["Yes", "No", "Up", "Down", "Left","Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four","Five", "Six", "Seven", "Eight", "Nine","Undefined"]
dir="..\..\project\speech_commands_v0.02"
file="..\..\project\\yes.wav"

'''
samples, sample_rate = librosa.load(file, sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)'''

frame_length=8000
frame_offset=1000

def smooth_posteriors(posteriors):
    w_smooth=5
    posteriors_smooth=np.zeros(posteriors.shape)
    for j in range(posteriors.shape[0]):
        h_smooth=np.amax([1,j+1-w_smooth+1])
        for i in range(posteriors.shape[1]):
            a=1/(j+1-h_smooth+1)
            b=posteriors[h_smooth,i]
            for k in range(h_smooth+1,j+1):
                b+=(posteriors[k,i])
            posteriors_smooth[j,i]=a*b
    return posteriors_smooth

def compute_confidence(posteriors):
    w_max=20
    n=posteriors.shape[1]
    confidences=np.zeros(posteriors.shape[0])
    for j in range(posteriors.shape[0]):
        m=1
        h_max=np.amax([1,(j+1)-w_max+1])
        for i in range(n):
            max=0#posteriors[0,i]
            for k in range(h_max,j+1):
                if(posteriors[k,i]>max):
                    max=posteriors[k,i]
            m*=max
        confidences[j]=(m)**(1/(n-1))
    return confidences

def split_file(samples,frame_length,frame_offset):
    split_samples=[]
    i = 0
    while i + frame_length < len(samples):
        split_samples.append(samples[i:i+frame_length])
        i += frame_offset
    return split_samples

def get_predictions(samples):
    labels_predicted=[]
    probs=[]
    for frame in samples:
        prob=model.predict(np.array(frame).reshape((-1,8000,1)))
        probs.append(prob)
    posteriors=np.array(probs).reshape((-1,21))
    #posteriors=smooth_posteriors(posteriors)
    #confidences=compute_confidence(posteriors)
    last_predictions=0
    indexes=[]
    predicted=[]
    for i in range(posteriors.shape[0]):
        indexes.append(np.argmax(probs[i]))
    print(indexes)
    for i in range(len(indexes)):
        if (i!=0):
            if(indexes[i]==indexes[i-1]):
                last_predictions+=1
            else:
                last_predictions=0
            if(last_predictions==1 and labels[indexes[i]]!="Undefined"):
                predicted.append(labels[indexes[i]])
    #print(labels[index])
    #print(confidences[i])
    print(predicted)


#-------------------------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 8000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

while(1):
  print( "recording")
  frames = []
  for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      data = stream.read(CHUNK)
      frames.append(data)

  waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
  waveFile.setnchannels(CHANNELS)
  waveFile.setsampwidth(audio.get_sample_size(FORMAT))
  waveFile.setframerate(RATE)
  waveFile.writeframes(b''.join(frames))
  waveFile.close()
  samples, sample_rate = librosa.load(WAVE_OUTPUT_FILENAME, sr = 8000)
  samples=split_file(samples,frame_length,frame_offset)
  get_predictions(samples)


# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
#-------------------------------------------

#samples=split_file(samples,frame_length,frame_offset)
