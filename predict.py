import tensorflow as tf

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

#from keras.layers import Bidirectional, BatchNormalization, CuDNNGRU, TimeDistributed
#from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Activation

#from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K

model = load_model('model.hdf5')
labels=["Yes", "No", "Up", "Down", "Left","Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four","Five", "Six", "Seven", "Eight", "Nine","Undefined"]
dir="..\..\project\speech_commands_v0.02"


def get_features(filename):
    (rate,sig) = wav.read(filename)
    fbank_feat = logfbank(sig,rate,winlen=0.025, winstep=0.01, nfilt=40)
    new_features = np.zeros((len(fbank_feat),32, 40))

    for i in range(len(fbank_feat)):
        k=0
        for j in reversed(range(1,24)):
            if(i-j>=0):
                new_features[i,k,:]=(fbank_feat[i-j])
            k+=1
        new_features[i,k,:]=(fbank_feat[i])
        k+=1
        for j in range(1,9):
            if(i+j<len(fbank_feat)):
                new_features[i,k,:]=(fbank_feat[i+j])
            k+=1
    return new_features

#j=frame, i=label
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

file="..\..\project\speech_commands_v0.02\\nine\\00b01445_nohash_0.wav"
#file="..\..\project\\nine.wav"
labels_predicted=[]
probs=[]
for frame in get_features(file):
    prob=model.predict(np.array(frame).reshape((-1,32,40,1)))
    probs.append(prob)

posteriors=np.array(probs).reshape((-1,21))
posteriors=smooth_posteriors(posteriors)
confidences=compute_confidence(posteriors)

for i in range(confidences.shape[0]):
    index=np.argmax(probs[i])
    print(labels[index])
    print(confidences[i])
