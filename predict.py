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
import librosa
import sklearn.metrics
#from keras.layers import Bidirectional, BatchNormalization, CuDNNGRU, TimeDistributed
#from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Activation

#from keras.models import Model
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K

model = load_model('funziona/modelCNN_raw.hdf5')
labels=["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four","five", "six", "seven", "eight", "nine","undefined"]
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

def split_file(samples,frame_length,frame_offset):
    split_samples=[]
    i = 0
    while i + frame_length < len(samples):
        split_samples.append(samples[i:i+frame_length])
        i += frame_offset
    return split_samples

dir="..\..\project\speech_commands_v0.02"
directories = [f.path for f in os.scandir(dir) if f.is_dir()]
files=[]
true_labels=[]
predicted_labels=[]
for subdir in directories:
    print(subdir)
    files = [subdir + "\\" + f for f in os.listdir(subdir)]
    count=0
    for file in files:
        features=[]
        count+=1
        if (count<300):
            continue
        if(count>350):
            break
        samples, sample_rate = librosa.load(file, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples) != 8000) :
            new_samples = np.zeros((8000))
            new_samples[:len(samples)]=np.array(samples).reshape((len(samples)))
        else:
            new_samples = np.array(samples)
        features=np.array(new_samples).reshape(-1,8000,1)
        prob=model.predict(features)
        index=np.argmax(prob)
        predicted_labels.append(index)
        label=subdir[len(dir)+1:]
        try:
            true_labels.append(labels.index(label))
        except ValueError:
            true_labels.append(len(labels)-1)

conf_matrix=sklearn.metrics.confusion_matrix(true_labels,predicted_labels,normalize="true")

disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation="horizontal", values_format=None)
plt.show()
exit()

#dir="..\..\project\speech_commands_v0.02\\dog"
file="..\..\project\\yes.wav"
labels_predicted=[]
probs=[]

samples, sample_rate = librosa.load(file, sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)

frame_length=8000
frame_offset=1000

samples=split_file(samples,frame_length,frame_offset)

for frame in samples:
    prob=model.predict(np.array(frame).reshape((-1,8000,1)))
    probs.append(prob)


posteriors=np.array(probs).reshape((-1,21))
posteriors=smooth_posteriors(posteriors)
confidences=compute_confidence(posteriors)

for i in range(confidences.shape[0]):
    index=np.argmax(probs[i])
    print(labels[index])
    print(confidences[i])


'''
for frame in get_features(file):
    prob=model.predict(np.array(frame).reshape((-1,32,40,1)))
    probs.append(prob)


posteriors=np.array(probs).reshape((-1,21))
posteriors=smooth_posteriors(posteriors)
confidences=compute_confidence(posteriors)

for i in range(confidences.shape[0]):
    index=np.argmax(probs[i])
    #if(labels[index]=="Undefined"):
        #probs[i][0][index]=0
        #index=np.argmax(probs[i])
    print(labels[index])
    print(confidences[i])
'''
