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
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional, BatchNormalization, CuDNNGRU, TimeDistributed
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, MaxPooling2D, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


labels=["Yes", "No", "Up", "Down", "Left","Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four","Five", "Six", "Seven", "Eight", "Nine"]
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

def get_labels_array(labels_data):
    new_labels = []
    for label in labels_data:
        label=label[22:]
        try:
            new_labels.append(labels.index(label))
        except ValueError:
            new_labels.append(len(labels))
    return new_labels


def init_dataset(dir_name):
    # Numpy matrix
    train_test = np.zeros(())
    labels=[]
    path = os.getcwd()
    # Get the list of all files and directories
    # in current working directory
    subdirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]

    features=[]
    for subdir in subdirs:
        files = [os.path.join(subdir, f) for f in os.listdir(subdir)]
        for file in files:
            if file.count(".wav")>0:
                file_features=get_features(file)
                features.append(file_features)
                labels.append(subdir)
        break#PER TESTARE CON UNA SOLA DIRECTORY
    #features=np.array(features)
    labels=np.array(get_labels_array(labels))
    return features, labels


#---------------------------------------PREPROCESSING AND DATA LOADING

features,data_labels=init_dataset(dir)
'''
features_new=np.zeros(len(features),32,40)
np.expand_dims(features_new, axis=-1)
features_new[:,:,:]=features'''
features=np.array(features).reshape((len(features),-1,32,40))

x_train, x_valid, y_train, y_valid = train_test_split(features,data_labels,stratify=data_labels,test_size = 0.2,random_state=777,shuffle=True)


#---------------------------------------NETWORK

K.clear_session()

inputs = Input(shape=(32,40,1))
#x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)

#First Conv2D layer
x = Conv2D(64,(20,8), padding='valid', activation='relu', strides=(1,1))(inputs)
x = MaxPooling2D((1,3))(x)
#x = Dropout(0.3)(x)

#Second Conv2D layer
x = Conv2D(64, (10,4), padding='valid', activation='relu', strides=(1,1))(x)
x = MaxPooling2D((1,1))(x)
#x = Dropout(0.3)(x)

x = Activation("relu")(x)

#Flatten layer
x = Flatten()(x)

x = Dense(128)(x)

outputs = Dense(len(labels), activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()

#-----------------------------TRAINING

model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

hist = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    batch_size=32,
    validation_data=(x_valid,y_valid)
)

pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

'''
from matplotlib import pyplot
pyplot.plot(fbank_feat)
pyplot.legend()
pyplot.show()
'''
