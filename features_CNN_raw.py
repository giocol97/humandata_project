import tensorflow as tf
'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)'''

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
from tensorflow.keras import backend as K
import librosa


labels=["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four","five", "six", "seven", "eight", "nine"]
dir="..\..\project\speech_commands_v0.02"
#dir="/nfsd/hda/DATASETS/Project_1"
savedir='modelCNN_raw.hdf5'
#savedir="/nfsd/hda/colottigio/models/modelCNN.hdf5"

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
        label=label[len(dir)+1:]
        try:
            new_labels.append(labels.index(label))
        except ValueError:
            new_labels.append(len(labels))
    return new_labels


def init_dataset(dir_name):
    # Numpy matrix
    max=300
    train_test = np.zeros(())
    features=[]
    labels=[]
    path = os.getcwd()
    # Get the list of all files and directories
    # in current working directory
    subdirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]
    for subdir in subdirs:
        if(subdir.count("_background_noise_")>0):
            continue
        cur=0
        files = [os.path.join(subdir, f) for f in os.listdir(subdir)]
        for file in files:
            if file.count(".wav")>0:
                cur+=1
                samples, sample_rate = librosa.load(file, sr = 16000)
                samples = librosa.resample(samples, sample_rate, 8000)
                if(len(samples) != 8000) :
                    new_samples = np.zeros((8000))
                    new_samples[:len(samples)]=np.array(samples).reshape((len(samples)))
                else:
                    new_samples = np.array(samples)
                features.append(new_samples)
                labels.append(subdir)

            #if(cur==max):
            #    break
        #break#PER TESTARE CON UNA SOLA DIRECTORY
    features=np.array(features).reshape(-1,8000,1)
    labels=np.array(get_labels_array(labels))
    '''
    train_test = np.zeros(())
    labels=[]
    path = os.getcwd()
    # Get the list of all files and directories
    # in current working directory
    subdirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]

    features=[]
    for subdir in subdirs:
        if(subdir.count("_background_noise_")>0):
            continue
        cur=0
        files = [os.path.join(subdir, f) for f in os.listdir(subdir)]
        for file in files:
            if file.count(".wav")>0:
                cur+=1
                file_features=get_features(file)
                for frame in file_features:
                    features.append(frame)
                    labels.append(subdir)
            if(cur==max):
                break
        #break#PER TESTARE CON UNA SOLA DIRECTORY
    #features=np.array(features)
    labels=np.array(get_labels_array(labels))'''
    return features, labels


#---------------------------------------PREPROCESSING AND DATA LOADING

features,data_labels=init_dataset(dir)

data_labels_matrix=tf.keras.utils.to_categorical(data_labels,len(labels)+1)

x_train, x_valid, y_train, y_valid = train_test_split(features,data_labels_matrix,stratify=data_labels,test_size = 0.2,random_state=777,shuffle=True)

#---------------------------------------NETWORK
'''
K.clear_session()

inputs = tf.keras.Input(shape=(32,40,1))
#x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)

#First Conv2D layer
x = tf.keras.layers.Conv2D(64,(20,8), padding='valid', activation='relu', strides=(1,1))(inputs)
x = tf.keras.layers.MaxPooling2D((1,3))(x)
x = Dropout(0.3)(x)

#Second Conv2D layer
x = tf.keras.layers.Conv2D(64, (10,4), padding='valid', activation='relu', strides=(1,1))(x)
x = tf.keras.layers.MaxPooling2D((1,1))(x)
x = Dropout(0.3)(x)

x = tf.keras.layers.Activation("relu")(x)

#Flatten layer
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(128)(x)

outputs = tf.keras.layers.Dense(len(labels)+1, activation="softmax")(x)

model = tf.keras.models.Model(inputs, outputs)
model.summary()
'''

K.clear_session()

inputs = tf.keras.layers.Input(shape=(8000,1))
x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)

#First Conv1D layer
x = tf.keras.layers.Conv1D(8,13, padding='valid', activation='relu', strides=1)(x)
x = tf.keras.layers.MaxPooling1D(3)(x)
x = tf.keras.layers.Dropout(0.3)(x)

#Second Conv1D layer
x = tf.keras.layers.Conv1D(16, 11, padding='valid', activation='relu', strides=1)(x)
x = tf.keras.layers.MaxPooling1D(3)(x)
x = tf.keras.layers.Dropout(0.3)(x)

#Third Conv1D layer
x = tf.keras.layers.Conv1D(32, 9, padding='valid', activation='relu', strides=1)(x)
x = tf.keras.layers.MaxPooling1D(3)(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True), merge_mode='sum')(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True), merge_mode='sum')(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=False), merge_mode='sum')(x)

x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

#Flatten layer
# x = Flatten()(x)

#Dense Layer 1
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(labels)+1, activation="softmax")(x)

model = tf.keras.models.Model(inputs, outputs)
model.summary()
#-----------------------------TRAINING

model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = tf.keras.callbacks.ModelCheckpoint(savedir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

hist = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    batch_size=32,
    validation_data=(x_valid,y_valid)
)

from matplotlib import pyplot
pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
