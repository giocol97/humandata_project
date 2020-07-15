import tensorflow as tf
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


labels=["yes", "no", "up", "down", "left","right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four","five", "six", "seven", "eight", "nine"]
dir="..\..\project\speech_commands_v0.02"
#dir="/nfsd/hda/DATASETS/Project_1"

def get_features(filename):
    (rate,sig) = wav.read(filename)
    fbank_feat = logfbank(sig,rate,winlen=0.025, winstep=0.01, nfilt=40)
    new_features = np.zeros((len(fbank_feat),32, 40))

    for i in range(23,len(fbank_feat)-8):
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
    max=10

    train_test = np.zeros(())
    labels=[]
    path = os.getcwd()
    # Get the list of all files and directories
    # in current working directory
    subdirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]

    features=[]
    for subdir in subdirs:
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
    labels=np.array(get_labels_array(labels))
    return features, labels


#---------------------------------------PREPROCESSING AND DATA LOADING

features,data_labels=init_dataset(dir)

features=np.array(features).reshape((-1,32,40,1))

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
#x = Dropout(0.3)(x)

#Second Conv2D layer
x = tf.keras.layers.Conv2D(64, (10,4), padding='valid', activation='relu', strides=(1,1))(x)
x = tf.keras.layers.MaxPooling2D((1,1))(x)
#x = Dropout(0.3)(x)

x = tf.keras.layers.Activation("relu")(x)

#Flatten layer
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(128)(x)

outputs = tf.keras.layers.Dense(len(labels)+1, activation="softmax")(x)

model = tf.keras.models.Model(inputs, outputs)
model.summary()
'''

def encoder(input):
    #encoder
    #input = 32 x 40
    conv1 = tf.keras.layers.Conv2D(64, (20, 8), activation='relu', padding='same', strides=(1,1))(input) #32 x 40 x 32
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) #16 x 20 x 32
    conv2 = tf.keras.layers.Conv2D(128, (10, 4), activation='relu', padding='same', strides=(1,1))(pool1) #16 x 20 x 64
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = tf.keras.layers.Conv2D(256, (5, 2), activation='relu', padding='same', strides=(1,1))(pool2) #8 x 10 x 128 (small and thick)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv4 = tf.keras.layers.Conv2D(512, (5, 2), activation='relu', padding='same', strides=(1,1))(conv3) #8 x 10 x 256 (small and thick)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = tf.keras.layers.Flatten()(enco)
    den = tf.keras.layers.Dense(256, activation='relu')(flat)
    out = tf.keras.layers.Dense(len(labels)+1, activation='softmax')(den)
    return out

def decoder(conv4):
    #decoder
    conv5 = tf.keras.layers.Conv2D(256, (5, 2), activation='relu', padding='same', strides=(1,1))(conv4) #8 x 10 x 128
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv6 = tf.keras.layers.Conv2D(128, (5, 2), activation='relu', padding='same', strides=(1,1))(conv5) #8 x 10 x 64
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    up1 = tf.keras.layers.UpSampling2D((2,2))(conv6) #16 x 20 x 64
    conv7 = tf.keras.layers.Conv2D(64, (10, 4), activation='relu', padding='same', strides=(1,1))(up1) #16 x 20 x 32
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    up2 = tf.keras.layers.UpSampling2D((2,2))(conv7) # 32 x 40 x 32
    decoded = tf.keras.layers.Conv2D(1, (20, 8), activation='sigmoid', padding='same', strides=(1,1))(up2) # 32 x 40 x 1
    return decoded

inputs = tf.keras.Input(shape=(32,40,1))

autoencoder = tf.keras.models.Model(inputs, decoder(encoder(inputs)))
autoencoder.compile(loss='mean_squared_error', optimizer = tf.keras.optimizers.RMSprop(),metrics=['accuracy'])

#-----------------------------TRAINING

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

hist = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=10,
    callbacks=[early_stop, checkpoint],
    batch_size=32,
    validation_data=(x_valid,x_valid)
)

from matplotlib import pyplot

pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
