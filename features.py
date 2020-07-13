from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle

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
    labels=["Yes", "No", "Up", "Down", "Left","Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four","Five", "Six", "Seven", "Eight", "Nine"]
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
        files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(path, f))]
        for file in files:
            features.append(get_features(file))
            labels.append(subdir)

    features=np.array(features)
    labels=np.array(get_labels_array(labels))
    return features, labels

dir="speech_commands_v0.02"

features,labels_data=init_dataset(dir)

print(features.shape)
'''
if(os.path.isfile('features.pkl')):
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
else:
    features=init_dataset(dir)
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

dir="speech_commands_v0.02"

labels=["Yes", "No", "Up", "Down", "Left","Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four","Five", "Six", "Seven", "Eight", "Nine"]

features=init_dataset(dir)

if(isfile('features.pkl')):
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
else:
    features=init_dataset(dir)
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)


x_train, x_valid, y_train, y_valid = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)


K.clear_session()

inputs = Input(shape=(8000,1))
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)

#First Conv1D layer
x = Conv1D(8,13, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

#Second Conv1D layer
x = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

#Third Conv1D layer
x = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

x = Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum')(x)
x = Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum')(x)
x = Bidirectional(CuDNNGRU(128, return_sequences=False), merge_mode='sum')(x)

x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)

#Flatten layer
# x = Flatten()(x)

#Dense Layer 1
x = Dense(256, activation='relu')(x)
outputs = Dense(len(labels), activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = ModelCheckpoint('speech2text_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

hist = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    batch_size=32,
    validation_data=(x_valid,y_valid)
)


from matplotlib import pyplot
pyplot.plot(fbank_feat)
pyplot.legend()
pyplot.show()
'''
