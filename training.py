import pandas as pd
import numpy as np
import wfdb
import ast


from tqdm import tqdm

def load_data(df, sampling_rate, path):
    raw_data = []
    if sampling_rate == 100:
        raw_data = [wfdb.rdsamp(path + file) for file in df.filename_lr]
    else:
        for i, file in enumerate(tqdm(df.filename_hr)):
            raw_data.append(wfdb.rdsamp(path + file))
    data = np.array([signal[0] for signal in raw_data])
    return data

path = 'ptbxl/data/'
sampling_rate=500

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    return list(set(agg_df.loc[key].diagnostic_class for key in y_dic.keys() if key in agg_df.index))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

def delete_ecg(x, y):
    x_processed = [x_val for x_val, dis in zip(x, y) if dis]
    y_processed = [dis for dis in y if dis]
    return x_processed, y_processed

X_train,y_train = delete_ecg(X_train,y_train)
X_test,y_test = delete_ecg(X_test,y_test)
            
y_train = np.array(y_train,dtype=object)
y_test = np.array(y_test,dtype=object)
X_train = np.array(X_train)
X_test = np.array(X_test)

print(y_train.shape)
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Activation, Add,
                          Flatten, Dense)
from keras.models import Model
import numpy as np


class ResidualUnit(object):
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_rate=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        if downsample > 1:
            y = MaxPooling1D(pool_size=downsample)(y)
        if n_filters_in != self.n_filters_out:
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1].value
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2].value
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]

kernel_size = 16
kernel_initializer = 'he_normal'

input_signal = Input(shape=(5000, 12), dtype=np.float32, name='signal')
input_age_range = Input(shape=(5,), dtype=np.float32, name='age_range')
input_is_male = Input(shape=(1,), dtype=np.float32, name='is_male')

processed_signal = input_signal
processed_signal = Conv1D(64, kernel_size, padding='same', use_bias=False, kernel_initializer=kernel_initializer)(processed_signal)
processed_signal = BatchNormalization()(processed_signal)
processed_signal = Activation('relu')(processed_signal)

residual_unit1_outputs = ResidualUnit(1024, 128, kernel_size=kernel_size, kernel_initializer=kernel_initializer)([processed_signal, processed_signal])
processed_signal, residual_unit1_output = residual_unit1_outputs

residual_unit2_outputs = ResidualUnit(256, 196, kernel_size=kernel_size, kernel_initializer=kernel_initializer)([processed_signal, residual_unit1_output])
processed_signal, residual_unit2_output = residual_unit2_outputs

residual_unit3_outputs = ResidualUnit(64, 256, kernel_size=kernel_size, kernel_initializer=kernel_initializer)([processed_signal, residual_unit2_output])
processed_signal, residual_unit3_output = residual_unit3_outputs

residual_unit4_output = ResidualUnit(16, 320, kernel_size=kernel_size, kernel_initializer=kernel_initializer)([processed_signal, residual_unit3_output])

flattened_signal = Flatten()(residual_unit4_output)
diagnostic = Dense(5, activation='sigmoid', kernel_initializer=kernel_initializer)(flattened_signal)

model = Model(input_signal, diagnostic)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


if __name__ == "__main__":
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

diseases = ["NORM","MI","HYP","CD","STTC"]

def Ribeiro_convert(y_train):
    
    y_train = list(y_train)

    y_train_ = [[0,0,0,0] for i in list(y_train)]
    for index, info in enumerate(y_train):

        for disease in info:

            if disease != 'NORM':
                y_train_[index][diseases.index(disease)-1] = 1

    y_train_ = np.array(y_train_)
    return y_train_

y_train = Ribeiro_convert(y_train)
y_test = Ribeiro_convert(y_test)

print(y_train.shape)
print(y_test.shape)

import sys   
import tensorflow.compat.v1 as tf 
from keras.optimizers import Adam 
from keras.callbacks import (ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)
from keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                            Dense, Dropout, BatchNormalization,
                            Activation, Add)

from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.utils import np_utils


loss = 'binary_crossentropy'
lr = 0.001
batch_size = 64
opt = Adam(lr)
callbacks = [ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=7,
                               min_lr=lr / 100),
             EarlyStopping(patience=9,
                           min_delta=0.00001)]
# Set session and compile model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

model.compile(loss=loss, optimizer=opt)

# Create log
callbacks += [TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=False),
              CSVLogger('training.log', append=False)]  # Change append to true if continuing training
# Save the BEST and LAST model
callbacks += [ModelCheckpoint('./backup.hdf5'),
              ModelCheckpoint('./backup.hdf5', save_best_only=True)]
# Train neural network

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=70,
                    initial_epoch=0,  # If you are continuing a interrupted section change here
                    validation_data = (X_test,y_test),
                    #validation_split= 0.02,
                    shuffle='batch',  # Because our dataset is an HDF5 file
                    #callbacks=callbacks,
                    verbose=1)
# Save final result
model.save("Ribeiro.hdf5")

import cv2 # opencv
import tqdm
print(X.shape)#Before conversion

def convert_X(X,scale):
    X_conv = np.empty((X.shape[0], 4096, 12))
    for index,array in enumerate(X):
        array = cv2.resize(array,(12,scale))
        X_conv[index,:,:]=array
    return X_conv
    
X_=convert_X(X,4096)    
print(X_.shape)#After Conversion

import numpy as np
import matplotlib.pyplot as plt
CHAR = 4096
y = np.linspace(0,10,5000)
y_ =np.linspace(0,10,CHAR)

for index, ECG_12 in enumerate(X):
    ECG_12 = np.transpose(ECG_12)
    plot = plt.figure(figsize=(20, 50))
    ECG_12_conv = np.transpose(X_[index])

    for n in range(12):
        plt.subplot(24, 1, 2 * n + 1)
        plt.plot(y, ECG_12[n])

        plt.subplot(24, 1, 2 * n + 2)
        plt.plot(y_, ECG_12_conv[n], color="r")
        break

    break

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as keras
from keras.models import load_model
from keras.optimizers import Adam
import h5py

path_to_model = 'path/to/model'

model = load_model(path_to_model)

model.compile(loss='binary_crossentropy', optimizer=Adam())
print("Ribeiro model available")

healthy = 0
for i in list(y_test)+list(y_train):
    if (i==np.array([0,0,0,0])).all():
        healthy += 1
print(healthy/(len(y_test)+len(y_train))*100,"%")

lo = []

for layer in model.layers[1:49]:

    layer.trainable = True 

    lo.append(layer.output)

    
model_adaptated = keras.Model(inputs = model.input, outputs = lo,name ='pre-trained')
model_adaptated.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=[])

model_adaptated.summary()

import numpy as np
import matplotlib.pyplot as plt
import statistics

conv_example = model_adaptated.predict(np.array([X_[0]]))[-2]

y = np.linspace(0,10,16)


ECG_12 = np.transpose(conv_example)

infos = [np.linalg.norm(x) for x in ECG_12]
infos_2 = infos.copy()
max_L2 = [infos.pop(infos.index(max(infos))) for i in range(5)] 

infos_max = [infos_2.index(i) for i in max_L2]

fig = plt.figure(figsize=(20,25))
y_l2 = np.linspace(0, 10, 320)

plt.subplot(6, 1, 1)
plt.plot(y_l2, infos_2)
plt.title("Density of Information for Each of the Convoluted Part")

print(f"Mean: {statistics.mean(infos_max)}")
print(f"Median: {statistics.median(infos_max)}")

for i, n in enumerate(infos_max):
    plt.subplot(6, 1, i+2)
    plt.plot(y, ECG_12[n])

from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense
from keras.utils import plot_model

kernel_initializer = 'he_normal'
x = model_adaptated.output[47]
preds = Dense(4,activation='sigmoid', kernel_initializer=kernel_initializer)(x)

new_model = keras.Model(inputs = model_adaptated.input, outputs = preds)
new_model.compile(loss='binary_crossentropy', optimizer=Adam(),metrics = [tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
new_model.summary()

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from datetime import datetime
import time 

CHAR = 4096
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
date_time = str(now.day)+"-"+ str(current_time) 

NAME = "New_Ribeiro_model_fulltrained" + date_time
beginning = time.time()
X_train = convert_X(X_train,CHAR)
X_test = convert_X(X_test,CHAR)

monitor = EarlyStopping(monitor ='val_loss',
                        min_delta=0.001,
                        patience=20,
                        verbose=0,
                        restore_best_weights=True)

tensorboard = TensorBoard(log_dir = "logs/"+ NAME,histogram_freq=2)
batch_size = 64
history = new_model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=50,
                    validation_data = (X_test,y_test),
                    verbose=1,
                    callbacks = [monitor,tensorboard])

new_model.save(NAME + ".hdf5")

y_predicted = new_model.predict(X_test,verbose=1)


from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)

def find_optimal_precision_recall(y_true, y_score):
    """Determine precision and recall values that result in the highest F1 score."""
    n = y_true.shape[1]
    optimal_precision = []
    optimal_recall = []
    optimal_threshold = []
    for k in range(n):
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
        max_index = np.argmax(f1)
        optimal_precision.append(precision[max_index])
        optimal_recall.append(recall[max_index])
        t = threshold[max_index-1] if max_index != 0 else threshold[0]-1e-10
        optimal_threshold.append(t)
    return np.array(optimal_precision), np.array(optimal_recall), np.array(optimal_threshold)

p,r,t = get_optimal_precision_recall(y_test.astype(float),y_predicted)
print("precision:",p)
print("recall:",r)
print('thresholds:',t)


import tqdm
y_score = []
thresholds = t


for pred in tqdm.tqdm(y_predicted):
    res = [0,0,0,0]
    for ind,val in enumerate(pred):

        if val > thresholds[ind]:
           
            res[ind] = 1 

    y_score.append(res)

import statistics

failed_prec = []
nb_fail = 0
fail_binary = 0

for i in range(len(y_score)):
    if not np.array_equal(y_test[i], y_score[i]):
        failed_prec.append(i)
        nb_fail += 1
    for index, pred in enumerate(y_score[i]):
        if pred != y_test[i, index]:
            fail_binary += 1

print(f"Total binary accuracy is {(1 - fail_binary / (len(y_score) * 4)) * 100}%")
print(f"Total precision is {(1 - nb_fail / len(y_score)) * 100}%")
print(f"Total recall is {statistics.mean(r) * 100}%")
