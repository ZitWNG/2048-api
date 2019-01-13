#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import array
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent, Agent
from keras.callbacks import ReduceLROnPlateau as RLRL
from game2048.displays import Display
from keras.callbacks import TensorBoard
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

game_size = 4
zit_score_to_win = 2048

n_epoch = 20 # 跑之前修改
n_steps_per_epoch = 7000
ite = 7000 # the number of iteration to generate

data_path = r"./cnn_train_data/"

count = 0
rnn_filter = 64

drop = 0.15

lr= 0.001
decay= lr/n_epoch
game = Game(game_size, zit_score_to_win)

from keras.callbacks import LearningRateScheduler as LRS
import keras.backend as K


def scheduler(epoch):
    if epoch% 3 == 0 and epoch != 0:
        lr = K.get_value(zit_model.optimizer.lr)
        K.set_value(zit_model.optimizer.lr, lr*0.1)
        print('lr change to {}'.format(lr*0.1))
    return K.get_value(zit_model.optimizer.lr)

reduce_lr = LRS(scheduler)


def generate_array_from_file(data_path):
    global count
    while 1:
        load_path = os.path.join(data_path, 'new_cnn_dataset_'+str(count)+'.npz')

        train_data = np.load(load_path)['arr_0']
        train_label = np.load(load_path)['arr_1']

        count = count+1
        if count >= ite:
            count = 0
        train_data = train_data.reshape((train_data.shape[0],4,4,12))
        yield (train_data, train_label)

def generate_val_dataset(data_path, num):
    load_path = os.path.join(data_path, 'new_cnn_dataset_'+str(num)+'.npz')

    val_data = np.load(load_path)['arr_0']
    val_label = np.load(load_path)['arr_1']

    val_data = val_data.reshape((val_data.shape[0], 4, 4, 12))

    return val_data, val_label

from keras.layers import  Flatten, Dense, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,LSTM,concatenate,Bidirectional
from keras import Model, Input


inputs = Input((4, 4, 12))
filters1 = 128

cnn1 = Conv2D(filters1, (1, 2),strides=(1, 1), kernel_initializer='he_uniform',
               padding='same',activation='relu')(inputs)

cnn2 = Conv2D(filters1, (2,1),strides=(1, 1), kernel_initializer='he_uniform',
               padding='same',activation='relu')(inputs)
cnn3 = Conv2D(filters1, (2,2),strides=(1, 1), kernel_initializer='he_uniform',
               padding='same',activation='relu')(inputs)
cnn4 = Conv2D(filters1, (3,3),strides=(1, 1), kernel_initializer='he_uniform',
               padding='same',activation='relu')(inputs)

cnn5 = Conv2D(filters1, (4,4),strides=(1, 1), kernel_initializer='he_uniform',
               padding='same',activation='relu')(inputs)

con1 = concatenate([cnn1,cnn2, cnn3, cnn4,cnn5])

con1 = Reshape(target_shape=(16, 640))(con1)

rnn1 = LSTM(128, return_sequences = True)(con1)
rnn1 = LSTM(128, return_sequences = False)(rnn1)
dense1 = Dense(256, activation='relu')(rnn1)
dense1 = Dense(64, activation='relu')(dense1)

outputs = Dense(4, activation='softmax')(dense1)
zit_model = Model(inputs=inputs, outputs=outputs)
zit_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

zit_model.summary()

_val_data, _val_label = generate_val_dataset(data_path, 7085)
val_data_, val_label_ = generate_val_dataset(data_path, 7085)

val_data = np.vstack((_val_data, val_data_))
val_label = np.vstack((_val_label, val_label_))

# x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, train_size=batch_size)
history = zit_model.fit_generator(generate_array_from_file(data_path),
                            steps_per_epoch=n_steps_per_epoch, 
                            epochs = n_epoch,
                            validation_data=(val_data,val_label),
                             callbacks=[reduce_lr])

zit_model.save(r'model0002.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
