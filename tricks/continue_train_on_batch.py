import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import array
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent, Agent
from keras.callbacks import ReduceLROnPlateau as RLRL
from game2048.displays import Display
from keras.callbacks import TensorBoard
from keras.models import Model,Input
from keras.layers import Dense, concatenate,Conv2D,Flatten
from keras.models import load_model

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder()
ohe.fit([[0], [2], [4], [8], [16], [32], [64], [128], [256], [512], [1024], [2048]])


def board_transform(board_tO_onehot, gamesize=4):
    """transform the board by one-hot encoding"""
    new_board = ohe.transform(board_tO_onehot[0][0]).toarray()
    for i in range(gamesize):
        for j in range(gamesize):
            if i == 0 and j == 0:
                continue
            element = board_tO_onehot[i][j]
            new_element = ohe.transform(element).toarray()
            new_board = np.vstack((new_board, new_element))
    # print(new_board)
    y = np.expand_dims(new_board, axis=0)
    return y


label_ohe = OneHotEncoder()
label_ohe.fit([[0], [1], [2], [3]])


def label_transform(label_to_onehot):
    tmp_label = label_ohe.transform(label_to_onehot).toarray()
    return np.array(tmp_label).reshape(1, -1)
    # return label_to_onehot


# In[3]:


game_size = 4
zit_score_to_win = 2048

n_epoch = 10
n_steps_per_epoch = 1000
every_epoch = 1
ite = 1000  # the number of iteration to generate

data_path = '/home/srtp/ML/2048-api-master/2048-api-master/cnn_train_data/'
tenb_path = "/home/srtp/ML/2048-api-master/2048-api-master/my_log_dir/"

count = 0
min_score = 0
max_score = 7
rnn_filter = 64

drop = 0.15

lr = 0.001
decay = lr / n_epoch
game = Game(game_size, zit_score_to_win)


def get_score(board):
    # input a 16*12 tensor
    # notice that return is a number between 0~11(reflect 0~2048)
    score = 0
    for number in board:
        max_num = np.argmax(number)
        if max_num > score:
            score = max_num
    return score


def generate_array_from_file(data_path):
    global count
    while 1:
        load_path = os.path.join(data_path, 'cnn_dataset_' + str(count) + '.npz')

        initial_data = np.load(load_path)['arr_0']
        initial_label = np.load(load_path)['arr_1']
        """
        if len(train_label)>max_index:
            train_data = train_data[min_index:max_index]
            train_label = train_label[min_index:max_index]
        else:
            train_data = train_data[min_index:]
            train_label = train_label[min_index:]
        """
        for i in range(initial_data.shape[0]):
            if get_score(initial_data[i]) >= min_score:
                break

        if max_score == 11:
            train_data = initial_data[i:]
            train_label = initial_label[i:]
        else:

            train_data = np.expand_dims(initial_data[i], axis=0)
            train_label = initial_label[i]
            i = i + 1

            while 1:
                if get_score(initial_data[i]) < max_score:
                    train_data = np.vstack((train_data, np.expand_dims(initial_data[i], axis=0)))
                    train_label = np.vstack((train_label, initial_label[i]))
                    i = i + 1
                else:
                    break
        count = count + 1
        if count >= ite:
            count = 0
        train_data = train_data.reshape((train_data.shape[0], 4, 4, 12))
        yield (train_data, train_label)


def generate_val_dataset(data_path, num):
    load_path = os.path.join(data_path, 'cnn_dataset_' + str(num) + '.npz')

    initial_data = np.load(load_path)['arr_0']
    initial_label = np.load(load_path)['arr_1']
    """
    if len(val_label)>max_index:
        val_data = val_data[min_index:max_index]
        val_label = val_label[min_index:max_index]

    else:
        val_data = val_data[min_index:]
        val_label = val_label[min_index:]
    """
    print(initial_data.shape)
    for i in range(initial_data.shape[0]):
        # print(get_score(initial_data[i]))
        if get_score(initial_data[i]) >= min_score:
            break

    if max_score == 11:
        val_data = initial_data[i:]
        val_label = initial_label[i:]
    else:
        val_data = np.expand_dims(initial_data[i], axis=0)
        val_label = initial_label[i]
        i = i + 1

        while 1:
            if get_score(initial_data[i]) < max_score:
                val_data = np.vstack((val_data, np.expand_dims(initial_data[i], axis=0)))
                val_label = np.vstack((val_label, initial_label[i]))
                i = i + 1
            else:
                break

    val_data = val_data.reshape((val_data.shape[0], 4, 4, 12))

    return val_data, val_label


model_path = '/home/srtp/PycharmProjects/2048_on_data/cnn_model_128.h5'
model = load_model(model_path)

_val_data, _val_label = generate_val_dataset(data_path, 1498)
val_data_, val_label_ = generate_val_dataset(data_path, 1499)

val_data = np.vstack((_val_data, val_data_))
val_label = np.vstack((_val_label, val_label_))

# x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, train_size=batch_size)
history = model.fit_generator(generate_array_from_file(data_path),
                              steps_per_epoch=n_steps_per_epoch,
                              epochs=n_epoch,
                              validation_data=(val_data, val_label))


model.save('/home/srtp/PycharmProjects/2048_on_data/cnn_model_' + str(2 ** max_score) + '.h5')

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


class ZitAgent(Agent):
    """my agent"""

    def __init__(self, game, display=None):
        super().__init__(game, display)

    def step(self):
        # print('board is :', board_transform(self.game.board))
        one_hot_direction = model.predict(board_transform(self.game.board, 4).reshape(1,4,4,12))
        direction = np.argmax(one_hot_direction[0])
        # print(direction)
        return direction


# test
from game2048.game import Game
from game2048.displays import Display


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=ZitAgent)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))


