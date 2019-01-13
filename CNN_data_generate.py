#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent, Agent

from game2048.displays import Display

import os


# In[2]:


from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder()
ohe.fit([ [0], [2], [4], [8], [16], [32], [64], [128], [256], [512], [1024], [2048]])

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
label_ohe.fit([[0],[1],[2],[3]])

def label_transform(label_to_onehot):
    tmp_label = label_ohe.transform(label_to_onehot).toarray()
    return np.array(tmp_label).reshape(1, -1)
    #return label_to_onehot


# In[3]:


class GenerateDataAgent(ExpectiMaxAgent):
    
    def __init__(self, game, display=None):
        super().__init__(game, display)
        
    def play(self, max_iter=np.inf, verbose=False):
        """edit paly function to generate training data"""
        n_iter = 0
        gamesize = self.game.size
        
        if (n_iter < max_iter) and (not self.game.end):
            # how to edit here?
            direction = self.step()
            train_data = board_transform(self.game.board, game_size)
            train_label = label_transform(direction)
            self.game.move(direction)
        
            while (n_iter < max_iter) and (not self.game.end):
                direction = self.step()
                n_iter = 1 + n_iter
                train_data = np.vstack((train_data, board_transform(self.game.board, gamesize)))
                train_label = np.vstack((train_label, label_transform(direction)))
                self.game.move(direction)
                """
                if self.game.end:
                    direction = self.step()
                    train_data = np.vstack((train_data, board_transform(self.game.board, gamesize)))
                    train_label = np.vstack((train_label, label_transform(direction)))
                    break
                """
        else:
            print("Oh! No!")
        return train_data, train_label


# In[4]:


data_path = '/home/srtp/ML/2048-api-master/2048-api-master/cnn_train_data/'
print(data_path)

game_size = 4
zit_score_to_win = 2048
AgentClass = GenerateDataAgent

game = Game(game_size, zit_score_to_win)
agent = AgentClass(game, display=Display())


# In[5]:


def data_generate(size=4,score_to_win=zit_score_to_win, AgentClass = GenerateDataAgent, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    train_data, train_label = agent.play()
    return train_data, train_label


# In[6]:


for i in range(2091,7001):
    save_path = os.path.join(data_path, 'cnn_dataset_'+str(i)+'.npz')
    print(save_path)
    train_data, train_label=data_generate()
    np.savez(save_path,train_data, train_label)


# In[ ]:




