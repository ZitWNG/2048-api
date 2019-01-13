import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(float(input("0: left, 1: down, 2: right, 3: up = "))) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction



# my agent


def board_transform(board_to_transe,board_dimension, gamesize=4):
    """transform the board by one-hot encoding"""
    if board_dimension == 3:
        inputboard = np.zeros((1, 16, 12))
        for p in range(4):
            for q in range(4):
                num = board_to_transe[p, q]
                if num == 0:
                    inputboard[0, 4*p+q, 0] = 1
                else:
                    inputboard[0, 4*p+q, int(np.log2(num))] = 1 
    elif board_dimension==4:
        inputboard = np.zeros(( 1, 4, 4, 12))
        for p in range(4):
            for q in range(4):
                num = board_to_transe[p, q]
                if num == 0:
                    inputboard[0, p, q, 0] = 1
                else:
                    inputboard[ 0, p, q, int(np.log2(num))] = 1
    else:
        print('OH NO! you data denmension is not ture!')

    return inputboard


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.models import load_model

class ZitAgent(Agent):
    """my agent"""
    def __init__(self, game, display = None):
        super().__init__(game, display)
        self.zmodel = load_model('/home/srtp/ML/2048-api-master/model0002.h5')
        self.zmodel.summary()

    def step(self):
        #this_board = board_transform(self.game.board, 4)
        
        one_hot_direction = self.zmodel.predict(np.array(board_transform(self.game.board,4)))
        direction = np.argmax(one_hot_direction[0])
        return direction

