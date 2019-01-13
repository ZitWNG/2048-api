# -*- coding: utf-8 -*-
# zit Wang
from game2048.game import Game
import numpy as np
ACTION_MAP = ['↑', '↓', '←', '→']


def state2tensor(state):
    new_board = np.zeros((4, 4, 16), dtype=np.float32)
    for i in range(4):
        for j in rang(4):
            num = state[i, j]
            if num == 0:
                new_board[i,j,0]=1
            else:
                new_board[i,j,int(np.log2(num))] = 1
    return new_board


class Env2048(object):
    def __init__(self,score_to_win=2048, dim=4, base=2, state=None):
        self.game = Game(size=dim, score_to_win=score_to_win)
        self.dim_ = dim
        self.base_ = base
        self.start_tiles_ = 2

        self.score_to_win = score_to_win
        """
        if state is None:
            self.state_ = self.game.board
        else:
            self.state_ = state.copy()
        """
    def __str__(self):
        conver2char = lambda num: '%5d' % (num) if num > 0 else ' ' * 5
        demarcation = ('+' + '-' * 5) * self.dim_ + '+\n'
        ret = demarcation
        ret += demarcation.join([
            '|' + '|'.join([conver2char(num) for num in row]) + '|\n'
            for row in self.game.board
        ])
        ret += demarcation
        return ret

    def __repr__(self):
        return self.__str__(self)

    def set_state(self, state):
        self.game.board = state

    def get_state(self):
        return self.game.board

    def to_tensor(self):
        return state2tensor(self.game.board)

    def reset(self):
        self.game = Game(size = self.dim_, score_to_win=self.score_to_win)
        return self.game.board

    def step(self, action):
        # 0 left; 1 down; 2 right; 3 up
        score0 = self.game.score
        len_1 = len(self.game._where_empty)
        """
        print('--------------')
        print(self.game.score)
        print(self.game.board)
        print('---------')
        """
        self.game.move(action)

        if self.is_terminate() == 2:
            reward = 20
        elif self.is_terminate() == 0:
            reward = 4
        else:
            reward = -8
        return self.game.board, reward, self.is_terminate(), ''

    def get_return(self):
        return self.game.score
    
    def is_terminate(self):
        return self.game.end

