import numpy as np
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent, Agent
import os
from game2048.displays import Display
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import load_model


board_dimension = 4


def board_transform(board_to_transe, gamesize=4):
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


label_ohe=OneHotEncoder()
label_ohe.fit([[0],[1],[2],[3]])


def label_transform(label_to_onehot):
    tmp_label = label_ohe.transform(label_to_onehot).toarray()
    return np.array(tmp_label).reshape(1, -1)
    #return label_to_onehot


game_size = 4
zit_score_to_win = 2048


class ZitAgent(Agent):
    """my agent"""
    def __init__(self, game, display = None):
        super().__init__(game, display)
        self.zmodel = load_model('/home/zitwng/2048-api/model0002.h5')
        self.zmodel.summary()

    def step(self):
        # this_board = board_transform(self.game.board, 4)

        one_hot_direction = self.zmodel.predict(np.array(board_transform(self.game.board)))
        direction = np.argmax(one_hot_direction[0])
        return direction
