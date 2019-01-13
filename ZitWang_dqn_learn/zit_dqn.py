from functools import namedtuple
import random
import numpy as np

from game2048.expectimax import board_to_move
from keras.models import Sequential, Model, Input
from keras.layers import Flatten, Dense, Dropout, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop

from env_edit_reward import state2tensor, Env2048

Guide = namedtuple('Guides', ('state', 'action', 'next_state', 'reward'))
model_path = '/home/srtp/2048-api-master/zit_dqn.h5'


class Guides:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def ready(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(
            self,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=2000,
            batch_size=128,
            e_greedy_increment=None,
            output_graph=False,
    ):

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        # 动作会有随机， 1-e_greedy的概率随机决策
        self.replace_target_iter = replace_target_iter

        self.batch_size = batch_size
        # 随着训练降低epsilon
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 1
        self.memory = Guides(memory_size)
        self._build_net()

    def _build_net(self):
        # 构建网络
        # 待修改
        # ==================evualte_network=========
        filters = 128
        eval_inputs = Input((4, 4, 16))
        conv1 = Conv2D(filters, (1, 2), padding='same', kernel_initializer='he_uniform')(eval_inputs)
        conv2 = Conv2D(filters, (2, 1), padding='same', kernel_initializer='he_uniform')(eval_inputs)
        conv3 = Conv2D(filters, (1, 3), padding='same', kernel_initializer='he_uniform')(eval_inputs)
        conv4 = Conv2D(filters, (3, 1), padding='same', kernel_initializer='he_uniform')(eval_inputs)
        conv5 = Conv2D(filters, (4, 4), padding='same', kernel_initializer='he_uniform')(eval_inputs)

        hidden = concatenate([Flatten()(x) for x in [conv1, conv2, conv3, conv4, conv5]])
        eval_hidden = BatchNormalization()(hidden)
        eval_hidden = Activation('relu')(eval_hidden)

        for width in [512, 128]:
            eval_hidden = Dense(width, kernel_initializer='he_uniform')(eval_hidden)
            eval_hidden = BatchNormalization()(eval_hidden)
            eval_hidden = Activation('relu')(eval_hidden)

        self.q_eval = Dense(4)(eval_hidden)

        # ==================target network=============

        target_inputs = Input(shape=(4, 4, 16))
        conv21 = Conv2D(filters, (1, 2), padding='same', kernel_initializer='he_uniform')(target_inputs)
        conv22 = Conv2D(filters, (2, 1), padding='same', kernel_initializer='he_uniform')(target_inputs)
        conv23 = Conv2D(filters, (1, 3), padding='same', kernel_initializer='he_uniform')(target_inputs)
        conv24 = Conv2D(filters, (3, 1), padding='same', kernel_initializer='he_uniform')(target_inputs)
        conv25 = Conv2D(filters, (4, 4), padding='same', kernel_initializer='he_uniform')(target_inputs)

        hidden2 = concatenate([Flatten()(x) for x in [conv21, conv22, conv23, conv24, conv25]])
        target_hidden = BatchNormalization()(hidden2)
        target_hidden = Activation('relu')(target_hidden)

        for width in [512, 128]:
            target_hidden = Dense(width, kernel_initializer='he_uniform')(target_hidden)
            target_hidden = BatchNormalization()(target_hidden)
            target_hidden = Activation('relu')(target_hidden)

        self.q_next = Dense(4)(target_hidden)

        self.model1 = Model(target_inputs, self.q_next)
        self.model2 = Model(eval_inputs, self.q_eval)
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

        self.model1.summary()
        self.model2.summary()

    def target_replace_op(self):
        model_params = self.model2.get_weights()
        self.model1.set_weights(model_params)
        print('+++++++=======================+++++++++++++++')
        print("params has changed")
        print('+++++++=======================+++++++++++++++')

    def choose_action(self, state, greedy=False):
        # 决策
        if greedy or np.random.random() < self.epsilon:
            board = np.expand_dims(state, axis=0)
            actions_value = self.model1.predict(board)
            action = np.argmax(actions_value)

            print('=====================================')
            print('action_value', actions_value)
            print('-------------------------------------')
            print(action)
            print('=====================================')

        else:
            # action = np.random.randint(0, 4)
            gameboard = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    num = np.argmax(state[i][j])
                    if num == 0:
                        gameboard[i][j] = 0
                    else:
                        gameboard[i][j] = 2 ** num

            action = board_to_move(gameboard)

        return action

    def learn(self):
        # 模型训练
        if self.memory.ready(self.batch_size):
            guides = self.memory.sample(self.batch_size)

            learn_state = []
            learn_q_target = []

            for guide in guides:
                q_eval = self.model2.predict(np.expand_dims(guide.state, axis=0))
                if guide.next_state is None:
                    q_next = 0
                else:
                    q_next = self.model1.predict(np.expand_dims(guide.next_state, axis=0))

                q_target = q_eval.copy()

                """
                print(q_next)
                print(q_target)
                """
                learn_state.append(guide.state)
                reward_now = guide.reward
                action = guide.action

                q_target[0, action] = reward_now + self.gamma * np.max(q_next)
                learn_q_target.append(q_target.reshape(4))

            self.model2.fit(np.array(learn_state), np.array(learn_q_target), epochs=10)
            print('traning')
            self.model2.save(model_path)

            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1

    def training(self, num_episodes):
        last50 = []
        env = Env2048(score_to_win=1024)

        for i_episode in range(num_episodes):
            env.reset()
            state = env.to_tensor()
            for i_step in range(100000):
                action = self.choose_action(state)
                _, reward, done, _ = env.step(action)

                if not done:
                    next_state = env.to_tensor()
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if done:
                    print(env.get_state())
                    break

                # print(self.memory.memory[i_step])

            self.learn()

            if i_episode % self.replace_target_iter == 0:
                self.target_replace_op()

            last50.append(env.get_return())
            if len(last50) > 50:
                last50.pop(0)
            self.average_ = np.average(last50)

            print('Episode %3d, steps %4d, score %5d, avg %5.0f, eps %.4f' % (
            self.learn_step_counter, i_step, env.get_return(), self.average_, self.epsilon))

            self.learn()


if __name__ == '__main__':
    n_episodes = 2000
    player = DQN(e_greedy_increment=0.001)

    player.training(n_episodes)