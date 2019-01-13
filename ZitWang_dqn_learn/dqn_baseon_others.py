import math, random, time
import tensorflow as tf
import numpy as np
import sys
from game2048.game import Game
from keras.models import load_model

from game2048.expectimax import board_to_move

model_score = 256

def board_to_oht(state):
    out_ = np.zeros((1,16,12))
    for i in range(4):
        for j in range(4):
            out_[0][i*4+j][state[i][j]] = 1
    return out_



class Memory:
    def __init__(self, size):
        self.size = size
        self.index = 0
        self.is_full = False
        self.s = np.empty([size, 4, 4], dtype=float)
        self.a = np.empty([size], dtype=int)
        self.r = np.empty([size])
        self.sp = np.empty([size, 4, 4], dtype=float)
        self.is_done = np.empty([size], dtype=bool)

    def push(self, s, a, r, sp, is_done):
        self.s[self.index] = s
        self.a[self.index] = a
        self.r[self.index] = r
        self.sp[self.index] = sp
        self.is_done[self.index] = is_done
        self.index = self.index + 1
        if self.index == self.size:
            self.is_full = True
            self.index = 0

    def isFull(self):
        return self.is_full

    def getBatch(self, batch_size):
        idx = np.random.choice(np.arange(self.size), batch_size)
        return np.take(self.s, idx, 0), np.take(self.a, idx), np.take(self.r, idx), np.take(self.sp, idx, 0), np.take(
            self.is_done, idx)

class Mygame(Game):
    @property
    def state(self):
        state = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                if self.board[i][j] != 0:
                    state[i][j] = math.log(self.board[i][j], 2)
        return state

    def emit_line(self, line):
        # [ i, _ , j , _ ]
        i, j = 0, 1
        score = 0
        while j < self.size:
            if line[j] == 0:
                j += 1
            elif line[i] == 0:
                line[i] = line[j]
                line[j] = 0
                j += 1
            elif line[i] == line[j]:
                line[i] += 1
                line[j] = 0
                score += 2 ** line[i]
                i += 1
                j += 1
            else:
                i += 1
            if j == i:
                j += 1
        return score

    def try_move(self, direct):
        self.score_ = 0
        if direct == 3:  # up
            for i in range(4):
                self.score_ += self.emit_line(self.state[:, i])

        elif direct == 1:  # down
            for i in range(4):
                self.score_ += self.emit_line(
                    self.state[-1::-1, i])

        elif direct == 0:  # left
            for i in range(4):
                self.score_ = self.emit_line(self.state[i, :])

        elif direct == 2:  # right
            for i in range(4):
                self.score_ = self.emit_line(
                    self.state[i, -1::-1])

        return self.score_


    def get_reward(self, dirction):
        reward_score = self.try_move(dirction)

        return reward_score


mem = Memory(262144)

batch = 128
eps_append = 0.99
eps_decay = 0.9997  # 0.99999
eps_base = 0.01
dis = 0.9
lr = 0.0001
qt_update_int = 64
double_dqn = True
sep_eps = True
# exp
outer_eps_base = 0.05
outer_eps_app = 0.85
outer_eps_dec = 0.998
# linear
inner_eps_init_frac = 0.1
inner_eps_inc_dur = 1
inner_eps_inc_dur_int = 10

train_counter = 0
print_counter = 0


model_256 = load_model('/home/srtp/ML/3convrnn_256.h5')
"""
x_ = tf.placeholder(tf.float32, [None, 4, 4])
xp_ = tf.reshape(x_, [-1, 4, 4, 1])
cb_ = tf.Variable(tf.zeros([128]))
cw_ = tf.Variable(tf.truncated_normal([2, 2, 1, 128], stddev=math.sqrt(2.0 / (2 * 2 * 1 + 128))))
co_ = tf.nn.relu(tf.nn.conv2d(xp_, cw_, [1, 1, 1, 1], 'VALID') + cb_)
cb2_ = tf.Variable(tf.zeros([64]))
cw2_ = tf.Variable(tf.truncated_normal([2, 2, 128, 64], stddev=math.sqrt(2.0 / (2 * 2 * 128 + 64))))
co2_ = tf.nn.relu(tf.nn.conv2d(co_, cw2_, [1, 1, 1, 1], 'VALID') + cb2_)
xf_ = tf.reshape(co2_, [-1, 4 * 64])
b_ = tf.Variable(tf.zeros([128]))
w_ = tf.Variable(tf.truncated_normal([256, 128], stddev=math.sqrt(2.0 / (256 + 128))))
o_ = tf.nn.relu(tf.matmul(xf_, w_) + b_)
# b1=tf.Variable(tf.zeros([32]))
# w1=tf.Variable(tf.truncated_normal([128,32],stddev=math.sqrt(2.0/(128+32))))
# o1=tf.nn.relu(tf.matmul(o,w1)+b1)
b2_ = tf.Variable(tf.zeros([4]))
w2_ = tf.Variable(tf.truncated_normal([128, 4], stddev=math.sqrt(2.0 / (128 + 4))))
y_ = tf.matmul(o_, w2_) + b2_
yp_ = tf.placeholder(tf.float32, [None, 4])
argmax_ = tf.argmax(y_, 1)

# saver = tf.train.Saver()
sess_ = tf.Session()
saver_ = tf.train.Saver()

saver_.restore(sess_, '/home/srtp/PycharmProjects/zit_dqn/dalao_model/model.1545678840.187.ckpt')
"""

# Q-value
x = tf.placeholder(tf.float32, [None, 4, 4])
xp = tf.reshape(x, [-1, 4, 4, 1])
cb = tf.Variable(tf.zeros([128]))
cw = tf.Variable(tf.truncated_normal([2, 2, 1, 128], stddev=math.sqrt(2.0 / (2 * 2 * 1 + 128))))
co = tf.nn.relu(tf.nn.conv2d(xp, cw, [1, 1, 1, 1], 'VALID') + cb)
cb2 = tf.Variable(tf.zeros([64]))
cw2 = tf.Variable(tf.truncated_normal([2, 2, 128, 64], stddev=math.sqrt(2.0 / (2 * 2 * 128 + 64))))
co2 = tf.nn.relu(tf.nn.conv2d(co, cw2, [1, 1, 1, 1], 'VALID') + cb2)
xf = tf.reshape(co2, [-1, 4 * 64])
b = tf.Variable(tf.zeros([128]))
w = tf.Variable(tf.truncated_normal([256, 128], stddev=math.sqrt(2.0 / (256 + 128))))
o = tf.nn.relu(tf.matmul(xf, w) + b)
# b1=tf.Variable(tf.zeros([32]))
# w1=tf.Variable(tf.truncated_normal([128,32],stddev=math.sqrt(2.0/(128+32))))
# o1=tf.nn.relu(tf.matmul(o,w1)+b1)
b2 = tf.Variable(tf.zeros([4]))
w2 = tf.Variable(tf.truncated_normal([128, 4], stddev=math.sqrt(2.0 / (128 + 4))))
y = tf.matmul(o, w2) + b2
yp = tf.placeholder(tf.float32, [None, 4])
ymax = tf.reduce_max(y, 1)
argmax = tf.argmax(y, 1)
loss = tf.losses.mean_squared_error(yp, y)
step = tf.train.RMSPropOptimizer(lr).minimize(loss)

# Q-target
qt_x = tf.placeholder(tf.float32, [None, 4, 4])
qt_xp = tf.reshape(qt_x, [-1, 4, 4, 1])
qt_cb = tf.Variable(cb.initialized_value())
qt_cw = tf.Variable(cw.initialized_value())
qt_co = tf.nn.relu(tf.nn.conv2d(qt_xp, qt_cw, [1, 1, 1, 1], 'VALID') + qt_cb)
qt_cb2 = tf.Variable(cb2.initialized_value())
qt_cw2 = tf.Variable(cw2.initialized_value())
qt_co2 = tf.nn.relu(tf.nn.conv2d(qt_co, qt_cw2, [1, 1, 1, 1], 'VALID') + qt_cb2)
qt_xf = tf.reshape(qt_co2, [-1, 4 * 64])
qt_b = tf.Variable(b.initialized_value())
qt_w = tf.Variable(w.initialized_value())
qt_o = tf.nn.relu(tf.matmul(qt_xf, qt_w) + qt_b)
# qt_b1=tf.Variable(b1.initialized_value())
# qt_w1=tf.Variable(w1.initialized_value())
# qt_o1=tf.nn.relu(tf.matmul(qt_o,qt_w1)+qt_b1)
qt_b2 = tf.Variable(b2.initialized_value())
qt_w2 = tf.Variable(w2.initialized_value())
qt_y = tf.matmul(qt_o, qt_w2) + qt_b2
qt_ymax = tf.reduce_max(qt_y, 1)
qt_update = [qt_cb.assign(cb), qt_cw.assign(cw),
             qt_cb2.assign(cb2), qt_cw2.assign(cw2),
             qt_b.assign(b), qt_w.assign(w),
             #        qt_b1.assign(b1),qt_w1.assign(w1),
             qt_b2.assign(b2), qt_w2.assign(w2)]

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))
saver = tf.train.Saver(max_to_keep=0)
tf.global_variables_initializer().run()


def train():
    global mem, sess, train_counter, print_counter
    if not mem.isFull(): return
    for j in range(1):
        s, a, r, sp, is_done = mem.getBatch(batch)
        q_ = sess.run(y, feed_dict={x: s})
        if double_dqn:
            ya = sess.run(argmax, feed_dict={x: sp})
            ynxt = sess.run(qt_y, feed_dict={qt_x: sp})
        else:
            ym = sess.run(qt_ymax, feed_dict={qt_x: sp})
        for i in range(batch):
            if is_done[i]:
                q_[i][a[i]] = r[i]
            else:
                if double_dqn:
                    q_[i][a[i]] = r[i] + dis * ynxt[i][ya[i]]
                else:
                    q_[i][a[i]] = r[i] + dis * ym[i]
        sess.run([step], feed_dict={x: s, yp: q_})
        train_counter += 1
        if train_counter == qt_update_int:
            train_counter = 0
            sess.run(qt_update)
        print_counter += 1
        if print_counter == 128:
            print_counter = 0
            oy, lo = sess.run([y, loss], feed_dict={x: s, yp: q_})
            if not double_dqn:
                print(lo, q_[0], oy[0], ym[0], r[0], s[0])
            else:
                print(lo, q_[0], oy[0], ynxt[0][ya[0]], r[0], s[0])
            # time.sleep(1)


def fillGame():
    game = Mygame(4, model_score)

    grid = game.state
    while True:
        dir = model_256.predict(board_to_oht(grid))
        game.move(dir)
        grid = game.state
        if game.end:
            return
        if game.score >= 256:
            break

    e_step = 0
    game_score = 0
    while True:
        dir = random.randint(0, 3)
        so = game.state
        r = game.get_reward(dir)
        game_score += r

        if r == 0:
            r = -1
        else:
            r = math.log(r, 2)

        game.move(dir)
        e_step+=1
        grid = game.state
        """
        print("======================")
        print(so, dir, r, grid)
        # eps_append*=eps_decay
        """
        if game.end:
            # r=0
            if game.end == 2:
                r += 100
            mem.push(so, dir, r, grid, True)
            # train()
            # print "\nGame Over"
            print("Filling score: " + str(game_score) + " max: " + str(game.score))
            # printGrid(grid)
            # time.sleep(1)
            return game_score, e_step
        else:
            mem.push(so, dir, r, grid, False)
            # train()


def testGame():
    # Create the game grid
    # The game should work for square grid of any size though
    game = Mygame(4, model_score)
    grid = game.state
    e_step = 0
    # printGrid(grid)

    while True:
        dir = model_256.predict(board_to_oht(grid))
        game.move(dir)
        grid = game.state
        if game.end:
            print("Testing  max: " + str(game.score))
            return 0, 0
        if game.score >= 512:
            print("Testing score: max: " + str(game.score) + " step: " + str(e_step))
            break

    game_score = 0  # Score of the user
    while True:
        dir = sess.run(argmax, feed_dict={x: [grid]})[0]
        r = game.get_reward(dir)
        game_score += r

        game.move(dir)
        grid = game.state
        e_step += 1

        if game.end:
            print("Testing score: " + str(game_score) + " max: " + str(game.score) + " step: " + str(e_step))

            return game_score,e_step


def trainGame():
    global eps_append
    game= Mygame(4, model_score)
    e_step = 0
    grid = game.state

    while True:
        dir = model_256.predict(board_to_oht(grid))
        game.move(dir)
        grid = game.state
        if game.end:
            return 0, 0
        if game.score >= 512:
            break

    outer_eps = outer_eps_base + outer_eps_app
    inner_eps = outer_eps * inner_eps_init_frac
    if not sep_eps: inner_eps = eps_append + eps_base

    game_score = 0
    while True:
        if random.random() < inner_eps:
            dir = random.randint(0, 3)
        else:
            dir = sess.run(argmax, feed_dict={x: [grid]})[0]
        e_step += 1
        # print(dir)
        so = grid
        r = game.get_reward(dir)
        game_score+=r

        if r == 0:
            r = -1
        else:
            r = math.log(r, 2)

        game.move( dir)
        grid = game.state

        if not sep_eps:
            # eps_append*=eps_decay
            inner_eps = eps_append + eps_base
        else:
            if inner_eps < outer_eps: inner_eps += outer_eps * (1 - inner_eps_init_frac) / inner_eps_inc_dur

        if game.end:
            if game.end == 2:
                r += 100
            mem.push(so, dir, r, grid, True)

            print("Training score: " + str(game_score) + " max: " + str(game.score) + " eps: " + str(
                inner_eps) + " step: " + str(e_step))

            return game_score, e_step
        else:
            mem.push(so, dir, r, grid, False)
            # train()


while not mem.isFull():
    fillGame()
print("Filled")

for i in range(200):
    t = 0
    s = 0
    for j in range(100):
        print(j)
        (tmp1, tmp2) = trainGame()
        t += tmp1
        s += tmp2
        for k in range(32):
            train()
        eps_append *= eps_decay
        outer_eps_app *= outer_eps_dec
        if (i * 100 + j) % inner_eps_inc_dur_int == 0: inner_eps_inc_dur += 1
    print("train " + str(i) + " avg: " + str(t / 100) + " step: " + str(s / 100))
    sys.stdout.flush()

    if i % 2 == 0:
        continue
    t = 0
    s = 0
    for j in range(100):
        tmp1, tmp2 = testGame()
        t += tmp1
        s += tmp2

    print("test " + " avg: " + str(t / 100) + " step: " + str(s / 100))
    saver.save(sess, '/home/srtp/PycharmProjects/zit_dqn/model_128/model0001.ckpt')
    sys.stdout.flush()

