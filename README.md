# 2048 api
## 作业提交：
* /model.py， 是我生成模型使用的程序（和model_code/model2.py一样）

* /model0002.h5，是我运行的agent中使用的模型

* /game2048/agents.py 在保留原来agents.py不变的基础上，增加了ZitAgent（我的agent）

* /evaulate.py 在原来的基础上，仅将调用expectimaxagent改成我的agent（ZitAgent）

* /generate_fingerprint.py:在原来的基础上，将待修改部分的expectimaxagent换成了
  我的模型zitagent

* /EE369_fingerprint.json

* /EE369_evaluation.log


## 代码结构：
  * model.py:
        * 1.scheduler是lr随着epoch下降的优化
        
        * 2.generate开头的两个函数分别是读取训练集（生成器），验证集（返回
        一个数组），读取范围为全局游戏

        * 3.从73行from keras.models import Sequential开始是神经网络模型
        
        * 4.从133行acc=history.history['acc']开始是acc和loss的训练过程图

## 运行方式：
  * 训练：
    * 1.数据集保存形式为n*16*12的数组。（注：内部有reshape成n*4*4*12的部分）

    * 2.训练时需修改路径及名字（以‘model.py’为例）
        ```python
          data_path = r"./cnn_train_data/" # line24
          zit_model.save(r'model002.h5')  # line131
          load_path = os.path.join(data_path, 'new_cnn_dataset_'+str(count)+'.npz')
          # line 52

          _val_data, _val_label = generate_val_dataset(data_path, 7085)
          val_data_, val_label_ = generate_val_dataset(data_path, 7085)
          # line 118
      ```
    * 3.命令行输入 python model.py
  * 测试：
    * 在/game2048/agents 中，将load_model的路径设置好。
    * 在命令行中输入 python evualate.py

## 其余内容目录：
* [`tricks/`](tricks/)：在训练过程中曾经尝试过的优化，但在最后未使用

  * [continiue_train_on_batch](): 继续训练模型。因为在估计新的模型是否有效的时候，我
  都会先少跑几个epoch，看看效果，旧的模型保存下来，如果觉得可用的话，就会继续训练，
  以节省时间。
  * [my_training_online.ipynb]()：在线学习，为了磨合多个在不同数据区间上训练的模型，
  如128、256、512分别用一个模型，为了使得拼接效果更好，使用online training


* [models_code/]():我预想的模型改良方案，但因时间关系，未能去实施

    * model2.py 网络结构如下，在助教给的cnn的基础上增加了rnn分支：
      ```python

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
      ```

    * conv3.py 网络结构如下，考虑到游戏时需考虑平滑性和梯度性，

      ```python        
        inputs = Input((4, 4, 12))
        filters1 = 128

        cnn1 = Conv2D(filters1, (1,4),strides=(1, 1), kernel_initializer='he_uniform',
                     padding='same',activation='relu')(inputs)

        cnn2 = Conv2D(filters1, (4,1),strides=(1, 1), kernel_initializer='he_uniform',
                     padding='same',activation='relu')(inputs)
        cnn3 = Conv2D(filters1, (2,1),strides=(1, 1), kernel_initializer='he_uniform',
                     padding='same',activation='relu')(inputs)
        cnn4 = Conv2D(filters1, (1,2),strides=(1, 1), kernel_initializer='he_uniform',
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
      ```
    * conv4.py 在model3的基础上增加了一个反向rnn分支。
      结构如下
      ```python
      inputs = Input((4, 4, 12))
      filters1 = 128

      cnn1 = Conv2D(filters1, (1,4),strides=(1, 1), kernel_initializer='he_uniform',
                   padding='same',activation='relu')(inputs)

      cnn2 = Conv2D(filters1, (4,1),strides=(1, 1), kernel_initializer='he_uniform',
                   padding='same',activation='relu')(inputs)
      cnn3 = Conv2D(filters1, (2,1),strides=(1, 1), kernel_initializer='he_uniform',
                   padding='same',activation='relu')(inputs)
      cnn4 = Conv2D(filters1, (1,2),strides=(1, 1), kernel_initializer='he_uniform',
                   padding='same',activation='relu')(inputs)

      cnn5 = Conv2D(filters1, (4,4),strides=(1, 1), kernel_initializer='he_uniform',
                   padding='same',activation='relu')(inputs)

      con1 = concatenate([cnn1,cnn2, cnn3, cnn4,cnn5])

      con1 = Reshape(target_shape=(16, 640))(con1)

      rnn1 = LSTM(128, return_sequences = True)(con1)
      rnn1 = LSTM(128, return_sequences = False)(rnn1)
      dense1 = Dense(256, activation='relu')(rnn1)
      dense1 = Dense(64, activation='relu')(dense1)

      rnn2 = Bidirectional(LSTM(128, return_sequences = True))(con1)
      rnn2 = Bidirectional(LSTM(128, return_sequences = False))(rnn2)
      dense2 = Dense(256 , activation='relu')(rnn2)
      dense2 = Dense(64, activation='relu')(dense2)

      con2 = concatenate([dense1, dense2])

      outputs = Dense(4, activation='softmax')(con2)
      zit_model = Model(inputs=inputs, outputs=outputs)
      zit_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

      zit_model.summary()
      ```
    * model5.py 在model2的基础上增加了一个反向rnn分支。大致同model4和model3的关系
      此不赘述。

* [history_models_code/]():保存了我在探索时使用的几个效果相对来说较好的几个模型。
  此处不赘述。

* [ZitWang_dqn_learn/]():我学习强化学习时写的代码。
  * [zit_dqn.py & zit_dqn.py]()：根据莫烦的教程和另一个kears版的dqn改写，
  并结合了我们的game，创建了环境。但是学习起来会发现越学越差，尚待完善，

    此处贴上借鉴的三个连接：

    https://github.com/yytdfc/DQN-2048
    https://blog.csdn.net/senjie_wang/article/details/82708381
    https://github.com/Jason33Wang/ReinforcementLearning_by_keras

  * [dqn_baseon_others.py]()：从git上找的另外一个大佬的dqn，使用tensorflow
  的框架，大佬在他的代码里能跑到20000+（折合下来应该有2400多）

    代码连接：https://github.com/xdavidwu/2048-dqn
    但是难度和我们这个游戏不同，他随机0.1的概率生成是4，
    而我们是0.5的概率是4。

      这篇代码主要是将我们的游戏接入了这个dqn框架，并未做很大的修改，
      但自己改写了能够生成score（按合并的数字总和）的模块。
      最后只能玩2000score左右（折合下来应该是两三百的样子）

      另外我还写了一篇大改的，但是最后也没能跑成功。
