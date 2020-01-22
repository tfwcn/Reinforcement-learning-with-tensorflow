"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        '''
        n_actions:4，动作数量(上下左右)
        n_features:2，状态数量(x,y)
        '''
        print('n_actions:', n_actions)
        print('n_features:', n_features)
        print('learning_rate:', learning_rate)
        print('reward_decay:', reward_decay)
        print('e_greedy:', e_greedy)
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.cost_his = []

    def _build_net(self):
        '''建立预测模型和target模型'''
        # ------------------ build evaluate_net ------------------
        s = tf.keras.Input([None, self.n_features], name='s')
        q_target = tf.keras.Input([None, self.n_actions], name='Q_target')
        # 预测模型
        x = tf.keras.layers.Dense(20, activation=tf.keras.activations.relu, name='l1')(s)
        x = tf.keras.layers.Dense(self.n_actions, name='l2')(x)
        self.eval_net = tf.keras.Model(inputs=s, outputs=x)
        # 损失计算函数
        self.loss = tf.keras.losses.MeanSquaredError()
        # 梯度下降方法
        self._train_op = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # ------------------ build target_net ------------------
        s_ = tf.keras.Input([None, self.n_features], name='s_')
        # target模型
        x = tf.keras.layers.Dense(20, activation=tf.keras.activations.relu, name='l1')(s_)
        x = tf.keras.layers.Dense(self.n_actions, name='l2')(x)
        self.target_net = tf.keras.Model(inputs=s_, outputs=x)
    
    def replace_target(self):
        '''预测模型权重更新到target模型权重'''
        self.target_net.get_layer(name='l1').set_weights(self.eval_net.get_layer(name='l1').get_weights())
        self.target_net.get_layer(name='l2').set_weights(self.eval_net.get_layer(name='l2').get_weights())

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_net(observation).numpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        with tf.GradientTape() as tape:
            q_next = self.target_net(batch_memory[:, -self.n_features:]).numpy()
            q_eval = self.eval_net(batch_memory[:, :self.n_features])

            # change q_target w.r.t q_eval's action
            q_target = q_eval.numpy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            """
            For example in this batch I have 2 samples and 3 actions:
            q_eval =
            [[1, 2, 3],
            [4, 5, 6]]

            q_target = q_eval =
            [[1, 2, 3],
            [4, 5, 6]]

            Then change q_target with the real q_target value w.r.t the q_eval's action.
            For example in:
                sample 0, I took action 0, and the max q_target value is -1;
                sample 1, I took action 2, and the max q_target value is -2:
            q_target =
            [[-1, 2, 3],
            [4, 5, -2]]

            So the (q_target - q_eval) becomes:
            [[(-1)-(1), 0, 0],
            [0, 0, (-2)-(6)]]

            We then backpropagate this error w.r.t the corresponding action to network,
            leave other action as error=0 cause we didn't choose it.
            """

            # train eval network
            self.cost = self.loss(y_true=q_target,y_pred=q_eval)
            # print('loss:', self.cost)
            
        gradients = tape.gradient(
            self.cost, self.eval_net.trainable_variables)

        self._train_op.apply_gradients(
            zip(gradients, self.eval_net.trainable_variables))
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



