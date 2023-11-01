"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym Pendulum-v1, continual action space
ENV介绍: https://gymnasium.farama.org/environments/classic_control/pendulum/
observation[0] -- cos(theta), [-1,1] float32
observation[1] -- sin(theta), [-1,1] float32
observation[2] -- angular velocity, [-8,8] float32
action -- torque, [-2,2] float32
reward -- r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2), theta \in [-pi, pi]
The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].
The episode truncates at 200 time steps.




Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
"""

import argparse
import os
import time
from tqdm import tqdm
import math

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train_or_test', dest='train_or_test', type=int, default=1)
args = parser.parse_args()


#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v1'    # environment name
THETA_TARGET = 0            # Pendulum 目标角度
RANDOMSEED = 1              # random seed

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 10000     # size of replay buffer
BATCH_SIZE = 32             # update batchsize

MAX_EPISODES = 1000          # total number of episodes for training
MAX_EP_STEPS = 200          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 3                     # control exploration


################################  DDPG  #####################################
from DDPG_pendulum import DDPG
External_controller = DDPG(1, 3, 2)
External_controller.load_ckpt()





###############################  AsDDPG  ####################################

class AsDDPG(object):
    """
    Asddpg class
    """
    def __init__(self, a_dim, s_dim, a_bound, switch_dim=2):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，一个switch，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound, self.switch_dim = a_dim, s_dim, a_bound, switch_dim

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            #注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name='Asddpg_Actor' + name)

        #建立Critic网络，输入s，a。输出Q(s,switch)和A(s,a)值
        def get_critic(input_state_shape, input_action_shape, switch_num=switch_dim, name=''):
            """
            Build critic_adv network
            :param input_state_shape: state
            :param input_action_shape: act
            :param input_switch_shape: switch
            :param name: name
            :return: Q value Q(s,switch) and adv value A(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='A_s_input')
            a = tl.layers.Input(input_action_shape, name='A_a_input')

            p = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='Q_l1')(s) 
            # 改成double dueling DQN，q1为policy，q2为PID
            avalue = tl.layers.Dense(n_units=2, act=None, W_init=W_init, b_init=b_init, name='Q_adv')(p)
            mean = tl.layers.Dense(n_units=1, act=None, W_init=W_init, b_init=b_init, name='Q_mean')(p)
            qvalue = tl.layers.ElementwiseLambda(lambda x,y: x+y)([avalue,mean])        # [bsz, 2]  
            # Q_value = mean + advantage     


            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(x)
            x = tl.layers.Dense(n_units=1, act=None, W_init=W_init, b_init=b_init, name='A_l2')(x)
            adv = tl.layers.ElementwiseLambda(lambda x,y: x-y)([x,mean])        # [bsz, 1]  
            return tl.models.Model(inputs=[s, a], outputs=[qvalue, adv], name='Asddpg_Critic' + name)
        


        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()
        

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和critic参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()


        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)



    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        #获取要更新的参数包括actor和critic_adv,critic_q的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # 选择动作，把s带进入，输出switch and action
    def choose_action(self, s):
        """
        Choose action
        :param s: state   
        :return: act
        """
        s = np.array([s], dtype=np.float32)
        switch = np.argmax(self.critic([s, np.zeros((1, a_dim), dtype=np.float32)])[0])
        # 选择critic_q值大的作为action，0--policy, 1--PID
        if switch == 0: 
            a = self.actor(s)[0]    # [1,]
        elif switch == 1: 
            a = External_controller.choose_action(s)[0] # [1,]
            a = np.clip(np.random.normal(a, VAR*2), -2, 2)                      #  这里要将external_controller调差一些
        return switch, a
        

    # 选择动作，把s带进入，输出a_Actor
    def testphase_choose_action(self, s):
        """
        Choose action
        :param s: state  
        :return: act
        """
        return 0, self.actor(np.array([s], dtype=np.float32))[0]


    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, (self.s_dim + self.a_dim):\
                (self.s_dim + self.a_dim + 1)]          #从bt获得数据r
        bs_ = bt[:, (self.s_dim + self.a_dim + 1):-1]   #从bt获得数据s'
        bswitch = bt[:, -1]           # [bsz, ]

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            qvalue_, adv_ = self.critic_target([bs_, a_])
            # action不重要因为只用Qvalue=[q1, q2]
            switch = tf.math.argmax(self.critic([bs_, a_])[0], axis=1, output_type=tf.dtypes.int32)
            switch_onehot = tf.one_hot(switch, depth=2)    # [bsz, 2]
            q_ = switch_onehot * qvalue_   # [bsz,2]
            q_ = tf.expand_dims(tf.reduce_sum(q_, axis=1), axis=1)  # [bsz, 1]
           
            
            y_adv = br + GAMMA * adv_
            y_q = br + GAMMA * q_   # [bsz, 1]
            y_q = y_q * tf.one_hot(bswitch, depth=2) # [bsz,2]  只对下标为bswitch的通道进行梯度反传
            qvalue, adv = self.critic([bs, ba])          
            q = qvalue * tf.one_hot(bswitch, depth=2)   # [bsz,2]
            td_error = tf.losses.mean_squared_error(y_q, q) + tf.losses.mean_squared_error(y_adv, adv)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))    # -0.50820506 --> 0.5062

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)             # 不是从Replay buffer里取值
            _, adv = self.critic([bs, a])
            a_loss = -tf.reduce_mean(adv)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()


    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_, switch):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_, [switch]))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_, [switch]存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/Asddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/Asddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/Asddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/Asddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/Asddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/Asddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/Asddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/Asddpg_critic_target.hdf5', self.critic_target)




if __name__ == '__main__':
    
    #初始化环境
    env = gym.make(ENV_NAME, render_mode='human')
    env = env.unwrapped

    # reproducible，设置随机种子，为了能够重现
    # env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    #定义状态空间，动作空间，动作幅度范围
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    switch_dim = 2      # 控制器个数
    a_bound = env.action_space.high

    print('s_dim',s_dim)        # 3
    print('a_dim',a_dim)        # 1
    print('switch_dim',switch_dim)        # 2
    print('a_bound', a_bound)       # 2

    #用AsDDPG算法
    Asddpg = AsDDPG(a_dim, s_dim, a_bound, switch_dim=switch_dim)

    #训练部分：
    if args.train_or_test:  # train
        
        reward_buffer = []      #用于记录每个TEST_EP的reward，统计变化
        switch_buffer = []      #用于记录每个TEST_EP的switch中policy占比
        t0 = time.time()        #统计时间

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xlabel('episode')
        ax1.set_ylabel('reward')
        ax1.set_title('AsDDPG')
        ax2.set_xlabel('episode')
        ax2.set_ylabel('rate of policy')


        for i in range(MAX_EPISODES):
            t1 = time.time()
            s, _ = env.reset(seed=RANDOMSEED)
            ep_reward = 0       #记录当前EP的reward
            switch_policy_distribution = 0
            for j in range(MAX_EP_STEPS):
                s2theta = math.atan2(s[1], s[0])
                # Add exploration noise
                switch, a = Asddpg.choose_action(s)       #这里很简单，直接用actor估算出a动作
                if switch==0:   
                    switch_policy_distribution += 1

                    # 为了能保持开发，这里用了另外一种方式增加探索。
                    # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                    # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                    # 然后进行裁剪
                    a = np.clip(np.random.normal(a, VAR), -2, 2)  

                # 与环境进行互动
                s_, r, done, truncated, info = env.step(a)
                s_2theta = math.atan2(s_[1], s_[0])
                if truncated:
                    raise ValueError("Env needs to call reset().")

                # 保存s，a，r，s_
                Asddpg.store_transition(s, a, r / 10, s_, switch)

                # 第一次数据满了，就可以开始学习
                if Asddpg.pointer > BATCH_SIZE*4:
                    Asddpg.learn()

                #输出数据记录
                s = s_  
                ep_reward += r  #记录当前EP的总reward
                if j == MAX_EP_STEPS - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Rate of switching to Policy: {:.4f} | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward, switch_policy_distribution/MAX_EP_STEPS,
                            time.time() - t1
                        ), end=''
                    )


            # validation
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s, _ = env.reset()
                ep_reward = 0
                switch_policy_distribution = 0
                for j in range(MAX_EP_STEPS):

                    switch, a = Asddpg.choose_action(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
                    if switch==0:   switch_policy_distribution += 1

                    s_, r, done, truncated, info = env.step(a)
                    if truncated:
                        raise ValueError("Env needs to call reset().")

                    s = s_
                    ep_reward += r
                    if j == MAX_EP_STEPS - 1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Rate of switching to Policy: {:.4f} | Running Time: {:.4f}'.format(
                                i, MAX_EPISODES, ep_reward, switch_policy_distribution/MAX_EP_STEPS,
                                time.time() - t1
                            )
                        )

                        reward_buffer.append(ep_reward)
                        switch_buffer.append(switch_policy_distribution / MAX_EP_STEPS)

            if reward_buffer:
                ax1.clear()
                ax2.clear()
                ax1.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer, label='Episode reward', color='g')  # plot the episode vt
                ax1.set_ylim([-2000,0])
                ax2.plot(np.array(range(len(switch_buffer))) * TEST_PER_EPISODES, switch_buffer, label='Policy rate', color='b')
                ax2.set_ylim([0,1])
                ax1.set_xlabel('episode')
                ax1.set_ylabel('reward')
                ax1.set_title('AsDDPG')
                ax2.set_xlabel('episode')
                ax2.set_ylabel('rate of policy')
                plt.pause(0.1)

        plt.ioff()
        plt.show()
        # plt.savefig('AsDDPG_temp_result.png')
        plt.close()
        print('\nRunning time: ', time.time() - t0)
        Asddpg.save_ckpt()

    # test
    Asddpg.load_ckpt()
    if True:
        s, _ = env.reset()
        ep_reward = 0
        for _ in tqdm(range(MAX_EP_STEPS)):
            env.render()
            time.sleep(0.1)  

            _, a = Asddpg.testphase_choose_action(s)
            s, r, done, _, info = env.step(a)
            ep_reward += r
            if done:
                print("done!")
                break
        print('Test phase total reward:', ep_reward)   
