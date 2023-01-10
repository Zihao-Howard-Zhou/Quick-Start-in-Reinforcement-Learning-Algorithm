import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import argparse

# 首先定义实验的超参数
def parse_args():
    parser = argparse.ArgumentParser(description='Parameters setting of DQN')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate for the net.')
    parser.add_argument('--num_episodes', type=int, default=500, help='the num of train epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--gamma', type=float, default=0.98, help='the discount rate')
    parser.add_argument('--epsilon', type=float, default=0.01, help='the epsilon rate')

    parser.add_argument('--target_update', type=float, default=10, help='the frequency of the target net')
    parser.add_argument('--buffer_size', type=float, default=10000, help='the size of the buffer')
    parser.add_argument('--minimal_size', type=float, default=500, help='the minimal size of the learning')

    parser.add_argument('--env_name', type=str, default="CartPole-v0", help='the name of the environment')

    args = parser.parse_args()
    return args

# 网络结构
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s):
        s = self.layer(s)
        return s

class DQN:
    def __init__(self, args):
        self.args = args
        self.hidden_dim = 128                   # Q网络隐藏层维度
        self.batch_size = args.batch_size
        self.minimum_size = args.minimal_size   # 当经验池中的经验到达某一数量时就可以开始学习
        self.lr = args.lr                       # 学习率
        self.gamma = args.gamma                 # 折扣因子
        self.epsilon = args.epsilon             # epsilon-贪婪策略
        self.target_update = args.target_update # 目标网络更新频率
        self.count = 0                          # 计数器,记录更新次数,指示目标网络是否应当被更新
        self.num_episodes = args.num_episodes   # 所需要采集的episode数量

        self.capacity = args.buffer_size         # 经验池的容量
        self.buffer = deque(maxlen=self.capacity) # 初始化经验池 (队列)

        self.env = gym.make(args.env_name)

        random.seed(args.seed)
        np.random.seed(args.seed)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_q_net = Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)

        self.optimizer = Adam(self.q_net.parameters(), lr=self.lr)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def store_transition(self,state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self):
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())        # 更新目标网络

        self.count += 1

        # 下面从经验池中随机抽取经验
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        states = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        actions = torch.tensor(action).view(-1, 1).to(self.device)
        rewards = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        dones = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)                           # Q value
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)   # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)         # TD target

        loss = torch.mean(F.mse_loss(q_values, q_targets))                         # 均方误差损失函数
        self.optimizer.zero_grad()                                                 # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        loss.backward()                                                            # 反向传播更新参数
        self.optimizer.step()

if __name__ == '__main__':
    args = parse_args()
    agent = DQN(args)
    return_list = []

    for episode in range(400):
        episode_return = 0
        state = agent.env.reset()
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = agent.env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_return += reward

            if len(agent.buffer) > agent.minimum_size:
                agent.update()

            if done:
                print('Episode index: ', episode, '| Episode_return: ', episode_return)
                break

        return_list.append(episode_return)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()









