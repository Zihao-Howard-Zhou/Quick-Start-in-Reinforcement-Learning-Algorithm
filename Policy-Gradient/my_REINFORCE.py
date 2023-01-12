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

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters setting of REINFORCE')

    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate for the net.')
    parser.add_argument('--num_episodes', type=int, default=500, help='the num of train epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--gamma', type=float, default=0.98, help='the discount rate')

    parser.add_argument('--env_name', type=str, default="CartPole-v0", help='the name of the environment')

    args = parser.parse_args()
    return args

# 定义策略网络结构
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s):
        s = self.layer(s)
        return F.softmax(s, dim=1)

class REINFORCE:
    # 值得注意的是: 在REINFORCE的算法实践中,我们并不需要使用BUFFER REPLAY,因为REINFORCE是基于MC的,
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)

        self.hidden_dim = 128                                  # Policy网络隐藏层维度
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.lr = args.lr                                      # 学习率
        self.gamma = args.gamma                                # 折扣因子

        self.num_episodes = args.num_episodes                  # 所需要采集的episode数量
        self.policynet = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim)

        self.device = torch.device("cpu")
        self.optimizer = Adam(self.policynet.parameters(), lr=self.lr)

    def take_action(self, state):
        states = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policynet(states)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        return action.item()

    def update(self, transition_dict):
        # 在更新过程中,基于MC方法,在episode里面没来一个step就计算累计奖励,并进行更新
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0            # 初始化当前episode的累计奖励
        self.optimizer.zero_grad()

        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor(action_list[i]).view(-1,1).to(self.device)

            G = self.gamma * G + reward
            probs = self.policynet(state)
            # 下面选出action对应的概率值
            action_probs = probs.gather(1, action)
            log_action_probs = torch.log(action_probs)

            # 计算loss函数,在一个episode的计算中,梯度是需要累加的,在整个episode计算完成之后就可以做梯度下降
            loss = -log_action_probs * G
            loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    args = parse_args()
    agent = REINFORCE(args)

    return_list = []
    for episode in range(1000):
        # 对于每一个episode都会有一个新的字典
        transition_dict = {'states':[], 'actions':[], 'next_states':[], 'rewards':[], 'dones':[]}
        episode_return = 0
        state = agent.env.reset()
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = agent.env.step(action)

            # 将此episode的这一个step加入到字典中
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            # 做状态转移
            state = next_state
            episode_return += reward

            if done:
                print('Episode index: ', episode, '| Episode_return: ', episode_return)
                break

        return_list.append(episode_return)

        # 在收集完一整条episode之后,就可以开始对policynet进行更新
        agent.update(transition_dict)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()


