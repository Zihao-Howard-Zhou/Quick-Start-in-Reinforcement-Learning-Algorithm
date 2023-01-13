import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters setting of A2C')

    parser.add_argument('--actor_lr', type=float, default=1e-3, help='Learning rate for the policy net.')
    parser.add_argument('--critic_lr', type=float, default=2e-3, help='Learning rate for the critic net')
    parser.add_argument('--num_episodes', type=int, default=500, help='the num of train epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--gamma', type=float, default=0.98, help='the discount rate')

    parser.add_argument('--env_name', type=str, default="CartPole-v0", help='the name of the environment')

    args = parser.parse_args()
    return args

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
        return F.softmax(s, dim=1)       # 按行做归一化

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, s):
        q = self.layer(s)
        return q

class A2C:
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)

        self.hidden_dim = 128                                 # Policy网络隐藏层维度
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim)
        self.critic = CriticNet(self.state_dim, self.hidden_dim)

        self.actor_lr = args.actor_lr                         # actor学习率
        self.critic_lr = args.critic_lr                       # critic学习率
        self.gamma = args.gamma                               # 折扣因子

        self.num_episodes = args.num_episodes                 # 所需要采集的episode数量

        self.device = torch.device("cpu")
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def take_action(self, state):
        states = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(states)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        """
        在QAC方法中是对episode中的每一个step都进行网络参数的更新
        QAC是一个on-policy的方法,当前的policynet用于产生下一个step的action
        """
        states = torch.tensor([state], dtype=torch.float).to(self.device)
        actions = torch.tensor(action).view(-1,1).to(self.device)
        rewards = torch.tensor(reward, dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor([next_state], dtype=torch.float).to(self.device)
        dones = torch.tensor(done, dtype=torch.float).view(-1,1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1-dones)
        td_delta = td_target - self.critic(states)                                      # 用于actor网络的更新

        probs = self.actor(states)
        action_probs = probs.gather(1, actions)
        log_action_probs = torch.log(action_probs)

        actor_loss = torch.mean(-log_action_probs * td_delta)

        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

if __name__ == '__main__':
    args = parse_args()
    agent = A2C(args)

    return_list = []
    for episode in range(300):
        episode_return = 0
        state = agent.env.reset()
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _ = agent.env.step(action)

            agent.update(state, action, reward, next_state, done)

            # 做状态转移
            state = next_state
            episode_return += reward

            if done:
                print('Episode index: ', episode, '| Episode_return: ', episode_return)
                break

        return_list.append(episode_return)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()

