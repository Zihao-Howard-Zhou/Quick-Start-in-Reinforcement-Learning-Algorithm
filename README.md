# Quick Start in Reinforcement Learning Algorithm
![](https://img.shields.io/badge/Github-%40ZihaoZhouSCUT-informational) ![](https://img.shields.io/badge/Category-Reinforcement%20Learning-black) ![](https://img.shields.io/badge/License-MIT-green)
## Introduction
This repository is a reproduction of the reinforcement learning algorithm I have learned. It is convenient to quickly understand the implementation and principle of the algorithm in combination with theory. Due to the difficulty in obtaining the system model in practice, the algorithms to be implemented in this repository mainly focus on model-free algorithms, including value-based methods, policy-based methods and actor-critic methods. <bar>
![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Algorithm%20classification.png)

## Install
This project is based on Pytorch 1.2.0

## Results
The following figure shows the convergence of DQN in the Cartpole-v0 environment, it can be found that after 400 episodes, reward has converged to the maximum of 200, indicating that DQN has achieved good results in this task.<bar>
![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Deep-Q-Network/Episode_return_DQN.png)

The following picture shows the convergence of REINFORCE in the Cartpole-v0 environment, since REINFORCE is based on MC, the variance of the estimation is relatively large, which makes the fluctuation of reward more significant.In addition, we note that the episode required for the reward convergence of REINFORCE algorithm to the highest score is much more than that of DQN (1000 vs 400), because REINFORCE will update the network parameters only after an episode is completed.<bar>

![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Policy-Gradient/Episode_return_REINFORCE.png)

Next is the convergence of A2C in the Cartpole-v0 environment. During the experiment, it was found that the effect of updating the network with all the data after one episode was better than training with separate sample after each action, and the convergence effect was also more stable. At the same time, we find that compared with REINFORCE algorithm, the reward jitter of A2C is smaller, indicating that the introduction of baseline reduces the variance of the algorithm.<bar>

![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Advantage%20Actor%20Critic/New_Episode_return_A2C.png)

The episode return of PPO-Clip is shown in the next picture. It can be seen that the jitter of return is significantly lower than that of A2C.
![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Proximal%20Policy%20Optimization/Episode_return_PPO_v1.png)
