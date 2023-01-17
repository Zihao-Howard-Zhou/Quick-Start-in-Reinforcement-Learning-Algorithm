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

The following picture shows the convergence of REINFORCE in the Cartpole-v0 environment, since REINFORCE is based on MC, the variance of the estimation is relatively large, which makes the fluctuation of reward more significant.<bar>

![image](https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm/blob/master/Policy-Gradient/Episode_return_REINFORCE.png)

Next is the convergence of A2C in the Cartpole-v0 environment, A2C is difficult to converge in the actual training process and needs to adjust parameters carefully. The reason should be that A2C does not use experience pool, resulting in too high correlation between samples of each training.<bar>

