[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This repository shows the reinforcement learning approaches appied in the first project ``Navigation'' (Value-Based Methods) as 
part of the udacity nanodegree: Deep Reinforcement Learning. The approaches used during this project are based 
on a Deep Q-Network (DQN). 
During the project I tested and compared different variation and additions to the basic DQN e.g. Double DQN, Dueling DQN, priority replay buffers and frame skipping.

The trained agent has the task to navigate a square environment, pick up yellow bananas and avoid blue bananas.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Install
This project requires **Python 3.5** and the following libaries:

- [NumPy](http://www.numpy.org/)
- [Torch](https://pytorch.org)
- [UnityAgents](https://github.com/Unity-Technologies/ml-agents)
- [OpenAI Gym](https://gym.openai.com)

### Instructions

Navigate to the `p1-navigation/` directory and start jupyter notebook:

```shell
$ ipython3 notebook
```
Follow the instructions in `Navigation.ipynb` to train your own agent or test best performing weights achieved during this project.

### Report

Read a detailed report, describing appied approaches and achieved results [here](https://github.com/bbroecker/p1-navigation/blob/master/Report.pdf) 


![Result](https://github.com/bbroecker/p1-navigation/blob/master/figures/network_types.png)
