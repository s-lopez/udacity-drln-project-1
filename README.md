# Udacity's Deep Reinforcement Learning Nanodegree: Navigation project

This repository contains my solution to the first project in [Udacity's DRL nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## The task

![Navigation](https://github.com/SIakovlev/Navigation/blob/master/results/navigation_short.gif)

Meet the Bananator! An agent that was trained with a [Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to collect yellow bananas and avoid blue bananas. The environment is a modified version of [Unity ML-agents' Banana Collector](https://github.com/Unity-Technologies/ml-agents) environment.

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. The agent will learn to choose the approppriate actions at each time step which will lead to the maximum cumulative reward.

### State and action spaces, goal

- **State space** is `37` dimensional and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

- **Action space** is `4` dimentional. Four discrete actions correspond to:
  - `0` - move forward
  - `1` - move backward
  - `2` - move left
  - `3` - move right


- **Solution criteria**: the environment is considered as solved when the agent gets an average score of **+13 over 100 consecutive episodes**.

## Set-up

To run this code in your own machine, please follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) and [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

Note: To develop in my machine, I used an updated version of Pytorch (`1.3.1`). You can reproduce the conda environment exactly following the instructions in `requirements-conda.txt`

## How to run

- **Report.ipynb** contains a detailed description of the implementation and allows you to visualize the performance of a trained agent.
- Running **main.ipynb** trains the agent from scratch
- The parameters needed to clone the trained agent can be found in  models/. Refer to the report for more details.
- The agent is defined in dqn_agent.py
- The actual DQN network is defined in model.py