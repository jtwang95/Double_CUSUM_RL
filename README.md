# A Robust Test for the Stationarity Assumption in Sequential Decision Making

This repository contains the implementation for the paper "A Robust Test for the Stationarity Assumption in Sequential Decision Making" (ICML 2023) in Python.

## Abstract of the paper

Reinforcement learning (RL) is a powerful technique that allows an autonomous agent to learn an optimal policy to maximize the expected return. The optimality of various RL algorithms relies on the stationarity assumption, which requires time-invariant state transition and reward functions. However, deviations from stationarity over extended periods often occur in real-world applications like robotics control, health care and digital marketing, resulting in sub-optimal policies learned under stationary assumptions. In this paper, we propose a model-based doubly robust procedure for testing the stationarity assumption and detecting change points in offline RL settings with certain degree of homogeneity. Our proposed testing procedure is robust to model misspecifications and can effectively control type-I error while achieving high statistical power, especially in high-dimensional settings. Extensive comparative simulations and a real-world interventional mobile health example illustrate the advantages of our method in detecting change points and optimizing long-term rewards in high-dimensional, non-stationary environments.

## Requirements
Run `conda env create -f environment.yml` to create the conda environment and install the required packages. 

## File overview
+ `./core` contains all main functions for the proposed test.
    + `calculate_test_statistic_discrete.py` contains functions for the proposed test in tabular case.
    + `calculate_test_statistic.py` contains functions for the proposed test for continuous state space and binary action.
+ `./grid_world` contains codes for reproducing results of Section 4.4 in the original paper.
    + __NOTE__: Please run `python setup.py install` under the folder `./grid_world/gym-examples` to install the simulation environment for the grid world before running any script. 
    + To reproduce the result in the paper, run `python s05_batch_run_all.py`.
+ `./simulation` contains codes for reproducing results of Section 4.3 in the original paper.
    + To reproduce the result in the paper, run `python s04_batch_run_all.py`. ==Important==: Remember to adjust the parameter __CORES__ before running the code. Set the value to be the number of CPU cores you want to use. 
    + To generate the results for different setting, set __TYPE__ to be one of ["pwc2ada_state","pwc2ada_reward"] and __SDIM__ to one of [1,10,20,30].
+ `./toy_example` contains codes for reproducing results of Section 4.2 in the original paper.
+ `./simple_demo.py` contains two simple demo examples for continuous case and discrete respectively.

## License
All content in this repository is licensed under the MIT license.