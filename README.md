# 2023_Learning_based_Iterative_Solver

## Abstract
Model predictive control (MPC) is a powerful control method for handling complex nonlinear systems that are subject to constraints.
However, the real-time application of this approach can be severely limited by the need to solve constrained nonlinear optimization problems at each sampling time.
To this end, this work introduces a novel learning-based iterative solver that provides highly accurate predictions, optimality certification, and fast evaluation of the MPC solution at each sampling time. 
To learn this iterative solver, we propose an unsupervised training algorithm that builds on the Karush-Kahn-Tucker optimality conditions, modified by a Fischer-Burmeister formulation, and eliminates the need for prior sampling of exact optimizer solutions.
By exploiting efficient vector-Jacobian and Jacobian-vector products via automatic differentiation, the proposed training algorithm can be efficiently executed.
We demonstrate the potential of the proposed learning-based iterative solver on the example of nonlinear model predictive control of a nonlinear double integrator. We show its advantages when compared to exact optimizer solutions and with an imitation learning-based approach that directly obtains a data-based approximation of the MPC control law.

## Introduction
Dear reader,

welcome to this repository. Here you will find the code for our work "Learning iterative solvers for accurate and fast nonlinear model predictive control via unsupervised training", which has recently be submitted for review.
Please don't hesitate to write us a message, if you have any questions.
To better understand the structure of this repository and our code please read the overview below.

## Structure

**Model predictive control of nonlinear double integrator** 
Folder: nl_double_int_nmpc
Files:
1. main.py
2. template_model.py
3. template_mpc.py
4. template_simulator.py

**Data Sampling**
Files:
1. generate_sampling_plans.py
2. data_sampling.py
3. data_processing.py

Folders:
1. datasets
   Here, the datasets used for generating the results in the paper are stored.
2. sampling
   Folder, in which individual data samples and sampling plans are stored when running the sampling script

**Learning-based iterative solver**

**Approximate MPC**

**Open-loop evaluation**

**Closed-loop evaluation**


## Note regarding use of Python>=3.11
- The sampling feature in do-mpc uses "inspect.getargspec", which has been removed from Python>=3.11
- The full functionality can be recovered by instead using "inspect.getfullargspec".
- This will be included in the next do-mpc release (version 4.6.3), which will be available in the next few days.
- To circumvent this in the meantime, when using Python>=3.11 and the current do-mpc version (4.6.2) you can replace "inspect.getargspec" with "inspect.getfullargspec" in "do_mpc/sampling/_sampler.py"
