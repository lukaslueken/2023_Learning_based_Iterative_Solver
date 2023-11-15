# 2023_Learning_based_Iterative_Solver

## Abstract
Model predictive control (MPC) is a powerful control method for handling complex nonlinear systems that are subject to constraints.
However, the real-time application of this approach can be severely limited by the need to solve constrained nonlinear optimization problems at each sampling time.
To this end, this work introduces a novel learning-based iterative solver that provides highly accurate predictions, optimality certification, and fast evaluation of the MPC solution at each sampling time. 
To learn this iterative solver, we propose an unsupervised training algorithm that builds on the Karush-Kahn-Tucker optimality conditions, modified by a Fischer-Burmeister formulation, and eliminates the need for prior sampling of exact optimizer solutions.
By exploiting efficient vector-Jacobian and Jacobian-vector products via automatic differentiation, the proposed training algorithm can be efficiently executed.
We demonstrate the potential of the proposed learning-based iterative solver on the example of nonlinear model predictive control of a nonlinear double integrator. We show its advantages when compared to exact optimizer solutions and with an imitation learning-based approach that directly obtains a data-based approximation of the MPC control law.
