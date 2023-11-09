"""
Example application to showcase the NMPC of the nonlinear double integrator using do_mpc.
"""


import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import do_mpc

from template_model import template_model # nonlinear double integrator
from template_mpc import template_mpc # NMPC settings
from template_simulator import template_simulator # simulator settings

# SETUP 
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# INFO
lbx, ubx = np.array(mpc.bounds['lower','_x','x']), np.array(mpc.bounds['upper','_x','x'])
lub, ubu = np.array(mpc.bounds['lower','_u','u']), np.array(mpc.bounds['upper','_u','u'])

# CONFIG
e = np.ones([model.n_x,1])
x0 = np.random.uniform(-4*e,4*e)
u0 = DM(0)

noise_level = 0.3

N_sim = 25

mpc.u0 = u0
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

# MAIN LOOP
for k in range(N_sim):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next) + np.random.normal(0,noise_level,[model.n_x,1])

fig1, axs_1 = plt.subplots(3,1)
axs_1[0].plot(simulator.data['_x','x',0],label='x1')
axs_1[0].hlines(lbx[0],0,N_sim,colors='r',linestyles='dashed',label="lower bound")
axs_1[0].hlines(ubx[0],0,N_sim,colors='r',linestyles='dashed',label="upper bound")
axs_1[0].hlines(0,0,N_sim,colors='k',linestyles='dashed',label="origin",linewidth=0.5)
axs_1[0].set_xlabel("time")

axs_1[1].plot(simulator.data['_x','x',1],label='x2')
axs_1[1].hlines(lbx[1],0,N_sim,colors='r',linestyles='dashed',label="lower bound")
axs_1[1].hlines(ubx[1],0,N_sim,colors='r',linestyles='dashed',label="upper bound")
axs_1[1].hlines(0,0,N_sim,colors='k',linestyles='dashed',label="origin",linewidth=0.5)
axs_1[1].set_xlabel("time")

axs_1[2].plot(simulator.data['_u','u'],label='u')
axs_1[2].hlines(lub,0,N_sim,colors='r',linestyles='dashed',label="lower bound")
axs_1[2].hlines(ubu,0,N_sim,colors='r',linestyles='dashed',label="upper bound")
axs_1[2].hlines(0,0,N_sim,colors='k',linestyles='dashed',label="origin",linewidth=0.5)
axs_1[2].set_xlabel("time")

axs_1[0].legend()
axs_1[1].legend()
axs_1[2].legend()


fig2, axs_2 = plt.subplots(1,1)
axs_2.plot(simulator.data['_x','x',0],simulator.data['_x','x',1],label="trajectory (x1,x2)")
axs_2.plot([0.0],[0.0],'ro',label="origin")
axs_2.set_xlabel("x1")
axs_2.set_ylabel("x2")
axs_2.legend()

fig3, axs_3 = plt.subplots(1,1)
axs_3.plot(simulator.data['_aux','stage_cost'])
axs_3.set_xlabel("time")
axs_3.set_ylabel("stage cost")
plt.show()
input("Press [enter] to continue.")