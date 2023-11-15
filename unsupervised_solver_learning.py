"""
Date: 2023-11-10
Author: Lukas Lüken

VJP/JVP-based Solver Learning NMPC.
"""
# %% 
# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from nlp_handler import NLPHandler

# Define control problem
from nl_double_int_nmpc.template_model import template_model
from nl_double_int_nmpc.template_mpc import template_mpc

# %%
# Setup
seed = 0
torch.manual_seed(seed)
device = torch.device('cpu')
# set default device
torch.set_default_device(device)

torch_data_type = torch.float64
torch.set_default_dtype(torch_data_type)   
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

# %% 
# Config

# MPC
eps = 1e-6

# NN
n_layers = 1 # number of hidden layers
n_neurons = 2000 # number of neurons per hidden layer

# NN Training
N_steps = 200 # number of solver forward steps per epoch
N_epochs = 100 # number of epochs
batch_size = 1024
lrs = [1e-3] # learning rates for learning rate scheduling: only one is used


# Numerics
offset = 1e-16 # offset for numeerical reasons
rangeF = 0.1 # direct scaling of output of NN (not necessary, interval in adaptive scaling of gamma can be adapted accourdingly)
filter_level = 1e3 # upper bound for scaling factor of output (not necessary for example presented in paper, for badly scaled NLPs this might improve training stability)

# %% 
# Functions
def generate_ffNN(n_layers,n_neurons,n_in,n_out):
    layers = []
    for i in range(n_layers):
        if i == 0:
            layers.append(torch.nn.Linear(n_in,n_neurons))
        else:
            layers.append(torch.nn.Linear(n_neurons,n_neurons))
        layers.append(torch.nn.ReLU())
    ann = torch.nn.Sequential(*layers,torch.nn.Linear(n_neurons,n_out))
    return ann

# batched version of dot product (pytorch) 
batched_dot = torch.func.vmap(torch.dot,chunk_size=None)

# full solver step (with calculation of scaling factor gamma)
def full_step(nlp_handler,solver_nn,z_k_i,p_i,eps_i,offset,rangeF):
    F_k_i = nlp_handler.F_FB_batch_func(z_k_i.numpy().T,p_i.numpy().T,eps_i.numpy().T)
    F_k_i = torch.tensor(np.array(F_k_i).T)
    norm_F_k_i = torch.norm(F_k_i,dim=1)+offset
    
    F_k_i_scaled = torch.divide(F_k_i,norm_F_k_i[:,None])
    
    with torch.no_grad():
        # Stack NN inputs
        nn_inputs_i = torch.hstack((p_i,F_k_i_scaled,torch.log(norm_F_k_i)[:,None]))

        # PREDICT
        dz_k_i = solver_nn(nn_inputs_i) * norm_F_k_i[:,None] * rangeF

        # Scaling Factor gamma
        low_gamma = torch.tensor(0.1)
        high_gamma = torch.tensor(20.0)
        g_k_batch = nlp_handler.g_k_func(z_k_i.numpy().T,p_i.numpy().T,eps_i.numpy().T,dz_k_i.numpy().T) # JVP of Gradient of F_FB and Step
        g_k_batch = torch.tensor(np.array(g_k_batch).T)
        jvpkTjvpk = batched_dot(g_k_batch,g_k_batch)+offset # dot product of JVP of Gradient of F_FB and Step with itself
        FkTjvpk = batched_dot(F_k_i,g_k_batch)
        gamma_i = - torch.divide(FkTjvpk,jvpkTjvpk)
        gamma_i = torch.clamp(gamma_i,min=low_gamma,max=high_gamma)

        # UPDATE
        z_k_i = z_k_i + gamma_i[:,None]*dz_k_i

    return z_k_i, norm_F_k_i

# Parameter Sampling
def gen_rand_p(N):
    x_0_batch = torch.rand(N,2)*20-10
    u_prev_batch = torch.rand(N,1)*4-2    
    return torch.hstack((x_0_batch, u_prev_batch))

def gen_rand_z_norm(N,nlp_handler):
    x_k_batch = torch.randn(N,nlp_handler.n_x)
    lam_k_batch = torch.randn(N,nlp_handler.n_g)
    nu_k_batch = torch.randn(N,nlp_handler.n_h)
    return torch.hstack((x_k_batch, lam_k_batch, nu_k_batch))


# %% 
# Initialization
model = template_model()
mpc = template_mpc(model)
nlp_handler = NLPHandler(mpc)

# %% NN
n_in = nlp_handler.n_p + nlp_handler.n_z + 1 # (p, F_scaled, F_norm)
n_out = nlp_handler.n_z # (dz_scaled)

# Setup NN
solver_nn = generate_ffNN(n_layers,n_neurons,n_in,n_out)
print("Neural network initialized with architecture: ", solver_nn)
# print number of nn parameters
n_params = sum(p.numel() for p in solver_nn.parameters() if p.requires_grad)
print("---------------------------------")
print("Number of trainable parameters: ", n_params)
print("---------------------------------")

# %% 
# Training

append_solver_steps = True

# batched version of casadi functions (for speedup) - evaluating casadi function on N_threads in parallel and returns data in batches
nlp_handler.setup_batch_functions(batch_size,N_threads=8)

mse = torch.nn.MSELoss() # not necessary, used for training option 2

history = {"loss": [], "eta": [], "lr": [], "epoch": [], "step": [],"step_loss": [], "step_eta": []}
# LOOP - LEARNING RATES
idx_lr= 0
###
for lr in lrs:
    optim = torch.optim.AdamW(solver_nn.parameters(),lr=lr) # adamW optimizer: optimizer Adam with decoupled weight decay regularization
    
    for epoch in range(N_epochs):
        # init training epoch 
        p_batch = gen_rand_p(batch_size) # sample random parameters
        z_batch = gen_rand_z_norm(batch_size,nlp_handler) # sample random z_0 (initial guesses of primal-dual solution)
        eps_batch = torch.ones(batch_size,1)*eps # set epsilon for all samples in batch (code implemented such that eps can be adapted on the fly, not necessary though as shown in paper, since solver converges even if eps is very small from the start)
        epoch_loss = 0.0
        epoch_eta = 0.0
        for step in range(N_steps):

            # calc F_k_batch
            F_k_batch = nlp_handler.F_FB_batch_func(z_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
            F_k_batch = torch.tensor(np.array(F_k_batch).T)

            # norm_F_k_batch
            norm_F_k_batch = torch.norm(F_k_batch,dim=1)

            # scaled F_k_batch
            F_k_batch_scaled = torch.divide(F_k_batch,norm_F_k_batch[:,None]+offset)

            # Stack NN inputs
            nn_inputs_batch = torch.hstack((p_batch,F_k_batch_scaled,torch.log(norm_F_k_batch+offset)[:,None]))

            # Upper Threshold Filter 
            norm_F_k_batch_filtered = norm_F_k_batch.detach().clone()[:,None]
            norm_F_k_batch_filtered[(norm_F_k_batch_filtered>=filter_level)] = filter_level # this is actually not necessary for the nonlinear double integrator case study. Might be necessary for badly scaled NLPs
            ####

            # Zero Grads
            optim.zero_grad(set_to_none=True)

            # PREDICT
            dz_hat_batch = solver_nn(nn_inputs_batch) * norm_F_k_batch_filtered * rangeF


            # Scaling Factor gamma (important note: in paper we use 0.01 and 2.0 as interval of gamma, this is because we omit rangeF in the paper. If rangeF is used, the interval of gamma can be adapted accordingly)
            dz_batch = dz_hat_batch.detach().clone() # important to detach here
            low_gamma = torch.tensor(0.1) # lower bound on gamma to mitigate taking no steps or negative steps 
            high_gamma = torch.tensor(20.0) # upper bound on gamma to mitigate taking too large steps
            g_k_batch = nlp_handler.g_k_func(z_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T,dz_batch.numpy().T)
            g_k_batch = torch.tensor(np.array(g_k_batch).T)
            jvpkTjvpk = batched_dot(g_k_batch,g_k_batch)+offset
            FkTjvpk = batched_dot(F_k_batch,g_k_batch)
            gamma_batch = - torch.divide(FkTjvpk,jvpkTjvpk)
            gamma_batch = torch.clamp(gamma_batch,min=low_gamma,max=high_gamma)
            dz_hat_batch = gamma_batch[:,None]*dz_hat_batch

            # LOSS
            V_tilde_batch = nlp_handler.V_tilde_batch_func(z_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T,dz_hat_batch.detach().clone().numpy().T)
            V_tilde_batch = torch.tensor(np.array(V_tilde_batch)).squeeze()

            d_k_batch = nlp_handler.d_k_batch_func(z_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T,dz_hat_batch.detach().clone().numpy().T)
            d_k_batch = torch.tensor(np.array(d_k_batch).T)
            
            loss_batch = V_tilde_batch + batched_dot(d_k_batch,dz_hat_batch - dz_hat_batch.detach().clone()) # implementation "trick" to avoid having to manually add gradients to optimizer and use pytorchs in-built functionality instead. The derivative of this expression is exactly as presented in the paper.
            
            eta_batch = torch.divide(loss_batch,0.5*batched_dot(F_k_batch,F_k_batch)) # "eta": measure of improvement of z_k+1 over z_k (if eta)

            # scale and mean loss 
            log_loss_batch = torch.log10(loss_batch+offset)
            loss = torch.mean(log_loss_batch) # Option 1: as described in paper
            # loss = mse(log_loss_batch,torch.tensor(-16.0)) # Option 2: needs a little less epochs for training; compared to option 1: effectivly scales loss gradient linearly with distance of log10-loss to -16 

            # eta
            eta = torch.mean(eta_batch) # as long as mean of eta is <1 solver is improving the iterates on average

            # Backprop
            loss.backward()

            epoch_loss += loss.item()
            epoch_eta += eta.item()
            
            # update z_k = z_k + dz_k --> initial guess of next step
            if append_solver_steps:
                dz_batch = dz_hat_batch.detach().clone()
                z_batch = z_batch + dz_batch

            # Update NN Params
            optim.step()

            # history
            history["step_loss"].append(loss.item())
            history["step_eta"].append(eta.item())
            history["lr"].append(lr)
            history["step"].append(step)
        
        # end of epoch
        epoch_loss = epoch_loss/N_steps
        epoch_eta = epoch_eta/N_steps
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["eta"].append(epoch_eta)

        # Print
        if epoch % 1 == 0:
            print("Epoch: ",epoch)
            print("Loss: ",epoch_loss)
            print("eta: ",epoch_eta)
            print("-------------------------------")

print("#----------------------------------#")
print("Training complete.")
print("#----------------------------------#")

# %%
# Visualization
fig, axs = plt.subplots(2,1)
axs[0].plot(history["loss"])
axs[0].legend(["loss"])
axs[1].plot(history["eta"])
axs[1].legend(["eta"])
axs[1].plot([0,len(history["eta"])],[1,1],"r-")
axs[1].plot([0,len(history["eta"])],[0.1,0.1],"r--")
axs[1].set_yscale("log")

# %%
# closed loop solver application to showcase convergence 
# Important to conclude the results: about over 43 % of all possible values for p lead to infeasible NLPs, therefore not improving the norm of optimality conditions

max_iter = 100
solver_batch_size = 64
nlp_handler.setup_batch_functions(solver_batch_size,N_threads=8)

p_batch = gen_rand_p(solver_batch_size)
z_k_batch = gen_rand_z_norm(solver_batch_size,nlp_handler)
eps_batch = torch.ones(solver_batch_size,1)*eps

solver_history = {"mae_k_batch": [],"mae_k_max": [], "mae_k": [], "mae_k_10perc": [], "mae_k_50perc": [], "mae_k_90perc": [], "mae_k_95perc": []}
solver_history["alpha_mean"] = []
solver_history["alpha_max"] = []

# apply solver batch-wise
for i in range(max_iter):
    # calc F_k_batch
    F_k_batch = nlp_handler.F_FB_batch_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
    F_k_batch = torch.tensor(np.array(F_k_batch).T)
    norm_F_k_batch = torch.norm(F_k_batch,dim=1)
    
    # logging
    mae_k_batch = torch.mean(torch.abs(F_k_batch),dim=1)
    mae_k_95perc = torch.quantile(mae_k_batch,0.95)
    mae_k_90perc = torch.quantile(mae_k_batch,0.9)
    mae_k_50perc = torch.quantile(mae_k_batch,0.5)
    mae_k_10perc = torch.quantile(mae_k_batch,0.1)
    mae_k_max = torch.max(mae_k_batch)
    mae_k = torch.mean(mae_k_batch)
    solver_history["mae_k_batch"].append(mae_k_batch.cpu())
    solver_history["mae_k_95perc"].append(mae_k_95perc.cpu())
    solver_history["mae_k_90perc"].append(mae_k_90perc.cpu())
    solver_history["mae_k_50perc"].append(mae_k_50perc.cpu())
    solver_history["mae_k_10perc"].append(mae_k_10perc.cpu())
    solver_history["mae_k_max"].append(mae_k_max.cpu())
    solver_history["mae_k"].append(mae_k.cpu())
    
    z_k_batch, _ = full_step(nlp_handler,solver_nn,z_k_batch,p_batch,eps_batch,offset,rangeF)

# 4.2 Plot Solver History
def plot_solver_history(solver_history,ylim=[1e-16, 1e2]):
    fig, ax = plt.subplots()
    ax.plot(solver_history["mae_k_95perc"])
    ax.plot(solver_history["mae_k_90perc"])
    ax.plot(solver_history["mae_k_50perc"])
    ax.plot(solver_history["mae_k_10perc"])
    ax.plot(solver_history["mae_k_max"])
    ax.plot(solver_history["mae_k"])
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("MAE")
    ax.set_title("mean KKT FB")
    ax.legend(["95%","90%","50%","10%", "max", "mean"])
    fig.show()

# plot solver history
plot_solver_history(solver_history)

# %% visualize individual solver trajectories
idx_list = torch.randperm(solver_history["mae_k_batch"][0].shape[0]).tolist() # legacy/ can be used if very large amount of solver trajectories are generated to make visualization more clear
n_traj = solver_batch_size
fig, ax = plt.subplots()
for j in range(n_traj):
    idx = idx_list[j]
    plt_list = []
    for i in range(max_iter):
        plt_list.append(solver_history["mae_k_batch"][i][idx])
    ax.plot(plt_list)
ax.set_yscale("log")
ax.set_title("mean KKT FB of random solver trajectories")
ax.set_xlabel("Iterations")
ax.set_ylabel("MAE")

# %% 
# Success Rate
mae_last = []
for j in range(n_traj):
    idx = idx_list[j]
    mae_last.append(solver_history["mae_k_batch"][max_iter-1][idx].item())
mae_last = np.array(mae_last)

for lvl in [1e-4,1e-5,1e-6,1e-7]:
    print(f"Success rate for lvl {lvl}: {sum(mae_last<lvl)/mae_last.shape[0]}")


# %% Save run
save_model = True
if save_model:
    print("Saving run...")
    run_hparams = {"n_layers": n_layers, "n_neurons": n_neurons, "n_in": n_in,
                    "n_out": n_out, "N_epochs": N_epochs, "batch_size": batch_size,	
                    "lrs": lrs, "N_steps": N_steps,
                    "optimizer": str(optim.__class__.__name__),
                    "eps": eps,
                    "offset": offset, "rangeF": rangeF, "filter_level": filter_level,
                    "n_z": nlp_handler.n_z, "n_x": nlp_handler.n_x, "n_g": nlp_handler.n_g,
                    "n_h": nlp_handler.n_h, "n_p": nlp_handler.n_p}

    for i in range(100):
        run_folder = file_pth.joinpath("learning_based_solver_model",f"run_{i}")
        if run_folder.exists():
            continue
        else:
            run_folder.mkdir()
            # save model dict
            torch.save(solver_nn.state_dict(), run_folder.joinpath("solver_nn_state_dict.pt"))
            fig.savefig(run_folder.joinpath("run_trajectories.png"))
            # save run_hparams as json
            with open(run_folder.joinpath("run_hparams.json"), 'w') as fp:
                json.dump(run_hparams, fp)
            break


# %%
# Load model
load_model = True
if load_model:
    print("Loading model...")
    run_folder_load = file_pth.joinpath("learning_based_solver_model",f"run_{i}")
    with open(run_folder_load.joinpath("run_hparams.json"),"r") as fp:
        run_hparams_loaded = json.load(fp)

    n_layers = run_hparams_loaded["n_layers"]
    n_neurons = run_hparams_loaded["n_neurons"]
    n_in = run_hparams_loaded["n_in"]
    n_out = run_hparams_loaded["n_out"]
    solver_nn_loaded = generate_ffNN(n_layers,n_neurons,n_in,n_out)

    state_dict_loaded = torch.load(run_folder_load.joinpath("solver_nn_state_dict.pt"))
    solver_nn_loaded.load_state_dict(state_dict_loaded)

# %%
