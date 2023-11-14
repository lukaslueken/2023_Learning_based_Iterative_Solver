"""
Date: 2023-11-02
Author: Lukas Lüken

Script to evaluate closed loop application of learned solver.
"""

# %%
# Config
eps = 1e-6

# test data
test_dataset_file_name = "dataset_test_1500.pt"

# solver
approx_solver_folder = "learning_based_solver_model"

# approx mpc
approx_mpc_folder = "approx_mpc_model"

closed_loop_steps = 25
max_iter = 40
n_runs = 1500

eval_ipopt = True
eval_approx_mpc = True

tolerance = 1e-6
noise_level = 0.3

save_runs = True


# %% Imports
import torch
import numpy as np
import subprocess
# import matplotlib.pyplot as plt
# import pandas as pd
from pathlib import Path
from approx_MPC import ApproxMPC, ApproxMPCSettings #, plot_history
import do_mpc
import json
from timeit import default_timer as timer
import pickle as pkl
from nlp_handler import NLPHandler

# Define control problem
from nl_double_int_nmpc.template_model import template_model
from nl_double_int_nmpc.template_mpc import template_mpc
from nl_double_int_nmpc.template_simulator import template_simulator

# %% 
# Functions
####################################################################################
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
batched_dot = torch.func.vmap(torch.dot,chunk_size=None)

def full_step(nlp_handler,solver_nn,z_k_i,p_i,eps_i,offset,rangeF):
    F_k_i = nlp_handler.F_FB_batch_func(z_k_i.numpy().T,p_i.numpy().T,eps_i.numpy().T)
    F_k_i = torch.tensor(np.array(F_k_i).T)
    norm_F_k_i = torch.norm(F_k_i,dim=1)+offset
    
    F_k_i_scaled = torch.divide(F_k_i,norm_F_k_i[:,None])
    
    with torch.no_grad():

        # Stack NN inputs
        nn_inputs_i = torch.hstack((p_i,F_k_i_scaled,torch.log(norm_F_k_i)[:,None]))

        # PREDICT
        dz_k_i = solver_nn(nn_inputs_i) * norm_F_k_i * rangeF

        low_gamma = torch.tensor(0.1)
        high_gamma = torch.tensor(20.0)
        g_k_batch = nlp_handler.g_k_func(z_k_i.numpy().T,p_i.numpy().T,eps_i.numpy().T,dz_k_i.numpy().T)
        g_k_batch = torch.tensor(np.array(g_k_batch).T)
        jvpkTjvpk = batched_dot(g_k_batch,g_k_batch)+offset
        FkTjvpk = batched_dot(F_k_i,g_k_batch)
        gamma_i = - torch.divide(FkTjvpk,jvpkTjvpk)
        gamma_i = torch.clamp(gamma_i,min=low_gamma,max=high_gamma)

        # UPDATE
        z_k_i = z_k_i + gamma_i[:,None]*dz_k_i

    return z_k_i, norm_F_k_i

def load_solver(file_pth,run_folder):
    print("Loading solver...")
    run_folder_load = file_pth.joinpath(run_folder)
    with open(run_folder_load.joinpath("run_hparams.json"),"r") as fp:
        run_hparams_loaded = json.load(fp)

    n_layers = run_hparams_loaded["n_layers"]
    n_neurons = run_hparams_loaded["n_neurons"]
    n_in = run_hparams_loaded["n_in"]
    n_out = run_hparams_loaded["n_out"]
    solver_nn_loaded = generate_ffNN(n_layers,n_neurons,n_in,n_out)

    state_dict_loaded = torch.load(run_folder_load.joinpath("solver_nn_state_dict.pt"))
    solver_nn_loaded.load_state_dict(state_dict_loaded)
    print("Solver loaded.")
    return solver_nn_loaded, run_hparams_loaded

def load_approx_mpc(file_pth,run_folder):
    print("Loading model...")
    folder_pth = file_pth.joinpath(run_folder)
    approx_mpc_settings_loaded = ApproxMPCSettings.from_json(folder_pth,"approx_MPC_settings")
    approx_mpc_loaded = ApproxMPC(approx_mpc_settings_loaded)
    approx_mpc_loaded.load_state_dict(folder_pth,file_name="approx_MPC_state_dict")
    print("Model loaded.")
    return approx_mpc_loaded

# %%
# Setup
seed = 0
torch.manual_seed(seed)
device = torch.device('cpu')
torch_data_type = torch.float64
torch.set_default_dtype(torch_data_type)

# set numpy seed
np.random.seed(seed)

file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)


# %% 
# Initialization
model = template_model()
mpc = template_mpc(model,silence_solver=True)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)
nlp_handler = NLPHandler(mpc)

def gen_rand_z_norm(N,nlp_handler):
    x_k_batch = torch.randn(N,nlp_handler.n_x)
    lam_k_batch = torch.randn(N,nlp_handler.n_g)
    nu_k_batch = torch.randn(N,nlp_handler.n_h)
    return torch.hstack((x_k_batch, lam_k_batch, nu_k_batch))

# load dataset
test_dataset_folder = file_pth.joinpath('datasets')
test_dataset = torch.load(test_dataset_folder.joinpath(test_dataset_file_name),map_location=device)
n_test = test_dataset.tensors[0].shape[0]
assert n_runs <= n_test, "n_runs must be smaller than n_test"

# load solver
solver_nn, solver_run_hparams = load_solver(file_pth,approx_solver_folder)

offset = solver_run_hparams["offset"]
rangeF = solver_run_hparams["rangeF"]
filter_level = solver_run_hparams["filter_level"]

# load approx mpc
approx_mpc = load_approx_mpc(file_pth,approx_mpc_folder)
approx_mpc.set_device(device="cpu")

# %% solver mpc closed loop 

nlp_handler.setup_batch_functions(1,N_threads=1)
u_pos_in_z = nlp_handler.n_x-mpc.settings.n_horizon
rand_indices = np.random.permutation(n_test)[:n_runs]

trajectories = []
# Different Runs
for k in range(n_runs):
    trajectory = {"run":k,
                  "norm_F_FB":[],"norm_KKT":[],
                  "N_iter":[],"u0":[],"x0":[],
                  "u0_ipopt":[],"diff_u0":[],
                  "u0_approx_mpc":[],"diff_u0_approx_mpc":[],
                  "approx_mpc_time":[],"ipopt_time":[],
                  "approx_solver_time":[]}

    # Solver Data
    idx = rand_indices[k]
    p_data = test_dataset.tensors[0][idx,:]
    eps_i = torch.ones(1,1)*eps

    #  MPC Settings
    x0 = p_data.numpy()[0:2].reshape(-1,1)
    u_prev = p_data.numpy()[2].reshape(-1,1)

    trajectory["x0"].append(x0)

    # Closed Loop
    for i in range(closed_loop_steps):
        z_k_i = gen_rand_z_norm(1,nlp_handler)

        mpc.u0 = u_prev
        mpc.x0 = x0
        # Use initial state to set the initial guess.
        mpc.set_initial_guess()
        simulator.x0 = x0
        estimator.x0 = x0

        if eval_ipopt:
            ipopt_start = timer()
            u0_ipopt = mpc.make_step(x0)
            ipopt_end = timer()
            trajectory["ipopt_time"].append(ipopt_end-ipopt_start)

        p_i = torch.tensor(np.vstack((x0,u_prev)).reshape(1,-1))

        if eval_approx_mpc:
            approx_mpc_start = timer()
            u0_approx_mpc = approx_mpc.make_step(p_i,scale_inputs=True,rescale_outputs=True,clip_outputs=False)
            approx_mpc_end = timer()
            trajectory["approx_mpc_time"].append(approx_mpc_end-approx_mpc_start)
            trajectory["u0_approx_mpc"].append(u0_approx_mpc)

        if eval_ipopt and eval_approx_mpc:
            diff_u0_approx_mpc = np.abs(u0_approx_mpc-u0_ipopt)
            trajectory["diff_u0_approx_mpc"].append(diff_u0_approx_mpc)

        # CLOSED LOOP SOLVER
        iteration = 0
        approx_solver_times = []
        for j in range(max_iter):
            iteration += 1            
            # kkt
            kkt_i = np.array(nlp_handler.KKT_batch_func(z_k_i.numpy().T,p_i.numpy().T)).T
            norm_kkt_i = np.linalg.norm(kkt_i,axis=1)

            approx_step_start = timer()
            z_k_i, norm_F_k_i = full_step(nlp_handler,solver_nn,z_k_i,p_i,eps_i,offset,rangeF)
            approx_step_end = timer()

            approx_solver_times.append(approx_step_end-approx_step_start)
            if norm_F_k_i <= tolerance:
                break            
        trajectory["approx_solver_time"].append(np.sum(approx_solver_times))

        # u0
        u0_i = z_k_i[:,u_pos_in_z].detach().clone().numpy()

        # diff_u0
        if eval_ipopt:
            diff_u0_i = np.abs(u0_i-u0_ipopt)
            trajectory["diff_u0"].append(diff_u0_i)
            trajectory["u0_ipopt"].append(u0_ipopt)

        trajectory["norm_F_FB"].append(norm_F_k_i)
        trajectory["norm_KKT"].append(norm_kkt_i)
        trajectory["N_iter"].append(iteration)
        trajectory["u0"].append(u0_i)

        y_next = simulator.make_step(u0_i.reshape(-1,1))
        x0 = estimator.make_step(y_next) + np.random.randn(2,1)*noise_level
        u_prev = u0_i
        trajectory["x0"].append(x0)

    trajectory["x0"] = np.array(trajectory["x0"]).squeeze()
    trajectory["u0"] = np.array(trajectory["u0"]).squeeze()
    trajectory["norm_F_FB"] = np.array(trajectory["norm_F_FB"]).squeeze()
    trajectory["norm_KKT"] = np.array(trajectory["norm_KKT"]).squeeze()
    trajectory["N_iter"] = np.array(trajectory["N_iter"])
    trajectory["u0_ipopt"] = np.array(trajectory["u0_ipopt"]).squeeze()

    if eval_approx_mpc:
        trajectory["u0_approx_mpc"] = np.array(trajectory["u0_approx_mpc"]).squeeze()

    if eval_ipopt:
        trajectory["diff_u0"] = np.array(trajectory["diff_u0"]).squeeze()
    
    if eval_approx_mpc and eval_ipopt:    
        trajectory["diff_u0_approx_mpc"] = np.array(trajectory["diff_u0_approx_mpc"]).squeeze()

    trajectories.append(trajectory)

    print("--------------------")
    print("Run: ",k)
    print("max_norm_F_k_i: ",np.max(trajectory["norm_F_FB"]))
    print("max_N_iter: ",np.max(trajectory["N_iter"]))
    print("median_norm_F_k_i: ",np.median(trajectory["norm_F_FB"]))
    print("median_N_iter: ",np.median(trajectory["N_iter"]))

# %%
# save settings and trajectories

if save_runs:
    # get git commit of file
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    # get dict with all settings of this script
    closed_loop_run_dict = {"eps":eps,
                            "tolerance":tolerance,
                            "noise_level":noise_level,
                            "n_runs":n_runs,
                            "n_test":n_test,
                            "closed_loop_steps":closed_loop_steps,
                            "max_iter":max_iter,
                            "eval_ipopt":eval_ipopt,
                            "eval_approx_mpc":eval_approx_mpc,
                            "approx_solver_folder":approx_solver_folder,
                            "approx_mpc_folder":approx_mpc_folder,
                            "test_dataset_file_name":test_dataset_file_name,
                            "seed":seed,
                            "offset":offset,
                            "rangeF":rangeF,
                            "filter_level":filter_level,
                            "git_commit":git_commit}
    
    # save trajectories
    for i in range(100):
        run_folder = file_pth.joinpath("results_export","closed_loop",f"run_{i}")
        if run_folder.exists():
            continue
        else:
            run_folder.mkdir()
            with open(run_folder.joinpath("run_dict.json"),"w") as fp:
                json.dump(closed_loop_run_dict,fp,indent=4)
            with open(run_folder.joinpath("trajectories.pkl"),"wb") as fp:
                pkl.dump(trajectories,fp)
            break

# %%
