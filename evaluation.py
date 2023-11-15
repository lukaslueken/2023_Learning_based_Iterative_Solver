"""
Date: 2023-10-21
Author: Lukas Lüken

Pytorch script to evaluate data from case studies.
"""
# %%
# Config
eps = 1e-6

test_dataset_file_name = "dataset_test_1500.pt"
eval_test_dataset = True

# approx MPC to evaluate
approx_mpc_folder = "approx_mpc_model"
eval_approx_mpc = True

# solver
approx_solver_folder = "learning_based_solver_model"
eval_approx_solver = True
eval_approx_solver_times = True
visualize_solver_history = True
visualize_bool = eval_approx_solver and visualize_solver_history

max_iter = 1000

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from approx_MPC import ApproxMPC, ApproxMPCSettings
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

def calc_gamma(nlp_handler,F_k_batch,z_k_batch,p_batch,eps_batch,dz_k_batch,offset):
    low_gamma = torch.tensor(0.1)
    high_gamma = torch.tensor(20.0)
    g_k_batch = nlp_handler.g_k_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T,dz_k_batch.numpy().T)
    g_k_batch = torch.tensor(np.array(g_k_batch).T)
    jvpkTjvpk = batched_dot(g_k_batch,g_k_batch)
    FkTjvpk = batched_dot(F_k_batch,g_k_batch)
    gamma_batch = - torch.divide(FkTjvpk,jvpkTjvpk+offset)
    gamma_batch = torch.clamp(gamma_batch,min=low_gamma,max=high_gamma)
    return gamma_batch

# Solver
def solver_step(nlp_handler,solver_nn,z_k_batch,p_batch,eps_batch,offset,rangeF,filter_level):
    # calc F_k_batch
    F_k_batch = nlp_handler.F_FB_batch_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
    F_k_batch = torch.tensor(np.array(F_k_batch).T)
    
    with torch.no_grad():
        # norm_F_k_batch
        norm_F_k_batch = torch.norm(F_k_batch,dim=1)

        # scaled F_k_batch
        F_k_batch_scaled = torch.divide(F_k_batch,norm_F_k_batch[:,None]+offset)

        # Stack NN inputs
        nn_inputs_batch = torch.hstack((p_batch,F_k_batch_scaled,torch.log(norm_F_k_batch+offset)[:,None]))

        # Upper Threshold Filter
        norm_F_k_batch_filtered = norm_F_k_batch.detach().clone()[:,None]
        norm_F_k_batch_filtered[(norm_F_k_batch_filtered>=filter_level)] = filter_level
        ####

        # PREDICT
        dz_k_batch = solver_nn(nn_inputs_batch) * norm_F_k_batch_filtered * rangeF

        # check for nan
        is_nan = torch.isnan(dz_k_batch).any(dim=1)
        dz_k_batch[is_nan,:] = 0.0

        return dz_k_batch
    
# Solver
def solver_step_time_tracking(nlp_handler,solver_nn,z_k_batch,p_batch,eps_batch,offset,rangeF,filter_level):
    # calc F_k_batch
    F_k_batch = nlp_handler.F_FB_batch_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
    F_k_batch = torch.tensor(np.array(F_k_batch).T)
    
    with torch.no_grad():
        # norm_F_k_batch
        norm_F_k_batch = torch.norm(F_k_batch,dim=1)

        # scaled F_k_batch
        F_k_batch_scaled = torch.divide(F_k_batch,norm_F_k_batch[:,None]+offset)

        # Stack NN inputs
        nn_inputs_batch = torch.hstack((p_batch,F_k_batch_scaled,torch.log(norm_F_k_batch+offset)[:,None]))

        # Upper Threshold Filter
        norm_F_k_batch_filtered = norm_F_k_batch.detach().clone()[:,None]
        norm_F_k_batch_filtered[(norm_F_k_batch_filtered>=filter_level)] = filter_level
        ####

        # PREDICT
        start = timer()
        dz_k_scaled_batch = solver_nn(nn_inputs_batch)
        end = timer()
        dz_k_batch = dz_k_scaled_batch * norm_F_k_batch_filtered * rangeF
        # d_zk_batch = torch.rand_like(d_zk_batch)*1e-1

        # check for nan
        is_nan = torch.isnan(dz_k_batch).any(dim=1)
        dz_k_batch[is_nan,:] = 0.0

        return dz_k_batch, end-start

####################################################################################
def load_approx_mpc(file_pth,approx_mpc_folder):
    print("Loading model...")
    folder_pth = file_pth.joinpath(approx_mpc_folder)
    approx_mpc_settings_loaded = ApproxMPCSettings.from_json(folder_pth,"approx_MPC_settings")
    approx_mpc_loaded = ApproxMPC(approx_mpc_settings_loaded)
    approx_mpc_loaded.load_state_dict(folder_pth,file_name="approx_MPC_state_dict")
    print("Model loaded.")
    return approx_mpc_loaded

def load_solver(file_pth,approx_solver_folder):
    print("Loading solver...")
    folder_pth = file_pth.joinpath(approx_solver_folder)
    with open(folder_pth.joinpath("run_hparams.json"),"r") as fp:
        run_hparams_loaded = json.load(fp)

    n_layers = run_hparams_loaded["n_layers"]
    n_neurons = run_hparams_loaded["n_neurons"]
    n_in = run_hparams_loaded["n_in"]
    n_out = run_hparams_loaded["n_out"]
    solver_nn_loaded = generate_ffNN(n_layers,n_neurons,n_in,n_out)

    state_dict_loaded = torch.load(folder_pth.joinpath("solver_nn_state_dict.pt"))
    solver_nn_loaded.load_state_dict(state_dict_loaded)
    print("Solver loaded.")
    return solver_nn_loaded, run_hparams_loaded

# %%
# Setup
seed = 0
torch.manual_seed(seed)
# np.random.seed(seed)
device = torch.device('cpu')
torch_data_type = torch.float64
torch.set_default_dtype(torch_data_type)   
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)


# %% 
# Initialization
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)
nlp_handler = NLPHandler(mpc)

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

# %% load dataset
test_dataset_folder = file_pth.joinpath('datasets')
test_dataset = torch.load(test_dataset_folder.joinpath(test_dataset_file_name),map_location=device)
n_test = test_dataset.tensors[0].shape[0]

# %% 
# eval test dataset
if eval_test_dataset:
    z_num = test_dataset.tensors[2].numpy().T
    p_num = test_dataset.tensors[3].numpy().T
    eps_num = np.ones((n_test,1))*eps

    nlp_handler.setup_batch_functions(n_test,N_threads=8)
    kkt_batch = np.array(nlp_handler.KKT_batch_func(z_num,p_num)).T
    F_FB_batch = np.array(nlp_handler.F_FB_batch_func(z_num,p_num,eps_num)).T

    norm_kkt_batch = np.linalg.norm(kkt_batch,axis=1)
    norm_F_FB_batch = np.linalg.norm(F_FB_batch,axis=1)

    norm_kkt_max = np.max(norm_kkt_batch)
    norm_kkt_min = np.min(norm_kkt_batch)
    norm_kkt_mean = np.mean(norm_kkt_batch)
    norm_kkt_std = np.std(norm_kkt_batch)
    frac_norm_kkt_1e2 = np.sum(norm_kkt_batch<=1e-2)/n_test
    frac_norm_kkt_1e3 = np.sum(norm_kkt_batch<=1e-3)/n_test
    frac_norm_kkt_1e4 = np.sum(norm_kkt_batch<=1e-4)/n_test
    frac_norm_kkt_1e5 = np.sum(norm_kkt_batch<=1e-5)/n_test
    frac_norm_kkt_1e6 = np.sum(norm_kkt_batch<=1e-6)/n_test
    frac_norm_kkt_1e7 = np.sum(norm_kkt_batch<=1e-7)/n_test
    frac_norm_kkt_1e8 = np.sum(norm_kkt_batch<=1e-8)/n_test

    norm_F_FB_max = np.max(norm_F_FB_batch)
    norm_F_FB_min = np.min(norm_F_FB_batch)
    norm_F_FB_mean = np.mean(norm_F_FB_batch)
    norm_F_FB_std = np.std(norm_F_FB_batch)
    frac_norm_F_FB_1e2 = np.sum(norm_F_FB_batch<=1e-2)/n_test
    frac_norm_F_FB_1e3 = np.sum(norm_F_FB_batch<=1e-3)/n_test
    frac_norm_F_FB_1e4 = np.sum(norm_F_FB_batch<=1e-4)/n_test
    frac_norm_F_FB_1e5 = np.sum(norm_F_FB_batch<=1e-5)/n_test
    frac_norm_F_FB_1e6 = np.sum(norm_F_FB_batch<=1e-6)/n_test
    frac_norm_F_FB_1e7 = np.sum(norm_F_FB_batch<=1e-7)/n_test
    frac_norm_F_FB_1e8 = np.sum(norm_F_FB_batch<=1e-8)/n_test

    ipopt_results_dict = {"eps": eps, "n_data": n_test}
    ipopt_results_dict["norm_kkt"] = {"max":norm_kkt_max,"min":norm_kkt_min,"mean":norm_kkt_mean,
                                      "std":norm_kkt_std,"frac_1e2":frac_norm_kkt_1e2,
                                      "frac_1e3":frac_norm_kkt_1e3,"frac_1e4":frac_norm_kkt_1e4,
                                      "frac_1e5":frac_norm_kkt_1e5,"frac_1e6":frac_norm_kkt_1e6,
                                      "frac_1e7":frac_norm_kkt_1e7,"frac_1e8":frac_norm_kkt_1e8}                                      
    ipopt_results_dict["norm_F_FB"] = {"max":norm_F_FB_max,"min":norm_F_FB_min,"mean":norm_F_FB_mean,
                                        "std":norm_F_FB_std,"frac_1e2":frac_norm_F_FB_1e2,
                                        "frac_1e3":frac_norm_F_FB_1e3,"frac_1e4":frac_norm_F_FB_1e4,
                                        "frac_1e5":frac_norm_F_FB_1e5,"frac_1e6":frac_norm_F_FB_1e6,
                                        "frac_1e7":frac_norm_F_FB_1e7,"frac_1e8":frac_norm_F_FB_1e8}
            
    # save to save file
    with open(file_pth.joinpath("results","ipopt_results.json"),"w") as fp:
        json.dump(ipopt_results_dict,fp,indent=4)


# %%

# load test dataset
X_test = test_dataset.tensors[0].to(device)
Y_test = test_dataset.tensors[1].to(device)

# approx MPC evaluation
if eval_approx_mpc:

    # load
    approx_mpc = load_approx_mpc(file_pth,approx_mpc_folder)
    approx_mpc.set_device(device="cpu")
    # predict on test dataset
    Y_pred = approx_mpc.make_step(X_test,scale_inputs=True,rescale_outputs=True,clip_outputs=False)

    u0_diff = Y_pred.squeeze()-Y_test.numpy()
    u0_diff_abs = np.abs(u0_diff)
    u0_diff_abs_max = np.max(u0_diff_abs,axis=0)
    u0_diff_abs_min = np.min(u0_diff_abs,axis=0)
    u0_diff_abs_mean = np.mean(u0_diff_abs,axis=0)
    u0_diff_abs_std = np.std(u0_diff_abs,axis=0)
    u0_diff_99 = np.quantile(u0_diff_abs,0.99,axis=0)
    u0_diff_95 = np.quantile(u0_diff_abs,0.95,axis=0)
    u0_diff_90 = np.quantile(u0_diff_abs,0.9,axis=0)
    u0_diff_50 = np.quantile(u0_diff_abs,0.5,axis=0)

    approx_mpc_results = {"u0_diff_abs_max":u0_diff_abs_max,
                            "u0_diff_abs_min":u0_diff_abs_min,
                            "u0_diff_abs_mean":u0_diff_abs_mean,
                            "u0_diff_abs_std":u0_diff_abs_std,
                            "u0_diff_99":u0_diff_99,
                            "u0_diff_95":u0_diff_95,
                            "u0_diff_90":u0_diff_90,
                            "u0_diff_50":u0_diff_50}

    with open(file_pth.joinpath("results",f"approx_mpc_results.json"),"w") as fp:
        json.dump(approx_mpc_results,fp,indent=4)

# eval times
if eval_approx_mpc:
    # load
    approx_mpc = load_approx_mpc(file_pth,approx_mpc_folder)
    approx_mpc.set_device(device="cpu")
    step_times = []
    # predict on test dataset
    for i in range(X_test.shape[0]):
        x_pred_i = X_test[i,:].unsqueeze(0)
        start = timer()
        y_pred_i = approx_mpc.make_step(x_pred_i,scale_inputs=True,rescale_outputs=True,clip_outputs=False)
        end = timer()
        t_step = end-start
        step_times.append(t_step)
    
    step_times = np.array(step_times)
    step_times_max = np.max(step_times)
    step_times_min = np.min(step_times)
    step_times_mean = np.mean(step_times)
    step_times_std = np.std(step_times)

    approx_mpc_solve_times_dict = {"max":step_times_max,
                                "min":step_times_min,
                                "mean":step_times_mean,
                                "std":step_times_std}
    approx_mpc_metrics = {"approx_mpc_solve_times":approx_mpc_solve_times_dict}

    with open(file_pth.joinpath("results",f"approx_mpc_metrics.json"),"w") as fp:
        json.dump(approx_mpc_metrics,fp,indent=4)

####################################################################################################################
# %% 
# Closed Loop Solver
if eval_approx_solver:
    # load solver
    solver_nn, solver_run_hparams = load_solver(file_pth,approx_solver_folder)

    # solver run on test set
    offset = solver_run_hparams["offset"]
    rangeF = solver_run_hparams["rangeF"]
    filter_level = solver_run_hparams["filter_level"]

    solver_batch_size = n_test
    nlp_handler.setup_batch_functions(solver_batch_size,N_threads=8)
    u_pos_in_z = nlp_handler.n_x-mpc.settings.n_horizon

    # test data for comparison
    p_batch = test_dataset.tensors[3].to(device)
    u0_opt = test_dataset.tensors[1].to(device)


    # generate random initial guesses
    z_k_batch = gen_rand_z_norm(solver_batch_size,nlp_handler)

    eps_batch = torch.ones(solver_batch_size,1)*eps

    solver_history = {"eps": eps, "iter": [], "norm_kkt_batch": [], "norm_F_FB_batch": [],
                    "u0_batch": [], "diff_u0_batch": []}
    # CLOSED LOOP SOLVER
    for i in range(max_iter+1):
        if i%100 == 0:
            print(f"iter: {i}/{max_iter}")
        # F_FB_batch
        F_k_batch = nlp_handler.F_FB_batch_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
        F_k_batch = torch.tensor(np.array(F_k_batch).T)
        norm_F_k_batch = torch.norm(F_k_batch,dim=1)
        
        # kkt
        kkt_batch = np.array(nlp_handler.KKT_batch_func(z_k_batch.numpy().T,p_batch.numpy().T)).T
        norm_kkt_batch = np.linalg.norm(kkt_batch,axis=1)

        # u0
        u0_batch = z_k_batch[:,u_pos_in_z].detach().clone()
        diff_u0 = torch.abs(u0_batch-u0_opt)

        # logging
        solver_history["iter"].append(i)
        solver_history["norm_kkt_batch"].append(norm_kkt_batch)
        solver_history["norm_F_FB_batch"].append(norm_F_k_batch.numpy())
        solver_history["u0_batch"].append(u0_batch.numpy())
        solver_history["diff_u0_batch"].append(diff_u0.numpy())

        # STEP
        dz_k_batch = solver_step(nlp_handler,solver_nn,z_k_batch,p_batch,eps_batch,offset,rangeF,filter_level)
        gamma_batch = calc_gamma(nlp_handler,F_k_batch,z_k_batch,p_batch,eps_batch,dz_k_batch,offset)

        # UPDATE
        z_k_batch = z_k_batch + gamma_batch[:,None]*dz_k_batch

    # store solver history
    with open(file_pth.joinpath("results",f"approx_solver_history.pkl"),"wb") as fp:
        pkl.dump(solver_history,fp)

# %% 
# eval solver history

# solver history extension
if eval_approx_solver:
    solver_history_extended = solver_history.copy()

    solver_results = {"eps": eps, "iter": [],"kkt_eval_dict": [], "F_FB_eval_dict": [], "diff_u0_eval_dict": []}

    for i in range(max_iter+1):
        # get solver data
        norm_kkt_batch = solver_history_extended["norm_kkt_batch"][i]
        norm_F_FB_batch = solver_history_extended["norm_F_FB_batch"][i]
        diff_u0_batch = solver_history_extended["diff_u0_batch"][i]

        # metrics
        norm_kkt_batch_max = np.max(norm_kkt_batch)
        norm_kkt_batch_min = np.min(norm_kkt_batch)
        norm_kkt_batch_mean = np.mean(norm_kkt_batch)
        norm_kkt_batch_std = np.std(norm_kkt_batch)
        norm_kkt_batch_99 = np.quantile(norm_kkt_batch,0.99)
        norm_kkt_batch_95 = np.quantile(norm_kkt_batch,0.95)
        norm_kkt_batch_90 = np.quantile(norm_kkt_batch,0.9)
        norm_kkt_batch_50 = np.quantile(norm_kkt_batch,0.5)
        frac_norm_kkt_1e2 = np.sum(norm_kkt_batch<=1e-2)/solver_batch_size
        frac_norm_kkt_1e3 = np.sum(norm_kkt_batch<=1e-3)/solver_batch_size
        frac_norm_kkt_1e4 = np.sum(norm_kkt_batch<=1e-4)/solver_batch_size
        frac_norm_kkt_1e5 = np.sum(norm_kkt_batch<=1e-5)/solver_batch_size
        frac_norm_kkt_1e6 = np.sum(norm_kkt_batch<=1e-6)/solver_batch_size
        frac_norm_kkt_1e7 = np.sum(norm_kkt_batch<=1e-7)/solver_batch_size
        frac_norm_kkt_1e8 = np.sum(norm_kkt_batch<=1e-8)/solver_batch_size

        kkt_eval_dict = {"norm_kkt_batch_max":norm_kkt_batch_max,
                        "norm_kkt_batch_min":norm_kkt_batch_min,
                        "norm_kkt_batch_mean":norm_kkt_batch_mean,
                        "norm_kkt_batch_std":norm_kkt_batch_std,
                        "norm_kkt_batch_99":norm_kkt_batch_99,
                        "norm_kkt_batch_95":norm_kkt_batch_95,
                        "norm_kkt_batch_90":norm_kkt_batch_90,
                        "norm_kkt_batch_50":norm_kkt_batch_50,
                        "frac_norm_kkt_batch_1e2":frac_norm_kkt_1e2,
                        "frac_norm_kkt_batch_1e3":frac_norm_kkt_1e3,
                        "frac_norm_kkt_batch_1e4":frac_norm_kkt_1e4,
                        "frac_norm_kkt_batch_1e5":frac_norm_kkt_1e5,
                        "frac_norm_kkt_batch_1e6":frac_norm_kkt_1e6,
                        "frac_norm_kkt_batch_1e7":frac_norm_kkt_1e7,
                        "frac_norm_kkt_batch_1e8":frac_norm_kkt_1e8}
        
        norm_F_FB_batch_max = np.max(norm_F_FB_batch)
        norm_F_FB_batch_min = np.min(norm_F_FB_batch)
        norm_F_FB_batch_mean = np.mean(norm_F_FB_batch)
        norm_F_FB_batch_std = np.std(norm_F_FB_batch)
        norm_F_FB_batch_99 = np.quantile(norm_F_FB_batch,0.99)
        norm_F_FB_batch_95 = np.quantile(norm_F_FB_batch,0.95)
        norm_F_FB_batch_90 = np.quantile(norm_F_FB_batch,0.9)
        norm_F_FB_batch_50 = np.quantile(norm_F_FB_batch,0.5)
        # number of norm_F_FB_batch elements smaller than 1tol
        frac_norm_F_FB_batch_1e2 = np.sum(norm_F_FB_batch<=1e-2)/solver_batch_size
        frac_norm_F_FB_batch_1e3 = np.sum(norm_F_FB_batch<=1e-3)/solver_batch_size
        frac_norm_F_FB_batch_1e4 = np.sum(norm_F_FB_batch<=1e-4)/solver_batch_size
        frac_norm_F_FB_batch_1e5 = np.sum(norm_F_FB_batch<=1e-5)/solver_batch_size
        frac_norm_F_FB_batch_1e6 = np.sum(norm_F_FB_batch<=1e-6)/solver_batch_size
        frac_norm_F_FB_batch_1e7 = np.sum(norm_F_FB_batch<=1e-7)/solver_batch_size
        frac_norm_F_FB_batch_1e8 = np.sum(norm_F_FB_batch<=1e-8)/solver_batch_size

        F_FB_eval_dict = {"norm_F_FB_batch_max":norm_F_FB_batch_max,
                        "norm_F_FB_batch_min":norm_F_FB_batch_min,
                        "norm_F_FB_batch_mean":norm_F_FB_batch_mean,
                        "norm_F_FB_batch_std":norm_F_FB_batch_std,
                        "norm_F_FB_batch_99":norm_F_FB_batch_99,
                        "norm_F_FB_batch_95":norm_F_FB_batch_95,
                        "norm_F_FB_batch_90":norm_F_FB_batch_90,
                        "norm_F_FB_batch_50":norm_F_FB_batch_50,
                        "frac_norm_F_FB_batch_1e2":frac_norm_F_FB_batch_1e2,
                        "frac_norm_F_FB_batch_1e3":frac_norm_F_FB_batch_1e3,
                        "frac_norm_F_FB_batch_1e4":frac_norm_F_FB_batch_1e4,
                        "frac_norm_F_FB_batch_1e5":frac_norm_F_FB_batch_1e5,
                        "frac_norm_F_FB_batch_1e6":frac_norm_F_FB_batch_1e6,
                        "frac_norm_F_FB_batch_1e7":frac_norm_F_FB_batch_1e7,
                        "frac_norm_F_FB_batch_1e8":frac_norm_F_FB_batch_1e8}
        
        diff_u0_max = np.max(diff_u0_batch)
        diff_u0_min = np.min(diff_u0_batch)
        diff_u0_mean = np.mean(diff_u0_batch)
        diff_u0_std = np.std(diff_u0_batch)
        diff_u0_99 = np.quantile(diff_u0_batch,0.99)
        diff_u0_95 = np.quantile(diff_u0_batch,0.95)
        diff_u0_90 = np.quantile(diff_u0_batch,0.9)
        diff_u0_50 = np.quantile(diff_u0_batch,0.5)
        diff_u0_eval_dict = {"diff_u0_max":diff_u0_max,
                        "diff_u0_min":diff_u0_min,
                        "diff_u0_mean":diff_u0_mean,
                        "diff_u0_std":diff_u0_std,
                        "diff_u0_99":diff_u0_99,
                        "diff_u0_95":diff_u0_95,
                        "diff_u0_90":diff_u0_90,
                        "diff_u0_50":diff_u0_50}
        
        solver_results["iter"].append(i)
        solver_results["kkt_eval_dict"].append(kkt_eval_dict)
        solver_results["F_FB_eval_dict"].append(F_FB_eval_dict)
        solver_results["diff_u0_eval_dict"].append(diff_u0_eval_dict)
        

    with open(file_pth.joinpath("results",f"approx_solver_results.json"),"w") as fp:
        json.dump(solver_results,fp,indent=4)

# %% visualize solver history
if visualize_bool:
    idx_list = torch.randperm(solver_history_extended["norm_F_FB_batch"][0].shape[0]).tolist()
    n_traj = 30
    fig, ax = plt.subplots()
    for j in range(n_traj):
        idx = idx_list[j]
        plt_list = []
        for i in range(max_iter):
            plt_list.append(solver_history_extended["norm_F_FB_batch"][i][idx])
        ax.plot(plt_list)
    ax.set_yscale("log")
    ax.set_title("2-Norm F_FB of random solver trajectories")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("2-Norm F_FB")


# %% visualization solver results
if visualize_bool:
    fig1, ax1 = plt.subplots(1,1)
    iterations = solver_results["iter"]
    mean_kkt_errors = [x["norm_kkt_batch_mean"] for x in solver_results["kkt_eval_dict"]]
    max_kkt_errors = [x["norm_kkt_batch_max"] for x in solver_results["kkt_eval_dict"]]
    min_kkt_errors = [x["norm_kkt_batch_min"] for x in solver_results["kkt_eval_dict"]]

    ax1.plot(iterations,mean_kkt_errors,"C0")
    ax1.plot(iterations,max_kkt_errors,"C1")
    ax1.plot(iterations,min_kkt_errors,"C2")
    ax1.set_yscale("log")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("KKT error (2-Norm)")
    ax1.legend(["mean","max","min"])
    ax1.set_title("KKT error")

    fig2, ax2 = plt.subplots(1,1)
    iterations = solver_results["iter"]
    mean_F_FB_errors = [x["norm_F_FB_batch_mean"] for x in solver_results["F_FB_eval_dict"]]
    max_F_FB_errors = [x["norm_F_FB_batch_max"] for x in solver_results["F_FB_eval_dict"]]
    min_F_FB_errors = [x["norm_F_FB_batch_min"] for x in solver_results["F_FB_eval_dict"]]

    ax2.plot(iterations,mean_F_FB_errors,"C0")
    ax2.plot(iterations,max_F_FB_errors,"C1")
    ax2.plot(iterations,min_F_FB_errors,"C2")
    ax2.set_yscale("log")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("KKT error (2-Norm)")
    ax2.legend(["mean","max","min"])
    ax2.set_title("F_FB error")

    fig3, ax3 = plt.subplots(1,1)
    iterations = solver_results["iter"]
    mean_diff_u0 = [x["diff_u0_mean"] for x in solver_results["diff_u0_eval_dict"]]
    max_diff_u0 = [x["diff_u0_max"] for x in solver_results["diff_u0_eval_dict"]]
    min_diff_u0 = [x["diff_u0_min"] for x in solver_results["diff_u0_eval_dict"]]

    ax3.plot(iterations,mean_diff_u0,"C0")
    ax3.plot(iterations,max_diff_u0,"C1")
    ax3.plot(iterations,min_diff_u0,"C2")
    ax3.set_yscale("log")
    ax3.set_xlabel("iterations")
    ax3.set_ylabel("u0 error (Absolute)")
    ax3.legend(["mean","max","min"])
    ax3.set_title("u0 error")


# %% 
# Eval solver times

if eval_approx_solver_times:
    # load solver
    solver_nn, solver_run_hparams = load_solver(file_pth,approx_solver_folder)

    # solver run on test set
    offset = solver_run_hparams["offset"]
    rangeF = solver_run_hparams["rangeF"]
    filter_level = solver_run_hparams["filter_level"]

    solver_batch_size = 1
    nlp_handler.setup_batch_functions(solver_batch_size,N_threads=8)


    full_step_times = []
    nn_step_times = []

    # generate 100 random indices in n_test
    idx_list = torch.randperm(n_test)[:100].tolist()

    # CLOSED LOOP SOLVER
    for idx in idx_list:
        p_batch = test_dataset.tensors[3].to(device)[idx,:].unsqueeze(0)
        z_k_batch = gen_rand_z_norm(solver_batch_size,nlp_handler)
        eps_batch = torch.ones(solver_batch_size,1)*eps

        for i in range(max_iter+1):
            start_full = timer()
            # F_FB_batch
            F_k_batch = nlp_handler.F_FB_batch_func(z_k_batch.numpy().T,p_batch.numpy().T,eps_batch.numpy().T)
            F_k_batch = torch.tensor(np.array(F_k_batch).T)
            norm_F_k_batch = torch.norm(F_k_batch,dim=1)

            # STEP
            dz_k_batch, nn_step_time = solver_step_time_tracking(nlp_handler,solver_nn,z_k_batch,p_batch,eps_batch,offset,rangeF,filter_level)
            gamma_batch = calc_gamma(nlp_handler,F_k_batch,z_k_batch,p_batch,eps_batch,dz_k_batch,offset)

            # UPDATE
            z_k_batch = z_k_batch + gamma_batch[:,None]*dz_k_batch
            end_full = timer()
            full_step_times.append(end_full-start_full)
            nn_step_times.append(nn_step_time)
    
    full_step_times = np.array(full_step_times)
    nn_step_times = np.array(nn_step_times)

    full_step_times_max = np.max(full_step_times)
    full_step_times_min = np.min(full_step_times)
    full_step_times_mean = np.mean(full_step_times)
    full_step_times_std = np.std(full_step_times)

    nn_step_times_max = np.max(nn_step_times)
    nn_step_times_min = np.min(nn_step_times)
    nn_step_times_mean = np.mean(nn_step_times)
    nn_step_times_std = np.std(nn_step_times)

    approx_solver_times_dict = {"full_step_times":{"max":full_step_times_max,
                                                "min":full_step_times_min,
                                                "mean":full_step_times_mean,
                                                "std":full_step_times_std},
                                "nn_step_times":{"max":nn_step_times_max,
                                                "min":nn_step_times_min,
                                                "mean":nn_step_times_mean,
                                                "std":nn_step_times_std}}
    
    with open(file_pth.joinpath("results",f"approx_solver_times.json"),"w") as fp:
        json.dump(approx_solver_times_dict,fp,indent=4)
# %%
