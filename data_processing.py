"""
Date: 2023-10-19
Author: Lukas Lüken

Script for processing the sampled data and splitting it into training, validation and test datasets, with training sets of various sizes.

"""

# %% Imports
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Control Problem
from nl_double_int_nmpc.template_model import template_model
from nl_double_int_nmpc.template_mpc import template_mpc

from nlp_handler import NLPHandler

file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

silence_solver = True
model = template_model()
mpc = template_mpc(model,silence_solver=silence_solver)
nlp_handler = NLPHandler(mpc)
data_dir = file_pth.joinpath('./sampling')

# %% Config

mode_val = False
mode_test = False
mode_train = False
assert not (mode_val and mode_test and mode_train), "Only one mode can be active at a time"

n_val = 1000
n_test = 1500
nn_train = [100,1000,10000]


# Samples
if mode_val:
    data_file_name = 'data_n1225_opt' # validation data
elif mode_test:
    data_file_name = 'data_n2378_opt' # test data
elif mode_train:
    data_file_name = 'data_n12127_opt' # training data
else:
    data_file_name = 'data_n62_opt' # data processing - for testing the script

# %% Main

# load from pickle in pandas
data_file = str(data_dir) +'/' + data_file_name+'.pkl'
df = pd.read_pickle(data_file)

# %% Split datasets

x_1_list = []
x_2_list = []
uprev_list = []
u0_list = []
z_num_list = []
p_num_list = []

for i in range(len(df)):
    if i%100==0:
        print("progress: ",i,"/",len(df))
    x_1_list.append(df["x0"][i][0].item())
    x_2_list.append(df["x0"][i][1].item())
    uprev_list.append(df["u_prev"][i].item())
    u0_list.append(df["u0"][i])
    z_num_list.append(df["z_num"][i])
    p_num_list.append(df["p_num"][i])

# Extract input output data as pytorch tensors
x_1 = torch.tensor(x_1_list)
x_2 = torch.tensor(x_2_list)
uprev = torch.tensor(uprev_list)
u0 = torch.tensor(u0_list)
z_num = torch.tensor(np.stack(z_num_list).squeeze())
p_num = torch.tensor(np.stack(p_num_list).squeeze())


nlp_handler.setup_batch_functions(len(df),N_threads=8)
eps = 1e-6
eps_batch = torch.ones(len(df),1)*eps
F_FB_batch = nlp_handler.F_FB_batch_func(z_num.numpy().T,p_num.numpy().T,eps_batch.numpy().T)
F_FB_batch = torch.tensor(np.array(F_FB_batch).T)
norm_F_FB_batch = torch.norm(F_FB_batch,dim=1)

KKT_batch = nlp_handler.KKT_batch_func(z_num.numpy().T,p_num.numpy().T)
KKT_batch = torch.tensor(np.array(KKT_batch).T)
norm_KKT_batch = torch.norm(KKT_batch,dim=1)

FB_filter_idx = ~(norm_F_FB_batch>1e-7)
KKT_filter_idx = ~(norm_KKT_batch>1e-6)
filter_idx = FB_filter_idx & KKT_filter_idx

# %% Save datasets
# stack
inputs = torch.stack([x_1,x_2,uprev],dim=1)
outputs = u0.reshape(-1)

# filter optimizer (ipopt) data which is suboptimal to consider only points, which can be solved to optimality
inputs_filtered = inputs[filter_idx,:]
outputs_filtered = outputs[filter_idx]
z_num_filtered = z_num[filter_idx,:]
p_num_filtered = p_num[filter_idx,:]

if mode_val:
    # get n_val random indices from dataset
    idx = torch.randperm(len(inputs))[:n_val]
    data = (inputs[idx,:],outputs[idx],z_num[idx,:],p_num[idx,:])
    dataset = torch.utils.data.TensorDataset(*data)
    torch.save(dataset,file_pth.joinpath("datasets",f"dataset_val_{n_val}.pt"))
elif mode_test:
    idx = torch.randperm(len(inputs_filtered))[:n_test]
    data = (inputs_filtered[idx,:],outputs_filtered[idx],z_num_filtered[idx,:],p_num_filtered[idx,:])
    dataset = torch.utils.data.TensorDataset(*data)
    torch.save(dataset,file_pth.joinpath("datasets",f"dataset_test_{n_test}.pt"))
elif mode_train:
    for n_train in nn_train:
        idx = torch.randperm(len(inputs))[:n_train]    
        data = (inputs[idx,:],outputs[idx],z_num[idx,:],p_num[idx,:])
        dataset = torch.utils.data.TensorDataset(*data)
        torch.save(dataset,file_pth.joinpath("datasets",f"dataset_train_{n_train}.pt"))
else:
    data = (inputs,outputs,z_num,p_num)
    dataset = torch.utils.data.TensorDataset(*data)
    torch.save(dataset,file_pth.joinpath("datasets","dataset_dummy.pt"))

# %%
