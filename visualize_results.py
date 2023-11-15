"""
Date: 2023-10-22
Author: Lukas Lüken

Pytorch script to visualize results for paper.
"""
# %%
# Config
eps = 1e-6

# Results Folder
results_folder_name = "results"
figures_folder_name = "figures"

# ipopt
ipopt_metrics_file_name = "ipopt_solver_metrics.json"
ipopt_results_file_name = "ipopt_results.json"

# approx MPC to evaluate
approx_mpc_folder = "approx_mpc_model"

# solver
approx_solver_folder = "learning_based_solver_model"
iteration_split = [10,20,40,50,100,200,500,1000]

# visualization
save_figures = True

##############################################################################################################

# %% Imports
# import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
# import pickle as pkl
import importlib
from decimal import Decimal
importlib.import_module("mpl_config")

# %% 
# Helper functions
def val_to_decim_str(val):
    if val == 0.0:
        dec_str = "0.0"
    else:
        dec_str = f"{Decimal(str(val)):.2e}"
    return dec_str

# %%
# Setup
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)
results_folder = file_pth.joinpath(results_folder_name)
figures_folder = file_pth.joinpath(figures_folder_name,"open_loop")

# %% Load ipopt metrics & results
ipopt_metrics_file = results_folder.joinpath(ipopt_metrics_file_name)
ipopt_results_file = results_folder.joinpath(ipopt_results_file_name)

# ipopt_metrics = pd.read_json(ipopt_metrics_file)
# ipopt_results = pd.read_json(ipopt_results_file)

with open(ipopt_metrics_file,"r") as fp:
    ipopt_metrics = json.load(fp)

with open(ipopt_results_file,"r") as fp:
    ipopt_results = json.load(fp)

# %% Load approx mpc results
approx_mpc_results = {}
approx_mpc_results_file = results_folder.joinpath(f"approx_mpc_results.json")
with open(approx_mpc_results_file,"r") as fp:
    approx_mpc_results[approx_mpc_folder] = json.load(fp)

# %% Load approx solver results
approx_solver_results_file = results_folder.joinpath(f"approx_solver_results.json")
with open(approx_solver_results_file,"r") as fp:
    approx_solver_results = json.load(fp)

# %%
df = pd.DataFrame(approx_solver_results)
for key in ['kkt_eval_dict', 'F_FB_eval_dict', 'diff_u0_eval_dict']:
    for k in df[key].iloc[0].keys():
        df[k] = df[key].apply(lambda x: x[k])
    df.drop(key,axis=1,inplace=True)

# %%
# 1. VISUALIZE U0-U0_IPOPT 

iterations = approx_solver_results["iter"]
diff_u0_mean = [x["diff_u0_mean"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_min = [x["diff_u0_min"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_max = [x["diff_u0_max"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_99 = [x["diff_u0_99"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_95 = [x["diff_u0_95"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_90 = [x["diff_u0_90"] for x in approx_solver_results["diff_u0_eval_dict"]] 
diff_u0_50 = [x["diff_u0_50"] for x in approx_solver_results["diff_u0_eval_dict"]] 
fig1, ax1 = plt.subplots(1,1)
# ax1.plot(iterations,diff_u0_mean,label="mean")
# ax1.plot(iterations,diff_u0_min,label="min")
ax1.plot(iterations,diff_u0_max,label="max")
ax1.plot(iterations,diff_u0_99,label="99th perc.")
ax1.plot(iterations,diff_u0_95,label="95th perc.")
ax1.plot(iterations,diff_u0_90,label="90th perc.")
ax1.plot(iterations,diff_u0_50,label="50th perc.")

ax1.set_xlabel("Iteration")
ax1.set_ylabel("$|\hat{u}_0 - u_{0,ipopt}|$")
# ax1.set_title("Absolute value of difference between $u_0$ and $u_{0,ipopt}$ over iterations")
ax1.set_title("Absolute error to optimal (ipopt) control action on test data")
ax1.legend()
ax1.set_yscale("log")
ax1.set_xscale("log")

# %%
# 2. VISUALIZE F_FB_norm
iterations = approx_solver_results["iter"]
norm_F_FB_batch_mean = [x["norm_F_FB_batch_mean"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_min = [x["norm_F_FB_batch_min"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_max = [x["norm_F_FB_batch_max"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_99 = [x["norm_F_FB_batch_99"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_95 = [x["norm_F_FB_batch_95"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_90 = [x["norm_F_FB_batch_90"] for x in approx_solver_results["F_FB_eval_dict"]] 
norm_F_FB_batch_50 = [x["norm_F_FB_batch_50"] for x in approx_solver_results["F_FB_eval_dict"]] 

fig2, ax2 = plt.subplots(1,1)
# ax2.plot(iterations,norm_F_FB_batch_mean,label="mean")
# ax2.plot(iterations,norm_F_FB_batch_min,label="min")
ax2.plot(iterations,norm_F_FB_batch_max,label="max")
ax2.plot(iterations,norm_F_FB_batch_99,label="99th perc.")
ax2.plot(iterations,norm_F_FB_batch_95,label="95th perc.")
ax2.plot(iterations,norm_F_FB_batch_90,label="90th perc.")
ax2.plot(iterations,norm_F_FB_batch_50,label="50th perc.")

ax2.set_xlabel("Iteration")
ax2.set_ylabel("$||F_{FB}||_{2}$")
ax2.set_title("2-Norm of $F_{FB}$ over iterations")
ax2.legend()
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylim([1e-8,1e2])

# %% 
# 3. VISUALIZE KKT_norm
iterations = approx_solver_results["iter"]
norm_kkt_batch_mean = [x["norm_kkt_batch_mean"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_min = [x["norm_kkt_batch_min"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_max = [x["norm_kkt_batch_max"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_99 = [x["norm_kkt_batch_99"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_95 = [x["norm_kkt_batch_95"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_90 = [x["norm_kkt_batch_90"] for x in approx_solver_results["kkt_eval_dict"]]
norm_kkt_batch_50 = [x["norm_kkt_batch_50"] for x in approx_solver_results["kkt_eval_dict"]]

fig3, ax3 = plt.subplots(1,1)
# ax3.plot(iterations,norm_kkt_batch_mean,label="mean")
# ax3.plot(iterations,norm_kkt_batch_min,label="min")
ax3.plot(iterations,norm_kkt_batch_max,label="max")
ax3.plot(iterations,norm_kkt_batch_99,label="99th perc.")
ax3.plot(iterations,norm_kkt_batch_95,label="95th perc.")
ax3.plot(iterations,norm_kkt_batch_90,label="90th perc.")
ax3.plot(iterations,norm_kkt_batch_50,label="50th perc.")

ax3.set_xlabel("Iteration")
ax3.set_ylabel("$||F_{\mathrm{KKT}}||_{2}$")
ax3.set_title("2-norm of KKT over iterations")
ax3.legend()
ax3.set_yscale("log")
ax3.set_xscale("log")
# add hline at 1e-6
# ax3.axhline(y=eps,color="black",linestyle="--")
# ax3.set_ylim([1e-8,1e2])
ax3.set_ylim([1e-6,1e2])

# %% 
# Save figures
if save_figures:
    for format in ["png","svg","eps","pdf"]:
        fig1.savefig(figures_folder.joinpath(f"solver_diff_u0.{format}"),format=format)
        fig2.savefig(figures_folder.joinpath(f"solver_F_FB.{format}"),format=format)
        fig3.savefig(figures_folder.joinpath(f"solver_KKT.{format}"),format=format)

# %%
# 1. Table of results
# rows: ipopt, approximate MPC, approximate solver - 1 step, approximate solver - 10 steps, approximate solver 20 steps, ... (see iteration_split)
# columns diff_u0: max, mean, 99-perc. 95-perc. 90-perc. 50-perc.

df = pd.DataFrame(columns=["max","mean","min","99-perc.","95-perc.","90-perc.","50-perc."])
df.index.name = "solver"

# ipopt
df.loc["ipopt"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# approx mpc
tmp_dict = approx_mpc_results[approx_mpc_folder]
df.loc[approx_mpc_folder] = [tmp_dict["u0_diff_abs_max"],
                            tmp_dict["u0_diff_abs_mean"],
                            tmp_dict["u0_diff_abs_min"],
                            tmp_dict["u0_diff_99"],
                            tmp_dict["u0_diff_95"],
                            tmp_dict["u0_diff_90"],
                            tmp_dict["u0_diff_50"]]
# approx solver
for iter in iteration_split:
    tmp_dict = approx_solver_results["diff_u0_eval_dict"][iter]
    df.loc[f"approx_solver_N{iter}"] = [tmp_dict["diff_u0_max"],
                                tmp_dict["diff_u0_mean"],
                                tmp_dict["diff_u0_min"],
                                tmp_dict["diff_u0_99"],
                                tmp_dict["diff_u0_95"],
                                tmp_dict["diff_u0_90"],
                                tmp_dict["diff_u0_50"]]
# apply everywhere to string
df = df.map(val_to_decim_str)

df_tex = df.to_latex()
print(df_tex)

if save_figures:
    with open(figures_folder.joinpath("table_u0_diff.tex"),"w") as f:
        f.write(df_tex)


# %%
