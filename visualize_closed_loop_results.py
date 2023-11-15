"""
Date: 2023-11-03
Author: Lukas Lüken

Script to visualize closed loop application of learned solver.
"""
# %%
# Config

closed_loop_run_folder = "closed_loop"

visualize_closed_loop = False # Visualization of MPC closed loop trajectories. Be careful, this can produce a lot of plots.
analyse_results = True  

# %% 
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import pickle as pkl
import importlib
importlib.import_module("mpl_config")
# %%
# Setup
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

# %% 
# Load run
run_dict_pth = file_pth.joinpath("results",closed_loop_run_folder,"run_dict.json")
with open(run_dict_pth, 'r') as f:
    run_dict = json.load(f)

trajectories_pth = file_pth.joinpath("results",closed_loop_run_folder,"trajectories.pkl")
with open(trajectories_pth, 'rb') as f:
    trajectories = pkl.load(f)


n_runs = run_dict["n_runs"]
noise_level = run_dict["noise_level"]
max_iter = run_dict["max_iter"]
print("-----------------------------------")
print(f"Loaded closed loop data with {n_runs} closed loop runs with noise: {noise_level} and max. iter.: {max_iter}")
print("-----------------------------------")


# %% visualize trajectories
if visualize_closed_loop:
    for trajectory in trajectories:
        fig, ax = plt.subplots(1,1)
        ax.plot(trajectory["x0"][:,0],label="x1")
        ax.plot(trajectory["x0"][:,1],label="x2")
        ax.plot(trajectory["u0"],label="u0")
        ax.plot(trajectory["u0_ipopt"],"--",label="u0_ipopt")
        ax.set_xlabel("Time")
        # ax.set_ylabel("states and control action")
        ax.legend()
        ax.set_title("Closed loop trajectory")

# %% analyze results

if analyse_results:
    # get a df with norm_F_FB, N_iter, diff_u0, diff_u0_approx_mpc and norm_KKT and run number
    df = []
    for k, trajectory in enumerate(trajectories):
        for i in range(len(trajectory["norm_F_FB"])):
            df.append({"run":k,
                    "closed_loop_iteration":i,
                    "norm_F_FB":trajectory["norm_F_FB"][i],
                    "N_iter":trajectory["N_iter"][i],
                    "diff_u0":trajectory["diff_u0"][i],
                    "diff_u0_approx_mpc":trajectory["diff_u0_approx_mpc"][i],
                    "norm_KKT":trajectory["norm_KKT"][i],
                    "ipopt_time":trajectory["ipopt_time"][i],
                        "approx_mpc_time":trajectory["approx_mpc_time"][i],
                        "approx_solver_time":trajectory["approx_solver_time"][i]})
    df = pd.DataFrame(df)

    # 50th perc.
    def q50(x):
        return x.quantile(0.5,interpolation="midpoint")

    # 90th perc.
    def q90(x):
        return x.quantile(0.9,interpolation="midpoint")

    # 95th perc.
    def q95(x):
        return x.quantile(0.95,interpolation="midpoint")

    # 99th perc.
    def q99(x):
        return x.quantile(0.99,interpolation="midpoint")

    # get metrics for each closed_loop_iteration
    df_grouped = df.groupby("closed_loop_iteration").agg({"norm_F_FB":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "N_iter":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "diff_u0":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "diff_u0_approx_mpc":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "norm_KKT":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "approx_mpc_time":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "ipopt_time":["mean","median","std","max","min",q50,q90,q95,q99],
                                                        "approx_solver_time":["mean","median","std","max","min",q50,q90,q95,q99]})

# %% 
# Latex Table
def format_scientific_power(x,mode="e",precision=2):
    if x == 0:
        return "0"
    else:
        if mode == "e":
            return f"{x:.{precision}e}"
        elif mode == "10^":
            return f"{x:.{precision}e}".replace("e","\\cdot 10^{")+"}"
        elif mode == None:
            return f"{x:.{precision}f}"
        else:
            raise ValueError("mode must be specified as 'e' or '10^' or None")

du_approx_mpc = {"max":df["diff_u0_approx_mpc"].max(),
              "q99":q99(df["diff_u0_approx_mpc"]),
                "q90":q90(df["diff_u0_approx_mpc"]),
                "q50":q50(df["diff_u0_approx_mpc"]),
                "min":df["diff_u0_approx_mpc"].min()}

du_approx_solver = {"max":df["diff_u0"].max(),
                "q99":q99(df["diff_u0"]),
                    "q90":q90(df["diff_u0"]),
                    "q50":q50(df["diff_u0"]),
                    "min":df["diff_u0"].min()}

# merge both to dataframe with indices approx_mpc and approx_solver and columns max, 99th perc., 90th perc., 50th perc. and min
df_diff_u0 = pd.DataFrame({"approx_mpc":du_approx_mpc,"approx_solver":du_approx_solver})
df_diff_u0 = df_diff_u0.transpose()
df_diff_u0 = df_diff_u0[["max","q99","q90","q50","min"]]
df_diff_u0 = df_diff_u0.rename(columns={"max":"max. error","q99":"99th perc.","q90":"90th perc.","q50":"50th perc.","min":"min. error"})


# reformat numbers to scientific notation
for col in df_diff_u0.columns:
    # df_diff_u0[col] = df_diff_u0[col].apply(lambda x: f"{x:.2e}")
    df_diff_u0[col] = df_diff_u0[col].apply(format_scientific_power)


print(df_diff_u0.to_latex())


# %%
# do same stuff for ratio between approx_mpc/ipopt_time and approx_solver/ipopt_time
df["ratio_approx_mpc"] = df["approx_mpc_time"]/df["ipopt_time"]
df["ratio_approx_solver"] = df["approx_solver_time"]/df["ipopt_time"]

ratio_approx_mpc = {"max":df["ratio_approx_mpc"].max(),
                "q99":q99(df["ratio_approx_mpc"]),
                    "q90":q90(df["ratio_approx_mpc"]),
                    "q50":q50(df["ratio_approx_mpc"]),
                    "min":df["ratio_approx_mpc"].min()}
ratio_approx_solver = {"max":df["ratio_approx_solver"].max(),
                "q99":q99(df["ratio_approx_solver"]),
                    "q90":q90(df["ratio_approx_solver"]),
                    "q50":q50(df["ratio_approx_solver"]),
                    "min":df["ratio_approx_solver"].min()}
df_relative_times = pd.DataFrame({"approx_mpc":ratio_approx_mpc,"approx_solver":ratio_approx_solver})
df_relative_times = df_relative_times.transpose()
df_relative_times = df_relative_times[["max","q99","q90","q50","min"]]
df_relative_times = df_relative_times.rename(columns={"max":"max. ratio","q99":"99th perc.","q90":"90th perc.","q50":"50th perc.","min":"min. ratio"})

for col in df_relative_times.columns:
    df_relative_times[col] = df_relative_times[col].apply(lambda x: format_scientific_power(x,mode=None,precision=3))

print(df_relative_times.to_latex())

# %% 
# do same stuff for absolute times
approx_mpc_time = {"max":df["approx_mpc_time"].max(),
                "q99":q99(df["approx_mpc_time"]),
                    "q90":q90(df["approx_mpc_time"]),
                    "q50":q50(df["approx_mpc_time"]),
                    "min":df["approx_mpc_time"].min()}

approx_solver_time = {"max":df["approx_solver_time"].max(),
                "q99":q99(df["approx_solver_time"]),
                    "q90":q90(df["approx_solver_time"]),
                    "q50":q50(df["approx_solver_time"]),
                    "min":df["approx_solver_time"].min()}

ipopt_time = {"max":df["ipopt_time"].max(),
                "q99":q99(df["ipopt_time"]),
                    "q90":q90(df["ipopt_time"]),
                    "q50":q50(df["ipopt_time"]),
                    "min":df["ipopt_time"].min()}

df_times = pd.DataFrame({"approx_mpc":approx_mpc_time,"approx_solver":approx_solver_time,"ipopt":ipopt_time})
df_times = df_times.transpose()
df_times = df_times[["max","q99","q90","q50","min"]]
df_times = df_times.rename(columns={"max":"max. time","q99":"99th perc.","q90":"90th perc.","q50":"50th perc.","min":"min. time"})
for col in df_times.columns:
    df_times[col] = df_times[col].apply(lambda x: format_scientific_power(x,mode=None,precision=6))

print(df_times.to_latex())


# %%
#  do same stuff for number of iterations of approx_solver
n_iter = {"max":df["N_iter"].max(),
                "q99":q99(df["N_iter"]),
                    "q90":q90(df["N_iter"]),
                    "q50":q50(df["N_iter"]),
                    "min":df["N_iter"].min()}
df_N_iter = pd.DataFrame({"approx_solver":n_iter})
df_N_iter = df_N_iter.transpose()
df_N_iter = df_N_iter[["max","q99","q90","q50","min"]]

df_N_iter = df_N_iter.rename(columns={"max":"max. iterations","q99":"99th perc.","q90":"90th perc.","q50":"50th perc.","min":"min. iterations"})
for col in df_N_iter.columns:
    df_N_iter[col] = df_N_iter[col].apply(lambda x: format_scientific_power(x,mode=None,precision=0))

print(df_N_iter.to_latex())

# %%
# export latex tables in text file to run folder
latex_tables_pth = file_pth.joinpath("results",closed_loop_run_folder,"latex_tables.txt")
with open(latex_tables_pth, 'w') as f:
    f.write(df_diff_u0.to_latex())
    f.write("\n")
    f.write(df_diff_u0.T.to_latex())
    f.write("\n")
    f.write(df_relative_times.to_latex())
    f.write("\n")
    f.write(df_relative_times.T.to_latex())
    f.write("\n")
    f.write(df_times.to_latex())
    f.write("\n")
    f.write(df_times.T.to_latex())
    f.write("\n")
    f.write(df_N_iter.to_latex())
    f.write("\n")
    f.write(df_N_iter.T.to_latex())
# %%
# export tables as json
df_diff_u0.to_json(file_pth.joinpath("results",closed_loop_run_folder,"df_diff_u0.json"),indent=4)
df_relative_times.to_json(file_pth.joinpath("results",closed_loop_run_folder,"df_relative_times.json"),indent=4)
df_times.to_json(file_pth.joinpath("results",closed_loop_run_folder,"df_times.json"),indent=4)
df_N_iter.to_json(file_pth.joinpath("results",closed_loop_run_folder,"df_N_iter.json"),indent=4)


# %%
# Visualization of Histogram of error distribution

# settings
n_bins = 100
alpha_trans = 0.3

# calculate histograms
log_diff_u0 = np.log10(df["diff_u0"])
log_diff_u0_approx_mpc = np.log10(df["diff_u0_approx_mpc"])
diff_u0_hist = np.histogram(log_diff_u0,bins=n_bins)
diff_u0_approx_mpc_hist = np.histogram(log_diff_u0_approx_mpc,bins=n_bins)

# figure
fig, ax = plt.subplots(1,1)
# fig.tight_layout()
# plot histogram
ax.plot(diff_u0_hist[1][:-1],diff_u0_hist[0],"C0",label="learning-based iterative solver")
ax.plot(diff_u0_approx_mpc_hist[1][:-1],diff_u0_approx_mpc_hist[0],"C1")
ax.hist(log_diff_u0,color="C0",bins=n_bins,alpha=alpha_trans)
ax.hist(log_diff_u0_approx_mpc,color="C1",bins=n_bins,alpha=alpha_trans)


# X-AXIS
# change xticks to 10^x
xticks = ax.get_xticks()
xticks_labels = [f"$10^{{{int(x)}}}$" for x in xticks]
ax.set_xticklabels(xticks_labels)
ax.set_xlabel("Absolute error to optimal control action $|\hat{u}_0 - u_{0}^{*}|$")

# Y-AXIS
# ax.set_yscale("log")
ax.set_ylabel("Frequency")
# ax.set_yticks([10,100,1000])

# LEGEND
ax.legend(["learning-based iterative solver","approximate mpc"])
# tighten legend box
leg = ax.get_legend()
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_alpha(0.0)

# TITLE
ax.set_title("Distribution of absolute error to optimal control action")

# SAVE
# for format in ["png","svg","eps","pdf"]:
#     fig.savefig(file_pth.joinpath("results",closed_loop_run_folder,f"histogram_error_distribution.{format}"),format=format)

# save in figure folder
figures_folder = file_pth.joinpath("figures","closed_loop")
for format in ["png","svg","eps","pdf"]:
    fig.savefig(figures_folder.joinpath(f"histogram_error_distribution.{format}"),format=format)

# %%
