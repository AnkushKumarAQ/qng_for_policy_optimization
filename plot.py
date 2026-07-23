import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import SOLVE_THRESHOLD


MIN_ACTIVE = 5

def plot_metric_vs_episode(files, metric, ylabel, filename, threshold=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    for name, file in files.items():
        if not os.path.exists(file):
            continue
            
        df = pd.read_csv(file)
        
        if metric not in df.columns:
            continue
            
        agg = df.groupby("global_episode").agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n=(metric, "count")
        ).reset_index()
        
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
        agg = agg[agg["n"] >= MIN_ACTIVE]
        
        if agg.empty:
            continue
            
        ax1.plot(agg["global_episode"], agg["mean"], lw=2, label=name)
        ax1.fill_between(agg["global_episode"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], alpha=0.25)
        ax2.plot(agg["global_episode"], agg["n"], lw=2, label=name)
        
    if threshold is not None:
        ax1.axhline(threshold, color="red", linestyle="--", label="Solved")
        
    ax1.set_ylabel(ylabel)
    ax1.legend()
    
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Active seeds")
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_metric_vs_steps(files, metric, ylabel, filename, threshold=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    step_grid = np.linspace(0, 50000, 250)
    
    for name, file in files.items():
        if not os.path.exists(file):
            continue
            
        df = pd.read_csv(file)
        
        if metric not in df.columns:
            continue
            
        interpolated = []
        
        for seed, g in df.groupby("seed"):
            steps = g["total_steps"].values
            vals = g[metric].values
            interp_vals = np.interp(step_grid, steps, vals, right=np.nan)
            
            for s, v in zip(step_grid, interp_vals):
                if not np.isnan(v):
                    interpolated.append({"step": s, "val": v})
                    
        df_interp = pd.DataFrame(interpolated)
        
        if df_interp.empty:
            continue
            
        agg = df_interp.groupby("step").agg(
            mean=("val", "mean"),
            std=("val", "std"),
            n=("val", "count")
        ).reset_index()
        
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
        agg = agg[agg["n"] >= MIN_ACTIVE]
        
        if agg.empty:
            continue
            
        ax1.plot(agg["step"], agg["mean"], lw=2, label=name)
        ax1.fill_between(agg["step"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], alpha=0.25)
        ax2.plot(agg["step"], agg["n"], lw=2, label=name)
        
    if threshold is not None:
        ax1.axhline(threshold, color="red", linestyle="--", label="Solved")
        
    ax1.set_ylabel(ylabel)
    ax1.legend()
    
    ax2.set_xlabel("Total Steps")
    ax2.set_ylabel("Active seeds")
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_architecture_plots(arch):
    arch_dir = f"logs/{arch}"
    
    if not os.path.exists(arch_dir):
        return
        
    files = {}
    
    for opt in os.listdir(arch_dir):
        log_path = os.path.join(arch_dir, opt, "training_logs.csv")
        
        if os.path.isfile(log_path):
            files[opt.upper()] = log_path
            
    plot_dir = f"plots/{arch}"
    
    plot_metric_vs_episode(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_episode.png", SOLVE_THRESHOLD)
    plot_metric_vs_steps(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_steps.png", SOLVE_THRESHOLD)
    
    plot_metric_vs_episode(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_episode.png")
    plot_metric_vs_steps(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_steps.png")
    
    plot_metric_vs_episode(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_episode.png")
    plot_metric_vs_steps(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_steps.png")

if __name__ == "__main__":
    generate_architecture_plots("mlp")
    generate_architecture_plots("pqc")

    print("Plotting complete. Plots saved in plots/.")
