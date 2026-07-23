import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import SOLVE_THRESHOLD


MIN_ACTIVE = 5

COLOR_MAP = {
    "MLP - SGD": "#00BFFF",
    "MLP - ADAM": "#FF8C00",
    "MLP - NGD": "#32CD32",
    "PQC - SGD": "#FF0033",
    "PQC - ADAM": "#8A2BE2",
    "PQC - NGD": "#FF00FF",
    "PQC - QNG_BLOCK-DIAG_ANALYTICAL": "#00FFFF",
    "PQC - QNG_EXACT_ANALYTICAL": "#FFD700",
    "SGD": "#00BFFF",
    "ADAM": "#FF8C00",
    "NGD": "#32CD32",
    "QNG_BLOCK-DIAG_ANALYTICAL": "#00FFFF",
    "QNG_EXACT_ANALYTICAL": "#FFD700"
}

ORDER_PRIORITY = [
    "MLP - SGD",
    "MLP - ADAM",
    "MLP - NGD",
    "PQC - SGD",
    "PQC - ADAM",
    "PQC - NGD",
    "PQC - QNG_BLOCK-DIAG_ANALYTICAL",
    "PQC - QNG_EXACT_ANALYTICAL",
    "SGD",
    "ADAM",
    "NGD",
    "QNG_BLOCK-DIAG_ANALYTICAL",
    "QNG_EXACT_ANALYTICAL"
]


def get_order_key(name):
    if name in ORDER_PRIORITY:
        return ORDER_PRIORITY.index(name)
    return len(ORDER_PRIORITY)

def plot_metric_vs_episode(files, metric, ylabel, filename, title, threshold=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    sorted_files = sorted(files.items(), key=lambda x: get_order_key(x[0]))
    
    for name, file in sorted_files:
        if not os.path.exists(file):
            continue
            
        df = pd.read_csv(file)
        
        if metric not in df.columns:
            continue
            
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])
        
        if df.empty:
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
            
        color = COLOR_MAP.get(name, None)
        ax1.plot(agg["global_episode"], agg["mean"], lw=2, label=name, color=color)
        ax1.fill_between(agg["global_episode"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], alpha=0.25, color=color)
        ax2.plot(agg["global_episode"], agg["n"], lw=2, label=name, color=color)
        
    if threshold is not None:
        ax1.axhline(threshold, color="red", linestyle="--", label="Solved")
        
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Active seeds")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_metric_vs_steps(files, metric, ylabel, filename, title, threshold=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    step_grid = np.linspace(0, 50000, 250)
    
    sorted_files = sorted(files.items(), key=lambda x: get_order_key(x[0]))
    
    for name, file in sorted_files:
        if not os.path.exists(file):
            continue
            
        df = pd.read_csv(file)
        
        if metric not in df.columns:
            continue
            
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric, "total_steps"])
        
        interpolated = []
        
        for seed, g in df.groupby("seed"):
            steps = g["total_steps"].values
            vals = g[metric].values
            
            if len(steps) == 0:
                continue
                
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
            
        color = COLOR_MAP.get(name, None)
        ax1.plot(agg["step"], agg["mean"], lw=2, label=name, color=color)
        ax1.fill_between(agg["step"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], alpha=0.25, color=color)
        ax2.plot(agg["step"], agg["n"], lw=2, label=name, color=color)
        
    if threshold is not None:
        ax1.axhline(threshold, color="red", linestyle="--", label="Solved")
        
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_xlabel("Total Steps")
    ax2.set_ylabel("Active seeds")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
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
    
    plot_metric_vs_episode(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_episode.png", f"{arch.upper()} - Evaluation Reward vs Episode", SOLVE_THRESHOLD)
    plot_metric_vs_steps(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_steps.png", f"{arch.upper()} - Evaluation Reward vs Steps", SOLVE_THRESHOLD)
    
    plot_metric_vs_episode(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_episode.png", f"{arch.upper()} - Gradient Norm vs Episode")
    plot_metric_vs_steps(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_steps.png", f"{arch.upper()} - Gradient Norm vs Steps")
    
    plot_metric_vs_episode(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_episode.png", f"{arch.upper()} - Condition Number vs Episode")
    plot_metric_vs_steps(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_steps.png", f"{arch.upper()} - Condition Number vs Steps")

def generate_compiled_plots():
    base_dir = "logs"
    
    if not os.path.exists(base_dir):
        return
        
    files = {}
    
    for arch in ["mlp", "pqc"]:
        arch_dir = os.path.join(base_dir, arch)
        
        if not os.path.exists(arch_dir):
            continue
            
        for opt in os.listdir(arch_dir):
            log_path = os.path.join(arch_dir, opt, "training_logs.csv")
            
            if os.path.isfile(log_path):
                label = f"{arch.upper()} - {opt.upper()}"
                files[label] = log_path
                
    if not files:
        return
        
    plot_dir = "plots/compiled"
    
    plot_metric_vs_episode(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_episode.png", "Compiled - Evaluation Reward vs Episode", SOLVE_THRESHOLD)
    plot_metric_vs_steps(files, "eval_mean_reward", "Evaluation reward", f"{plot_dir}/reward_vs_steps.png", "Compiled - Evaluation Reward vs Steps", SOLVE_THRESHOLD)
    
    plot_metric_vs_episode(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_episode.png", "Compiled - Gradient Norm vs Episode")
    plot_metric_vs_steps(files, "grad_norm", "Gradient Norm", f"{plot_dir}/grad_norm_vs_steps.png", "Compiled - Gradient Norm vs Steps")
    
    plot_metric_vs_episode(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_episode.png", "Compiled - Condition Number vs Episode")
    plot_metric_vs_steps(files, "condition_number", "Condition Number", f"{plot_dir}/condition_vs_steps.png", "Compiled - Condition Number vs Steps")

if __name__ == "__main__":
    generate_architecture_plots("mlp")
    generate_architecture_plots("pqc")
    generate_compiled_plots()
    
    print("Plotting complete. Plots saved in plots/.")
