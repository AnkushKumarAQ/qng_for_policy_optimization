import os
import csv
import time
import pandas as pd

from config import SOLVE_THRESHOLD


def init_logger(filepath, header=["seed", "lr", "damping", "global_episode", "total_steps", "train_mean_reward", "eval_mean_reward", "time_seconds", "grad_norm", "condition_number"]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

def get_best_hyperparameters(csv_path):
    df = pd.read_csv(csv_path, keep_default_na=False)
    best_lr = None
    best_damp = None
    best_success_rate = -1
    best_speed = float('inf')
    
    for (lr, damp), group in df.groupby(['lr', 'damping'], dropna=False):
        solves = 0
        episodes_to_solve = []
        
        for seed, seed_group in group.groupby('seed'):
            if seed_group['eval_mean_reward'].max() >= SOLVE_THRESHOLD:
                solves += 1
                episodes_to_solve.append(seed_group['global_episode'].max())
                
        success_rate = solves / len(group['seed'].unique())
        mean_speed = sum(episodes_to_solve) / len(episodes_to_solve) if episodes_to_solve else float('inf')
        
        if success_rate > best_success_rate or (success_rate == best_success_rate and mean_speed < best_speed):
            best_success_rate = success_rate
            best_speed = mean_speed
            best_lr = lr
            best_damp = damp
            
    return best_lr, best_damp

def generate_text_report(tuning_logs, training_logs, report_path, best_lr, best_damp, script_start_time=None):
    train_df = pd.read_csv(training_logs)
    
    with open(report_path, 'w') as f:
        f.write("=== ANALYSIS REPORT ===\n\n")
        f.write("1. HYPERPARAMETER TUNING\n")
        f.write(f"Best Learning Rate Selected: {best_lr}\n")
        
        if best_damp and best_damp != 'N/A' and str(best_damp).lower() != 'nan':
            f.write(f"Best Damping Selected: {best_damp}\n")
            
        f.write("\n")
        f.write("2. TRAINING RESULTS\n")
        
        unique_seeds = train_df['seed'].unique()
        num_seeds = len(unique_seeds)
        solves = 0
        stats = {'steps': [], 'eps': [], 'times': [], 'grads': []}
        
        for seed, seed_group in train_df.groupby('seed'):
            max_eval = seed_group['eval_mean_reward'].max()
            
            if max_eval >= SOLVE_THRESHOLD:
                solves += 1
                stats['steps'].append(seed_group['total_steps'].max())
                stats['eps'].append(seed_group['global_episode'].max())
                stats['times'].append(seed_group['time_seconds'].max())
                
                if 'grad_norm' in seed_group.columns:
                    stats['grads'].append(seed_group['grad_norm'].mean())
                    
        success_rate = (solves / num_seeds) * 100 if num_seeds > 0 else 0
        f.write(f"Overall Success Rate: {success_rate:.1f}% ({solves}/{num_seeds} seeds)\n\n")
        
        if solves > 0:
            f.write("--- Metrics for Successful Seeds ---\n")
            f.write(f"Mean Steps to Solve: {sum(stats['steps'])/len(stats['steps']):.1f}\n")
            f.write(f"Mean Episodes to Solve: {sum(stats['eps'])/len(stats['eps']):.1f}\n")
            f.write(f"Mean Time to Solve (s): {sum(stats['times'])/len(stats['times']):.2f}\n")
            
            if stats['grads']:
                f.write(f"Mean Gradient Norm: {sum(stats['grads'])/len(stats['grads']):.4f}\n")
                
        if script_start_time is not None:
            total_script_time = time.time() - script_start_time
            f.write(f"\n--- EXECUTION TIME ---\n")
            f.write(f"Total Execution Time: {total_script_time:.2f} seconds\n")
