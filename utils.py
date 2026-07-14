import os
import csv
import random
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from config import SOLVE_THRESHOLD


def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

def init_logger(filepath, header=["seed", "lr", "global_episode", "total_steps", "train_mean_reward", "eval_mean_reward", "time_seconds"]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not os.path.isfile(filepath):
        with open(filepath, mode='a', newline='') as f:
            csv.writer(f).writerow(header)
            
def env_log_probs_and_rewards(env, policy, initial_state, inference_mode=False):
    state = initial_state
    log_probs, rewards = [], []
    done = False
    
    while not done:
        state_t = torch.FloatTensor(state)
        action_probs = policy(state_t)
        dist = Categorical(action_probs)
        
        action = torch.argmax(action_probs) if inference_mode else dist.sample()
        log_probs.append(None if inference_mode else dist.log_prob(action))

        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
        
    return log_probs, rewards

def compute_reinforce_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
        
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns

def evaluate_policy(env, policy, eval_seeds):
    eval_rewards = []
    for seed in eval_seeds:
        state, _ = env.reset(seed=seed)
        _, rewards = env_log_probs_and_rewards(env, policy, state, inference_mode=True)
        eval_rewards.append(sum(rewards))
    return sum(eval_rewards) / len(eval_rewards)
    
def get_best_learning_rate(csv_path):
    df = pd.read_csv(csv_path)
    best_lr = None
    best_success_rate = -1
    best_speed = float('inf')
    
    for lr, group in df.groupby('lr'):
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
            
    return best_lr

def generate_text_report(tuning_csv, training_csv, output_txt, best_lr):
    tune_df = pd.read_csv(tuning_csv)
    train_df = pd.read_csv(training_csv)
    
    with open(output_txt, 'w') as f:
        f.write("=== ANALYSIS REPORT ===\n\n")
        f.write(f"1. HYPERPARAMETER TUNING\n")
        f.write(f"Best Learning Rate Selected: {best_lr}\n\n")
        
        f.write("2. FINAL TRAINING RESULTS\n")
        solves = 0
        total_steps_list = []
        unique_seeds = train_df['seed'].unique()
        num_seeds = len(unique_seeds)
        
        for seed, seed_group in train_df.groupby('seed'):
            max_eval = seed_group['eval_mean_reward'].max()
            if max_eval >= SOLVE_THRESHOLD:
                solves += 1
                total_steps_list.append(seed_group['total_steps'].max())
                
        success_rate = (solves / num_seeds) * 100 if num_seeds > 0 else 0
        mean_steps = sum(total_steps_list) / len(total_steps_list) if total_steps_list else 0
        
        f.write(f"Overall Success Rate: {success_rate:.1f}% ({solves}/{num_seeds} seeds)\n")
        f.write(f"Mean Steps to Solve (for successful seeds): {mean_steps:.1f}\n")
