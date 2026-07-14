import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def get_best_hyperparameters(csv_path):
    df = pd.read_csv(csv_path)
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

def generate_text_report(tuning_csv, training_csv, output_txt, best_lr, best_damp=None):
    train_df = pd.read_csv(training_csv)
    
    with open(output_txt, 'w') as f:
        f.write("=== ANALYSIS REPORT ===\n\n")
        f.write("1. HYPERPARAMETER TUNING\n")
        f.write(f"Best Learning Rate Selected: {best_lr}\n")
        if best_damp and best_damp != 'N/A':
            f.write(f"Best Damping Selected: {best_damp}\n")
        f.write("\n")
        
        f.write("2. FINAL TRAINING RESULTS\n")
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

def generate_plots(training_csv, output_dir):
    df = pd.read_csv(training_csv)
    sns.set_theme(style="darkgrid")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='global_episode', y='train_mean_reward', label='Train Reward')
    sns.lineplot(data=df, x='global_episode', y='eval_mean_reward', label='Eval Reward')
    plt.title('Mean Reward vs Global Episode')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(output_dir, 'reward_vs_episode.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='total_steps', y='train_mean_reward')
    plt.title('Train Reward vs Total Steps')
    plt.ylabel('Train Reward')
    plt.xlabel('Total Steps')
    plt.savefig(os.path.join(output_dir, 'train_vs_steps.png'))
    plt.close()
    
    if 'grad_norm' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='global_episode', y='grad_norm')
        plt.title('Gradient Norm vs Global Episode')
        plt.ylabel('Gradient Norm')
        plt.xlabel('Episode')
        plt.savefig(os.path.join(output_dir, 'grad_norm_vs_episode.png'))
        plt.close()

def apply_natural_gradient(policy, batch_log_probs, damping):
    standard_grad = torch.cat([p.grad.clone().view(-1) for p in policy.parameters()])
    
    grads = []
    for log_prob in batch_log_probs:
        policy.zero_grad()
        log_prob.backward(retain_graph=True)
        g = torch.cat([p.grad.view(-1) for p in policy.parameters() if p.grad is not None])
        grads.append(g)
        
    grads = torch.stack(grads)
    FIM = (grads.T @ grads) / len(batch_log_probs)
    
    FIM += damping * torch.eye(FIM.size(0), device=FIM.device)
    
    eigenvalues = torch.linalg.eigvalsh(FIM)
    cond_num = (eigenvalues[-1] / (eigenvalues[0] + 1e-8)).item()
    
    FIM_inv = torch.linalg.inv(FIM)
    nat_grad = FIM_inv @ standard_grad
    
    idx = 0
    policy.zero_grad()
    for p in policy.parameters():
        length = p.numel()
        if p.grad is None:
            p.grad = nat_grad[idx:idx+length].view(p.shape).clone()
        else:
            p.grad.copy_(nat_grad[idx:idx+length].view(p.shape))
        idx += length
        
    return cond_num
