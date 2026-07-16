import os
import csv
import time
import argparse
import torch
import gymnasium as gym
import torch.optim as optim

from config import ENV_NAME, HIDDEN_DIM, GAMMA, MAX_EPISODE_BUDGET, BATCH_SIZE, SOLVE_THRESHOLD, LEARNING_RATE_LIST, FIM_DAMPING_LIST, TUNE_SEEDS, TRAIN_SEEDS, EVAL_SEEDS
from models.classical import ClassicalPolicyNetwork
from utils import set_seed, init_logger, env_log_probs_and_rewards, compute_reinforce_returns, evaluate_policy, get_best_hyperparameters, generate_text_report, generate_plots, apply_natural_gradient


# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Train Classical Policy Network")
parser.add_argument('--opt', type=str, choices=['adam', 'sgd', 'fim'], required=True, help="Optimizer to use: 'adam', 'sgd', or 'fim'")
args = parser.parse_args()

# --- PATH SETUP ---
LOGS_DIR = f"logs/classical/{args.opt}/"
POLICY_DIR = f"policies/classical/{args.opt}/"
os.makedirs(POLICY_DIR, exist_ok=True)

tuning_logs_path = os.path.join(LOGS_DIR, "tuning_logs.csv")
training_logs_path = os.path.join(LOGS_DIR, "training_logs.csv")
report_path = os.path.join(LOGS_DIR, "analysis_report.txt")

log_headers = ["seed", "lr", "damping", "global_episode", "total_steps", "train_mean_reward", "eval_mean_reward", "time_seconds", "grad_norm", "fim_condition_number"]
init_logger(tuning_logs_path, header=log_headers)
init_logger(training_logs_path, header=log_headers)

env = gym.make(ENV_NAME)

# --- HYPERPARAMETER GRID SETUP ---
if args.opt == 'fim':
    tuning_grid = [(lr, damp) for lr in LEARNING_RATE_LIST for damp in FIM_DAMPING_LIST]
else:
    tuning_grid = [(lr, 'N/A') for lr in LEARNING_RATE_LIST]

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================
print("\n" + "="*50)
print(f"HYPERPARAMETER TUNING ({args.opt.upper()})")
print("="*50)

for lr, damp in tuning_grid:
    print(f"\nTesting Learning Rate: {lr} | Damping: {damp}")
    
    for seed in TUNE_SEEDS:
        print(f"  --- Tuning Seed: {seed} ---")
        set_seed(env, seed)
        
        policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
        
        if args.opt == 'adam':
            optimizer = optim.Adam(policy.parameters(), lr=lr)
        elif args.opt in ['sgd', 'fim']:
            optimizer = optim.SGD(policy.parameters(), lr=lr)
        
        global_episodes = 0
        total_steps = 0
        start_time = time.time()
        
        while global_episodes < MAX_EPISODE_BUDGET:
            batch_log_probs, batch_returns, batch_rewards_tracking = [], [], []
            episodes_to_play = min(BATCH_SIZE, MAX_EPISODE_BUDGET - global_episodes)
            
            for _ in range(episodes_to_play):
                state, _ = env.reset()
                log_probs, rewards = env_log_probs_and_rewards(env, policy, state, inference_mode=False)
                returns = compute_reinforce_returns(rewards, GAMMA)
                
                batch_log_probs.extend(log_probs)
                batch_returns.extend(returns)
                batch_rewards_tracking.append(sum(rewards))
                
                global_episodes += 1
                total_steps += len(rewards)
                
            train_mean = sum(batch_rewards_tracking) / len(batch_rewards_tracking)
            
            optimizer.zero_grad()
            policy_loss = [-lp * G for lp, G in zip(batch_log_probs, batch_returns)]
            loss = torch.stack(policy_loss).sum()
            
            loss.backward(retain_graph=(args.opt == 'fim'))
            
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in policy.parameters() if p.grad is not None]), 2).item()
            
            cond_num = 'N/A'
            if args.opt == 'fim':
                cond_num = apply_natural_gradient(policy, batch_log_probs, damping=damp)
                
            optimizer.step()
            
            eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
            elapsed = time.time() - start_time
            
            with open(tuning_logs_path, mode='a', newline='') as f:
                csv.writer(f).writerow([seed, lr, damp, global_episodes, total_steps, train_mean, eval_mean, elapsed, grad_norm, cond_num])
            
            if eval_mean >= SOLVE_THRESHOLD:
                print(f"      >>> Solved! (Eps: {global_episodes} | Steps: {total_steps})")
                break
                
        if eval_mean < SOLVE_THRESHOLD:
            print(f"      >>> Couldn't solve within {MAX_EPISODE_BUDGET} episodes.")

# ==========================================
# HYPERPARAMETER SELECTION
# ==========================================
print("\n" + "="*50)
print("HYPERPARAMETER SELECTION")
print("="*50)

best_lr, best_damp = get_best_hyperparameters(tuning_logs_path)
print(f">>> Selected Best Learning Rate: {best_lr}")
if args.opt == 'fim':
    print(f">>> Selected Best Damping: {best_damp}")

# ==========================================
# TRAINING
# ==========================================
print("\n" + "="*50)
print(f"TRAINING FINAL POLICIES ({len(TRAIN_SEEDS)} SEEDS) - {args.opt.upper()}")
print("="*50)

for seed in TRAIN_SEEDS:
    print(f"\n--- Training Final Seed: {seed} ---")
    set_seed(env, seed)
    
    policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
    
    if args.opt == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=best_lr)
    elif args.opt in ['sgd', 'fim']:
        optimizer = optim.SGD(policy.parameters(), lr=best_lr)
    
    global_episodes = 0
    total_steps = 0
    start_time = time.time()
    
    while global_episodes < MAX_EPISODE_BUDGET:
        batch_log_probs, batch_returns, batch_rewards_tracking = [], [], []
        episodes_to_play = min(BATCH_SIZE, MAX_EPISODE_BUDGET - global_episodes)
        
        for _ in range(episodes_to_play):
            state, _ = env.reset()
            log_probs, rewards = env_log_probs_and_rewards(env, policy, state, inference_mode=False)
            returns = compute_reinforce_returns(rewards, GAMMA)
            
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)
            batch_rewards_tracking.append(sum(rewards))
            
            global_episodes += 1
            total_steps += len(rewards)
            
        train_mean = sum(batch_rewards_tracking) / len(batch_rewards_tracking)
        
        optimizer.zero_grad()
        policy_loss = [-lp * G for lp, G in zip(batch_log_probs, batch_returns)]
        loss = torch.stack(policy_loss).sum()
        
        loss.backward(retain_graph=(args.opt == 'fim'))
        
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in policy.parameters() if p.grad is not None]), 2).item()
        
        cond_num = 'N/A'
        if args.opt == 'fim':
            cond_num = apply_natural_gradient(policy, batch_log_probs, damping=best_damp)
            
        optimizer.step()
        
        eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
        elapsed = time.time() - start_time
        
        with open(training_logs_path, mode='a', newline='') as f:
            csv.writer(f).writerow([seed, best_lr, best_damp, global_episodes, total_steps, train_mean, eval_mean, elapsed, grad_norm, cond_num])
        
        print(f"Eps: {global_episodes:3d}/{MAX_EPISODE_BUDGET} | Train: {train_mean:5.1f} | Eval: {eval_mean:5.1f}")
        
        if eval_mean >= SOLVE_THRESHOLD:
            print(f">>> Solved in {global_episodes} episodes! Saving policy...")
            model_path = os.path.join(POLICY_DIR, f"policy_seed_{seed}.pth")
            torch.save(policy.state_dict(), model_path)
            break
            
    if eval_mean < SOLVE_THRESHOLD:
        print(f">>> Couldn't solve within {MAX_EPISODE_BUDGET} episodes.")

# ==========================================
# ANALYSIS
# ==========================================
print("\n" + "="*50)
print("ANALYSIS")
print("="*50)

generate_text_report(tuning_logs_path, training_logs_path, report_path, best_lr, best_damp)
generate_plots(training_logs_path, LOGS_DIR)
print(f"Done! Full report and plots saved to: {LOGS_DIR}")

env.close()
