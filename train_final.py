import gymnasium as gym
import torch
import torch.optim as optim
import time
import csv
import os

from model import ClassicalPolicyNetwork
from utils import env_log_probs_and_rewards, compute_reinforce_returns, set_seed, evaluate_policy
from seeds import FINAL_SEEDS, EVAL_SEEDS

# --- LOCKED HYPERPARAMETERS ---
OPTIMAL_LR = 0.005
HIDDEN_DIM = 64
GAMMA = 0.99
MAX_EPISODE_BUDGET = 500 
BATCH_EPISODES = 10        
SOLVE_THRESHOLD = 475.0

CSV_FILENAME = "classical_sgd_final_logs.csv"

# Initialize CSV with Workplan Metrics
with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "global_episode", "total_steps", "train_mean_reward", 
                     "eval_mean_reward", "grad_norm", "time_seconds"])

env = gym.make("CartPole-v1")
print(f"Starting Final Evaluation (20 Seeds) with LR: {OPTIMAL_LR}...")

for seed in FINAL_SEEDS:
    print(f"\n--- Training Seed: {seed} ---")
    set_seed(env, seed)
    
    policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
    optimizer = optim.SGD(policy.parameters(), lr=OPTIMAL_LR)
    
    global_episodes = 0
    total_steps = 0  # Proxy for "Circuit Evaluations"
    start_time = time.time()
    
    while global_episodes < MAX_EPISODE_BUDGET:
        batch_log_probs = []
        batch_returns = []
        batch_rewards_tracking = []
        
        episodes_to_play = min(BATCH_EPISODES, MAX_EPISODE_BUDGET - global_episodes)
        
        # --- 1. GATHER BATCH ---
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
        
        # --- 2. UPDATE BRAIN & GET GRAD NORM ---
        optimizer.zero_grad()
        policy_loss = []
        for log_prob, G in zip(batch_log_probs, batch_returns):
            policy_loss.append(-log_prob * G)
            
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        
        # Calculate Gradient Norm (Workplan Requirement)
        grad_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        optimizer.step()
        
        # --- 3. EVALUATE & LOG ---
        eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
        elapsed_time = time.time() - start_time
        
        with open(CSV_FILENAME, mode='a', newline='') as f:
            csv.writer(f).writerow([seed, global_episodes, total_steps, train_mean, 
                                    eval_mean, grad_norm, elapsed_time])
        
        print(f"Eps: {global_episodes:3d} | Steps: {total_steps:5d} | Eval: {eval_mean:5.1f} | Grad: {grad_norm:5.2f}")
        
        if eval_mean >= SOLVE_THRESHOLD:
            print(f">>> Solved! (Seed {seed} finished in {global_episodes} episodes)")
            break
            
env.close()
print("\nFinal training complete. Data saved to classical_final_logs.csv.")
