import gymnasium as gym
import torch
import torch.optim as optim
import time
import csv
import os

from model import ClassicalPolicyNetwork
from utils import env_log_probs_and_rewards, compute_reinforce_returns, set_seed, evaluate_policy
from seeds import TUNE_SEEDS, EVAL_SEEDS

# --- HYPERPARAMETERS ---
LEARNING_RATE_LIST = [0.1, 0.05, 0.01, 0.005, 0.001]
HIDDEN_DIM = 64
GAMMA = 0.99
MAX_EPISODE_BUDGET = 500 
BATCH_EPISODES = 10        
SOLVE_THRESHOLD = 475.0

CSV_FILENAME = "sgd_baseline_logs.csv"
file_exists = os.path.isfile(CSV_FILENAME)
with open(CSV_FILENAME, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", "lr", "global_episode", "train_mean_reward", "eval_mean_reward", "time_seconds"])

env = gym.make("CartPole-v1")
print("Starting Hyperparameter Tuning...")

for lr in LEARNING_RATE_LIST:
    print(f"\n{'='*40}\nTesting Learning Rate: {lr}\n{'='*40}")
    
    for seed in TUNE_SEEDS:
        print(f"\n--- Run with Seed: {seed} ---")
        set_seed(env, seed)
        
        policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
        optimizer = optim.SGD(policy.parameters(), lr=lr)
        
        global_episodes = 0
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
                
            train_mean = sum(batch_rewards_tracking) / len(batch_rewards_tracking)
            
            # --- 2. UPDATE BRAIN ---
            optimizer.zero_grad()
            policy_loss = []
            for log_prob, G in zip(batch_log_probs, batch_returns):
                policy_loss.append(-log_prob * G)
                
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
            
            # --- 3. EVALUATE & LOG ---
            # We evaluate against the 100 explicitly defined evaluation seeds
            eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
            elapsed_time = time.time() - start_time
            
            with open(CSV_FILENAME, mode='a', newline='') as f:
                csv.writer(f).writerow([seed, lr, global_episodes, train_mean, eval_mean, elapsed_time])
            
            print(f"Eps: {global_episodes:3d}/{MAX_EPISODE_BUDGET} | Train: {train_mean:5.1f} | Eval: {eval_mean:5.1f}")
            
            if eval_mean >= SOLVE_THRESHOLD:
                print(f">>> Solved in {global_episodes} episodes! Moving to next seed.")
                break
                
        if global_episodes >= MAX_EPISODE_BUDGET:
            print(f">>> Failed to solve within budget ({MAX_EPISODE_BUDGET} eps).")

env.close()
print("\nAll tuning complete! Check adam_baseline_logs.csv")
