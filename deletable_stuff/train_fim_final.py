import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import time
import csv

from model import ClassicalPolicyNetwork
from utils import env_log_probs_and_rewards, compute_reinforce_returns, set_seed, evaluate_policy
from seeds import FINAL_SEEDS, EVAL_SEEDS
FINAL_SEEDS = [10, 20, 30, 40, 50]

# --- HYPERPARAMETERS ---
LEARNING_RATES = [0.5]
FIM_DAMPING = 1e-2  # Required to make the matrix invertible
HIDDEN_DIM = 64
GAMMA = 0.99
MAX_EPISODE_BUDGET = 500
BATCH_EPISODES = 10
SOLVE_THRESHOLD = 475.0

CSV_FILENAME = "classical_fim_logs.csv"

with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["lr", "seed", "global_episode", "train_mean_reward", "eval_mean_reward", "time_seconds"])

env = gym.make("CartPole-v1")

def get_flat_grads(log_prob):
    """Calculates gradients for a single step and flattens them into a 1D vector."""
    policy.zero_grad()
    log_prob.backward(retain_graph=True)
    grads = []
    for param in policy.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)

for lr in LEARNING_RATES:
    for seed in FINAL_SEEDS:
        print(f"\n--- Tuning LR: {lr} | Seed: {seed} ---")
        set_seed(env, seed)
        
        policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
        # We don't use a PyTorch optimizer here because we are doing the FIM math manually!
        
        global_episodes = 0
        start_time = time.time()
        
        while global_episodes < MAX_EPISODE_BUDGET:
            batch_log_probs = []
            batch_returns = []
            batch_rewards = []
            
            episodes_to_play = min(BATCH_EPISODES, MAX_EPISODE_BUDGET - global_episodes)
            
            # --- 1. GATHER BATCH ---
            for _ in range(episodes_to_play):
                state, _ = env.reset()
                log_probs, rewards = env_log_probs_and_rewards(env, policy, state, inference_mode=False)
                returns = compute_reinforce_returns(rewards, GAMMA)
                
                batch_log_probs.extend(log_probs)
                batch_returns.extend(returns)
                batch_rewards.append(sum(rewards))
                
                global_episodes += 1
                
            train_mean = sum(batch_rewards) / len(batch_rewards)
            
            # --- 2. CALCULATE EMPIRICAL FISHER (The heavy lifting) ---
            num_params = sum(p.numel() for p in policy.parameters())
            FIM = torch.zeros((num_params, num_params))
            
            # For each step, get the gradient of the log prob to build the FIM
            step_grads = []
            for lp in batch_log_probs:
                g = get_flat_grads(lp)
                step_grads.append(g)
                # Outer product: g * g^T
                FIM += torch.outer(g, g) 
            
            FIM /= len(batch_log_probs)
            # Add damping to diagonal to ensure we can invert it
            FIM += torch.eye(num_params) * FIM_DAMPING 
            
            # --- 3. STANDARD POLICY GRADIENT ---
            policy.zero_grad()
            policy_loss = []
            for log_prob, G in zip(batch_log_probs, batch_returns):
                policy_loss.append(-log_prob * G)
            loss = torch.stack(policy_loss).sum() / len(batch_log_probs)
            
            # Get the standard gradient (flattened)
            loss.backward()
            standard_grad = []
            for param in policy.parameters():
                if param.grad is not None:
                    standard_grad.append(param.grad.view(-1))
            standard_grad = torch.cat(standard_grad)
            
            # --- 4. NATURAL GRADIENT UPDATE (FIM^-1 * grad) ---
            FIM_inv = torch.linalg.inv(FIM)
            natural_grad = torch.matmul(FIM_inv, standard_grad)
            
            # Apply the update manually
            idx = 0
            with torch.no_grad():
                for param in policy.parameters():
                    numel = param.numel()
                    # Subtract because it's gradient descent
                    param -= lr * natural_grad[idx:idx+numel].view(param.shape)
                    idx += numel
            
            # --- 5. EVALUATE ---
            eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
            elapsed = time.time() - start_time
            
            with open(CSV_FILENAME, mode='a', newline='') as f:
                csv.writer(f).writerow([lr, seed, global_episodes, train_mean, eval_mean, elapsed])
            
            print(f"Eps: {global_episodes:3d} | Train: {train_mean:5.1f} | Eval: {eval_mean:5.1f}")
            
            if eval_mean >= SOLVE_THRESHOLD:
                print(f">>> Solved in {global_episodes} episodes!")
                break
            