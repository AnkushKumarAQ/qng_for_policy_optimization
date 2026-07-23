import os
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import gymnasium as gym

from config import ENV_NAME, HIDDEN_DIM, GAMMA, MAX_EPISODE_BUDGET, BATCH_SIZE, SOLVE_THRESHOLD, EVAL_SEEDS
from models.mlp import MLPPolicyNetwork
from utils.rl import set_seed, env_log_probs_and_rewards, compute_reinforce_returns, evaluate_policy
from utils.ngd import apply_natural_gradient


def run_mlp_seed_worker(seed, lr, damp, opt_type, policy_dir, position, lock, save_policy=False):
    torch.set_num_threads(1)
    tqdm.set_lock(lock)
    env = gym.make(ENV_NAME)

    set_seed(env, seed)
    policy = MLPPolicyNetwork(hidden_dim=HIDDEN_DIM)

    if opt_type == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=lr)
    elif opt_type in ['sgd', 'ngd']:
        optimizer = optim.SGD(policy.parameters(), lr=lr)

    global_episodes = 0
    total_steps = 0
    start_time = time.time()
    solved = False
    logs_to_write = []

    if opt_type == 'ngd':
        desc = f"LR: {lr:<5} | Damp: {damp:<5.0e} | Seed: {seed:<6}"
    else:
        desc = f"LR: {lr:<5} | Seed: {seed:<6}"

    pbar = tqdm(total=MAX_EPISODE_BUDGET, position=position, desc=desc, leave=False, ascii=True)

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
            total_steps += len(rewards)

        train_mean = sum(batch_rewards_tracking) / len(batch_rewards_tracking)

        optimizer.zero_grad()
        policy_loss = [-lp * G for lp, G in zip(batch_log_probs, batch_returns)]
        loss = torch.stack(policy_loss).sum()
        loss.backward(retain_graph=(opt_type == 'ngd'))

        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in policy.parameters() if p.grad is not None]), 2).item()

        cond_num = 'N/A'
        if opt_type == 'ngd':
            cond_num = apply_natural_gradient(policy, batch_log_probs, damping=damp)

        optimizer.step()

        eval_mean = evaluate_policy(env, policy, EVAL_SEEDS)
        elapsed = time.time() - start_time

        logs_to_write.append([seed, lr, damp, global_episodes + episodes_to_play, total_steps, train_mean, eval_mean, elapsed, grad_norm, cond_num])

        global_episodes += episodes_to_play
        pbar.update(episodes_to_play)

        if eval_mean >= SOLVE_THRESHOLD:
            solved = True
            break

    pbar.close()

    if save_policy and solved:
        model_path = os.path.join(policy_dir, f"policy_seed_{seed}.pth")
        torch.save(policy.state_dict(), model_path)

    return logs_to_write, seed, lr, damp, solved, global_episodes, total_steps
    