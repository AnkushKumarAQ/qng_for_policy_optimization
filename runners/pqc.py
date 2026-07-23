import os
import time
from tqdm import tqdm
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.optim as optim
import gymnasium as gym

from config import ENV_NAME, GAMMA, MAX_EPISODE_BUDGET, BATCH_SIZE, SOLVE_THRESHOLD, EVAL_SEEDS, BOUNDS_PNP
from models.pqc import PQCPolicyNetwork, create_qng_qnode
from utils.rl import set_seed, env_log_probs_and_rewards, compute_reinforce_returns, evaluate_policy
from utils.ngd import apply_natural_gradient
from utils.qng import calculate_condition_number, evaluate_qng_policy


def run_pqc_seed_worker(seed, lr, damp, opt_type, policy_dir, position, lock, save_policy=False):
    torch.set_num_threads(1)
    tqdm.set_lock(lock)
    env = gym.make(ENV_NAME)

    set_seed(env, seed)
    policy = PQCPolicyNetwork()

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


def run_pqc_qng_seed_worker(seed, lr, damp, approx, noise, policy_dir, position, lock, save_policy=False):
    if approx == "exact":
        approx = None

    torch.set_num_threads(1)
    tqdm.set_lock(lock)
    env = gym.make(ENV_NAME)

    set_seed(env, seed)

    qnode = create_qng_qnode(noise_type=noise)
    params = pnp.array(np.random.uniform(-np.pi, np.pi, size=(2, 4, 2)), requires_grad=True)
    opt = qml.QNGOptimizer(stepsize=lr, approx=approx, lam=damp)

    global_episodes = 0
    total_steps = 0
    start_time = time.time()
    solved = False
    logs_to_write = []

    desc = f"LR: {lr:<5} | Damp: {damp:<5.0e} | Seed: {seed:<6}"
    pbar = tqdm(total=MAX_EPISODE_BUDGET, position=position, desc=desc, leave=False, ascii=True)

    while global_episodes < MAX_EPISODE_BUDGET:
        batch_states, batch_actions, batch_returns, batch_rewards_tracking = [], [], [], []
        episodes_to_play = min(BATCH_SIZE, MAX_EPISODE_BUDGET - global_episodes)

        for _ in range(episodes_to_play):
            state, _ = env.reset()
            states_ep, actions_ep, rewards_ep = [], [], []
            done = False

            while not done:
                s_scaled = pnp.clip(state / BOUNDS_PNP, -1.0, 1.0) * pnp.pi
                logits = pnp.stack(qnode(params, s_scaled))
                exp_l = pnp.exp(logits)
                probs = (exp_l / pnp.sum(exp_l)).unwrap()
                
                action = np.random.choice(2, p=probs)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                
                states_ep.append(state)
                actions_ep.append(action)
                rewards_ep.append(reward)
                state = next_state

            returns_ep = compute_reinforce_returns(rewards_ep, GAMMA).numpy()
            
            batch_states.extend(states_ep)
            batch_actions.extend(actions_ep)
            batch_returns.extend(returns_ep)
            batch_rewards_tracking.append(sum(rewards_ep))
            total_steps += len(rewards_ep)

        train_mean = sum(batch_rewards_tracking) / len(batch_rewards_tracking)

        def objective(p):
            loss = 0.0

            for s, a, G in zip(batch_states, batch_actions, batch_returns):
                s_scaled = pnp.clip(s / BOUNDS_PNP, -1.0, 1.0) * pnp.pi
                logits = pnp.stack(qnode(p, s_scaled))
                exp_l = pnp.exp(logits)
                probs = exp_l / pnp.sum(exp_l)
                loss = loss - pnp.log(probs[a]) * G

            return loss / len(batch_states)

        mt_fn_base = qml.metric_tensor(qnode, approx=approx)
        mt_cache = []

        def batch_metric_tensor(p):
            sample_size = min(len(batch_states), 64)
            sampled_indices = np.random.choice(len(batch_states), sample_size, replace=False)
            batch_metrics = [mt_fn_base(p, pnp.clip(batch_states[i] / BOUNDS_PNP, -1.0, 1.0) * pnp.pi) for i in sampled_indices]
            avg_mt = tuple(sum(m[i] for m in batch_metrics) / len(batch_metrics) for i in range(len(batch_metrics[0])))
            
            mt_cache.clear()
            mt_cache.append(avg_mt)
            
            return avg_mt

        try:
            grad_fn = qml.grad(objective)
            grad_norm = pnp.linalg.norm(grad_fn(params)).item()
        except Exception:
            grad_norm = 0.0

        params, loss = opt.step_and_cost(objective, params, metric_tensor_fn=batch_metric_tensor)
        
        cond_num = calculate_condition_number(mt_cache, damp)
        eval_mean = evaluate_qng_policy(env, qnode, params, EVAL_SEEDS, BOUNDS_PNP)
        elapsed = time.time() - start_time

        logs_to_write.append([seed, lr, damp, global_episodes + episodes_to_play, total_steps, train_mean, eval_mean, elapsed, grad_norm, cond_num])

        global_episodes += episodes_to_play
        pbar.update(episodes_to_play)

        if eval_mean >= SOLVE_THRESHOLD:
            solved = True
            break

    pbar.close()

    if save_policy and solved:
        model_path = os.path.join(policy_dir, f"policy_seed_{seed}.npy")
        np.save(model_path, params.unwrap())

    return logs_to_write, seed, lr, damp, solved, global_episodes, total_steps
