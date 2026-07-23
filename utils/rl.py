import numpy as np
import torch
from torch.distributions import Categorical


def set_seed(env, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

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

@torch.no_grad()
def evaluate_policy(env, policy, eval_seeds):
    eval_rewards = []
    for seed in eval_seeds:
        state, _ = env.reset(seed=seed)
        _, rewards = env_log_probs_and_rewards(env, policy, state, inference_mode=True)
        eval_rewards.append(sum(rewards))
    return sum(eval_rewards) / len(eval_rewards)
    