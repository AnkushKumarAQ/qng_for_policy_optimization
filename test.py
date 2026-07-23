import os
import argparse
import torch
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import gymnasium as gym
from config import ENV_NAME, HIDDEN_DIM, BOUNDS_PNP
from models.mlp import MLPPolicyNetwork
from models.pqc import PQCPolicyNetwork, create_qng_qnode


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, choices=["mlp", "pqc"], required=True)
    parser.add_argument("--opt", type=str, choices=["adam", "sgd", "ngd", "qng"], required=True)
    parser.add_argument("--approx", type=str, choices=["block-diag", "exact"], default="block-diag")
    parser.add_argument("--noise", type=str, choices=["analytical", "noisy"], default="analytical")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test_seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    run_sig = f"qng_{args.approx}_{args.noise}" if args.opt == "qng" else args.opt
    ext = "npy" if args.opt == "qng" else "pth"
    model_path = f"policies/{args.arch}/{run_sig}/policy_seed_{args.seed}.{ext}"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return
        
    print(f"Loading model from: {model_path}")
    render_mode = "human" if args.render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    state, _ = env.reset(seed=args.test_seed)
    
    total_reward = 0
    done = False
    step = 0
    
    if args.arch == "mlp":
        policy = MLPPolicyNetwork(hidden_dim=HIDDEN_DIM)
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        policy.eval()
    elif args.arch == "pqc" and args.opt != "qng":
        policy = PQCPolicyNetwork()
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        policy.eval()
    elif args.arch == "pqc" and args.opt == "qng":
        params = pnp.array(np.load(model_path), requires_grad=False)
        qnode = create_qng_qnode(noise_type=args.noise)
        
    with torch.no_grad():
        while not done:
            step += 1
            print(f"\rSurvived time step: {step}", end="", flush=True)
            
            if args.opt == "qng":
                s_scaled = pnp.clip(state / BOUNDS_PNP, -1.0, 1.0) * pnp.pi
                logits = pnp.stack(qnode(params, s_scaled))
                action = int(pnp.argmax(logits))
            else:
                state_t = torch.FloatTensor(state)
                action_probs = policy(state_t)
                action = torch.argmax(action_probs).item()
                
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
    print(f"\n\nTest Episode Finished!")
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    run_test()
