import time
import torch
import gymnasium as gym

from config import ENV_NAME, HIDDEN_DIM
from models.classical import ClassicalPolicyNetwork


def test_policy(model_path):
    env = gym.make(ENV_NAME, render_mode="human")
    
    policy = ClassicalPolicyNetwork(hidden_dim=HIDDEN_DIM)
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    print(f"--- Starting Render Test ---")
    print(f"Model: {model_path}")
    
    while not done:
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = policy(state_t)
            action = torch.argmax(action_probs).item()
            
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        if step % 10 == 0:
            print(f"Step: {step:3d} | Action Taken: {action} | Cumulative Reward: {total_reward:.1f}")
        time.sleep(0.02)
        
    print(f">>> Test Finished! Total Steps Survived: {step} | Final Reward: {total_reward}")
    env.close()

test_policy("./policies/classical/fim/policy_seed_2000.pth")
