import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def calculate_condition_number(mt_cache, damp):
    try:
        all_eigvals = []
        
        for block in mt_cache[0]:
            block_np = np.array(qml.math.unwrap(block))
            
            if block_np.ndim >= 2:
                dim = int(np.sqrt(block_np.size))
                block_mat = block_np.reshape(dim, dim)
                all_eigvals.extend(np.linalg.eigvalsh(block_mat))
            elif block_np.ndim in [0, 1]:
                all_eigvals.append(float(block_np.flatten()[0]))
                
        damped_eigvals = [abs(e) + damp for e in all_eigvals]
        cond_num = max(damped_eigvals) / (min(damped_eigvals) + 1e-12)
        
        return cond_num
    except Exception:
        return 'N/A'

def evaluate_qng_policy(env, qnode, params, eval_seeds, bounds):
    eval_rewards = []
    
    for e_seed in eval_seeds:
        s, _ = env.reset(seed=e_seed)
        ep_reward = 0
        d = False
        
        while not d:
            s_scaled = pnp.clip(s / bounds, -1.0, 1.0) * pnp.pi
            logits = pnp.stack(qnode(params, s_scaled))
            action = int(pnp.argmax(logits))
            s, r, term, trunc, _ = env.step(action)
            ep_reward += r
            d = term or trunc
            
        eval_rewards.append(ep_reward)
        
    return sum(eval_rewards) / len(eval_rewards)
    