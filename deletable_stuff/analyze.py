import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. LOAD DATA
df = pd.read_csv("classical_fim_logs.csv")
SOLVE_THRESHOLD = 475.0

# 2. EXTRACT SOLVE METRICS PER SEED
results = []
solved_eps = []
solved_steps = []
solved_times = []

for seed, group in df.groupby('seed'):
    solved_rows = group[group['eval_mean_reward'] >= SOLVE_THRESHOLD]
    
    if not solved_rows.empty:
        # It solved! Grab the exact moment it crossed the threshold
        solve_row = solved_rows.iloc[0]
        status = "Solved"
        
        # Store for our mean/CI calculations
        solved_eps.append(solve_row['global_episode'])
        # solved_steps.append(solve_row['total_steps'])
        solved_times.append(solve_row['time_seconds'])
    else:
        # It failed! Grab its final state at 500 episodes
        solve_row = group.iloc[-1]
        status = "Failed"
        
    results.append({
        'Seed': str(seed),
        'Episodes': solve_row['global_episode'],
        # 'Steps': solve_row['total_steps'],
        'Time (s)': solve_row['time_seconds'],
        'Status': status
    })

results_df = pd.DataFrame(results)

# 3. HELPER FUNCTION FOR 95% CI (Using exact t-score)
def get_mean_ci(data):
    n = len(data)
    if n < 2: return np.mean(data), 0 # Safety catch
    mean = np.mean(data)
    se = stats.sem(data)
    t_score = stats.t.ppf(0.975, n - 1)
    ci = se * t_score
    return mean, ci

m_ep, ci_ep = get_mean_ci(solved_eps)
# m_step, ci_step = get_mean_ci(solved_steps)
m_time, ci_time = get_mean_ci(solved_times)

# 4. PLOT BAR CHARTS WITH MEAN & CI OVERLAYS
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

palette = {"Solved": "royalblue", "Failed": "crimson"}

# Plot 1: Episodes
sns.barplot(data=results_df, x='Seed', y='Episodes', hue='Status', ax=axes[0], palette=palette, dodge=False)
axes[0].axhline(m_ep, color='black', linestyle='--', linewidth=2, label=f'Mean: {m_ep:.1f}')
axes[0].axhspan(m_ep - ci_ep, m_ep + ci_ep, color='black', alpha=0.15, label=f'95% CI (±{ci_ep:.1f})')
axes[0].set_title('Episodes to Solve per Seed')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('Total Episodes')
axes[0].legend()

# # Plot 2: Steps (Circuit Evaluations)
# sns.barplot(data=results_df, x='Seed', y='Steps', hue='Status', ax=axes[1], palette=palette, dodge=False)
# axes[1].axhline(m_step, color='black', linestyle='--', linewidth=2, label=f'Mean: {m_step:.0f}')
# axes[1].axhspan(m_step - ci_step, m_step + ci_step, color='black', alpha=0.15, label=f'95% CI (±{ci_step:.0f})')
# axes[1].set_title('Total Steps (Circuit Evals) per Seed')
# axes[1].tick_params(axis='x', rotation=45)
# axes[1].set_ylabel('Total Steps')
# axes[1].legend()

# Plot 3: Wall-clock Time
sns.barplot(data=results_df, x='Seed', y='Time (s)', hue='Status', ax=axes[1], palette=palette, dodge=False)
axes[1].axhline(m_time, color='black', linestyle='--', linewidth=2, label=f'Mean: {m_time:.1f}s')
axes[1].axhspan(m_time - ci_time, m_time + ci_time, color='black', alpha=0.15, label=f'95% CI (±{ci_time:.1f}s)')
axes[1].set_title('Wall-Clock Time per Seed')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel('Seconds')
axes[1].legend()

plt.tight_layout()
plt.savefig("seed_performance_bars_with_ci.png", dpi=300)
print("Saved clean bar charts with CI to 'seed_performance_bars_with_ci.png'")
