# import matplotlib.pyplot as plt

# # Your dataset
# data = [
#     [None, None, None, None, None, None, None, None, None, None], 
#     [None, None, None, 19, None, None, None, None, None, 51], 
#     [290, 60, 203, 183, 69, None, 145, 144, 92, 79], 
#     [161, 267, 143, 173, 94, 74, 56, 148, 262, 277], 
#     [None, 256, 340, 369, 377, 268, 315, 291, 421, 355]
# ]

# # Learning rates for the legend
# learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]

# # X-axis ticks (1 to 10)
# x = list(range(1, 11))

# plt.figure(figsize=(10, 6))

# # Plotting each list
# for i, y_values in enumerate(data):
#     # marker='o' ensures that single isolated points show up as dots
#     plt.plot(x, y_values, marker='o', label=f"lr: {learning_rates[i]}")

# # Formatting the plot
# plt.title("Hyperparameter Search", fontsize=14)
# plt.xlabel("Index", fontsize=12)
# plt.ylabel("Episodes to Solve", fontsize=12)
# plt.xticks(x)  # Force exactly 10 ticks on the x-axis
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Learning Rates")

# # Display the plot
# plt.show()

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Let's simulate 20 "episodes to solve" results for lr=0.005
# (In your real code, this will be your actual list of 20 results)
np.random.seed(42)
episodes_to_solve = np.random.normal(loc=150, scale=40, size=20).astype(int).tolist()

# 2. Calculate Statistics
n = len(episodes_to_solve)
mean_val = np.mean(episodes_to_solve)

# Calculate the Standard Error of the Mean (SEM)
sem = stats.sem(episodes_to_solve) 

# Calculate the 95% CI bounds using the t-distribution
ci_bounds = stats.t.interval(0.95, df=n-1, loc=mean_val, scale=sem)

# The "half-width" is the stringency value after the ± sign
ci_half_width = mean_val - ci_bounds[0]

print(f"Raw Data (20 seeds): {episodes_to_solve}")
print(f"Result to Report: {mean_val:.2f} ± {ci_half_width:.2f} episodes")

# 3. Plotting
plt.figure(figsize=(6, 6), dpi=100)

# Plot the individual 20 seeds as a scatter plot with slight horizontal jitter so they don't stack
jitter = np.random.uniform(-0.05, 0.05, size=n)
plt.scatter(np.zeros(n) + jitter, episodes_to_solve, color='gray', alpha=0.6, label='Individual Seeds')

# Plot the Mean with 95% CI Error Bars
plt.errorbar(x=0, y=mean_val, yerr=ci_half_width, fmt='o', color='red', 
             markersize=10, capsize=8, elinewidth=3, label='Mean ± 95% CI')

# Formatting the Plot
plt.title("Agent Performance Evaluation (20 Seeds)", fontsize=14, fontweight='bold')
plt.ylabel("Episodes to Solve", fontsize=12)
plt.xticks([0], ["lr: 0.005"]) # Rename the X-axis tick
plt.xlim(-0.5, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
