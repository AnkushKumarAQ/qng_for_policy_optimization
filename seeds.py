# 5 seeds for sweeping hyperparameter tuning
TUNE_SEEDS = [42, 1337, 2024, 777, 999]  

# 20 seeds for the final evaluation of the best learning rate
FINAL_SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
               110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

# 100 explicitly defined seeds for the 100-episode solve evaluations
EVAL_SEEDS = list(range(1000, 1100))
