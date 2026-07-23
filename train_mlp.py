import os
import time
import argparse
import multiprocessing
import concurrent.futures
import csv
import torch
from config import MLP_SGD_LR_LIST, MLP_ADAM_LR_LIST, MLP_NGD_LR_LIST, MLP_NGD_DAMP_LIST, TUNE_SEEDS, TRAIN_SEEDS, MAX_EPISODE_BUDGET
from utils.reporting import init_logger, get_best_hyperparameters, generate_text_report
from runners.mlp import run_mlp_seed_worker


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if __name__ == "__main__":
    script_start_time = time.time()

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Train MLP Policy Network")
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd', 'ngd'], required=True, help="Optimizer to use")
    parser.add_argument('--workers', type=int, default=None, help="Number of CPU cores to use")
    args = parser.parse_args()

    num_workers = args.workers if args.workers is not None else os.cpu_count()

    LOGS_DIR = f"logs/mlp/{args.opt}/"
    POLICY_DIR = f"policies/mlp/{args.opt}/"
    os.makedirs(POLICY_DIR, exist_ok=True)

    tuning_logs_path = os.path.join(LOGS_DIR, "tuning_logs.csv")
    training_logs_path = os.path.join(LOGS_DIR, "training_logs.csv")
    report_path = os.path.join(LOGS_DIR, "analysis_report.txt")

    init_logger(tuning_logs_path)
    init_logger(training_logs_path)

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    print("\n" + "=" * 50)
    print(f"HYPERPARAMETER TUNING ({args.opt.upper()})")
    print("=" * 50)

    if args.opt == 'ngd':
        tuning_tasks = [(seed, lr, damp) for lr in MLP_NGD_LR_LIST for damp in MLP_NGD_DAMP_LIST for seed in TUNE_SEEDS]
    elif args.opt == 'adam':
        tuning_tasks = [(seed, lr, 'N/A') for lr in MLP_ADAM_LR_LIST for seed in TUNE_SEEDS]
    elif args.opt == 'sgd':
        tuning_tasks = [(seed, lr, 'N/A') for lr in MLP_SGD_LR_LIST for seed in TUNE_SEEDS]

    print(f"\nSubmitting all {len(tuning_tasks)} tuning tasks across {num_workers} cores...\n")

    tuning_summaries = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_mlp_seed_worker, task[0], task[1], task[2], args.opt, POLICY_DIR, i % num_workers, lock, False) for i, task in enumerate(tuning_tasks)]
        
        for future in concurrent.futures.as_completed(futures):
            logs, seed, lr, damp, solved, eps, steps = future.result()
            
            with open(tuning_logs_path, mode='a', newline='') as f:
                csv.writer(f).writerows(logs)
                
            tuning_summaries.append((seed, lr, damp, solved, eps, steps))

    tuning_summaries.sort(key=lambda x: (-x[1], x[0]))

    print("--- TUNING RESULTS ---")

    for seed, lr, damp, solved, eps, steps in tuning_summaries:
        damp_str = f" | Damp: {damp:<5.0e}" if args.opt == 'ngd' else ""
        
        if solved:
            print(f"  --- LR: {lr:<5}{damp_str} | Seed: {seed:<6} >>> Solved! (Eps: {eps} | Steps: {steps})")
        else:
            print(f"  --- LR: {lr:<5}{damp_str} | Seed: {seed:<6} >>> Couldn't solve within {MAX_EPISODE_BUDGET} episodes.")

    print("\n" + "=" * 50)
    print("HYPERPARAMETER SELECTION")
    print("=" * 50)

    best_lr, best_damp = get_best_hyperparameters(tuning_logs_path)
    print(f">>> Selected Best Learning Rate: {best_lr}")
    
    if args.opt == 'ngd':
        print(f">>> Selected Best Damping: {best_damp}")

    print("\n" + "=" * 50)
    print(f"TRAINING POLICIES ({len(TRAIN_SEEDS)} SEEDS) - {args.opt.upper()}")
    print("=" * 50)

    print(f"\nSubmitting all {len(TRAIN_SEEDS)} training tasks across {num_workers} cores...\n")

    training_summaries = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_mlp_seed_worker, seed, best_lr, best_damp, args.opt, POLICY_DIR, i % num_workers, lock, True) for i, seed in enumerate(TRAIN_SEEDS)]
        
        for future in concurrent.futures.as_completed(futures):
            logs, seed, lr, damp, solved, eps, steps = future.result()
            
            with open(training_logs_path, mode='a', newline='') as f:
                csv.writer(f).writerows(logs)
                
            training_summaries.append((seed, solved, eps))

    training_summaries.sort(key=lambda x: x[0])

    print("--- TRAINING RESULTS ---")

    for seed, solved, eps in training_summaries:
        if solved:
            print(f">>> Seed {seed:<6} solved in {eps} episodes! Policy saved.")
        else:
            print(f">>> Seed {seed:<6} couldn't solve within {MAX_EPISODE_BUDGET} episodes.")

    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    generate_text_report(tuning_logs_path, training_logs_path, report_path, best_lr, best_damp, script_start_time)
    print(f"Done! Report saved to: {LOGS_DIR}")
    