# Classical Policy Gradient Baselines

Reinforcement learning ablation study training a Classical Policy Network (MLP) on `CartPole-v1`. This repository establishes performance baselines using standard first-order optimizers (Adam, SGD) and explores second-order Natural Policy Gradients (FIM).

## Prerequisites
* **Python:** 3.13.14
* **Dependencies:** `torch`, `gymnasium`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## Usage
The `train_classical.py` script automatically runs the full pipeline: hyperparameter tuning, selection of hyperparameters, final 20-seed training, and report generation.

Run the script by specifying your target optimizer using the `--opt` flag:

```bash
# Train classical model with SGD
python train_classical.py --opt sgd

# Train classical model with Adam
python train_classical.py --opt adam

# Train classical model with FIM
python train_classical.py --opt fim

```

*Note: Logs (.csv, .txt, .png) and policies (.pth) are automatically saved to dynamically generated `logs/` and `policies/` folders.*
