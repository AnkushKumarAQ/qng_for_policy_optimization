# Classical Policy Gradient Baselines

Reinforcement learning ablation study training a Classical Policy Network (MLP) on `CartPole-v1`. This repository establishes performance baselines using standard first-order optimizers (Adam, SGD) and explores second-order Natural Policy Gradients (FIM).

## Prerequisites
* **Python:** 3.13.14
* **Dependencies:** `torch`, `gymnasium`, `pandas`, `numpy`

## Usage
The `train.py` script automatically runs the full pipeline: hyperparameter tuning, automagical best-LR selection, final 20-seed training, and report generation.

Run the script by specifying your target optimizer using the `--opt` flag:

```bash
# Train with Adam
python train.py --opt adam

# Train with SGD
python train.py --opt sgd
```

*Note: Logs (.csv, .txt) and trained policies (.pth) are automatically saved to dynamically generated `logs/` and `trained_policies/` folders.*
