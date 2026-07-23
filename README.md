# Classical & Quantum Policy Gradients

Reinforcement learning ablation study benchmarking Classical (MLP) and Quantum (PQC) Policy Networks on `CartPole-v1`. This repository establishes performance baselines using standard first-order optimizers (Adam, SGD) and explores second-order Natural Gradient Descent (NGD) and Quantum Natural Gradient (QNG).

## Prerequisites
* **Python:** 3.13.14
* **Dependencies:** Install via `pip install -r requirements.txt`

## Usage: Training

The training scripts automatically run the full pipeline: hyperparameter tuning, selection of the best hyperparameters, final training across multiple seeds, and text report generation.

### Classical Policy Network (MLP)
Run `train_mlp.py` specifying your target optimizer (`sgd`, `adam`, or `ngd`):

```bash
python train_mlp.py --opt sgd
python train_mlp.py --opt adam
python train_mlp.py --opt ngd

```

### Parameterized Quantum Circuit (PQC)
Run `train_pqc.py` specifying your target optimizer (`sgd`, `adam`, `ngd`, or `qng`). 

When using `qng`, you can also specify the metric tensor approximation (`block-diag` or `exact`) and noise simulation type (`analytical` or `noisy`):

```bash
python train_pqc.py --opt sgd
python train_pqc.py --opt adam
python train_pqc.py --opt ngd
python train_pqc.py --opt qng --approx block-diag --noise analytical
python train_pqc.py --opt qng --approx exact --noise analytical

```

## Usage: Analysis & Plotting

Once your training runs are complete, use the standalone utility scripts to evaluate the metrics across all architectures and optimizers.

* **Text Summary:** Run `python analyze.py` to parse all logs and generate a unified ASCII table (`analysis_summary.txt`) containing success rates, mean steps to solve, and gradient norms.
* **Visualizations:** Run `python plot.py` to generate aggregated performance graphs (rewards, condition numbers, gradient norms) saved dynamically to the `plots/` directory.

*Note: Logs (.csv, .txt), policies (.pth, .npy), and plots (.png) are automatically saved to dynamically generated `logs/`, `policies/`, and `plots/` folders structured symmetrically by architecture and optimizer.*
