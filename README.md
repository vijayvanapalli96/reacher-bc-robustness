# Reacher Behavior Cloning Robustness

Fast robotics-learning mini-project for Google Colab.

This project trains a behavior cloning policy on Gymnasium's MuJoCo `Reacher-v5` environment, then evaluates how the learned policy degrades under deployment-like perturbations:

- observation noise
- actuator noise
- control latency
- dynamics mismatch

The point is not to claim state-of-the-art RL. The point is to show a practical robotics instinct:

> Clean simulation performance is not enough. Learned policies need to be tested under the kinds of imperfections that show up on real robots.

## Run In Colab With Scripts

Open a fresh Google Colab notebook and run these as separate cells.

### Cell 1: Clone The Repo

```python
!git clone https://github.com/vijayvanapalli96/reacher-bc-robustness.git
%cd reacher-bc-robustness
```

### Cell 2: Install Dependencies

```python
!pip -q install -r requirements.txt
```

### Cell 3: Collect Expert Demonstrations

```python
!python scripts/01_collect_demos.py --episodes 250
```

This creates:

- `artifacts/demos.npz`

### Cell 4: Train Behavior Cloning

```python
!python scripts/02_train_bc.py --epochs 35
```

This creates:

- `artifacts/bc_policy.pt`
- `artifacts/training_loss.png`

### Cell 5: Evaluate Robustness

```python
!python scripts/03_evaluate_robustness.py --episodes 60
```

This creates:

- `artifacts/eval_results.csv`

### Cell 6: Plot Results

```python
!python scripts/04_plot_results.py
```

This creates:

- `artifacts/return_by_condition.png`
- `artifacts/success_rate_by_condition.png`

### Cell 7: Show Results In Colab

```python
import pandas as pd
from IPython.display import display, Image

display(pd.read_csv("artifacts/eval_results.csv"))
display(Image("artifacts/training_loss.png"))
display(Image("artifacts/return_by_condition.png"))
display(Image("artifacts/success_rate_by_condition.png"))
```

## Run The Original Single Notebook

You can also upload `reacher_bc_robustness_colab.ipynb` to Google Colab and run it from top to bottom. The script flow above is better for reporting progress cell by cell.

## GPU Recommendation

Use this order:

1. `T4 GPU`: best default for free/standard Colab. More than enough.
2. `L4 GPU`: great if Colab offers it, but unnecessary.
3. `A100`: overkill for this project.
4. `CPU`: acceptable. The notebook is small and should still run, just slower.

For the interview, it is totally fine to say you used a T4 or CPU. The substance is the experiment design, not the hardware.

## What The Notebook Does

```text
Reacher-v5 simulation
       ↓
Jacobian-transpose expert controller generates demonstrations
       ↓
Behavior cloning model learns observation -> action
       ↓
Policy is evaluated under clean and perturbed conditions
       ↓
Results table and robustness plots are produced
```

## How To Talk About It

Short version:

> I built a small behavior cloning setup in MuJoCo Reacher, but the part I cared about most was robustness. I tested how the cloned policy degrades under observation noise, actuator noise, latency, and dynamics mismatch because those are the kinds of problems that matter when learned policies meet real hardware.

Good discussion question:

> At Index, when learned policies fail on real hardware, is the bottleneck more often perception and representation, action prediction, data coverage, or hardware variability?

## Files

- `reacher_bc_robustness_colab.ipynb`: main Colab notebook
- `scripts/01_collect_demos.py`: collect expert demonstrations
- `scripts/02_train_bc.py`: train the behavior cloning policy
- `scripts/03_evaluate_robustness.py`: evaluate clean and perturbed conditions
- `scripts/04_plot_results.py`: create result plots
- `src/reacher_bc/core.py`: shared environment, expert, and policy code
- `requirements.txt`: Python package list
