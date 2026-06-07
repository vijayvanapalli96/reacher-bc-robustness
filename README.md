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
!mkdir -p artifacts
!python scripts/01_collect_demos.py --episodes 250 2>&1 | tee artifacts/01_collect_demos.log
```

This creates:

- `artifacts/demos.npz`

### Cell 4: Train Behavior Cloning

```python
!python scripts/02_train_bc.py --epochs 35 2>&1 | tee artifacts/02_train_bc.log
```

This creates:

- `artifacts/bc_policy.pt`
- `artifacts/training_loss.png`

### Cell 5: Evaluate Robustness

```python
!python scripts/03_evaluate_robustness.py --episodes 60 2>&1 | tee artifacts/03_evaluate_robustness.log
```

This creates:

- `artifacts/eval_results.csv`

### Cell 6: Plot Results

```python
!python scripts/04_plot_results.py 2>&1 | tee artifacts/04_plot_results.log
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

### Cell 8: Make A Pasteable Report

```python
!python scripts/05_make_report.py
```

Paste the output of this cell back to Codex.

### Optional Cell 9: Render The Robot Arm

```python
!python scripts/07_render_rollout.py --out artifacts/reacher_policy_rollout.gif
```

Display it:

```python
from IPython.display import Image, display
display(Image("artifacts/reacher_policy_rollout.gif"))
```

Render a failure-ish latency case:

```python
!python scripts/07_render_rollout.py --latency-steps 2 --out artifacts/reacher_latency_rollout.gif
display(Image("artifacts/reacher_latency_rollout.gif"))
```

### Optional Cell 10: Push The Report Back To GitHub

This lets Codex pull the report from GitHub and inspect it.

Create a fine-grained GitHub token with write access to this repo, then run:

```python
from getpass import getpass
import os

os.environ["GITHUB_TOKEN"] = getpass("GitHub token: ")
```

Then run:

```python
!python scripts/06_push_report.py
```

This pushes only logs, CSV results, and plots to a branch named `colab-report`.
It does not push the large demo/model files.

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
- `scripts/05_make_report.py`: print a compact report for sharing/debugging
- `scripts/06_push_report.py`: push report artifacts to the `colab-report` branch
- `scripts/07_render_rollout.py`: render the learned policy as a GIF
- `src/reacher_bc/core.py`: shared environment, expert, and policy code
- `requirements.txt`: Python package list
