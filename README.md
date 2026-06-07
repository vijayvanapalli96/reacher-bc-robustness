# Reacher Behavior Cloning Robustness

Fast robotics-learning mini-project for Google Colab.

This project trains a behavior cloning policy on Gymnasium's MuJoCo `Reacher-v5` environment, then evaluates how the learned policy degrades under deployment-like perturbations:

- observation noise
- actuator noise
- control latency
- dynamics mismatch

The point is not to claim state-of-the-art RL. The point is to show a practical robotics instinct:

> Clean simulation performance is not enough. Learned policies need to be tested under the kinds of imperfections that show up on real robots.

## Run In Colab

1. Upload `reacher_bc_robustness_colab.ipynb` to Google Colab.
2. In Colab, choose `Runtime > Change runtime type`.
3. Use `T4 GPU` if available. CPU is also fine for this notebook.
4. Run all cells from top to bottom.

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
- `requirements.txt`: Python package list

