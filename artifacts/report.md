# Reacher BC Robustness Run Report

## Demo Collection Log

```text
_Missing: `artifacts/01_collect_demos.log`_
```

## Training Log

```text
_Missing: `artifacts/02_train_bc.log`_
```

## Evaluation Log

```text
_Missing: `artifacts/03_evaluate_robustness.log`_
```

## Evaluation CSV

| condition           |   return_mean |   return_std |   final_distance_mean |   success_rate_dist_lt_0.07 |
|:--------------------|--------------:|-------------:|----------------------:|----------------------------:|
| clean               |      -81.5745 |      6.30445 |             0.0376282 |                    0.883333 |
| obs_noise_0.02      |      -76.8028 |      6.40181 |             0.0440452 |                    0.833333 |
| obs_noise_0.05      |      -74.8847 |      6.3357  |             0.0609702 |                    0.683333 |
| actuator_noise_0.10 |      -76.2378 |      5.72689 |             0.0388858 |                    0.85     |
| latency_2_steps     |      -88.7019 |      6.0934  |             0.0718082 |                    0.533333 |
| dynamics_mismatch   |      -81.3665 |      6.36875 |             0.0372231 |                    0.883333 |

## Quick Interpretation Helpers

- Clean success rate: `0.883`
- Clean mean return: `-81.575`
- Best condition by return: `obs_noise_0.05`
- Worst condition by return: `latency_2_steps`
