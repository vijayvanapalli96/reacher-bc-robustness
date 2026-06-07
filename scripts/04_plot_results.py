import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reacher_bc.core import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="artifacts/eval_results.csv")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    results = pd.read_csv(args.results)

    plt.figure(figsize=(10, 4))
    plt.bar(results["condition"], results["return_mean"], yerr=results["return_std"], capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Episode return, higher is better")
    plt.title("Behavior Cloning Policy Robustness on MuJoCo Reacher")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return_plot = out_dir / "return_by_condition.png"
    plt.savefig(return_plot, dpi=160)

    plt.figure(figsize=(10, 4))
    plt.bar(results["condition"], results["success_rate_dist_lt_0.07"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.title("Success Rate Under Deployment Perturbations")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    success_plot = out_dir / "success_rate_by_condition.png"
    plt.savefig(success_plot, dpi=160)

    print(f"saved_plot: {return_plot}")
    print(f"saved_plot: {success_plot}")


if __name__ == "__main__":
    main()

