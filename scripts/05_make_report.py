import argparse
from pathlib import Path

import pandas as pd

from reacher_bc.core import ensure_dir


def read_text(path: Path) -> str:
    if not path.exists():
        return f"_Missing: `{path}`_"
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--out", type=str, default="artifacts/report.md")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    ensure_dir(artifacts)

    lines = [
        "# Reacher BC Robustness Run Report",
        "",
        "## Demo Collection Log",
        "",
        "```text",
        read_text(artifacts / "01_collect_demos.log"),
        "```",
        "",
        "## Training Log",
        "",
        "```text",
        read_text(artifacts / "02_train_bc.log"),
        "```",
        "",
        "## Evaluation Log",
        "",
        "```text",
        read_text(artifacts / "03_evaluate_robustness.log"),
        "```",
        "",
    ]

    results_path = artifacts / "eval_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        lines.extend(
            [
                "## Evaluation CSV",
                "",
                results.to_markdown(index=False),
                "",
                "## Quick Interpretation Helpers",
                "",
            ]
        )
        clean = results.loc[results["condition"] == "clean"]
        if not clean.empty:
            clean_success = float(clean.iloc[0]["success_rate_dist_lt_0.07"])
            clean_return = float(clean.iloc[0]["return_mean"])
            lines.append(f"- Clean success rate: `{clean_success:.3f}`")
            lines.append(f"- Clean mean return: `{clean_return:.3f}`")

        best = results.sort_values("return_mean", ascending=False).iloc[0]
        worst = results.sort_values("return_mean", ascending=True).iloc[0]
        lines.append(f"- Best condition by return: `{best['condition']}`")
        lines.append(f"- Worst condition by return: `{worst['condition']}`")
        lines.append("")
    else:
        lines.extend(["## Evaluation CSV", "", "_Missing `artifacts/eval_results.csv`._", ""])

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()

