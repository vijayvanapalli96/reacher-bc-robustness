import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPORT_FILES = [
    "artifacts/report.md",
    "artifacts/eval_results.csv",
    "artifacts/01_collect_demos.log",
    "artifacts/02_train_bc.log",
    "artifacts/03_evaluate_robustness.log",
    "artifacts/04_plot_results.log",
    "artifacts/training_loss.png",
    "artifacts/return_by_condition.png",
    "artifacts/success_rate_by_condition.png",
]


def run(cmd, check=True, capture=False):
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode != 0:
        if capture:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def github_slug_from_remote(remote_url: str) -> str:
    remote_url = remote_url.strip()
    patterns = [
        r"github\.com[:/](?P<slug>[^/]+/[^/.]+)(?:\.git)?$",
        r"github\.com/(?P<slug>[^/]+/[^/.]+)(?:\.git)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, remote_url)
        if match:
            return match.group("slug")
    raise SystemExit(f"Could not parse GitHub repo from remote URL: {remote_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="colab-report")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--message", default="Update Colab run report")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"Missing ${args.token_env}. In Colab, set it with getpass before running this script."
        )

    # Ensure report.md reflects the latest CSV/log files.
    run([sys.executable, "scripts/05_make_report.py"], check=True)

    remote = run(["git", "remote", "get-url", "origin"], capture=True).stdout.strip()
    slug = github_slug_from_remote(remote)
    authed_remote = f"https://x-access-token:{token}@github.com/{slug}.git"

    run(["git", "config", "user.name", "Colab Report Bot"])
    run(["git", "config", "user.email", "colab-report@example.com"])
    run(["git", "checkout", "-B", args.branch])

    existing_files = [path for path in REPORT_FILES if Path(path).exists()]
    if not existing_files:
        raise SystemExit("No report artifacts found. Run the experiment first.")

    run(["git", "add", *existing_files])
    status = run(["git", "status", "--short"], capture=True).stdout.strip()
    if status:
        run(["git", "commit", "-m", args.message])
    else:
        print("No report changes to commit.")

    run(["git", "push", "--force", authed_remote, f"HEAD:{args.branch}"])
    print(f"Pushed report branch: https://github.com/{slug}/tree/{args.branch}")
    print(f"Report URL: https://github.com/{slug}/blob/{args.branch}/artifacts/report.md")


if __name__ == "__main__":
    main()

