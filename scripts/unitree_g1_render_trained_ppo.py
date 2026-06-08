import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from unitree_g1_mujoco_ppo_train import render_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/g1_mujoco_ppo/ppo_g1_mujoco_final.zip")
    parser.add_argument("--unitree-root", default="unitree_rl_gym")
    parser.add_argument("--out", default="artifacts/g1_mujoco_ppo/ppo_g1_mujoco_rollout.gif")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    render_policy(
        model_path=args.model,
        unitree_root=args.unitree_root,
        out_path=args.out,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()

