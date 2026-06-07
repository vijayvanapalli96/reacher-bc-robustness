import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from reacher_bc.core import BCPolicy, ensure_dir, policy_action, set_seed


def make_render_env(seed: int):
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/bc_policy.pt")
    parser.add_argument("--out", type=str, default="artifacts/reacher_policy_rollout.gif")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    policy = BCPolicy(checkpoint["obs_dim"], checkpoint["act_dim"]).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    env = make_render_env(args.seed)
    obs, _ = env.reset(seed=args.seed)
    obs_buffer = [obs.copy() for _ in range(args.latency_steps + 1)]
    frames = []
    total_return = 0.0
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)

        policy_obs = obs_buffer[0].copy()
        if args.obs_noise > 0:
            policy_obs += np.random.normal(0, args.obs_noise, size=policy_obs.shape)

        action = policy_action(
            policy,
            policy_obs,
            checkpoint["obs_mean"],
            checkpoint["obs_std"],
            device=device,
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_buffer.append(obs.copy())
        obs_buffer = obs_buffer[-(args.latency_steps + 1) :]
        total_return += reward
        done = terminated or truncated

    env.close()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    imageio.mimsave(out_path, frames, fps=args.fps)

    print(f"device: {device}")
    print(f"saved_gif: {out_path}")
    print(f"frames: {len(frames)}")
    print(f"episode_return: {total_return:.3f}")


if __name__ == "__main__":
    main()
