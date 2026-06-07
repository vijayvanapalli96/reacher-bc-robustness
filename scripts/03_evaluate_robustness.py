import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from reacher_bc.core import (
    BCPolicy,
    EvalCondition,
    ensure_dir,
    final_distance_from_obs,
    make_env,
    policy_action,
    set_seed,
)


CONDITIONS = [
    EvalCondition("clean"),
    EvalCondition("obs_noise_0.02", obs_noise=0.02),
    EvalCondition("obs_noise_0.05", obs_noise=0.05),
    EvalCondition("actuator_noise_0.10", action_noise=0.10),
    EvalCondition("latency_2_steps", latency_steps=2),
    EvalCondition("dynamics_mismatch", damping_scale=1.8, mass_scale=1.25),
]


def evaluate(policy, obs_mean, obs_std, condition: EvalCondition, episodes: int, seed: int, device: str):
    env = make_env(
        seed=seed,
        damping_scale=condition.damping_scale,
        mass_scale=condition.mass_scale,
    )
    returns, final_dists = [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_buffer = [obs.copy() for _ in range(condition.latency_steps + 1)]
        done = False
        total = 0.0

        while not done:
            policy_obs = obs_buffer[0].copy()
            if condition.obs_noise > 0:
                policy_obs += np.random.normal(0, condition.obs_noise, size=policy_obs.shape)

            action = policy_action(policy, policy_obs, obs_mean, obs_std, device=device)
            if condition.action_noise > 0:
                action += np.random.normal(0, condition.action_noise, size=action.shape)

            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs_buffer.append(obs.copy())
            obs_buffer = obs_buffer[-(condition.latency_steps + 1):]
            done = terminated or truncated
            total += reward

        returns.append(total)
        final_dists.append(final_distance_from_obs(obs))

    env.close()
    final_dists = np.array(final_dists)
    return {
        "condition": condition.name,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "final_distance_mean": float(np.mean(final_dists)),
        "success_rate_dist_lt_0.07": float(np.mean(final_dists < 0.07)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=str, default="artifacts/eval_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    policy = BCPolicy(checkpoint["obs_dim"], checkpoint["act_dim"]).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    rows = []
    for condition in tqdm(CONDITIONS, desc="Evaluating robustness"):
        rows.append(
            evaluate(
                policy=policy,
                obs_mean=checkpoint["obs_mean"],
                obs_std=checkpoint["obs_std"],
                condition=condition,
                episodes=args.episodes,
                seed=args.seed,
                device=device,
            )
        )

    results = pd.DataFrame(rows)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index=False)

    print(f"device: {device}")
    print(f"saved_results: {out_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

