import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from reacher_bc.core import ensure_dir, expert_action, make_env, set_seed


def collect_demos(num_episodes: int, seed: int):
    env = make_env(seed=seed)
    obs_list, act_list, returns = [], [], []

    for ep in tqdm(range(num_episodes), desc="Collecting demos"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0

        while not done:
            action = expert_action(env)
            obs_list.append(obs.astype(np.float32))
            act_list.append(action.astype(np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        returns.append(total)

    env.close()
    return np.array(obs_list), np.array(act_list), np.array(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="artifacts/demos.npz")
    args = parser.parse_args()

    set_seed(args.seed)
    x, y, returns = collect_demos(args.episodes, args.seed)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    np.savez_compressed(out_path, observations=x, actions=y, expert_returns=returns)

    print(f"Saved demos to {out_path}")
    print(f"observations: {x.shape}")
    print(f"actions: {y.shape}")
    print(f"expert_return_mean: {returns.mean():.3f}")
    print(f"expert_return_std: {returns.std():.3f}")


if __name__ == "__main__":
    main()

