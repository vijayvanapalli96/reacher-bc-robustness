from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class EvalCondition:
    name: str
    obs_noise: float = 0.0
    action_noise: float = 0.0
    latency_steps: int = 0
    damping_scale: float = 1.0
    mass_scale: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int, damping_scale: float = 1.0, mass_scale: float = 1.0):
    env = gym.make("Reacher-v5")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.unwrapped.model.dof_damping[:] *= damping_scale
    env.unwrapped.model.body_mass[:] *= mass_scale
    return env


def expert_action(env, kp: float = 12.0, kd: float = 1.2) -> np.ndarray:
    """Simple Jacobian-transpose controller used to generate demonstrations.

    Gymnasium Reacher-v5 exposes a compact 10D observation:
    [cos(q0), cos(q1), sin(q0), sin(q1), target_x, target_y, qvel0, qvel1,
    fingertip_minus_target_x, fingertip_minus_target_y].

    We use that observation instead of MuJoCo site names, which changed across
    environment versions.
    """
    obs = env.unwrapped._get_obs()
    q0 = np.arctan2(obs[2], obs[0])
    q1 = np.arctan2(obs[3], obs[1])
    qvel = obs[6:8]

    # Reacher's default links are 0.1m each. The Jacobian maps joint velocity to
    # fingertip velocity for a planar 2-link arm.
    l1, l2 = 0.1, 0.1
    s0, c0 = np.sin(q0), np.cos(q0)
    s01, c01 = np.sin(q0 + q1), np.cos(q0 + q1)
    jac = np.array(
        [
            [-l1 * s0 - l2 * s01, -l2 * s01],
            [l1 * c0 + l2 * c01, l2 * c01],
        ],
        dtype=np.float32,
    )

    # Observation stores fingertip - target, so target - fingertip is negative.
    err = -obs[-2:]
    torque = kp * jac.T @ err - kd * qvel
    return np.clip(torque, env.action_space.low, env.action_space.high).astype(np.float32)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def normalize_obs(obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    return ((obs - obs_mean) / obs_std).astype(np.float32)


@torch.no_grad()
def policy_action(policy, obs, obs_mean, obs_std, device: str) -> np.ndarray:
    x = normalize_obs(obs[None, :], obs_mean, obs_std)
    action = policy(torch.tensor(x, device=device)).cpu().numpy()[0]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def final_distance_from_obs(obs: np.ndarray) -> float:
    # Reacher-v5 observations end with fingertip-target x/y distance.
    return float(np.linalg.norm(obs[-2:]))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
