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


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def expert_action(env, kp: float = 18.0, kd: float = 1.6) -> np.ndarray:
    """Simple inverse-kinematics PD controller used for demonstrations.

    Gymnasium Reacher-v5 exposes a compact 10D observation:
    [cos(q0), cos(q1), sin(q0), sin(q1), target_x, target_y, qvel0, qvel1,
    fingertip_minus_target_x, fingertip_minus_target_y].

    We solve a tiny 2-link planar inverse-kinematics problem to pick desired
    joint angles, then use joint-space PD control. This is intentionally simple:
    it gives behavior cloning a decent teacher without requiring RL training.
    """
    obs = env.unwrapped._get_obs()
    q0 = np.arctan2(obs[2], obs[0])
    q1 = np.arctan2(obs[3], obs[1])
    qvel = obs[6:8]
    target = obs[4:6]

    l1, l2 = 0.1, 0.1
    radius = np.linalg.norm(target)
    max_radius = l1 + l2 - 1e-4
    if radius > max_radius:
        target = target * (max_radius / radius)
        radius = max_radius

    cos_q1 = (radius**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q1 = np.clip(cos_q1, -1.0, 1.0)
    q1_options = np.array([np.arccos(cos_q1), -np.arccos(cos_q1)])
    q0_options = []
    for q1_des in q1_options:
        q0_des = np.arctan2(target[1], target[0]) - np.arctan2(
            l2 * np.sin(q1_des), l1 + l2 * np.cos(q1_des)
        )
        q0_options.append(q0_des)

    candidates = np.stack([q0_options, q1_options], axis=1)
    current = np.array([q0, q1])
    errors = _wrap_angle(candidates - current)
    best = candidates[np.argmin(np.linalg.norm(errors, axis=1))]

    joint_error = _wrap_angle(best - current)
    torque = kp * joint_error - kd * qvel
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
