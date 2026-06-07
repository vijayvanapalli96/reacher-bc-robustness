from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
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


def _site_id(model, name: str) -> int:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
        raise RuntimeError(f"MuJoCo site not found: {name}")
    return site


def expert_action(env, kp: float = 12.0, kd: float = 1.2) -> np.ndarray:
    """Simple Jacobian-transpose controller used to generate demonstrations."""
    unwrapped = env.unwrapped
    model, data = unwrapped.model, unwrapped.data
    fingertip = _site_id(model, "fingertip")
    target = _site_id(model, "target")

    err = data.site_xpos[target] - data.site_xpos[fingertip]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, fingertip)

    torque = kp * jacp[:, :2].T @ err - kd * data.qvel[:2]
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
    # Reacher observations end with fingertip-target x/y/z distance.
    return float(np.linalg.norm(obs[-3:]))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

