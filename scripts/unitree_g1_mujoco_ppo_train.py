import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_unitree_path(unitree_root: Path, value: str) -> str:
    return value.replace("{LEGGED_GYM_ROOT_DIR}", str(unitree_root))


class UnitreeG1MujocoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        unitree_root="unitree_rl_gym",
        config="g1.yaml",
        episode_seconds=10.0,
        render_mode=None,
        seed=None,
    ):
        super().__init__()
        self.unitree_root = Path(unitree_root).resolve()
        self.config_path = self.unitree_root / "deploy" / "deploy_mujoco" / "configs" / config
        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.xml_path = resolve_unitree_path(self.unitree_root, self.cfg["xml_path"])
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg["simulation_dt"]

        self.sim_dt = self.cfg["simulation_dt"]
        self.decimation = self.cfg["control_decimation"]
        self.policy_dt = self.sim_dt * self.decimation
        self.max_policy_steps = int(episode_seconds / self.policy_dt)
        self.render_mode = render_mode
        self.renderer = None

        self.kps = np.array(self.cfg["kps"], dtype=np.float32)
        self.kds = np.array(self.cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(self.cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = self.cfg["ang_vel_scale"]
        self.dof_pos_scale = self.cfg["dof_pos_scale"]
        self.dof_vel_scale = self.cfg["dof_vel_scale"]
        self.action_scale = self.cfg["action_scale"]
        self.cmd_scale = np.array(self.cfg["cmd_scale"], dtype=np.float32)
        self.num_actions = self.cfg["num_actions"]
        self.num_obs = self.cfg["num_obs"]
        self.cmd = np.array(self.cfg["cmd_init"], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.policy_step = 0

    def _get_obs(self):
        qj = self.data.qpos[7:]
        dqj = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        qj_scaled = (qj - self.default_angles) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.ang_vel_scale

        period = 0.8
        sim_time = self.policy_step * self.policy_dt
        phase = sim_time % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[:3] = omega_scaled
        obs[3:6] = gravity_orientation
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9 : 9 + self.num_actions] = qj_scaled
        obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj_scaled
        obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
        obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array(
            [sin_phase, cos_phase]
        )
        return obs

    def _is_fallen(self):
        height = self.data.qpos[2]
        gravity = get_gravity_orientation(self.data.qpos[3:7])
        return bool(height < 0.45 or gravity[2] < 0.25)

    def _reward(self, action):
        forward_vel = self.data.qvel[0]
        lateral_vel = self.data.qvel[1]
        yaw_rate = self.data.qvel[5]
        height = self.data.qpos[2]
        gravity = get_gravity_orientation(self.data.qpos[3:7])

        target_vx, target_vy, target_yaw = self.cmd
        tracking = np.exp(-2.0 * (forward_vel - target_vx) ** 2)
        lateral_penalty = lateral_vel**2
        yaw_penalty = (yaw_rate - target_yaw) ** 2
        upright = max(0.0, gravity[2])
        height_penalty = (height - 0.78) ** 2
        action_penalty = float(np.mean(np.square(action)))
        joint_vel_penalty = float(np.mean(np.square(self.data.qvel[6:])))

        reward = (
            1.0 * tracking
            + 0.8 * upright
            + 0.2
            - 0.5 * lateral_penalty
            - 0.2 * yaw_penalty
            - 3.0 * height_penalty
            - 0.02 * action_penalty
            - 0.001 * joint_vel_penalty
        )
        if self._is_fallen():
            reward -= 5.0
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.8
        self.data.qpos[7:] = self.default_angles + self.np_random.normal(
            0.0, 0.03, size=self.num_actions
        )
        self.data.qvel[:] = self.np_random.normal(0.0, 0.02, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.policy_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

        for _ in range(self.decimation):
            tau = pd_control(
                self.target_dof_pos,
                self.data.qpos[7:],
                self.kps,
                np.zeros_like(self.kds),
                self.data.qvel[6:],
                self.kds,
            )
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.policy_step += 1
        reward = self._reward(self.action)
        terminated = self._is_fallen()
        truncated = self.policy_step >= self.max_policy_steps
        info = {
            "height": float(self.data.qpos[2]),
            "forward_vel": float(self.data.qvel[0]),
            "target_forward_vel": float(self.cmd[0]),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=540, width=960)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def render_policy(model_path, unitree_root, out_path, duration=10.0):
    env = UnitreeG1MujocoEnv(unitree_root=unitree_root, episode_seconds=duration)
    model = PPO.load(model_path, env=None, device="cpu")
    obs, _ = env.reset(seed=123)
    frames = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        done = terminated or truncated
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    env.close()
    print(f"saved_gif: {out_path}")
    print(f"frames: {len(frames)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitree-root", type=str, default="unitree_rl_gym")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--out-dir", type=str, default="artifacts/g1_mujoco_ppo")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        check_env(UnitreeG1MujocoEnv(unitree_root=args.unitree_root, episode_seconds=2.0))

    def make_env():
        return UnitreeG1MujocoEnv(
            unitree_root=args.unitree_root, episode_seconds=args.episode_seconds
        )

    env = VecMonitor(DummyVecEnv([make_env for _ in range(args.num_envs)]))
    callback = CheckpointCallback(
        save_freq=max(10_000 // args.num_envs, 1),
        save_path=str(out_dir),
        name_prefix="ppo_g1_mujoco",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(out_dir / "tb"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)

    final_model = out_dir / "ppo_g1_mujoco_final.zip"
    model.save(final_model)
    print(f"saved_model: {final_model}")
    render_policy(final_model, args.unitree_root, out_dir / "ppo_g1_mujoco_rollout.gif")


if __name__ == "__main__":
    main()

