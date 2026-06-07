import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitree-root", type=str, default="unitree_rl_gym")
    parser.add_argument("--config", type=str, default="g1.yaml")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--out", type=str, default="artifacts/unitree_g1_mujoco_rollout.gif")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    unitree_root = Path(args.unitree_root).resolve()
    config_path = unitree_root / "deploy" / "deploy_mujoco" / "configs" / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Unitree config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = resolve_unitree_path(unitree_root, config["policy_path"])
    xml_path = resolve_unitree_path(unitree_root, config["xml_path"])

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    frames = []
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    total_steps = int(args.duration / simulation_dt)
    render_every = max(1, int(1.0 / (args.fps * simulation_dt)))

    for counter in range(total_steps):
        tau = pd_control(
            target_dof_pos,
            data.qpos[7:],
            kps,
            np.zeros_like(kds),
            data.qvel[6:],
            kds,
        )
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        if counter % control_decimation == 0:
            qj = data.qpos[7:]
            dqj = data.qvel[6:]
            quat = data.qpos[3:7]
            omega = data.qvel[3:6]

            qj_scaled = (qj - default_angles) * dof_pos_scale
            dqj_scaled = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega_scaled = omega * ang_vel_scale

            period = 0.8
            sim_time = counter * simulation_dt
            phase = sim_time % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega_scaled
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj_scaled
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj_scaled
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                [sin_phase, cos_phase]
            )

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        if counter % render_every == 0:
            renderer.update_scene(data)
            frames.append(renderer.render())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.fps)

    print(f"unitree_root: {unitree_root}")
    print(f"policy_path: {policy_path}")
    print(f"xml_path: {xml_path}")
    print(f"saved_gif: {out_path}")
    print(f"frames: {len(frames)}")
    print(f"duration: {args.duration}")


if __name__ == "__main__":
    main()

