import argparse
from pathlib import Path

print("importing numpy/mujoco/yaml...", flush=True)
import mujoco
import numpy as np
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
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print("loading config...", flush=True)
    unitree_root = Path(args.unitree_root).resolve()
    config_path = unitree_root / "deploy" / "deploy_mujoco" / "configs" / args.config
    print(f"config_path: {config_path}", flush=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = resolve_unitree_path(unitree_root, config["xml_path"])
    print(f"xml_path: {xml_path}", flush=True)

    print("loading MuJoCo model...", flush=True)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = config["simulation_dt"]

    print("initializing G1 state...", flush=True)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    data.qpos[2] = 0.8
    data.qpos[7:] = default_angles
    mujoco.mj_forward(model, data)

    print("stepping physics...", flush=True)
    for step in range(args.steps):
        tau = pd_control(
            default_angles,
            data.qpos[7:],
            kps,
            np.zeros_like(kds),
            data.qvel[6:],
            kds,
        )
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if step in {0, args.steps - 1}:
            gravity = get_gravity_orientation(data.qpos[3:7])
            print(
                f"step={step} height={data.qpos[2]:.3f} "
                f"forward_vel={data.qvel[0]:.3f} gravity_z={gravity[2]:.3f}",
                flush=True,
            )

    print("smoke_test: passed", flush=True)


if __name__ == "__main__":
    main()

