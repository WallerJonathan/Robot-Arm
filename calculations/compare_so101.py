"""Compare URDF calculations with the matching SO-101 MuJoCo model."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution

from robot import Robot, load_config
from urdf_import import apply_urdf
from torque import static_torque
from reports import write_json

BASE = Path(__file__).resolve().parent


def rotation_error(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b)-1)/2, -1, 1)))


def compare(urdf_path, mjcf_path, samples, seed):
    import mujoco
    config = apply_urdf(load_config(BASE/"configs/robot_config.json"), urdf_path)
    robot = Robot(config)
    model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
    data = mujoco.MjData(model)
    names = config["robot"]["active_joint_names"]
    qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names]
    dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names]
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_qpos = model.jnt_qposadr[gripper_id]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    rng = np.random.default_rng(seed)
    lower, upper = robot.limits.T
    configurations = np.vstack((robot.reference_q, rng.uniform(lower, upper, (samples, robot.n))))
    rows, positions = [], []
    contacts = 0
    for q in configurations:
        data.qpos[:] = 0
        data.qpos[qpos] = q
        data.qpos[gripper_qpos] = 0
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        pose = robot.tool_pose(q)
        mj_position = data.site_xpos[site_id].copy()
        mj_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
        mj_gripper_rotation = data.xmat[gripper_body_id].reshape(3, 3).copy()
        urdf_torque = static_torque(robot, q)[0]
        mj_torque = data.qfrc_bias[dofs].copy()
        rows.append({"q_rad": q.tolist(),
                     "tip_position_error_m": float(np.linalg.norm(pose[:3, 3]-mj_position)),
                     "tip_rotation_error_rad": rotation_error(pose[:3, :3], mj_rotation),
                     "arm_chain_gripper_rotation_error_rad": rotation_error(robot.frames(q)[-1, :3, :3], mj_gripper_rotation),
                     "gravity_torque_urdf_Nm": urdf_torque.tolist(),
                     "gravity_torque_mujoco_Nm": mj_torque.tolist(),
                     "gravity_torque_error_Nm": (urdf_torque-mj_torque).tolist(),
                     "mujoco_contacts": int(data.ncon)})
        positions.append(pose[:3, 3])
        contacts += bool(data.ncon)
    positions = np.asarray(positions)
    position_errors = np.array([x["tip_position_error_m"] for x in rows])
    rotation_errors = np.array([x["tip_rotation_error_rad"] for x in rows])
    chain_rotation_errors = np.array([x["arm_chain_gripper_rotation_error_rad"] for x in rows])
    torque_errors = np.array([x["gravity_torque_error_Nm"] for x in rows])
    radius = np.linalg.norm(positions[:, :2], axis=1)
    bounds = list(map(tuple, robot.limits))
    def maximize(metric):
        answer = differential_evolution(lambda q: -metric(robot.tool_pose(q)[:3, 3]), bounds,
                                        seed=seed, popsize=12, maxiter=160, polish=True, tol=1e-9)
        point = robot.tool_pose(answer.x)[:3, 3]
        return {"value_m": float(metric(point)), "q_rad": answer.x.tolist(), "position_m": point.tolist(),
                "method": "bounded differential evolution; numerical estimate, not proof"}
    optimized = {"distance_from_base_origin": maximize(lambda p: np.linalg.norm(p)),
                 "horizontal_radius": maximize(lambda p: np.linalg.norm(p[:2]))}
    result = {"source_urdf": str(Path(urdf_path).resolve()), "source_mjcf": str(Path(mjcf_path).resolve()),
              "active_tool_joints": names, "gripper_joint_excluded_from_tool_dof": True,
              "samples_including_zero": len(rows), "seed": seed,
              "urdf_total_mass_kg": sum(x["mass_kg"] for x in robot.masses),
              "mujoco_total_mass_kg": float(model.body_mass.sum()),
              "comparison": {"maximum_tip_position_error_m": float(position_errors.max()),
                             "maximum_tip_rotation_error_rad": float(rotation_errors.max()),
                             "maximum_arm_chain_rotation_error_rad": float(chain_rotation_errors.max()),
                             "maximum_abs_gravity_torque_error_Nm": float(abs(torque_errors).max()),
                             "rms_gravity_torque_error_Nm": float(np.sqrt(np.mean(torque_errors**2)))},
              "sampled_tip_position_bounds_m": {"minimum": positions.min(0).tolist(), "maximum": positions.max(0).tolist()},
              "sampled_horizontal_radius_m": {"minimum": float(radius.min()), "maximum": float(radius.max())},
              "sampled_distance_from_base_origin_m": {"minimum": float(np.linalg.norm(positions, axis=1).min()),
                                                       "maximum": float(np.linalg.norm(positions, axis=1).max())},
              "numerically_optimized_reach": optimized,
              "mujoco_samples_with_contacts": contacts,
              "reach_note": "Random joint-limit sample of the gripper-frame origin; not a guaranteed solid workspace or oriented drawing workspace.",
              "model_caveats": ["SO-101 provides five tool-pose joints plus a gripper actuator.",
                                "The source URDF gripper_frame fixed joint uses 180 degrees about Y, while the matching MJCF gripperframe site uses 90 degrees about Y; position is consistent but tip-frame orientation is not.",
                                "Repository says base collision meshes were removed due to simulation/planning problems.",
                                "The URDF/MJCF gripper mapping does not yet match LeRobot's 0-to-100 linear convention.",
                                "Reach samples use gripper joint zero and repository joint limits."],
              "samples": rows}
    return config, positions, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = BASE.parent.parent/"SO-ARM100"/"Simulation"/"SO101"
    parser.add_argument("--urdf", default=str(source/"so101_new_calib.urdf"))
    parser.add_argument("--mjcf", default=str(source/"so101_new_calib.xml"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", default="outputs/so101_comparison")
    args = parser.parse_args()
    folder = (BASE/args.output).resolve()
    if not folder.is_relative_to(BASE):
        parser.error("Output must remain inside calculations")
    folder.mkdir(parents=True, exist_ok=True)
    config, positions, result = compare(args.urdf, args.mjcf, args.samples, args.seed)
    write_json(folder/"imported_config.json", config)
    write_json(folder/"comparison.json", result)
    import os
    os.environ["MPLCONFIGDIR"] = str(BASE/".venv"/"matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*positions.T, s=2, alpha=.18, color="#1967a3")
    ax.scatter(0, 0, 0, marker="^", color="black", s=70, label="base")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="SO-101 sampled gripper-frame positions\n(random joint-limit samples; not guaranteed workspace)")
    ax.legend()
    fig.savefig(folder/"sampled_reach.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    keys = ("active_tool_joints","urdf_total_mass_kg","mujoco_total_mass_kg","comparison","sampled_tip_position_bounds_m","sampled_horizontal_radius_m","sampled_distance_from_base_origin_m","numerically_optimized_reach","mujoco_samples_with_contacts")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
