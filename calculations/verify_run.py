"""Focused numerical export and accepted-witness verification for a saved run."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

from robot import Robot
from workspace import surface_target
from collision import check_self_collision
from reports import write_json


def verify(folder):
    """Recheck exports and every accepted surface/stroke witness with independent FK."""
    config = json.loads((folder/"config_snapshot.json").read_text())
    summary = json.loads((folder/"run_summary.json").read_text())
    robot = Robot(config)
    counts = {"surface": 0, "stroke": 0, "intermediate_collision": 0}
    max_pos = max_axis = max_step = 0.
    if (folder/"workspace.json").exists():
        workspace = json.loads((folder/"workspace.json").read_text())
        for placement in workspace["placements"]:
            if placement["status"] != "feasible":
                continue
            assert placement["surface_samples_passed"] == placement["surface_samples_tested"]
            if config["numerics"]["workspace"]["continuity"]["enabled"]:
                assert placement["continuity"]["success"]
            forward = placement.get("side_direction_world", workspace["fixed_side_direction_world"])
            groups = [("surface", placement["surface_solutions"])]
            if placement["continuity"]:
                groups += [("stroke", trace) for trace in placement["continuity"]["traces"]]
            for kind, trace in groups:
                for index, target in enumerate(trace):
                    p, axis = surface_target(placement["center_m"], config["task"]["cylinder"]["radius_m"],
                                             target["phi_rad"], target["height_m"], forward)
                    actual_p, actual_axis, _, _ = robot.tip_state(target["q_rad"])
                    pos_error = float(np.linalg.norm(actual_p-p))
                    axis_error = float(np.arccos(np.clip(actual_axis @ axis, -1., 1.)))
                    assert pos_error <= config["numerics"]["ik"]["position_tolerance_m"]
                    assert axis_error <= config["numerics"]["ik"]["axis_tolerance_rad"]
                    assert not check_self_collision(robot, target["q_rad"])["collision"]
                    counts[kind] += 1
                    max_pos, max_axis = max(max_pos, pos_error), max(max_axis, axis_error)
                    if kind == "stroke" and index:
                        previous = np.asarray(trace[index-1]["q_rad"])
                        delta = np.asarray(target["q_rad"])-previous
                        step = float(max(abs(delta)))
                        assert step <= config["numerics"]["workspace"]["continuity"]["max_joint_step_rad"]
                        max_step = max(max_step, step)
                        substeps = config["numerics"]["workspace"]["continuity"].get("joint_interpolation_substeps", 3)
                        for fraction in np.linspace(0, 1, substeps+2)[1:-1]:
                            assert not check_self_collision(robot, previous+fraction*delta)["collision"]
                            counts["intermediate_collision"] += 1
        leaves = [cell for cell in workspace["cells"] if cell["is_leaf"]]
        counter = Counter(cell["status"] for cell in leaves)
        assert {status: counter.get(status, 0) for status in ("feasible", "invalid", "unresolved")} == workspace["counts_leaf_cells"]
        assert workspace["counts_leaf_cells"] == summary["workspace_counts"]
        volume = sum(cell["volume_m3"] for cell in leaves if cell["status"] == "feasible")
        assert np.isclose(volume, summary["estimated_feasible_volume_m3"])
        front = workspace.get("front_region_verification")
        if front:
            low, high = np.asarray(front["lower_m"]), np.asarray(front["upper_m"])
            clipped = {status: 0. for status in counter}
            for cell in leaves:
                widths = np.maximum(0., np.minimum(high, cell["upper_m"])-np.maximum(low, cell["lower_m"]))
                clipped[cell["status"]] += float(np.prod(widths))
            for status, expected in front["estimated_volume_m3_by_status"].items():
                assert np.isclose(clipped.get(status, 0.), expected)
            assert np.isclose(sum(clipped.values()), front["volume_m3"])
            fraction = clipped.get("feasible", 0.)/front["volume_m3"]
            assert np.isclose(fraction, front["estimated_feasible_fraction"])
            assert front["most_of_sampled_region_passed"] == (fraction > .5)
        with (folder/"workspace_placements.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(workspace["placements"])
        assert Counter(row["status"] for row in rows) == Counter(p["status"] for p in workspace["placements"])
    if (folder/"torque.json").exists():
        torque = json.loads((folder/"torque.json").read_text())
        with (folder/"torque_joints.csv").open() as stream:
            for row, result in zip(csv.DictReader(stream), torque["maxima"]):
                assert float(row["estimated_max_abs_Nm"]) == result["estimated_max_abs_Nm"]
        np.testing.assert_allclose(np.sum(torque["reference"]["component_contributions_Nm"], axis=0),
                                   torque["reference"]["signed_holding_Nm"], atol=1e-12)
    record = {"passed": True, "run": str(folder), "witnesses_rechecked": counts,
              "maximum_position_error_m": max_pos, "maximum_axis_error_rad": max_axis,
              "maximum_joint_step_rad": max_step, "cross_format_exports_agree": True,
              "scope": "Checks sampled model witnesses, not physical collision certification"}
    write_json(folder/"verification.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = Path(args.run_folder)
    folder = (path if path.is_absolute() else root/path).resolve()
    if not folder.is_relative_to(root):
        parser.error("Verification paths must remain inside calculations")
    print(json.dumps(verify(folder), indent=2))
