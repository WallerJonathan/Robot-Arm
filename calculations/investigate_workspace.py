"""Compare ordinary tip reach, cylinder constraints, and IK backends on one grid."""

from copy import deepcopy
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import os
import time
import numpy as np
from scipy.spatial import cKDTree

from robot import Robot, load_config
from workspace import PlacementEvaluator
from ik_backend import GeometricIK, nearby_configuration
from reports import write_json, write_csv

BASE = Path(__file__).resolve().parent
HISTORICAL = BASE/"outputs"/"20261002T080348Z_both_6be13914"


def main():
    started = time.perf_counter()
    old_workspace = json.loads((HISTORICAL/"workspace.json").read_text())
    old_config = old_workspace["config_snapshot"]
    old_points = {tuple(p["center_m"]): p for p in old_workspace["placements"]}
    cells = [c for c in old_workspace["cells"] if c["level"] == 0]
    config = deepcopy(old_config)
    config["numerics"]["ik"].update(backend="ik_geo", free_roll_samples=16)
    config["numerics"]["workspace"]["continuity"]["enabled"] = False
    config["task"]["collision_checking"] = False
    robot = Robot(config)
    adapter = GeometricIK(robot)
    evaluator = PlacementEvaluator(robot)
    folder = BASE/"outputs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_ik_investigation_"+uuid4().hex[:8])
    folder.mkdir()
    rng = np.random.default_rng(18500)
    sampled_q = rng.uniform(robot.limits[:, 0], robot.limits[:, 1], size=(3000, 6))
    poses = np.asarray([robot.tool_pose(q) for q in sampled_q])
    cloud = poses[:, :3, 3]
    tree = cKDTree(cloud)
    rows = []
    for index, cell in enumerate(cells, 1):
        center = np.asarray(cell["center_m"])
        old = old_points[tuple(cell["center_m"])]
        position_passed = False
        for nearest in tree.query(center, k=8)[1]:
            target = poses[nearest].copy()
            target[:3, 3] = center
            for q, least_squares in adapter.full_pose_candidates(target):
                q = nearby_configuration(robot, q, robot.reference_q)
                if q is not None and np.linalg.norm(robot.tool_pose(q)[:3, 3]-center) <= config["numerics"]["ik"]["position_tolerance_m"]:
                    position_passed = True
                    break
            if position_passed:
                break
        result = evaluator.evaluate(center)
        rows.append({"x_m": center[0], "y_m": center[1], "z_m": center[2],
                     "position_only_validated_solution": position_passed,
                     "old_surface_samples_passed": old["surface_samples_passed"],
                     "old_surface_all_pass": old["surface_samples_passed"] == old["surface_samples_tested"],
                     "old_complete_placement_status": old["status"],
                     "ik_geo_surface_all_pass": result["surface_samples_passed"] == result["surface_samples_tested"],
                     "ik_geo_surface_status": result["status"], "ik_geo_reason": result["reason"]})
        if index % 20 == 0 or index == len(cells):
            print(f"IK comparison {index}/{len(cells)}", flush=True)
    summary = {"illustrative": True, "same_grid_centers": len(rows),
               "position_only_validated_solutions": sum(r["position_only_validated_solution"] for r in rows),
               "position_only_note": "Eight nearest sampled orientations attempted per center using IK-Geo; accepted solutions verified by DH FK. Failures remain unestablished, not proof of unreachability.",
               "old_scipy_all_surface_targets_passed": sum(r["old_surface_all_pass"] for r in rows),
               "ik_geo_all_surface_targets_passed_same_original_settings": sum(r["ik_geo_surface_all_pass"] for r in rows),
               "old_complete_placement_status_counts": dict(Counter(r["old_complete_placement_status"] for r in rows)),
               "ik_geo_surface_status_counts": dict(Counter(r["ik_geo_surface_status"] for r in rows)),
               "ik_geo_rejection_reasons": dict(Counter(r["ik_geo_reason"] for r in rows if r["ik_geo_reason"])),
               "reach_cloud_samples": len(cloud),
               "sampled_tip_extent_lower_m": cloud.min(axis=0).tolist(),
               "sampled_tip_extent_upper_m": cloud.max(axis=0).tolist(),
               "sampled_max_horizontal_tip_radius_m": float(np.max(np.linalg.norm(cloud[:, :2], axis=1))),
               "cylinder_direction_world": evaluator.forward.tolist(), "configuration": config,
               "historical_source": str(HISTORICAL), "elapsed_seconds": time.perf_counter()-started,
               "limitations": ["Tip point-cloud is sampled, not a guarantee of a solid sphere",
                               "Position-only search leaves unsolved positions uncertain",
                               "This comparison isolates solvers using the original limits and no collisions; the current CLI also applies shoulder/collision/stroke checks"]}
    write_csv(folder/"ik_comparison.csv", rows, list(rows[0]))
    write_json(folder/"investigation.json", summary)
    os.environ["MPLCONFIGDIR"] = str(BASE/".venv"/"matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 7))
    for number, field, title in [(1, "position_only_validated_solution", "Position-only center reach: any tested orientation"),
                                 (2, "ik_geo_surface_all_pass", "Whole sampled cylinder side: inward normals")]:
        ax = fig.add_subplot(1, 2, number, projection="3d")
        ax.scatter(*cloud[::3].T, color="gray", alpha=.08, s=3, label="sampled tip reach")
        for value, color, legend in [(True, "#18845b", "validated solution"), (False, "#b64b4b", "no solution established" if number == 1 else "surface rejected by analytic model")]:
            points = np.asarray([[r["x_m"], r["y_m"], r["z_m"]] for r in rows if r[field] == value]).reshape(-1, 3)
            ax.scatter(*points.T, color=color, s=18, label=legend)
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
               xlim=(-.7,.7), ylim=(-.7,.7), zlim=(-.4,1.0), title=title)
        ax.set_box_aspect((1,1,1))
        ax.view_init(24,-55)
        ax.legend(fontsize=8)
    fig.suptitle("ILLUSTRATIVE: ordinary tip reach versus cylinder-placement requirements\nSame geometry/grid; original full-turn limits; no collision screening", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder/"reach_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k != "configuration"}, indent=2), flush=True)
    print("Saved investigation:", folder, flush=True)


if __name__ == "__main__":
    main()
