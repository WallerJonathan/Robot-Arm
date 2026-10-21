"""Interactive/noninteractive entry point; all default paths are script-relative."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
from uuid import uuid4
import numpy as np

from robot import Robot, load_config, apply_dh_table
from torque import analyze_torque
from workspace import analyze_workspace
from reports import write_json, export_torque, export_workspace, plot_workspace, export_latex

BASE = Path(__file__).resolve().parent


def local_path(value):
    """Resolve explicit relative overrides under calculations and enforce that scope."""
    path = Path(value)
    resolved = (path if path.is_absolute() else BASE/path).resolve()
    if not resolved.is_relative_to(BASE):
        raise ValueError("Paths must remain inside calculations/")
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Spatial robot gravity torque and sampled cylinder-placement kinematics")
    parser.add_argument("--mode", choices=("torque", "workspace", "both"))
    parser.add_argument("--config", default="configs/robot_config.json", help="JSON configuration; relative to calculations")
    parser.add_argument("--dh-table", help="Optional CSV/TSV geometry and reference-pose override; theta is NOT a zero offset")
    parser.add_argument("--urdf", help="Optional URDF input; may be outside calculations and is read-only")
    parser.add_argument("--tip-link", default="gripper_frame_link", help="URDF tool-tip link")
    parser.add_argument("--output-dir", default="outputs", help="Run parent folder within calculations; each run gets a unique directory")
    parser.add_argument("--latex", action="store_true", help="Generate an optional .tex report; no PDF compiler required")
    parser.add_argument("--quick", action="store_true", help="Small illustrative verification scan; override sampling/search settings in saved snapshot")
    args_list = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(args_list)
    if not args_list:
        args.mode = input("Analysis [torque/workspace/both] (both): ").strip() or "both"
        if args.mode not in ("torque", "workspace", "both"):
            parser.error("Choose torque, workspace, or both")
        args.config = input("Configuration path (configs/robot_config.json): ").strip() or args.config
        args.output_dir = input("Output parent folder (outputs): ").strip() or args.output_dir
        args.latex = input("Export LaTeX? [y/N]: ").strip().lower() in ("y", "yes")
    elif args.mode is None:
        parser.error("--mode is required for noninteractive runs")
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = None
    started = time.perf_counter()
    try:
        config = load_config(local_path(args.config))
        if args.urdf:
            from urdf_import import apply_urdf
            urdf_path = Path(args.urdf)
            config = apply_urdf(config, urdf_path if urdf_path.is_absolute() else BASE/urdf_path, args.tip_link)
        if args.dh_table:
            if args.urdf:
                raise ValueError("Choose either --urdf or --dh-table")
            apply_dh_table(config, local_path(args.dh_table))
        if args.quick:
            config["numerics"]["torque"].update(samples=48, optimization_starts=2, max_iterations=60)
            config["numerics"]["workspace"].update(spacing_m=[.3,.3,.3], surface_phi_samples=3,
                                                   surface_height_samples=2, boundary_refinement_levels=0)
            config["numerics"]["workspace"]["continuity"].update(horizontal_strokes=2, vertical_strokes=2)
        robot = Robot(config)
        provenance = config["provenance"]
        illustrative = bool(provenance["demonstration_values_to_replace"])
        label = provenance["result_label"]
        if illustrative and "ILLUSTRATIVE" not in label.upper():
            label = "ILLUSTRATIVE: "+label
            provenance["result_label"] = label
        print(label, flush=True)
        scope = "Approximate self-collision screening; table/object collision unchecked" if config["task"]["collision_checking"] else "Collision screening disabled"
        print(f"Gravity only; {scope}; no drawing-force capacity claim.", flush=True)
        print("IK backend:", config["numerics"]["ik"].get("backend", "scipy"), flush=True)
        parent = local_path(args.output_dir)
        parent.mkdir(parents=True, exist_ok=True)
        folder = parent/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+args.mode+"_"+uuid4().hex[:8])
        folder.mkdir()
        write_json(folder/"config_snapshot.json", config)
        metadata = {"result_label": label, "illustrative": illustrative, "config_snapshot": config,
                    "units": {"length": "m", "mass": "kg", "angles": "rad", "torque": "N*m"},
                    "mode": args.mode, "quick": args.quick, "source_config": str(local_path(args.config))}
        torque = workspace = None
        images = []
        if args.mode in ("torque", "both"):
            print("Estimating gravity holding torque maxima...", flush=True)
            torque = {**metadata, **analyze_torque(robot)}
            export_torque(folder, torque, label)
            print("Joint   reference N*m   horizontal N*m   estimated max |N*m|", flush=True)
            for j, entry in enumerate(torque["maxima"]):
                print(f"{j+1:5d} {torque['reference']['signed_holding_Nm'][j]:15.6f} {torque['horizontal_reference']['signed_holding_Nm'][j]:16.6f} {entry['estimated_max_abs_Nm']:21.6f}", flush=True)
        if args.mode in ("workspace", "both"):
            workspace = {**metadata, **analyze_workspace(robot, progress=lambda msg: print(msg, flush=True))}
            export_workspace(folder, workspace, label)
            images = plot_workspace(folder, robot, workspace)
            print("Workspace leaf counts:", workspace["counts_leaf_cells"], flush=True)
            print(f"Estimated feasible volume: {workspace['estimated_feasible_volume_m3']:.6f} m^3", flush=True)
            print("Feasible cell extents (holes may exist):", workspace["feasible_cell_extents_m"], flush=True)
            print("Effective coarse spacing (m):", workspace["effective_coarse_spacing_m"], flush=True)
            if workspace.get("front_region_verification"):
                front = workspace["front_region_verification"]
                print("Requested front box:", front["lower_m"], "to", front["upper_m"], flush=True)
                print(f"Estimated fraction passing surface/stroke checks: {front['estimated_feasible_fraction']:.1%}; most passed: {front['most_of_sampled_region_passed']}", flush=True)
            if workspace["cube"]["found"]:
                print("0.4 m cube bounds:", workspace["cube"]["lower_m"], "to", workspace["cube"]["upper_m"], flush=True)
            else:
                print(workspace["cube"]["reason"], flush=True)
        if args.latex:
            export_latex(folder, config, torque, workspace, images)
        summary = {**metadata, "elapsed_seconds": time.perf_counter()-started,
                   "output_directory": str(folder), "completed": True,
                   "torque_estimated_max_abs_Nm": None if torque is None else [m["estimated_max_abs_Nm"] for m in torque["maxima"]],
                   "workspace_counts": None if workspace is None else workspace["counts_leaf_cells"],
                   "estimated_feasible_volume_m3": None if workspace is None else workspace["estimated_feasible_volume_m3"],
                   "cube": None if workspace is None else workspace["cube"], "images": images,
                   "front_region_verification": None if workspace is None else workspace.get("front_region_verification"),
                   "python_version": sys.version, "numpy_version": np.__version__}
        write_json(folder/"run_summary.json", summary)
        print(f"Saved run: {folder}\nElapsed: {summary['elapsed_seconds']:.1f} s", flush=True)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Analysis error: {exc}", file=sys.stderr)
        if folder is not None:
            write_json(folder/"run_error.json", {"completed": False, "error": str(exc),
                                                "elapsed_seconds": time.perf_counter()-started})
            print(f"Partial outputs preserved: {folder}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
