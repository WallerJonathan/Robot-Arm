"""JSON/CSV, plotting, and optional LaTeX exports from the same run data."""

import csv
import json
import os
from pathlib import Path
import numpy as np


def write_json(path, data):
    """Save strict JSON: missing numerical values are null, never NaN."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows, fields):
    """Write a table with explicit headers even when no rows are present."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_torque(folder, data, label):
    """Save per-joint maxima and reference/maximizing component contributions."""
    write_json(folder/"torque.json", data)
    rows = [{"result_label": label, "joint": m["joint"],
             "reference_signed_Nm": data["reference"]["signed_holding_Nm"][m["joint"]-1],
             "horizontal_reference_signed_Nm": data["horizontal_reference"]["signed_holding_Nm"][m["joint"]-1],
             "estimated_max_abs_Nm": m["estimated_max_abs_Nm"],
             "maximizing_signed_Nm": m["signed_holding_Nm"],
             "maximizing_q_rad": json.dumps(m["q_rad"]),
             "continuous_joint_output_limit_Nm": m.get("continuous_joint_output_limit_Nm"),
             "estimated_utilization": m.get("estimated_utilization")} for m in data["maxima"]]
    write_csv(folder/"torque_joints.csv", rows, list(rows[0]))
    contributions = []
    for pose in (data["reference"], data["horizontal_reference"]):
        for k, name in enumerate(data["mass_names"]):
            for j, value in enumerate(pose["component_contributions_Nm"][k]):
                contributions.append({"result_label": label, "pose": pose["name"], "joint": j+1,
                                      "component": name, "holding_Nm": value})
    for maximum in data["maxima"]:
        for name, value in zip(data["mass_names"], maximum["component_contributions_Nm"]):
            contributions.append({"result_label": label, "pose": f"joint_{maximum['joint']}_maximizer",
                                  "joint": maximum["joint"], "component": name, "holding_Nm": value})
    write_csv(folder/"torque_components.csv", contributions, list(contributions[0]))


def export_workspace(folder, data, label):
    """Save every evaluated center, leaf geometry, surface solves, and path witnesses."""
    write_json(folder/"workspace.json", data)
    placements = []
    for point in data["placements"]:
        continuity = point["continuity"] or {}
        placements.append({"result_label": label, "x_m": point["center_m"][0],
                           "y_m": point["center_m"][1], "z_m": point["center_m"][2],
                           "status": point["status"], "reason": point["reason"],
                           "surface_samples_tested": point["surface_samples_tested"],
                           "surface_samples_passed": point["surface_samples_passed"],
                           "ik_attempts": point["ik_attempts"],
                           "max_successful_position_error_m": point["max_successful_position_error_m"],
                           "max_successful_axis_error_rad": point["max_successful_axis_error_rad"],
                           "continuity_passed": continuity.get("success"),
                           "stroke_samples_tested": continuity.get("samples_tested"),
                           "max_joint_step_rad": continuity.get("max_joint_step_rad")})
    write_csv(folder/"workspace_placements.csv", placements,
              list(placements[0]) if placements else ["result_label", "x_m", "y_m", "z_m", "status", "reason"])
    rows = [{"result_label": label, "center_m": json.dumps(c["center_m"]),
             "lower_m": json.dumps(c["lower_m"]), "upper_m": json.dumps(c["upper_m"]),
             "status": c["status"], "reason": c["reason"], "level": c["level"],
             "is_leaf": c["is_leaf"], "volume_m3": c["volume_m3"]} for c in data["cells"]]
    write_csv(folder/"workspace_cells.csv", rows, list(rows[0]) if rows else ["result_label", "status"])


def _box_edges(lower, upper):
    from itertools import product
    vertices = list(product(*zip(lower, upper)))
    for i, p in enumerate(vertices):
        for q in vertices[i+1:]:
            if sum(abs(np.asarray(p)-q) > 1e-10) == 1:
                yield np.array([p, q])


def plot_workspace(folder, robot, data):
    """Create three equal-scale 3D views, planar slices, and a cube detail if found."""
    # Keep matplotlib's font cache inside calculations, including headless runs.
    cache = Path(__file__).resolve().parent/".venv"/"matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = {"feasible": "#18845b", "invalid": "#b64b4b", "unresolved": "#dc9c20"}
    cells = [c for c in data["cells"] if c["is_leaf"]]
    bounds = np.asarray(data["settings"]["bounds_m"])
    base = robot.base[:3, 3]
    drawing_q = robot.config["robot"].get("visualization_q_rad", robot.reference_q)
    frames = robot.frames(drawing_q)
    images = []
    label = "ILLUSTRATIVE" if data["illustrative"] else "Configured model"
    collision_note = "Approximate self-collision checked; table/object unchecked" if data["collision_checking"] else "Collisions unchecked"
    table = robot.config["task"]["table"]

    def draw(ax, cube_only=False):
        for status, color in colors.items():
            points = np.asarray([c["center_m"] for c in cells if c["status"] == status]).reshape(-1, 3)
            if len(points):
                ax.scatter(*points.T, c=color, s=16, alpha=.7, label=status)
        ax.scatter(*base, c="black", marker="^", s=55, label="robot base")
        origins = frames[:, :3, 3]
        ax.plot(*origins.T, c="black", linewidth=2, label="illustrative drawing pose" if "visualization_q_rad" in robot.config["robot"] else "supplied reference pose")
        tip = robot.tool_pose(drawing_q)[:3, 3]
        ax.plot(*np.array([origins[-1], tip]).T, c="purple", linewidth=2)
        xmin, xmax = table["bounds_xy_m"][0]
        ymin, ymax = table["bounds_xy_m"][1]
        z = table["top_z_m"]
        plane = [(xmin, ymin, z), (xmax, ymin, z), (xmax, ymax, z), (xmin, ymax, z)]
        ax.add_collection3d(Poly3DCollection([plane], alpha=.07, facecolor="gray"))
        for edge in _box_edges(bounds[:, 0], bounds[:, 1]):
            ax.plot(*edge.T, color="gray", linewidth=.6, alpha=.5)
        front = data.get("front_region_verification")
        if front:
            for edge in _box_edges(front["lower_m"], front["upper_m"]):
                ax.plot(*edge.T, color="#2458df", linewidth=1.5)
            ax.text(*front["upper_m"], f"requested 0.4 m box\n{front['estimated_feasible_fraction']:.1%} sampled pass", fontsize=8)
        cube = data["cube"]
        if cube["found"]:
            for edge in _box_edges(cube["lower_m"], cube["upper_m"]):
                ax.plot(*edge.T, color="#2458df", linewidth=2.5)
            ax.text(*cube["upper_m"], "0.4 × 0.4 × 0.4 m\nreference-point box", fontsize=8)
        if cube_only:
            low, high = np.array(cube["lower_m"])-.08, np.array(cube["upper_m"])+.08
        else:
            low = np.minimum(bounds[:, 0], np.minimum(base, origins.min(axis=0)))
            high = np.maximum(bounds[:, 1], np.maximum(base, origins.max(axis=0)))
            low[2] = min(low[2], z)
        middle = (low+high)/2
        half = max(high-low)*.55
        ax.set_xlim(middle[0]-half, middle[0]+half)
        ax.set_ylim(middle[1]-half, middle[1]+half)
        ax.set_zlim(middle[2]-half, middle[2]+half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_zlabel("world z (m)")
        ax.legend(fontsize=7, loc="upper left")

    for index, (elevation, azimuth) in enumerate([(24, -55), (22, 40), (65, 140)], 1):
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax)
        ax.view_init(elevation, azimuth)
        cube_note = "cube found" if data["cube"]["found"] else "no qualifying cube found at these settings"
        fig.suptitle(f"{label}: sampled kinematic feasibility\n{collision_note}; strokes sampled\n{cube_note}", fontsize=10)
        filename = f"workspace_view_{index}.png"
        fig.savefig(folder/filename, dpi=140, bbox_inches="tight")
        plt.close(fig)
        images.append(filename)
    if data["cube"]["found"]:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax, cube_only=True)
        cube = data["cube"]
        ax.set_title(f"{label}: sampled cube containment\nBounds (m): {np.round(cube['lower_m'], 3)} to {np.round(cube['upper_m'], 3)}")
        fig.savefig(folder/"workspace_cube.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        images.append("workspace_cube.png")
    # Leaf-cell rectangles reveal holes; unlike a point cloud these expose extents.
    from matplotlib.patches import Rectangle, Patch
    z_levels = np.linspace(bounds[2, 0], bounds[2, 1], 5)[1:-1]
    fig, axes = plt.subplots(1, len(z_levels), figsize=(14, 5), squeeze=False)
    for ax, z in zip(axes[0], z_levels):
        for cell in cells:
            low, high = cell["lower_m"], cell["upper_m"]
            if low[2] <= z < high[2]:
                ax.add_patch(Rectangle(low[:2], high[0]-low[0], high[1]-low[1],
                                       facecolor=colors[cell["status"]], edgecolor="white", alpha=.6))
        cube = data["cube"]
        if cube["found"] and cube["lower_m"][2] <= z <= cube["upper_m"][2]:
            low, high = cube["lower_m"], cube["upper_m"]
            ax.add_patch(Rectangle(low[:2], high[0]-low[0], high[1]-low[1],
                                   fill=False, edgecolor="#2458df", linewidth=2))
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="x (m)", ylabel="y (m)", title=f"z = {z:.3f} m")
        ax.set_aspect("equal")
    fig.legend(handles=[Patch(color=c, label=s) for s, c in colors.items()], loc="lower center", ncol=3)
    fig.suptitle(f"{label}: leaf-cell slices; {collision_note}", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(folder/"workspace_slices.png", dpi=140)
    plt.close(fig)
    images.append("workspace_slices.png")
    return images


def tex_escape(text):
    """Escape configuration names without recursively escaping replacements."""
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(text))


def export_latex(folder, config, torque=None, workspace=None, images=()):
    """Generate a standalone .tex report using the same exported numerical data."""
    lines = [r"\documentclass{article}", r"\usepackage[margin=20mm]{geometry}",
             r"\usepackage{amsmath,graphicx,longtable}", r"\begin{document}",
             r"\section*{Robot analysis}", tex_escape(config["provenance"]["result_label"]),
             r"\subsection*{Provenance and assumptions}"]
    for entry in config["provenance"]["interpretation_assumptions"]:
        lines.append(tex_escape(entry)+r"\par")
    lines += ["Replace: "+tex_escape("; ".join(config["provenance"]["demonstration_values_to_replace"])),
              tex_escape(config["provenance"]["mass_inventory_note"])]
    if config["robot"]["convention"] == "standard_dh":
        lines += [r"\[T_i=R_z(q_i+\theta_{0,i})T_z(d_i)T_x(a_i)R_x(\alpha_i).\]",
                  r"\subsection*{DH parameters and reference pose}",
                  r"\begin{longtable}{rrrrrr}", r"Joint & $a$ (m) & $d$ (m) & $\alpha$ (rad) & $q_{ref}$ (rad) & $\theta_0$ (rad)\\\hline"]
        for i, row in enumerate(config["robot"]["joints"]):
            values = [i+1, row["a_m"], row["d_m"], row["alpha_rad"], config["robot"]["reference_q_rad"][i], row["zero_offset_rad"]]
            lines.append(" & ".join(f"{v:.9g}" for v in values)+r"\\")
    else:
        lines += [r"\subsection*{URDF joint chain}", tex_escape(f"Source: {config['robot']['urdf_source']}"),
                  r"\begin{longtable}{rllrr}", r"Joint & Name & Axis & Lower (rad) & Upper (rad)\\\hline"]
        for i, row in enumerate(config["robot"]["joints"]):
            lines.append(f"{i+1} & {tex_escape(row['name'])} & {tex_escape(str(row['axis_local']))} & {row['limits_rad'][0]:.9g} & {row['limits_rad'][1]:.9g}"+r"\\")
    lines += [r"\end{longtable}", r"\subsection*{Mass inventory}",
              r"\begin{longtable}{lrrl}", r"Component & Mass (kg) & Frame & Local COM (m)\\\hline"]
    for mass in config["masses"]:
        lines.append(f"{tex_escape(mass['name'])} & {mass['mass_kg']:.6g} & {mass['frame_index']} & {tex_escape(str([round(v, 6) for v in mass['com_local_m']]))}"+r"\\")
    lines += [r"\end{longtable}", r"World/base and tool transforms, joint limits, gravity, attachment indices, and all numerical settings are in config\_snapshot.json."]
    if torque is not None:
        lines += [r"\section*{Gravity-only holding torque}",
                  "These are numerical estimates over joint-limited poses with no collision filtering. They do not represent drawing-force capacity or inverse dynamics.",
                  r"\[\tau_{hold,i}=-\sum_{k\in D_i}\hat{a}_i\cdot[(p_k-o_i)\times m_k g].\]",
                  r"\begin{longtable}{rrrr}", r"Joint & Reference (N m) & Horizontal reference (N m) & Estimated max $|\tau|$ (N m)\\\hline"]
        for i, entry in enumerate(torque["maxima"]):
            lines.append(f"{i+1} & {torque['reference']['signed_holding_Nm'][i]:.9g} & {torque['horizontal_reference']['signed_holding_Nm'][i]:.9g} & {entry['estimated_max_abs_Nm']:.9g}"+r"\\")
        lines += [r"\end{longtable}", tex_escape(torque["horizontal_reference"]["name"]),
                  r"\[|\tau|=\sum_k m_k g r_k\quad\text{only for aligned horizontal gravity lever arms.}\]",
                  "Wrist offsets are transformed at the reference pose. They are not all positive horizontal lever arms.",
                  r"\subsection*{Reference component substitutions}",
                  r"\begin{longtable}{l"+"r"*len(torque["maxima"])+"}",
                  "Component & "+" & ".join(f"$\\tau_{i+1}$" for i in range(len(torque["maxima"])))+r"\\\hline"]
        for name, row in zip(torque["mass_names"], torque["reference"]["component_contributions_Nm"]):
            lines.append(tex_escape(name)+" & "+" & ".join(f"{v:.8g}" for v in row)+r"\\")
        lines += [r"\end{longtable}", "Each column sums to the reference holding torque. All entries are N m."]
        for entry in torque["maxima"]:
            lines += [tex_escape(f"Joint {entry['joint']} maximizing q (rad): {entry['q_rad']}"), r"\par",
                      tex_escape(f"Component substitutions (N m), in inventory order: {entry['component_contributions_Nm']}"), r"\par"]
        lines += [r"\subsection*{Horizontal-reference weight moment check}",
                  "The signed effective gravity lever is the component holding torque divided by its weight. It includes the transformed wrist geometry.",
                  r"\[\tau_{hold,2}=\sum_k m_k\lVert g\rVert r_{signed,k,2}.\]",
                  r"\begin{longtable}{lrrr}", r"Component & Mass (kg) & Signed lever (m) & Holding (N m)\\\hline"]
        horizontal = torque["horizontal_reference"]
        for index, name in enumerate(torque["mass_names"]):
            lines.append(tex_escape(name)+f" & {horizontal['component_masses_kg'][index]:.6g} & {horizontal['effective_signed_gravity_levers_m'][index][1]:.9g} & {horizontal['component_contributions_Nm'][index][1]:.9g}"+r"\\")
        lines += [r"\end{longtable}",
                  f"Numerical sum at $g={horizontal['gravity_magnitude_m_s2']:.9g}$ m/s$^2$: $\\tau_{{hold,2}}={horizontal['signed_holding_Nm'][1]:.9g}$ N m."]
    if workspace is not None:
        direction_note = (f"Fixed world direction f = {workspace['fixed_side_direction_world']}" if workspace["fixed_side_direction_world"] is not None else
                          "The selected side faces horizontally toward the robot separately at each placement; each direction is saved with that placement.")
        lines += [r"\section*{Sampled object-placement kinematic feasibility}",
                  tex_escape(workspace["collision_scope"])+". Representative strokes are sampled; this is not proof of continuous drawable motion.",
                  r"\[p(\phi,h)=c+R(\cos\phi\,f+\sin\phi\,(\hat{z}\times f))+h\hat{z},\quad -\pi/2\leq\phi\leq\pi/2.\]",
                  r"\[u_{tool}=-(\cos\phi\,f+\sin\phi\,(\hat{z}\times f)),\quad -H/2\leq h\leq H/2.\]",
                  tex_escape(direction_note), r"\par",
                  tex_escape(f"Cylinder: radius {config['task']['cylinder']['radius_m']} m, drawing height {config['task']['cylinder']['drawing_height_m']} m."), r"\par",
                  tex_escape(f"Leaf counts: {workspace['counts_leaf_cells']}; estimated volume {workspace['estimated_feasible_volume_m3']:.9g} cubic meters."), r"\par",
                  tex_escape(workspace["volume_method"]), r"\par",
                  tex_escape(f"Requested grid spacing (m): {workspace['requested_spacing_m']}; effective grid (m): {workspace['effective_coarse_spacing_m']}"), r"\par",
                  tex_escape(f"IK position tolerance {config['numerics']['ik']['position_tolerance_m']} m; axis tolerance {config['numerics']['ik']['axis_tolerance_rad']} rad."), r"\par",
                  tex_escape(f"Surface samples: {workspace['settings']['surface_phi_samples']} angles x {workspace['settings']['surface_height_samples']} heights; continuity: {workspace['settings']['continuity']}"), r"\par",
                  r"\[b+[-0.2,0.2]^3\subseteq\mathcal W_{object}\quad\text{(sampling-based verification).}\]"]
        cube = workspace["cube"]
        if workspace.get("front_region_verification"):
            front = workspace["front_region_verification"]
            lines += [tex_escape(f"Requested front box: {front['lower_m']} to {front['upper_m']} m; sampled passing fraction {front['estimated_feasible_fraction']:.6%}."), r"\par",
                      tex_escape(front["meaning"]), r"\par"]
        lines.append(tex_escape(f"Cube dimensions (m): {cube['dimensions_m']}, margin: {cube['margin_m']} m."))
        if cube["found"]:
            lines += [tex_escape(f"Cube center (m): {cube['center_m']}; bounds (m): {cube['lower_m']} to {cube['upper_m']}"), r"\par", tex_escape(cube["verification"])]
        else:
            lines.append(tex_escape(cube["reason"]))
        for filename in images:
            lines += [r"\begin{figure}[ht]\centering", r"\includegraphics[width=0.95\linewidth]{"+filename+"}", r"\end{figure}"]
    lines.append(r"\end{document}")
    (folder/"report.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
