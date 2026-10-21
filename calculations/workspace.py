"""Sampled cylinder-placement kinematics, representative strokes, and cube search.

Configured capsules screen approximate self-collision. Table/object collision is
not checked. Finite sampling does not prove continuous physical drawability.
"""

from itertools import product
import numpy as np


def cylinder_direction(robot, settings, center=None):
    """Compute the selected side's fixed horizontal world direction once.

    Parameters
    ----------
    robot : Robot
    settings : dict
        Cylinder settings including a configured reference center in meters.

    Returns
    -------
    ndarray, shape (3,)
        Unit horizontal outward normal at the middle of the selected side.
    """
    mode = settings["side_direction_mode"]
    if mode == "toward_base_per_placement":
        position = settings["reference_center_world_m"] if center is None else center
        direction = robot.base[:3, 3]-np.asarray(position, float)
    elif mode == "toward_base_from_reference_then_fixed":
        direction = robot.base[:3, 3]-np.asarray(settings["reference_center_world_m"], float)
    elif mode == "fixed_world_direction":
        direction = np.asarray(settings["fixed_world_direction"], float)
    else:
        raise ValueError("Unknown cylinder side_direction_mode")
    direction[2] = 0.
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("Cylinder direction needs a nonzero horizontal reference-to-base vector")
    return direction/np.linalg.norm(direction)


def surface_target(center, radius, phi, height, forward):
    """Return tip position (meters) and inward unit direction, each shape (3,)."""
    forward = np.asarray(forward, dtype=float)
    tangent = np.cross([0., 0., 1.], forward)
    normal = np.cos(phi)*forward + np.sin(phi)*tangent
    return np.asarray(center)+radius*normal+np.array([0., 0., height]), -normal


def check_strokes(robot, center, forward, surface_solutions, seed):
    """Connect all available IK branches along sampled representative strokes.

    A layered graph searches branch connections, avoiding greedy branch jumps.
    Edges enforce small joint steps and configured intermediate self-collision
    checks. These are finite samples, not proof of all continuous drawing motion.
    """
    from ik_backend import nearby_configuration
    from collision import check_self_collision
    settings = robot.config["numerics"]["workspace"]["continuity"]
    cylinder = robot.config["task"]["cylinder"]
    radius, height = cylinder["radius_m"], cylinder["drawing_height_m"]
    steps = int(np.ceil(max(np.pi*radius/settings["max_surface_step_m"],
                            np.pi/settings["max_normal_step_rad"])))
    vertical_steps = int(np.ceil(height/settings["max_surface_step_m"]))
    strokes = []
    for h in np.linspace(-height/2, height/2, settings["horizontal_strokes"]):
        strokes.append([(phi, h) for phi in np.linspace(-np.pi/2, np.pi/2, max(steps, 1)+1)])
    for phi in np.linspace(-np.pi/2, np.pi/2, settings["vertical_strokes"]):
        strokes.append([(phi, h) for h in np.linspace(-height/2, height/2, max(vertical_steps, 1)+1)])
    max_jump, tested, traces = 0., 0, []
    for index, stroke in enumerate(strokes):
        active = []
        for point_index, (phi, h) in enumerate(stroke):
            position, direction = surface_target(center, radius, phi, h, forward)
            nearest = min(surface_solutions, key=lambda item: (item[0]-phi)**2+(item[1]-h)**2)
            previous = nearest[2] if not active else min(active, key=lambda item: item["cost"])["q"]
            result = robot.ik(position, direction, seeds=[previous, nearest[2]],
                              random_seed=seed+index*100+point_index, attempts=2)
            if not result["success"] and result.get("backend", "scipy") == "scipy":
                result = robot.ik(position, direction, seeds=[previous, nearest[2]],
                                  random_seed=seed+index*100+point_index+10000,
                                  attempts=settings["extra_seed_attempts"])
            tested += 1
            if not result["success"]:
                return {"success": False, "reason": "stroke_ik_unresolved" if
                        not result.get("free_roll_exhaustive_for_axial_tool", False) else "stroke_target_rejected",
                        "target_failure_reason": result["failure_reason"],
                        "samples_tested": tested, "strokes_checked": index,
                        "max_joint_step_rad": max_jump, "traces": traces}
            candidates = result.get("valid_candidates", [result])
            next_states = []
            for candidate in candidates:
                best = None
                for state in active or [None]:
                    q = np.asarray(candidate["q_rad"])
                    jump, cost, history = 0., 0., []
                    if state is not None:
                        q = nearby_configuration(robot, q, state["q"])
                        if q is None:
                            continue
                        delta = q-state["q"]
                        jump = float(max(abs(delta)))
                        if jump > settings["max_joint_step_rad"]:
                            continue
                        substeps = settings.get("joint_interpolation_substeps", 3)
                        if any(check_self_collision(robot, state["q"]+fraction*delta)["collision"]
                               for fraction in np.linspace(0, 1, substeps+2)[1:-1]):
                            continue
                        cost, history = state["cost"]+float(delta@delta), state["trace"]
                    entry = {"phi_rad": float(phi), "height_m": float(h), "q_rad": q.tolist(),
                             "position_error_m": candidate["position_error_m"],
                             "axis_error_rad": candidate["axis_error_rad"]}
                    option = {"q": q, "cost": cost, "trace": history+[entry],
                              "max_jump": max(jump, 0. if state is None else state["max_jump"])}
                    if best is None or cost < best["cost"]:
                        best = option
                if best is not None:
                    next_states.append(best)
            if not next_states:
                return {"success": False, "reason": "stroke_branch_connection_unresolved",
                        "samples_tested": tested, "strokes_checked": index,
                        "max_joint_step_rad": max_jump, "traces": traces}
            active = next_states
        best = min(active, key=lambda item: item["cost"])
        traces.append(best["trace"])
        max_jump = max(max_jump, best["max_jump"])
    return {"success": True, "reason": None, "samples_tested": tested,
            "strokes_checked": len(strokes), "max_joint_step_rad": max_jump,
            "branch_method": "layered graph with sampled joint-interpolation self-collision screening",
            "traces": traces}


class PlacementEvaluator:
    """Evaluate all required sampled surface targets and representative strokes."""

    def __init__(self, robot):
        self.robot = robot
        task = robot.config["task"]
        if task["collision_checking"] and not task.get("self_collision", {}).get("enabled", False):
            raise ValueError("Collision checking requires configured self-collision capsules; environment collision is not implemented")
        if task["drawing_contact_force_enabled"]:
            raise ValueError("Contact-force analysis is not implemented; select gravity only")
        cylinder = task["cylinder"]
        if not cylinder["free_tool_roll"]:
            raise ValueError("This version constrains tool axis only and requires free roll")
        if cylinder["radius_m"] <= 0 or cylinder["drawing_height_m"] <= 0:
            raise ValueError("Cylinder dimensions must be positive")
        self.forward = cylinder_direction(robot, cylinder)
        self.direction_mode = cylinder["side_direction_mode"]
        self.settings = robot.config["numerics"]["workspace"]
        continuity = self.settings["continuity"]
        if continuity["enabled"]:
            if any(continuity[k] <= 0 for k in ("horizontal_strokes", "vertical_strokes",
                                               "max_surface_step_m", "max_normal_step_rad",
                                               "max_joint_step_rad", "extra_seed_attempts")):
                raise ValueError("Enabled representative stroke checks need positive counts and step limits")
        if self.settings["cube_verification_spacing_m"] <= 0 or self.settings["max_cube_candidates_to_verify"] < 1:
            raise ValueError("Cube refinement needs positive spacing and a positive candidate budget")
        if self.settings["boundary_refinement_levels"] < 0 or self.settings["max_refined_cells_per_level"] < 0:
            raise ValueError("Boundary refinement levels and budgets must be nonnegative")
        if self.settings["surface_phi_samples"] < 3 or self.settings["surface_height_samples"] < 2:
            raise ValueError("Surface sampling needs at least 3 azimuths and 2 heights")
        self.phis = np.linspace(-np.pi/2, np.pi/2, self.settings["surface_phi_samples"])
        self.heights = np.linspace(-cylinder["drawing_height_m"]/2,
                                   cylinder["drawing_height_m"]/2,
                                   self.settings["surface_height_samples"])
        self.radius = cylinder["radius_m"]
        self.cache = {}

    def evaluate(self, center):
        """Return feasible/invalid/unresolved with diagnostics and path witnesses."""
        center = np.asarray(center, dtype=float)
        key = tuple(np.round(center, 10))
        if key in self.cache:
            return self.cache[key]
        seed = self.settings["seed"]+len(self.cache)*1000
        forward = cylinder_direction(self.robot, self.robot.config["task"]["cylinder"], center)
        surface_solutions, failures = [], []
        max_pos, max_angle, attempts = 0., 0., 0
        for phi, h in product(self.phis, self.heights):
            position, direction = surface_target(center, self.radius, phi, h, forward)
            # Solve neighboring targets from nearby successful configurations.
            starts = [] if not surface_solutions else [min(
                surface_solutions, key=lambda item: (item[0]-phi)**2+(item[1]-h)**2)[2]]
            result = self.robot.ik(position, direction, seeds=starts, random_seed=seed+attempts)
            attempts += result["attempts"]
            if not result["success"] and result.get("backend", "scipy") == "scipy" and result["failure_reason"] != "outside_conservative_reach_bound":
                retry = self.robot.config["numerics"]["ik"]["retry_attempts"]
                result = self.robot.ik(position, direction, seeds=starts,
                                       random_seed=seed+attempts+500, attempts=retry)
                attempts += result["attempts"]
            if result["success"]:
                surface_solutions.append((float(phi), float(h), np.asarray(result["q_rad"])))
                max_pos = max(max_pos, result["position_error_m"])
                max_angle = max(max_angle, result["axis_error_rad"])
            else:
                failures.append({"phi_rad": float(phi), "height_m": float(h), **result})
        rejection_reasons = {"outside_conservative_reach_bound", "no_valid_analytic_pose_branch",
                             "analytic_branches_outside_joint_limits", "all_valid_analytic_branches_self_collide"}
        rejected = [f for f in failures if f["failure_reason"] in rejection_reasons]
        invalid = bool(rejected)
        status = "invalid" if invalid else "unresolved" if failures else "feasible"
        reason = rejected[0]["failure_reason"] if invalid else failures[0]["failure_reason"] if failures else None
        continuity = None
        if not failures and self.settings["continuity"]["enabled"]:
            continuity = check_strokes(self.robot, center, forward, surface_solutions, seed+50000)
            if not continuity["success"]:
                status = "invalid" if continuity["reason"] == "stroke_target_rejected" else "unresolved"
                reason = continuity["reason"]
        result = {"center_m": center.tolist(), "status": status, "reason": reason,
                  "side_direction_world": forward.tolist(),
                  "surface_samples_tested": len(self.phis)*len(self.heights),
                  "surface_samples_passed": len(surface_solutions), "ik_attempts": attempts,
                  "max_successful_position_error_m": max_pos if surface_solutions else None,
                  "max_successful_axis_error_rad": max_angle if surface_solutions else None,
                  "failures": failures, "continuity": continuity,
                  "surface_solutions": [{"phi_rad": p, "height_m": h, "q_rad": q.tolist()}
                                        for p, h, q in surface_solutions]}
        self.cache[key] = result
        return result


def cell_cube_contained(cells, lower, upper, domain_bounds, margin=0.):
    """Check every overlapping represented leaf cell, including interior holes.

    Parameters
    ----------
    cells : sequence of dict
        Complete disjoint leaf partition with lower_m/upper_m and status.
    lower, upper : ndarray, shape (3,)
        Cube bounds for the object reference point, in meters.
    domain_bounds : ndarray, shape (3, 2)
    margin : float
        Expand the box by this clearance before testing containment.

    Returns
    -------
    bool
        False for any invalid/unknown overlapping cell or missing coverage.
        Feasible cell classifications themselves are sampling approximations.
    """
    lower, upper = np.asarray(lower)-margin, np.asarray(upper)+margin
    domain = np.asarray(domain_bounds)
    if np.any(lower < domain[:, 0]-1e-10) or np.any(upper > domain[:, 1]+1e-10):
        return False
    covered = 0.
    for cell in cells:
        widths = np.minimum(upper, cell["upper_m"])-np.maximum(lower, cell["lower_m"])
        if np.all(widths > 1e-10):
            if cell["status"] != "feasible":
                return False
            covered += float(np.prod(widths))
    return bool(np.isclose(covered, np.prod(upper-lower), atol=1e-10, rtol=1e-7))


def find_cube(cells, bounds, dimensions, margin, evaluator, settings):
    """Find and refine a sampled-contained fixed box; never infer from corners."""
    dimensions = np.asarray(dimensions, dtype=float)
    bounds = np.asarray(bounds)
    if dimensions.shape != (3,) or np.any(dimensions <= 0) or margin < 0:
        raise ValueError("Cube dimensions must be positive and margin nonnegative")
    axes = [np.unique([v for c in cells for v in
                       [c["lower_m"][j], c["center_m"][j], c["upper_m"][j]]])
            for j in range(3)]
    candidates = []
    for center in product(*axes):
        center = np.asarray(center)
        lower, upper = center-dimensions/2, center+dimensions/2
        if cell_cube_contained(cells, lower, upper, bounds, margin):
            candidates.append(center)
    candidates.sort(key=lambda c: np.linalg.norm(c-np.asarray(evaluator.robot.config["task"]["cylinder"]["reference_center_world_m"])))
    tested, refinements = 0, []
    for center in candidates[:settings["max_cube_candidates_to_verify"]]:
        lower, upper = center-dimensions/2, center+dimensions/2
        counts = np.ceil((dimensions+2*margin)/settings["cube_verification_spacing_m"]).astype(int)
        points = [np.linspace(lower[j]-margin, upper[j]+margin, counts[j]+1) for j in range(3)]
        passed = True
        for point in product(*points):
            result = evaluator.evaluate(point)
            refinements.append({"center_m": list(point), "status": result["status"], "reason": result["reason"]})
            if result["status"] != "feasible":
                passed = False
                break
        tested += 1
        if passed:
            return {"found": True, "center_m": center.tolist(), "dimensions_m": dimensions.tolist(),
                    "lower_m": lower.tolist(), "upper_m": upper.tolist(), "margin_m": margin,
                    "cell_containment_candidates": len(candidates), "candidates_refined": tested,
                    "refinement_samples": refinements,
                    "verification": "All overlapping represented leaf cells plus sampled interior/boundary refinement; not a continuous guarantee"}
    return {"found": False, "dimensions_m": dimensions.tolist(), "margin_m": margin,
            "cell_containment_candidates": len(candidates), "candidates_refined": tested,
            "refinement_samples": refinements,
            "reason": "No qualifying cube found at the evaluated resolution and candidate/search settings; not a proof of impossibility"}


def analyze_workspace(robot, progress=print):
    """Sample a disjoint cell grid, refine limited boundaries, and estimate volume."""
    evaluator = PlacementEvaluator(robot)
    settings = evaluator.settings
    bounds, spacing = np.asarray(settings["bounds_m"], float), np.asarray(settings["spacing_m"], float)
    if bounds.shape != (3, 2) or spacing.shape != (3,) or not np.all(np.isfinite(bounds)) or np.any(spacing <= 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Workspace needs finite increasing bounds and positive spacing")
    counts = np.ceil((bounds[:, 1]-bounds[:, 0])/spacing).astype(int)
    # Exact domain partition: effective spacings can be smaller than requested.
    edges = [np.linspace(bounds[j, 0], bounds[j, 1], counts[j]+1) for j in range(3)]
    cells = []
    history = []

    def evaluate_cell(lower, upper, level, grid_index=None):
        center = (lower+upper)/2
        result = evaluator.evaluate(center)
        cell = {"lower_m": lower.tolist(), "upper_m": upper.tolist(),
                "center_m": center.tolist(), "level": level, "grid_index": grid_index,
                "status": result["status"], "reason": result["reason"], "is_leaf": True,
                "volume_m3": float(np.prod(upper-lower))}
        history.append(cell)
        return cell

    total = int(np.prod(counts))
    for index, grid_index in enumerate(product(*(range(n) for n in counts)), 1):
        lower = np.array([edges[j][grid_index[j]] for j in range(3)])
        upper = np.array([edges[j][grid_index[j]+1] for j in range(3)])
        cells.append(evaluate_cell(lower, upper, 0, list(grid_index)))
        if index == 1 or index % 10 == 0 or index == total:
            progress(f"Workspace coarse cells: {index}/{total}; feasible {sum(c['status']=='feasible' for c in cells)}")
    for level in range(settings["boundary_refinement_levels"]):
        # Touching cells with differing statuses define sampled boundaries.
        boundary = []
        for cell in cells:
            low, high = np.array(cell["lower_m"]), np.array(cell["upper_m"])
            for other in cells:
                if other["status"] == cell["status"]:
                    continue
                overlap = np.minimum(high, other["upper_m"])-np.maximum(low, other["lower_m"])
                if np.all(overlap >= -1e-10):
                    boundary.append(cell)
                    break
        boundary.sort(key=lambda c: (c["status"] != "feasible", c["center_m"]))
        chosen = boundary[:settings["max_refined_cells_per_level"]]
        for index, cell in enumerate(chosen, 1):
            cell["is_leaf"] = False
            cells.remove(cell)
            low, high = np.array(cell["lower_m"]), np.array(cell["upper_m"])
            middle = (low+high)/2
            for octant in product([0, 1], repeat=3):
                lower = np.where(octant, middle, low)
                upper = np.where(octant, high, middle)
                cells.append(evaluate_cell(lower, upper, level+1))
            progress(f"Boundary refinement level {level+1}: {index}/{len(chosen)} cells split")
    counts_by_status = {s: sum(c["status"] == s for c in cells) for s in ("feasible", "invalid", "unresolved")}
    volumes = {s: sum(c["volume_m3"] for c in cells if c["status"] == s) for s in counts_by_status}
    feasible = [c for c in cells if c["status"] == "feasible"]
    extents = None if not feasible else {"lower_m": np.min([c["lower_m"] for c in feasible], axis=0).tolist(),
                                       "upper_m": np.max([c["upper_m"] for c in feasible], axis=0).tolist()}
    cube_settings = robot.config["task"]["cube"]
    progress("Checking all represented cells for the fixed placement cube...")
    cube = find_cube(cells, bounds, cube_settings["dimensions_m"], cube_settings["margin_m"], evaluator, settings)
    front_region = robot.config["task"].get("front_region")
    front_verification = None
    if front_region and front_region["enabled"]:
        low, high = np.asarray(front_region["lower_m"]), np.asarray(front_region["upper_m"])
        clipped = {status: 0. for status in counts_by_status}
        for cell in cells:
            widths = np.maximum(0., np.minimum(high, cell["upper_m"])-np.maximum(low, cell["lower_m"]))
            clipped[cell["status"]] += float(np.prod(widths))
        region_volume = float(np.prod(high-low))
        fraction = clipped["feasible"]/region_volume
        front_verification = {"lower_m": low.tolist(), "upper_m": high.tolist(),
                              "dimensions_m": (high-low).tolist(), "volume_m3": region_volume,
                              "estimated_volume_m3_by_status": clipped,
                              "estimated_feasible_fraction": fraction,
                              "most_of_sampled_region_passed": bool(fraction > .5),
                              "all_overlapping_cells_passed_with_margin": cell_cube_contained(cells, low, high, bounds, cube_settings["margin_m"]),
                              "meaning": "Clipped disjoint cell-volume estimate, not proof of continuous or physical feasibility. Feasible includes all configured surface and representative stroke checks."}
    return {"analysis": "sampled kinematic feasibility with representative stroke checks" if settings["continuity"]["enabled"] else "sampled surface kinematic feasibility",
            "collision_checking": robot.config["task"]["collision_checking"],
            "collision_scope": "Approximate self-collision capsules; environment/object/table unchecked" if robot.config["task"]["collision_checking"] else "No collision screening",
            "ik_backend": robot.config["numerics"]["ik"].get("backend", "scipy"), "free_tool_roll": True,
            "analytic_rejection_scope": "Enumerated prescribed nominal targets; no exhaustive search over all tolerance-relaxed orientations",
            "status_definitions": {"feasible": "All surface samples and enabled representative stroke checks passed; see collision_scope",
                                   "invalid": "A required surface/stroke target was rejected by geometric IK, configured limits, approximate self-collision screening, or the conservative reach bound",
                                   "unresolved": "Numerical IK or representative continuation did not establish feasibility"},
            "side_direction_mode": evaluator.direction_mode,
            "fixed_side_direction_world": evaluator.forward.tolist() if evaluator.direction_mode != "toward_base_per_placement" else None,
            "side_reference_center_m": robot.config["task"]["cylinder"]["reference_center_world_m"],
            "requested_spacing_m": spacing.tolist(),
            "effective_coarse_spacing_m": ((bounds[:, 1]-bounds[:, 0])/counts).tolist(),
            "settings": settings, "counts_leaf_cells": counts_by_status,
            "volume_m3_by_status": volumes, "estimated_feasible_volume_m3": volumes["feasible"],
            "volume_method": "Sum disjoint leaf-cell volumes classified by center surface/stroke samples. Refined parents excluded. Invalid/unresolved cells excluded from feasible volume. Sampling estimate, not a lower bound or continuous guarantee.",
            "feasible_cell_extents_m": extents,
            "extents_note": "Enclosing extents only; interior holes and unresolved cells may exist",
            "cube": cube, "cells": history, "placements": list(evaluator.cache.values()),
            "front_region_verification": front_verification,
            "limitations": ["Self-collision uses illustrative capsules when enabled; table/object/obstacle collision remains unchecked",
                            "Finite surface and placement sampling",
                            "Only configured representative strokes checked; intervening continuous motion not proven",
                            "No drawing-force, dynamics, or actuator-capacity evaluation",
                            "Boundary refinement and cube candidate searches are budget limited"]}
