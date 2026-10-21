"""Established IK-Geo adapter, using this robot's configured geometry.

No built-in UR model/dimensions are substituted. DH is converted to POE, and every
candidate is validated by the independent DH FK and configured collision model.
"""

import numpy as np
from collision import check_self_collision


def nearby_configuration(robot, q, previous):
    """Choose allowed 2*pi equivalents near a previous pose; unwrap continuous axes."""
    q = np.asarray(q, float).copy()
    for j in range(robot.n):
        if robot.continuous[j]:
            q[j] += 2*np.pi*round((previous[j]-q[j])/(2*np.pi))
            continue
        turns = np.arange(int(np.ceil((robot.limits[j, 0]-q[j]-1e-10)/(2*np.pi))),
                          int(np.floor((robot.limits[j, 1]-q[j]+1e-10)/(2*np.pi)))+1)
        choices = q[j]+2*np.pi*turns
        if not len(choices):
            return None
        q[j] = choices[np.argmin(abs(choices-previous[j]))]
    return q


def _axis_basis(axis, preferred=None):
    reference = np.eye(3)[np.argmin(abs(axis))] if preferred is None else preferred
    x = reference-axis*np.dot(axis, reference)
    if np.linalg.norm(x) < 1e-8:
        x = np.eye(3)[np.argmin(abs(axis))]
        x -= axis*np.dot(axis, x)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(axis, x), axis))


class GeometricIK:
    """Closed-form branch enumeration for the configured three-parallel arm family."""

    def __init__(self, robot):
        from ik_geo import Robot as LibraryRobot
        from robot import dh_transform
        if robot.n != 6:
            raise ValueError("IK-Geo three-parallel backend requires six revolute joints")
        self.robot = robot
        frames = [np.eye(4)]
        for a, d, alpha in robot.geometry:
            frames.append(frames[-1] @ dh_transform(0., a, d, alpha))
        frames = np.asarray(frames)
        self.home_rotation = frames[-1, :3, :3]
        axes = frames[:-1, :3, 2].copy()
        translations = np.vstack(([0., 0., 0.], np.diff(frames[:, :3, 3], axis=0)))
        if not (np.allclose(axes[1], axes[2], atol=1e-8) and
                np.allclose(axes[2], axes[3], atol=1e-8)):
            raise ValueError("Geometry does not have three parallel axes; use scipy backend or another IK-Geo family")
        # Shift origins along their own axes into the solver's canonical form.
        # This changes coordinates, never physical axes or link dimensions.
        shift = np.dot(translations[1], axes[0])*axes[0]
        translations[0] += shift
        translations[1] -= shift
        shift = np.dot(translations[5], axes[4])*axes[4]
        translations[4] += shift
        translations[5] -= shift
        if np.linalg.norm(translations[5]) > 1e-8:
            raise ValueError("Last two joint axes must intersect for this analytic family")
        # Supplied pi/2 twists have 10-digit rounding; remove only sub-nanometer
        # coordinate noise. Independent FK still uses the original exact inputs.
        axes[abs(axes) < 1e-9] = 0.
        translations[abs(translations) < 1e-9] = 0.
        self.solver = LibraryRobot.three_parallel_two_intersecting(axes.tolist(), translations.tolist())
        self.axial_tool = (np.linalg.norm(robot.tool[:2, 3]) < 1e-10 and
                           np.allclose(robot.tool[:3, :3] @ robot.pointing_axis, [0,0,1], atol=1e-10))
        self.roll_invariant_collision = all(
            shape["frame_index"] != 6 or
            (np.linalg.norm(shape["start_local_m"][:2]) < 1e-10 and
             np.linalg.norm(shape["end_local_m"][:2]) < 1e-10)
            for shape in robot.config["task"].get("self_collision", {}).get("capsules", []))

    def full_pose_candidates(self, world_tip_pose):
        """Return all library branches, including least-squares flags, before filters."""
        robot = self.robot
        flange = np.linalg.inv(robot.base) @ world_tip_pose @ np.linalg.inv(robot.tool)
        rotation = flange[:3, :3] @ self.home_rotation.T
        # ik-geo 1.0.3's Python binding transposes rotation arrays internally
        # despite its row-major comment; source and FK/IK round trips confirm it.
        solutions = self.solver.get_ik(rotation.T.tolist(), flange[:3, 3].tolist())
        return [(np.asarray(q)-robot.offsets, least_squares) for q, least_squares in solutions]

    def wrist_center_diagnostic(self, position, direction):
        """Explain the UR-layout inner wrist-center exclusion using current dimensions."""
        robot = self.robot
        expected_twists = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
        if not (self.axial_tool and np.allclose(robot.geometry[:, 2], expected_twists, atol=1e-8)
                and np.allclose(robot.geometry[[0,3,4,5], 0], 0.)
                and np.allclose(robot.geometry[[1,2], 1], 0.)):
            return None
        extension = robot.geometry[5, 1]+robot.tool[2, 3]
        wrist_world = np.asarray(position)-extension*np.asarray(direction)
        wrist_base = (np.linalg.inv(robot.base) @ np.r_[wrist_world, 1.])[:3]
        radius = float(np.linalg.norm(wrist_base[:2]))
        minimum = float(abs(robot.geometry[3, 1]))
        settings = robot.config["numerics"]["ik"]
        allowance = settings["position_tolerance_m"]+abs(extension)*2*np.sin(settings["axis_tolerance_rad"]/2)
        return {"wrist_center_world_m": wrist_world.tolist(),
                "wrist_center_horizontal_radius_m": radius,
                "minimum_horizontal_wrist_center_radius_m": minimum,
                "maximum_center_displacement_from_tolerances_m": float(allowance),
                "inside_exclusion_even_with_tolerances": bool(radius+allowance < minimum),
                "meaning": "The configured UR-layout shoulder/wrist offset creates an inner wrist-center exclusion; this is not the tip's position-only workspace"}

    def solve(self, position, direction, seeds=None):
        """Solve position/inward axis, filter limits/collisions, rank nearby branches."""
        robot = self.robot
        settings = robot.config["numerics"]["ik"]
        previous = np.asarray(seeds[0] if seeds else robot.reference_q)
        preferred = robot.tool_pose(previous)[:3, 0]
        target_rotation = _axis_basis(direction, preferred) @ _axis_basis(robot.pointing_axis).T
        # With an axial tool and unrestricted joint 6, tool roll is exactly
        # decoupled: all feasible arm branches exist for any selected target roll.
        roll_count = 1 if self.axial_tool and self.roll_invariant_collision else settings["free_roll_samples"]
        all_candidates, rejected_limit, rejected_collision, rejected_pose = [], 0, 0, 0
        branches = 0
        for roll in np.linspace(0., 2*np.pi, roll_count, endpoint=False):
            basis = _axis_basis(direction)
            c, s = np.cos(roll), np.sin(roll)
            roll_rotation = basis @ np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ basis.T
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = roll_rotation @ target_rotation, position
            for raw_q, least_squares in self.full_pose_candidates(pose):
                branches += 1
                q = nearby_configuration(robot, raw_q, previous)
                if q is None:
                    rejected_limit += 1
                    continue
                if self.axial_tool and self.roll_invariant_collision and robot.continuous[-1]:
                    q[-1] = previous[-1]  # Tool roll is free: avoid needless wrist rotation.
                p, axis, _, _ = robot.tip_state(q)
                position_error = float(np.linalg.norm(p-position))
                axis_error = float(np.arccos(np.clip(axis @ direction, -1., 1.)))
                if position_error > settings["position_tolerance_m"] or axis_error > settings["axis_tolerance_rad"]:
                    rejected_pose += 1
                    continue
                collision = check_self_collision(robot, q)
                if collision["collision"]:
                    rejected_collision += 1
                    continue
                all_candidates.append({"q_rad": q.tolist(), "position_error_m": position_error,
                                       "axis_error_rad": axis_error, "library_least_squares": bool(least_squares)})
        all_candidates.sort(key=lambda r: np.linalg.norm(np.asarray(r["q_rad"])-previous))
        exhaustive = self.axial_tool and self.roll_invariant_collision and (robot.continuous[-1] or np.ptp(robot.limits[-1]) >= 2*np.pi-1e-8)
        reason = None
        if not all_candidates:
            if not exhaustive:
                reason = "free_roll_search_unresolved"
            elif rejected_collision:
                reason = "all_valid_analytic_branches_self_collide"
            elif rejected_limit and not rejected_pose:
                reason = "analytic_branches_outside_joint_limits"
            else:
                reason = "no_valid_analytic_pose_branch"
        return {"success": bool(all_candidates), **(all_candidates[0] if all_candidates else
                {"q_rad": None, "position_error_m": None, "axis_error_rad": None}),
                "failure_reason": reason, "backend": "ik_geo", "attempts": roll_count,
                "evaluations": branches, "analytic_branches_tested": branches,
                "branches_rejected_limits": rejected_limit, "branches_rejected_pose": rejected_pose,
                "branches_rejected_self_collision": rejected_collision,
                "free_roll_exhaustive_for_axial_tool": bool(exhaustive),
                "wrist_center_diagnostic": self.wrist_center_diagnostic(position, direction),
                "rejection_criterion": "Enumerated branches for prescribed exact targets, independently checked with FK tolerances; not a search over all tolerance-relaxed orientations",
                "valid_candidates": all_candidates}
