"""Spatial standard-DH/URDF-chain kinematics and free-roll inverse kinematics.

All internal quantities use SI units. Frame 0 is the base; frame i+1 is after
joint i's configured transform. Joint indices are zero based.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def dh_transform(theta, a, d, alpha):
    """Return standard DH Rz(theta) Tz(d) Tx(a) Rx(alpha), shape (4, 4)."""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct],
                     [st, ct*ca, -ct*sa, a*st],
                     [0., sa, ca, d], [0., 0., 0., 1.]])


def rpy_transform(xyz, rpy):
    """Return URDF origin transform T(xyz) Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]]
    transform[:3, 3] = [x, y, z]
    return transform


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    return result


def load_config(path):
    """Read the shared JSON configuration; never fill in missing hardware data."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    Robot(config)  # Validate the mathematical model immediately.
    return config


def apply_dh_table(config, path):
    """Replace DH geometry and reference angles from CSV or tab-separated input.

    Parameters
    ----------
    config : dict
        Shared configuration; masses/limits/transforms remain explicit inputs.
    path : path-like
        Columns theta_rad, a_m, d_m, alpha_rad, or the supplied table's
        Theta/Greek-letter headers with bracketed SI units. Theta is a reference
        pose, never silently interpreted as fixed zero offsets.

    Returns
    -------
    dict
        The updated configuration, also modified in place, validated as a Robot.
    """
    content = Path(path).read_text(encoding="utf-8-sig")
    delimiter = "\t" if "\t" in content.splitlines()[0] else ","
    rows = list(csv.DictReader(content.splitlines(), delimiter=delimiter))
    if len(rows) != len(config["robot"]["joints"]):
        raise ValueError("DH table row count must match the configured joint count")
    aliases = {"theta_rad": {"theta_rad", "theta [rad]", "θ [rad]", "θ_rad"},
               "a_m": {"a_m", "a [m]"}, "d_m": {"d_m", "d [m]"},
               "alpha_rad": {"alpha_rad", "alpha [rad]", "α [rad]", "α_rad"}}
    for index, row in enumerate(rows):
        normalized = {key.strip().lower(): value for key, value in row.items() if key is not None}
        values = {}
        for field, names in aliases.items():
            keys = names.intersection(normalized)
            if len(keys) != 1:
                raise ValueError(f"DH table needs one unambiguous {field} column")
            values[field] = float(normalized[next(iter(keys))])
        config["robot"]["reference_q_rad"][index] = values.pop("theta_rad")
        config["robot"]["joints"][index].update(values)
    config["provenance"]["dh_table_override"] = str(Path(path).resolve())
    Robot(config)
    return config


def _vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _transform(value, name):
    matrix = _vector(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if not (np.allclose(matrix[3], [0, 0, 0, 1]) and
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) and
            np.isclose(np.linalg.det(rotation), 1., atol=1e-7)):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return matrix


class Robot:
    """Shared spatial model with explicit mass attachment and joint limits.

    Parameters
    ----------
    config : dict
        Standard-DH rows, base/tool transforms, masses, and settings.
    """

    def __init__(self, config):
        self.config = config
        model = config["robot"]
        if model["convention"] not in ("standard_dh", "urdf_serial_chain"):
            raise ValueError("Convention must be standard_dh or urdf_serial_chain")
        self.convention = model["convention"]
        rows = model["joints"]
        if not rows:
            raise ValueError("At least one joint is required")
        self.n = len(rows)
        self.continuous = np.asarray([row.get("continuous", False) for row in rows], dtype=bool)
        self._geometric_ik = None
        if any(row["type"] != "revolute" for row in rows):
            raise ValueError("This version supports revolute joints only")
        if self.convention == "standard_dh":
            self.geometry = _vector([[r["a_m"], r["d_m"], r["alpha_rad"]]
                                     for r in rows], (self.n, 3), "DH geometry")
            self.offsets = _vector([r["zero_offset_rad"] for r in rows], (self.n,), "joint zero offsets")
            self.joint_origins = self.joint_axes_local = None
        else:
            self.geometry = self.offsets = None
            self.joint_origins = np.asarray([_transform(r["origin_transform"], "URDF joint origin") for r in rows])
            self.joint_axes_local = _vector([r["axis_local"] for r in rows], (self.n, 3), "URDF joint axes")
            self.joint_axes_local /= np.linalg.norm(self.joint_axes_local, axis=1)[:, None]
        self.limits = _vector([r["limits_rad"] for r in rows],
                              (self.n, 2), "joint limits")
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Each joint needs lower < upper limit")
        self.base = _transform(model["world_from_base"], "world_from_base")
        self.tool = _transform(model["flange_from_tip"], "flange_from_tip")
        self.gravity = _vector(model["gravity_m_s2"], (3,), "gravity")
        self.pointing_axis = _vector(model["tip_pointing_axis"], (3,), "pointing axis")
        if not np.isclose(np.linalg.norm(self.pointing_axis), 1.):
            raise ValueError("tip_pointing_axis must be a unit vector")
        self.reference_q = _vector(model["reference_q_rad"], (self.n,), "reference pose")
        self.check_q(self.reference_q)
        self.masses = config["masses"]
        if not self.masses:
            raise ValueError("A supported mass inventory is required; missing masses cannot imply zero torque")
        names = set()
        for mass in self.masses:
            if mass["name"] in names:
                raise ValueError("Mass names must be unique; avoid duplicate components")
            names.add(mass["name"])
            weight = float(mass["mass_kg"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Mass must be finite and nonnegative")
            frame = mass["frame_index"]
            if not isinstance(frame, int) or not 0 <= frame <= self.n:
                raise ValueError("Mass frame_index must be between 0 and joint count")
            _vector(mass["com_local_m"], (3,), "mass center")
            # Standard-DH frame k is moved by precisely joints 0 through k-1.
            # A motor housing on the upstream body belongs to frame k-1.
            if mass["affected_joints"] != list(range(frame)):
                raise ValueError(f"{mass['name']}: affected_joints must match its frame")
        limits = model.get("continuous_joint_torque_limits_Nm")
        self.torque_limits = None if limits is None else _vector(limits, (self.n,), "joint torque ratings")
        if self.torque_limits is not None and np.any(self.torque_limits <= 0):
            raise ValueError("Continuous joint output ratings must be positive")

    def check_q(self, q):
        """Validate a joint vector (n,) in radians against configured limits."""
        q = _vector(q, (self.n,), "joint angles")
        bounded = ~self.continuous
        if np.any(bounded & (q < self.limits[:, 0]-1e-10)) or np.any(bounded & (q > self.limits[:, 1]+1e-10)):
            raise ValueError("Joint configuration violates limits")
        return q

    def frames(self, q):
        """Return world transforms, shape (n+1, 4, 4), including base."""
        q = self.check_q(q)
        frames = [self.base.copy()]
        if self.convention == "standard_dh":
            for angle, offset, (a, d, alpha) in zip(q, self.offsets, self.geometry):
                frames.append(frames[-1] @ dh_transform(angle+offset, a, d, alpha))
        else:
            for angle, origin, axis in zip(q, self.joint_origins, self.joint_axes_local):
                frames.append(frames[-1] @ origin @ axis_rotation(axis, angle))
        return np.asarray(frames)

    def axes(self, q):
        """Return joint origins and world unit axes, each shape (n, 3)."""
        frames = self.frames(q)
        if self.convention == "standard_dh":
            return frames[:-1, :3, 3], frames[:-1, :3, 2]
        joint_frames = np.asarray([frames[i] @ self.joint_origins[i] for i in range(self.n)])
        origins = joint_frames[:, :3, 3]
        axes = np.einsum("nij,nj->ni", joint_frames[:, :3, :3], self.joint_axes_local)
        return origins, axes

    def tool_pose(self, q):
        """Return the world drawing-tip transform, shape (4, 4)."""
        return self.frames(q)[-1] @ self.tool

    def mass_centers(self, q):
        """Return mass-center world positions, shape (number of masses, 3)."""
        frames = self.frames(q)
        return np.asarray([(frames[m["frame_index"]] @
                            np.r_[m["com_local_m"], 1.])[:3] for m in self.masses]).reshape(-1, 3)

    def tip_state(self, q):
        """Return position, direction, geometric Jacobian, and direction Jacobian.

        Returns
        -------
        position, direction : ndarray, shape (3,)
        geometric : ndarray, shape (6, n)
            Rows are linear velocity (m/s) then angular velocity (rad/s).
        direction_jacobian : ndarray, shape (3, n)
            Derivative of the unit tool axis with respect to q (radians).
        """
        frames = self.frames(q)
        pose = frames[-1] @ self.tool
        position = pose[:3, 3]
        direction = pose[:3, :3] @ self.pointing_axis
        origins, axes = self.axes(q)
        linear = np.cross(axes, position-origins).T
        return position, direction, np.vstack((linear, axes.T)), np.cross(axes, direction).T

    @property
    def reach_bound_m(self):
        """Conservative spherical reach bound, including drawing-tip offset."""
        if self.convention == "standard_dh":
            return float(np.sum(np.abs(self.geometry[:, :2])) + np.linalg.norm(self.tool[:3, 3]))
        return float(sum(np.linalg.norm(t[:3, 3]) for t in self.joint_origins)+np.linalg.norm(self.tool[:3, 3]))

    def ik(self, position, direction, seeds=None, random_seed=0, attempts=None):
        """Solve tip position and pointing direction with free tool roll.

        Parameters
        ----------
        position, direction : ndarray, shape (3,)
            World target in meters and unit pointing direction.
        seeds : sequence of ndarray, shape (n,), optional
            Warm starts, followed by reference/center/random branches.

        Returns
        -------
        dict
            Success, q, position/orientation errors, attempts, and failure reason.
            Solver termination alone never establishes success.
        """
        target = _vector(position, (3,), "IK target position")
        axis = _vector(direction, (3,), "IK target direction")
        if not np.isclose(np.linalg.norm(axis), 1.):
            raise ValueError("IK direction must be a unit vector")
        settings = self.config["numerics"]["ik"]
        pos_tol, angle_tol = settings["position_tolerance_m"], settings["axis_tolerance_rad"]
        count = attempts if attempts is not None else settings["initial_attempts"]
        if count < 1 or pos_tol <= 0 or angle_tol <= 0 or settings["max_evaluations"] < 1:
            raise ValueError("IK attempts and tolerances must be positive")
        if np.linalg.norm(target-self.base[:3, 3]) > self.reach_bound_m+pos_tol:
            return {"success": False, "q_rad": None, "position_error_m": None,
                    "axis_error_rad": None, "failure_reason": "outside_conservative_reach_bound",
                    "attempts": 0, "evaluations": 0}
        backend = settings.get("backend", "scipy")
        if backend == "ik_geo" and self.convention != "standard_dh":
            raise ValueError("ik_geo requires the standard-DH arm family; use scipy for URDF chains")
        if backend == "ik_geo":
            from ik_backend import GeometricIK
            if self._geometric_ik is None:
                self._geometric_ik = GeometricIK(self)
            return self._geometric_ik.solve(target, axis, seeds)
        if backend != "scipy":
            raise ValueError("IK backend must be ik_geo or scipy")
        lower, upper = self.limits.T
        starts = list(seeds or []) + [self.reference_q, (lower+upper)/2]
        rng = np.random.default_rng(random_seed)
        starts += [rng.uniform(lower, upper) for _ in range(count)]
        best, evaluations = None, 0
        scale = settings["position_residual_scale_m"]
        if scale <= 0:
            raise ValueError("IK residual scale must be positive")
        posture_weight = settings["posture_regularization_weight"]
        if posture_weight < 0:
            raise ValueError("Posture regularization must be nonnegative")
        for attempt, seed in enumerate(starts[:count], 1):
            start = np.clip(seed, lower+1e-9, upper-1e-9)
            cache_q, cache_state = None, None

            def state(q):
                nonlocal cache_q, cache_state
                if cache_q is None or not np.array_equal(q, cache_q):
                    cache_q, cache_state = q.copy(), self.tip_state(q)
                return cache_state

            def residual(q):
                p, a, _, _ = state(q)
                # Vector difference avoids accepting an antiparallel axis, for
                # which a cross-product-only residual would incorrectly be zero.
                # Small seed-distance regularization stabilizes the free-roll
                # null space. Acceptance still uses the actual task errors.
                return np.r_[(p-target)/scale, a-axis, posture_weight*(q-start)]

            def jac(q):
                _, _, geometric, direction_jac = state(q)
                return np.vstack((geometric[:3]/scale, direction_jac,
                                  posture_weight*np.eye(self.n)))

            result = least_squares(residual, start,
                                   jac=jac, bounds=(lower, upper),
                                   max_nfev=settings["max_evaluations"],
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            p, a, _, _ = self.tip_state(result.x)
            pos_error = float(np.linalg.norm(p-target))
            angle_error = float(np.arccos(np.clip(a @ axis, -1., 1.)))
            evaluations += result.nfev
            success = pos_error <= pos_tol and angle_error <= angle_tol
            collision = None
            if success and self.config["task"].get("self_collision", {}).get("enabled", False):
                from collision import check_self_collision
                collision = check_self_collision(self, result.x)
                success = not collision["collision"]
            candidate = {"success": bool(success), "q_rad": result.x.tolist(),
                         "position_error_m": pos_error, "axis_error_rad": angle_error,
                         "failure_reason": None if success else "numerical_ik_unresolved",
                         "attempts": attempt, "evaluations": evaluations,
                         "solver_message": str(result.message), "backend": "scipy",
                         "self_collision": collision}
            if best is None or pos_error/pos_tol+angle_error/angle_tol < best[0]:
                best = (pos_error/pos_tol+angle_error/angle_tol, candidate)
            if success:
                return candidate
        best[1]["attempts"] = count
        best[1]["evaluations"] = evaluations
        return best[1]
