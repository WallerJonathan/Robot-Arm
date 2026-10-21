"""Gravity-only actuator holding torque, with per-component weight moments.

TODO: Add inverse dynamics using component inertia tensors and joint positions,
velocities, and accelerations. Evaluate gravity, acceleration, velocity-dependent
effects, and external contact loads along trajectories. Include actuator speed
dependence and continuous/peak limits. Masses and centers of mass alone do not
determine full dynamics. Gravity-only results are not drawing-force capacity.
"""

import numpy as np
from scipy.optimize import minimize


def static_torque(robot, q):
    """Return signed holding torque and component contributions in N*m.

    Parameters
    ----------
    robot : Robot
    q : ndarray, shape (n,)
        Joint angles in radians.

    Returns
    -------
    total : ndarray, shape (n,)
    contributions : ndarray, shape (number of masses, n)
        Actuator torque opposing each component's gravitational moment.
    """
    origins, axes = robot.axes(q)
    positions = robot.mass_centers(q)
    contributions = np.zeros((len(robot.masses), robot.n))
    for index, (mass, position) in enumerate(zip(robot.masses, positions)):
        moment = np.cross(position-origins, mass["mass_kg"]*robot.gravity)
        holding = -np.einsum("ij,ij->i", axes, moment)
        contributions[index, mass["affected_joints"]] = holding[mass["affected_joints"]]
    return contributions.sum(axis=0), contributions


def potential_energy(robot, q):
    """Return gravitational potential energy in joules for gradient checks."""
    centers = robot.mass_centers(q)
    return float(sum(-m["mass_kg"]*np.dot(robot.gravity, p)
                     for m, p in zip(robot.masses, centers)))


def analyze_torque(robot):
    """Evaluate reference poses and estimate each joint's own worst holding torque.

    Sampling and multistart bounded optimization are estimates, not global proofs.
    All joint-limited poses are considered; collision filtering is not implemented.
    """
    settings = robot.config["numerics"]["torque"]
    rng = np.random.default_rng(settings["seed"])
    lower, upper = robot.limits.T
    reference_q = robot.reference_q
    horizontal = robot.config["robot"]["horizontal_reference"]
    horizontal_q = robot.check_q(horizontal["q_rad"])
    candidates = [reference_q, horizontal_q, (lower+upper)/2]
    candidates += list(rng.uniform(lower, upper, size=(settings["samples"], robot.n)))
    candidates = np.asarray(candidates)
    values = np.asarray([static_torque(robot, q)[0] for q in candidates])
    maxima = []
    for joint in range(robot.n):
        order = np.argsort(np.abs(values[:, joint]))[::-1]
        best_q = candidates[order[0]].copy()
        best_value = float(values[order[0], joint])
        # A vertical gravity-parallel axis has exactly zero moment and needs no search.
        if np.max(np.abs(values[:, joint])) > 1e-12:
            for start in candidates[order[:settings["optimization_starts"]]]:
                for sign in (-1., 1.):
                    result = minimize(lambda q: -sign*static_torque(robot, q)[0][joint],
                                      start, method="L-BFGS-B", bounds=robot.limits,
                                      options={"maxiter": settings["max_iterations"], "ftol": 1e-12})
                    value = float(static_torque(robot, result.x)[0][joint])
                    if abs(value) > abs(best_value):
                        best_value, best_q = value, result.x.copy()
        _, component = static_torque(robot, best_q)
        entry = {"joint": joint+1, "signed_holding_Nm": best_value,
                 "estimated_max_abs_Nm": abs(best_value), "q_rad": best_q.tolist(),
                 "q_deg": np.rad2deg(best_q).tolist(),
                 "component_contributions_Nm": component[:, joint].tolist()}
        if robot.torque_limits is not None:
            entry["continuous_joint_output_limit_Nm"] = float(robot.torque_limits[joint])
            entry["estimated_utilization"] = abs(best_value)/robot.torque_limits[joint]
        maxima.append(entry)

    def reference(name, q):
        total, components = static_torque(robot, q)
        weight = np.array([m["mass_kg"]*np.linalg.norm(robot.gravity) for m in robot.masses])
        lever = np.divide(components, weight[:, None], out=np.zeros_like(components),
                          where=weight[:, None] != 0)
        return {"name": name, "q_rad": np.asarray(q).tolist(),
                "signed_holding_Nm": total.tolist(), "absolute_holding_Nm": abs(total).tolist(),
                "component_contributions_Nm": components.tolist(),
                "effective_signed_gravity_levers_m": lever.tolist(),
                "gravity_magnitude_m_s2": float(np.linalg.norm(robot.gravity)),
                "component_masses_kg": [m["mass_kg"] for m in robot.masses]}

    return {"analysis": "gravity_only_static_holding", "units": "N*m",
            "scope": "All joint-limited poses; no collision filtering; estimated numerical maxima",
            "limitations": ["Not drawing-force capacity", "Not inverse dynamics",
                            "Not a proof of global maxima"],
            "search_settings": settings, "maxima": maxima,
            "reference": reference("supplied reference pose", reference_q),
            "horizontal_reference": reference(horizontal["description"], horizontal_q),
            "mass_names": [m["name"] for m in robot.masses],
            "total_mass_kg": sum(m["mass_kg"] for m in robot.masses)}
