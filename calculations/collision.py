"""Approximate self-collision using explicitly configured capsules and housings.

Shapes are demonstration envelopes, not CAD. Adjacent connected bodies may
overlap at joints and are excluded explicitly. Object/table checks remain absent.
"""

import numpy as np


def segment_distance(p0, p1, q0, q1):
    """Return the minimum distance (meters) between two closed 3D segments."""
    p0, p1, q0, q1 = map(lambda x: np.asarray(x, float), (p0, p1, q0, q1))
    u, v, w = p1-p0, q1-q0, p0-q0
    a, b, c, d, e = u@u, u@v, v@v, u@w, v@w
    candidates = []
    # Minima occur in the interior or on one of the four segment boundaries.
    for s in (0., 1.):
        t = np.clip((e+s*b)/c, 0., 1.) if c > 1e-20 else 0.
        candidates.append(np.linalg.norm(w+s*u-t*v))
    for t in (0., 1.):
        s = np.clip((t*b-d)/a, 0., 1.) if a > 1e-20 else 0.
        candidates.append(np.linalg.norm(w+s*u-t*v))
    denominator = a*c-b*b
    if denominator > 1e-20:
        s, t = (b*e-c*d)/denominator, (a*e-b*d)/denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w+s*u-t*v))
    return float(min(candidates))


def check_self_collision(robot, q):
    """Return intersecting nonadjacent capsule pairs and signed clearance in m."""
    settings = robot.config["task"].get("self_collision", {})
    if not settings.get("enabled", False) or not robot.config["task"]["collision_checking"]:
        return {"checked": False, "collision": False, "pairs": []}
    frames = robot.frames(q)
    capsules = []
    for shape in settings["capsules"]:
        if shape["radius_m"] <= 0 or not 0 <= shape["frame_index"] <= robot.n:
            raise ValueError("Self-collision capsule needs a positive radius and valid frame")
        transform = frames[shape["frame_index"]]
        p0 = (transform @ np.r_[shape["start_local_m"], 1.])[:3]
        p1 = (transform @ np.r_[shape["end_local_m"], 1.])[:3]
        capsules.append((shape, p0, p1))
    excluded = {frozenset(pair) for pair in settings["excluded_pairs"]}
    pairs = []
    for index, (first, p0, p1) in enumerate(capsules):
        for second, q0, q1 in capsules[index+1:]:
            if frozenset((first["name"], second["name"])) in excluded:
                continue
            clearance = segment_distance(p0, p1, q0, q1)-first["radius_m"]-second["radius_m"]
            if clearance < settings["clearance_m"]-1e-10:
                pairs.append({"first": first["name"], "second": second["name"],
                              "clearance_m": clearance})
    return {"checked": True, "collision": bool(pairs), "pairs": pairs}
