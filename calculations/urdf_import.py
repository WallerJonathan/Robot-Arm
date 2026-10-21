"""Import a serial tool chain, limits, and inertial data from a URDF file."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def _numbers(text, default):
    return [float(x) for x in (text or default).split()]


def _origin(element):
    from robot import rpy_transform
    if element is None:
        return np.eye(4)
    return rpy_transform(_numbers(element.get("xyz"), "0 0 0"),
                         _numbers(element.get("rpy"), "0 0 0"))


def apply_urdf(config, path, tip_link="gripper_frame_link", reference_q=None):
    """Replace demonstration DH/COM data using a URDF tool chain.

    Meshes are not copied. Masses on side branches are folded onto the closest
    active-chain frame at their URDF zero configuration (the SO-101 moving jaw).
    """
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    links = {x.get("name"): x for x in root.findall("link")}
    joints = {x.get("name"): x for x in root.findall("joint")}
    child_joint = {x.find("child").get("link"): x for x in joints.values()}
    if tip_link not in links:
        raise ValueError(f"URDF tip link {tip_link!r} does not exist")
    chain = []
    cursor = tip_link
    while cursor in child_joint:
        joint = child_joint[cursor]
        chain.append(joint)
        cursor = joint.find("parent").get("link")
    base_link = cursor
    chain.reverse()

    active = [j for j in chain if j.get("type") in ("revolute", "continuous")]
    unsupported = [j.get("name") for j in chain if j.get("type") not in ("fixed", "revolute", "continuous")]
    if unsupported:
        raise ValueError(f"Tool chain has unsupported joints: {unsupported}")
    if not active:
        raise ValueError("URDF tool chain contains no revolute joints")

    serial = []
    pending = np.eye(4)
    frame_for_link = {base_link: (0, np.eye(4))}
    frame_index = 0
    for joint in chain:
        pending = pending @ _origin(joint.find("origin"))
        child = joint.find("child").get("link")
        if joint.get("type") == "fixed":
            frame_for_link[child] = (frame_index, pending.copy())
            continue
        limit = joint.find("limit")
        lower = -np.pi if joint.get("type") == "continuous" else float(limit.get("lower"))
        upper = np.pi if joint.get("type") == "continuous" else float(limit.get("upper"))
        serial.append({"name": joint.get("name"), "type": "revolute",
                       "origin_transform": pending.tolist(),
                       "axis_local": _numbers(joint.find("axis").get("xyz"), "1 0 0"),
                       "limits_rad": [lower, upper],
                       "continuous": joint.get("type") == "continuous",
                       "urdf_effort_limit": None if limit is None or limit.get("effort") is None else float(limit.get("effort")),
                       "urdf_velocity_limit": None if limit is None or limit.get("velocity") is None else float(limit.get("velocity"))})
        frame_index += 1
        pending = np.eye(4)
        frame_for_link[child] = (frame_index, np.eye(4))
    flange_from_tip = pending

    # Resolve every link's zero-pose transform to its nearest active frame. This
    # also handles side branches such as the SO-101 moving jaw at gripper q=0.
    unresolved = set(links)-set(frame_for_link)
    while unresolved:
        progress = False
        for child in list(unresolved):
            joint = child_joint.get(child)
            if joint is None:
                unresolved.remove(child)
                continue
            parent = joint.find("parent").get("link")
            if parent in frame_for_link:
                idx, transform = frame_for_link[parent]
                frame_for_link[child] = (idx, transform @ _origin(joint.find("origin")))
                unresolved.remove(child)
                progress = True
        if not progress:
            break

    masses = []
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None or name not in frame_for_link:
            continue
        mass = float(inertial.find("mass").get("value"))
        idx, link_from_frame = frame_for_link[name]
        local = (link_from_frame @ _origin(inertial.find("origin")))[:3, 3]
        inertia = inertial.find("inertia")
        masses.append({"name": f"urdf_{name}", "mass_kg": mass,
                       "frame_index": idx, "com_local_m": local.tolist(),
                       "affected_joints": list(range(idx)),
                       "source": "URDF inertial",
                       "inertia_kg_m2": None if inertia is None else {key: float(inertia.get(key)) for key in ("ixx","ixy","ixz","iyy","iyz","izz")}})

    result = deepcopy(config)
    q = [0.] * len(serial) if reference_q is None else list(reference_q)
    result["robot"] = {"convention": "urdf_serial_chain", "joints": serial,
                       "reference_q_rad": q,
                       "world_from_base": np.eye(4).tolist(),
                       "gravity_m_s2": [0, 0, -9.80665],
                       "flange_from_tip": flange_from_tip.tolist(),
                       "tip_pointing_axis": [0, 0, 1],
                       "continuous_joint_torque_limits_Nm": None,
                       "horizontal_reference": {"q_rad": q, "description": "URDF calibrated zero pose"},
                       "urdf_source": str(path), "urdf_robot_name": root.get("name"),
                       "urdf_base_link": base_link, "urdf_tip_link": tip_link,
                       "active_joint_names": [j["name"] for j in serial]}
    result["masses"] = masses
    result["numerics"]["ik"]["backend"] = "scipy"
    result["task"]["collision_checking"] = False
    result["task"]["self_collision"]["enabled"] = False
    result["provenance"]["result_label"] = f"SO-101 URDF model: {path.name}"
    result["provenance"]["supplied_actual"] = ["URDF joint origins/axes/limits", "URDF link masses/COMs/inertias"]
    result["provenance"]["interpretation_assumptions"] = [
        f"URDF serial chain from {base_link} to {tip_link}",
        "Five revolute joints affect the SO-101 tool pose; gripper actuation is a side branch",
        "Side-branch inertials use their zero joint configuration",
        "URDF calibrated joint limits are used"]
    result["provenance"]["demonstration_values_to_replace"] = ["drawing tool transform and pointing axis", "object/table placement", "collision model for modified hardware"]
    result["provenance"]["mass_inventory_note"] = f"URDF inertial inventory: {sum(m['mass_kg'] for m in masses):.9g} kg, including base and moving jaw."
    result["provenance"]["urdf_import_notes"] = [
        "Five arm joints control the tool pose; the sixth SO-101 servo actuates the gripper side branch.",
        "Side-branch inertials are attached at their URDF zero configuration.",
        "URDF collision meshes are not approximated by the older demonstration capsules."]
    return result
