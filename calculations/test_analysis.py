"""Focused checks for the mathematical contracts and important failure cases."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from robot import Robot, load_config, apply_dh_table
from torque import static_torque, potential_energy
from workspace import (cylinder_direction, surface_target, cell_cube_contained,
                       PlacementEvaluator, analyze_workspace)
from reports import tex_escape

CONFIG = Path(__file__).resolve().parent/"configs"/"robot_config.json"


def test_supplied_dh_table_preserves_geometry_and_reference_meaning():
    config = load_config(CONFIG)
    original = deepcopy(config["robot"])
    updated = apply_dh_table(config, CONFIG.parent/"dh_parameters.csv")
    assert updated["robot"]["joints"] == original["joints"]
    assert updated["robot"]["reference_q_rad"] == original["reference_q_rad"]
    assert all(joint["zero_offset_rad"] == 0 for joint in updated["robot"]["joints"])


def single_mass_robot(vertical=False):
    config = deepcopy(load_config(CONFIG))
    config["robot"]["joints"] = [{"type": "revolute", "a_m": 1., "d_m": 0.,
                                    "alpha_rad": 0., "zero_offset_rad": 0., "limits_rad": [-4., 4.]}]
    config["robot"]["reference_q_rad"] = [0.]
    config["robot"]["world_from_base"] = np.eye(4).tolist()
    if not vertical:
        config["robot"]["world_from_base"] = [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]
    config["robot"]["flange_from_tip"] = np.eye(4).tolist()
    config["masses"] = [{"name": "test_mass", "mass_kg": 2., "frame_index": 1,
                          "com_local_m": [0,0,0], "affected_joints": [0]}]
    return Robot(config)


def test_hand_moment_and_zero_lever_arm():
    robot = single_mass_robot()
    np.testing.assert_allclose(static_torque(robot, [0])[0], [-2*9.80665], atol=1e-10)
    np.testing.assert_allclose(static_torque(robot, [np.pi/2])[0], [0], atol=1e-10)
    np.testing.assert_allclose(static_torque(single_mass_robot(vertical=True), [.4])[0], [0], atol=1e-10)


def test_actual_geometry_torque_matches_energy_gradient():
    robot = Robot(load_config(CONFIG))
    q, h = robot.reference_q, 1e-6
    basis = np.eye(robot.n)*h
    gradient = [(potential_energy(robot, q+e)-potential_energy(robot, q-e))/(2*h) for e in basis]
    total, components = static_torque(robot, q)
    np.testing.assert_allclose(total, gradient, atol=2e-8)
    np.testing.assert_allclose(components.sum(axis=0), total, atol=1e-12)
    assert abs(total[0]) < 1e-12


def test_tool_jacobians_and_ik_diagnostics():
    robot = Robot(load_config(CONFIG))
    q, h = robot.reference_q, 1e-6
    position, direction, jac, axis_jac = robot.tip_state(q)
    numeric_position, numeric_axis = [], []
    for e in np.eye(robot.n)*h:
        p1, a1, _, _ = robot.tip_state(q+e)
        p0, a0, _, _ = robot.tip_state(q-e)
        numeric_position.append((p1-p0)/(2*h))
        numeric_axis.append((a1-a0)/(2*h))
    np.testing.assert_allclose(jac[:3], np.asarray(numeric_position).T, atol=2e-8)
    np.testing.assert_allclose(axis_jac, np.asarray(numeric_axis).T, atol=2e-8)
    result = robot.ik(position, direction, seeds=[q], attempts=1)
    assert result["success"] and result["position_error_m"] < 1e-8
    assert result["axis_error_rad"] < 1e-7
    impossible = robot.ik([10,0,0], direction)
    assert not impossible["success"]
    assert impossible["failure_reason"] == "outside_conservative_reach_bound"
    # Exhausting the solver on a reachable-sized target must not accept an iterate.
    robot.config["numerics"]["ik"]["max_evaluations"] = 1
    robot.config["numerics"]["ik"]["backend"] = "scipy"
    unresolved = robot.ik(position+[.03,.02,0], -direction, seeds=[q], attempts=1)
    assert not unresolved["success"]
    assert unresolved["failure_reason"] == "numerical_ik_unresolved"


def test_mass_attachment_is_explicit_and_validated():
    config = load_config(CONFIG)
    config["masses"][0]["affected_joints"] = [0]
    with pytest.raises(ValueError, match="affected_joints"):
        Robot(config)


def test_cylinder_fixed_direction_and_distinct_normals():
    robot = Robot(load_config(CONFIG))
    forward = cylinder_direction(robot, robot.config["task"]["cylinder"])
    np.testing.assert_allclose(forward, [-1,0,0])
    center = np.array([.3,.1,.4])
    p0, n0 = surface_target(center, .035, 0, 0, forward)
    p1, n1 = surface_target(center+[.1,.2,0], .035, 0, 0, forward)
    np.testing.assert_allclose(p1-p0, [.1,.2,0])
    np.testing.assert_allclose(n0, n1)
    _, n2 = surface_target(center, .035, np.pi/2, 0, forward)
    assert abs(np.dot(n0,n2)) < 1e-10
    settings = dict(robot.config["task"]["cylinder"], side_direction_mode="toward_base_per_placement")
    rotated = cylinder_direction(robot, settings, center)
    np.testing.assert_allclose(rotated, [-3/np.sqrt(10), -1/np.sqrt(10), 0])
    assert not np.allclose(rotated, cylinder_direction(robot, settings, center+[0,.2,0]))


def test_so101_urdf_import_uses_five_tool_joints_and_cad_inertials():
    from urdf_import import apply_urdf
    source = CONFIG.parents[3]/"SO-ARM100"/"Simulation"/"SO101"/"so101_new_calib.urdf"
    if not source.exists():
        pytest.skip("read-only SO-ARM100 source repository is unavailable")
    imported = apply_urdf(load_config(CONFIG), source)
    robot = Robot(imported)
    assert robot.n == 5
    assert imported["robot"]["active_joint_names"] == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    assert np.isclose(sum(m["mass_kg"] for m in robot.masses), .632006001)
    assert any(m["name"] == "urdf_moving_jaw_so101_v1_link" and m["frame_index"] == 5 for m in robot.masses)
    np.testing.assert_allclose(robot.tool_pose(np.zeros(5))[:3, 3], [.39136147, -9.2121e-6, .22646971], atol=2e-7)


def test_cube_rejects_interior_holes_unknown_and_missing_coverage():
    from itertools import product
    cells = [{"lower_m": list(c), "upper_m": list(np.array(c)+1), "status": "feasible"}
             for c in product(range(3), repeat=3)]
    domain = [[0,3]]*3
    assert cell_cube_contained(cells, [0,0,0], [3,3,3], domain)
    center = next(c for c in cells if c["lower_m"] == [1,1,1])
    center["status"] = "invalid"
    assert not cell_cube_contained(cells, [0,0,0], [3,3,3], domain)
    center["status"] = "unresolved"
    assert not cell_cube_contained(cells, [0,0,0], [3,3,3], domain)
    cells.remove(center)
    assert not cell_cube_contained(cells, [0,0,0], [3,3,3], domain)


def test_empty_workspace_tests_all_samples_and_no_cube():
    config = load_config(CONFIG)
    settings = config["numerics"]["workspace"]
    settings.update(bounds_m=[[10,10.1],[10,10.1],[10,10.1]], spacing_m=[.1]*3,
                    boundary_refinement_levels=0)
    robot = Robot(config)
    result = analyze_workspace(robot, progress=lambda _: None)
    assert result["counts_leaf_cells"]["feasible"] == 0
    assert result["estimated_feasible_volume_m3"] == 0
    assert result["feasible_cell_extents_m"] is None
    assert not result["cube"]["found"]
    for p in result["placements"]:
        assert p["surface_samples_tested"] == settings["surface_phi_samples"]*settings["surface_height_samples"]
    assert sum(result["volume_m3_by_status"].values()) == pytest.approx(.001)


def test_unsupported_physics_fails_explicitly_and_tex_escaping():
    config = load_config(CONFIG)
    config["task"]["collision_checking"] = True
    config["task"]["self_collision"]["enabled"] = False
    with pytest.raises(ValueError, match="Collision checking"):
        PlacementEvaluator(Robot(config))
    assert tex_escape("tool_A&B") == r"tool\_A\&B"


def test_established_solver_fk_and_ik_agree_with_independent_dh():
    from ik_backend import GeometricIK, nearby_configuration
    config = load_config(CONFIG)
    config["task"]["collision_checking"] = False
    robot = Robot(config)
    adapter = GeometricIK(robot)
    rng = np.random.default_rng(18500)
    for q in rng.uniform(robot.limits[:, 0], robot.limits[:, 1], size=(20, 6)):
        pose = robot.tool_pose(q)
        foreign_rotation, foreign_position = adapter.solver.forward_kinematics((q+robot.offsets).tolist())
        flange = np.linalg.inv(robot.base) @ robot.frames(q)[-1]
        np.testing.assert_allclose(np.array(foreign_rotation).T @ adapter.home_rotation, flange[:3,:3], atol=2e-8)
        np.testing.assert_allclose(foreign_position, flange[:3,3], atol=2e-8)
        candidates = [nearby_configuration(robot, candidate, q)
                      for candidate, ls in adapter.full_pose_candidates(pose) if not ls]
        candidates = [candidate for candidate in candidates if candidate is not None]
        assert candidates
        assert min(np.linalg.norm(robot.tool_pose(candidate)-pose) for candidate in candidates) < 2e-8


def test_capsule_distance_and_actual_reference_screening():
    from collision import segment_distance, check_self_collision
    assert segment_distance([0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]) == pytest.approx(0.)
    assert segment_distance([0,0,0],[1,0,0],[0,1,0],[1,1,0]) == pytest.approx(1.)
    assert segment_distance([0,0,0],[0,0,0],[1,0,0],[1,0,0]) == pytest.approx(1.)
    robot = Robot(load_config(CONFIG))
    assert not check_self_collision(robot, robot.reference_q)["collision"]
    assert check_self_collision(robot, [0, 2.1, np.pi, 0, 0, 0])["collision"]


def test_full_turn_continuity_and_shoulder_exclusion():
    from ik_backend import nearby_configuration
    robot = Robot(load_config(CONFIG))
    before = robot.reference_q.copy()
    before[0] = 2*np.pi-.01
    target = before.copy()
    target[0] = .01
    adjusted = nearby_configuration(robot, target, before)
    assert adjusted[0]-before[0] == pytest.approx(.02)
    robot.check_q(adjusted)  # Continuous base may cross the angular display interval.
    target[1] = np.pi/2
    assert nearby_configuration(robot, target, before) is None
