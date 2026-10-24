# SO-101 reference model

Source: https://github.com/TheRobotStudio/SO-ARM100
Local source revision: `5f6d2b876a53a4872e405b991dd925556c9e38a4` (copied from the sibling SO-ARM100 checkout).
Original URDF: `Simulation/SO101/so101_new_calib.urdf`.
Meshes: the 13 STL files referenced by that URDF, from its `assets/` directory.
License: Apache-2.0; preserved in `LICENSE-SO-ARM100`.
Original generated URDF attribution comments are retained.

Modification: mesh paths use `package://pinchy_description/meshes/` so the
installed ROS package is self-contained. Geometry, joints, limits, and inertias
are otherwise unchanged. This is the stock follower with its original gripper,
not the custom writing arm. It has five positioning joints plus a gripper joint.
The new calibration places joint zero at the middle of its range.

Validate collision coverage, joint limits, calibration, and inertias against the
built arm before using this reference for planning or hardware control.
