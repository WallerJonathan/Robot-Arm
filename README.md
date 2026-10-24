# Robot-Arm
For CMU ECE Capstone (18-500); Fall 2026, Group 2B - Leonardo Da Pinchy

## Arm hardware

The custom SO-101 derivative has **six arm degrees of freedom**, with six
independently actuated joints driven by **Feetech smart bus servo motors**.
A **seventh servo** operates the end-effector gripper to grip and release the
pen. The gripper actuator is separate from the six arm joints: seven servos
total, with six degrees of freedom for positioning and orienting the pen.

## Repo Layout

- `calculations/` — Workspace and torque calculations based on robot specifications.
- [`robot_software/`](robot_software/README.md) — ROS 2 arm workspace, reference
  SO-101 description, and visualization launch files.
