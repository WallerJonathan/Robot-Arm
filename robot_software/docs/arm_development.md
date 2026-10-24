# Arm development notes

## Project context

Based on `Proposal Presentation - Leonardo Da Pinchy.pptx` and
`Design Presentation - Leonardo Da Pinchy.pptx` in the parent capstone directory.
The design deck marks slides 12 onward as old material; slides 2–10 describe the
current approach.

The project transfers SVG line artwork onto a known curved object, drawing only
on its reachable front region. The design targets a 0.4 m cube working area,
at least 0.50 m reach, and at most 2 mm end-effector and camera pose errors.
Those are project targets, not properties verified for the SO-101 reference.
The proposed stack uses a Jetson Orin Nano, smart serial bus servos,
ros2_control, MoveIt 2, and RViz. Camera/AprilTag alignment, SVG surface mapping,
and browser UI remain outside this starter workspace's scope.

## Confirmed custom arm hardware

The custom SO-101 derivative has six independently actuated arm joints, each
driven by a Feetech smart bus servo motor. These six joints provide the arm's
six degrees of freedom for positioning and orienting the pen. A seventh servo
operates the end-effector gripper to grip and release the pen. There are seven
servos in total; the gripper is separate from the six arm degrees of freedom.

Future URDF, hardware interfaces, controller configuration, and MoveIt planning
groups must account for six arm joints and a separate gripper actuator. Exact
servo models, bus IDs, joint geometry, and limits remain to be documented.

## Reference model and remaining modeling work

The stock reference has five arm joints (`shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`) and a separate `gripper` actuator.
It is a reference model only and does not yet represent the confirmed six-joint
custom arm with its seventh gripper servo. Add the sixth arm joint using the
actual joint arrangement before creating IK or controller configurations.

The reference is unmodified apart from mesh resource paths. Do not infer the
custom arm's reach or accuracy from it. Replace the gripper geometry with the
measured pen mount and define a calibrated `tool0`/`pen_tip` frame. Record its
position and orientation relative to the wrist, along with tool mass and inertia.
Use metres and radians, and preserve consistent joint names across URDF,
controllers, and planning configuration.

## Suggested next packages

1. Extend `pinchy_description` with the actual derivative model, pen mount,
   measured joint limits, collision geometry, and tip transform.
2. Add `pinchy_hardware` for the serial servo ros2_control hardware interface
   once motor IDs, serial protocol, calibration, and stop behavior are defined.
3. Add controller configuration and a separate hardware launch to
   `pinchy_bringup`; verify feedback before accepting motion commands.
4. Generate `pinchy_moveit_config` from the finalized arm model for planning,
   collision checking, and trajectory execution.
5. Add drawing path execution once surface coordinates and pen frames agree.

The display launch is a kinematic preview, with no servo communication or force
control. Keep it independent of future hardware bringup.
