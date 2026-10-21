# Robot calculations

Reusable spatial, gravity-only statics and cylinder-placement kinematic analysis.
Approximate self-collision screening uses configured demonstration capsules.
All project files, caches, and run outputs stay inside `calculations/`. No Git
commands are used by these programs.

## Run

From this directory, using the installed Linux/WSL environment:

```bash
.venv/bin/python analyze_robot.py --mode torque
.venv/bin/python analyze_robot.py --mode workspace --latex
.venv/bin/python analyze_robot.py --mode both --config configs/robot_config.json --latex
.venv/bin/python analyze_robot.py --mode workspace --config configs/robot_config_front_facing_robot.json --latex
.venv/bin/python analyze_robot.py --mode both --dh-table configs/dh_parameters.csv --latex
.venv/bin/python analyze_robot.py --mode both --quick --latex
.venv/bin/python analyze_robot.py --mode torque --urdf ../../SO-ARM100/Simulation/SO101/so101_new_calib.urdf
.venv/bin/python compare_so101.py --samples 5000
.venv/bin/python analyze_robot.py --help
```

Running without arguments opens a small menu. A complete CLI call never prompts.
Default paths and relative overrides are resolved relative to this directory,
even when the command is launched from another working directory. `--output-dir`
changes the parent folder; each run creates a new timestamp/UUID directory.
Output/config overrides are accepted only within `calculations/`. A URDF is a
read-only input and may be outside this folder.

The virtual environment is a **venv**, not a pyenv-managed Python installation.
Python 3.12, NumPy, SciPy, Matplotlib, IK-Geo 1.0.3, MuJoCo 3.3.7, and pytest are installed. Recreate the
dependencies using `requirements.txt`, or `requirements-dev.txt` for checks.
`requirements-lock.txt` records the exact versions used here. The WSL environment
is not a native Windows environment. Its installation on the Windows-mounted
drive can be slow.

## Current input and provenance

`configs/robot_config.json` contains all model and demonstration values. The six
DH rows and reference angles preserve the values supplied in the conversation.
The assumed convention is **standard DH**:

```text
T(i-1 to i) = Rz(q_i + zero_offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
```

The supplied theta column is **one reference pose**, not a fixed zero offset.
Signed negative `a` values are preserved. All six joints are revolute. Modified-DH
import is not implemented. URDF serial-chain import is described below.

The optional CSV/TSV importer accepts `theta_rad,a_m,d_m,alpha_rad` or the original
Greek-letter/table headers with bracketed SI units. It overrides joint geometry
and reference angles, while masses, centers, offsets, limits, and tool transforms
remain explicit JSON inputs. **Update local centers of mass if geometry changes**;
the importer does not infer new mass properties or updated collision shapes.

### URDF input and SO-101

`--urdf PATH` replaces the JSON's demonstration DH model, masses, centers of mass,
and joint limits with values parsed from a URDF. `--tip-link` selects the endpoint
and defaults to `gripper_frame_link`. Input URDFs are read-only and may live outside
`calculations/`; generated files remain here. Edits to joint origins, axes, limits,
link inertials, or topology therefore flow into the calculations without deriving
another DH table.

The importer supports revolute/continuous serial tool chains and follows fixed
joints to the requested tip. It folds inertials on side branches onto their nearest
tool-chain frame at the branch joint's URDF zero. For SO-101, the moving jaw's 12 g
mass contributes to arm gravity torque at gripper angle zero. Prismatic joints are
rejected explicitly. Imported runs disable the old illustrative capsule collision
model; use the matching MuJoCo mesh model until URDF mesh collision is integrated.

SO-101 has five joints affecting the gripper pose: shoulder pan/lift, elbow flex,
wrist flex, and wrist roll. Its sixth actuator moves the gripper jaw. It is a
five-axis arm with a gripper, rather than a six-axis pose arm. The repository URDF
provides CAD-derived masses, COMs, inertia tensors, transforms, and calibrated
limits, replacing the 55 g/70 g demonstration inventory for `--urdf` runs.

`compare_so101.py` loads the matching MuJoCo MJCF and evaluates identical joint
configurations in both models. It checks total mass, gripper-frame position, arm
orientation, gravity torque, sampled reach, and MuJoCo contacts. Results go to
`outputs/so101_comparison/`. The source repository notes that base collision meshes
were removed and its gripper joint does not yet implement LeRobot's linear 0–100
command convention.

The results are **ILLUSTRATIVE** because the following are filler values:

| Input | Current demonstration value |
| --- | --- |
| Joint/motor assemblies | Six at 0.055 kg each |
| Link assemblies | Six at 0.070 kg each |
| Joint motion | Full 360° angular period per motor; continuous axes unwrap across angle boundaries |
| Shoulder clearance sector | Demonstration exclusion `[60°, 120°]` downward; allowed interval `[120°, 420°]` |
| Self-collision shapes | 12 mm link radii, 18 mm joint housing radii, 25 mm base radius, 4 mm tool radius; 2 mm clearance |
| Joint zero offsets | Zero; require confirmation from the actual angle convention |
| Joint assembly centers | Upstream frame origin |
| Link centers | Midpoint between adjacent DH origins, expressed in attached frame |
| Tool | 0.012 kg, COM 0.02 m along flange z, tip 0.04 m along flange z |
| Payload | No additional payload |
| Base | World origin, identity orientation |
| Gravity | `[0, 0, -9.80665]` m/s² |
| Table | Top at world z = 0; x/y span −0.8 to +0.8 m, visual reference only |
| Cylinder | Radius 0.035 m, required drawing height 0.10 m |
| Cylinder reference center | `[0.35, 0, 0.30]` m |
| Selected side | Reference-to-base horizontal direction, calculated once: world −x |
| Tool roll | Free |
| Placement box | x = 0…0.4, y = −0.2…+0.2, z = 0.05…0.45 m; 0.005 m containment margin |

The assumed total supported inventory is **0.762 kg**, including the tool.
Brackets/wrist hardware are assumed included in the joint/link assemblies; they
are not added twice. Confirm this allocation by weighing the real components.
If separate hardware or payload exists, add it explicitly and remove it from any
assembly mass that already includes it.

Frame 0 is the base. Frame k is after joint k (joint indices in arrays start at
zero). A mass attached to frame k moves with joints `0..k-1`. Every mass lists its
`affected_joints`, and validation enforces consistency. Motor housings are attached
to the upstream body; their own joint does not rotate their COM. Confirm that the
actual motors are mounted this way. An explicitly transformed upstream COM can
represent a housing not located at a DH origin.

Replace actual limits, offsets, mass/COM inventory, base and tool transforms,
cylinder dimensions, and table/obstacle geometry before interpreting hardware
suitability. Continuous joint output torque ratings are optional and currently
absent. Motor-shaft ratings must be converted to joint output ratings with the
actual transmission. All internal angles are radians and distances are meters.

## What is calculated

`robot.py` provides FK, world joint axes, mass centers, tool pose, geometric and
pointing-axis Jacobians, and IK. The default backend is the established IK-Geo
library, configured from the supplied DH dimensions through `ik_backend.py`.
It enumerates the three-parallel/two-intersecting arm family's branches; no canned
UR model is substituted. Every returned candidate is checked by independent DH
FK, joint limits, and enabled approximate self-collision. A SciPy multistart backend
is retained by setting `numerics.ik.backend` to `scipy`. IK accepts a result only if
its measured position and pointing-axis errors meet the configured tolerances.
The SciPy backend does not accept termination merely because a gradient is small;
a configured seed-distance penalty stabilizes its unconstrained tool roll. For the
current axial tool, free roll is decoupled from arm reach: the analytic backend
can solve one roll and hold joint 6 constant. Non-axial tools use a sampled roll
search and leave unsolved cases unresolved. The IK-Geo binding's column-oriented
rotation arrays are explicitly handled and verified by independent FK/IK checks.

`torque.py` calculates the **actuator torque opposing gravity**:

```text
holding_i = −sum(axis_i · ((COM_k − origin_i) × (mass_k × gravity)))
```

Signs and magnitudes are both exported. Reference and zero-angle horizontal-link
poses are included, followed by random samples and bounded multistart optimization
for each joint independently. These maxima cover all joint-limited poses with
**no collision filtering** and are numerical estimates, not global proofs. Wrist
offsets are transformed at each pose. Joint 6 has zero gravity torque here because
the modeled flange/tool COMs lie on its rotation axis; an offset tool would change
that. Results are **not drawing-force capacity** and **not inverse dynamics**.

`workspace.py` samples positions of the **cylinder reference point at drawing-height
midpoint**, rather than just tip reach. For every accepted placement it tests the
entire configured surface-sample grid of the selected 180° side, with inward
pointing directions specific to each surface normal. The world side direction
stays fixed in the default profile. The separate
`configs/robot_config_front_facing_robot.json` profile recalculates the horizontal
direction toward the base for each cylinder placement. Its chosen half turns with
the object; each accepted placement records its direction. Independent targets enumerate analytic branches with IK-Geo; the optional SciPy
backend uses multiple seeds and additional retries before leaving a numerical
failure unresolved.

Representative horizontal arcs and vertical strokes use a layered graph of IK
branches with bounded sampled joint steps. Graph edges check three intermediate
joint-space samples for approximate self-collision. This avoids greedily choosing
a branch that later reaches a dead end. Allowed 2π-equivalent configurations are chosen near
the preceding pose and must remain within limits. All path witness angles and
errors are saved. These sampled checks do not prove continuity between samples
or that arbitrary drawing paths connect. Velocity, acceleration, and contact loads
are not evaluated.

**Workspace results are sampled feasibility with approximate self-collision
screening** when enabled. `collision.py` tests configured link/joint/tool capsules
and base housing. Connected neighboring bodies are excluded explicitly so their
shared attachment envelopes can overlap. These exclusions and dimensions must be
reviewed against real hardware or CAD. This is not a complete physical collision
model, especially near adjacent hinges. Cylinder, table, and obstacle collision
remain unchecked; the table in plots is only a reference. Contact-force analysis
is still unsupported and fails explicitly if enabled.

Placement status meanings:

- `feasible`: all required surface samples and enabled representative strokes pass.
- `invalid`: a required nominal target is rejected by analytic branch enumeration,
  configured limits/capsules, or a conservative reach bound. This describes the
  prescribed pose under the configured model, not ordinary position-only reach.
  Nominal analytic-pose rejection is not an exhaustive search over all orientations
  inside the allowed numerical tolerance neighborhood.
- `unresolved`: local IK or stroke continuation did not establish feasibility.
  This is not proof that the placement is unreachable.

The default front placement box spans x = 0…0.4, y = −0.2…+0.2, and
z = 0.05…0.45 m. Scan bounds include a 5 mm padding for containment checks.
Requested spacing is 0.1 m; exact domain partitioning produces five 0.082 m cells
per axis before refinement. It uses 9 surface azimuths × 3 heights, three horizontal
and three vertical strokes, deterministic seeds, and one boundary refinement
level limited to six parent-cell splits. The height was chosen after comparing
five 0.4 m tall boxes: it tied for the best sampled surface coverage and keeps the
lowest cylinder drawing extent above the illustrative tabletop. The blue plot
outline marks the requested box even when no qualifying cube is found.
`front_region_verification` clips leaf volumes to that exact requested box;
its fraction includes surface and enabled stroke checks. Full settings are recorded in JSON.
`--quick` uses a smaller scan and saves its overridden settings in the snapshot.

Volumes sum non-overlapping leaf-cell volumes assigned the status of their center's
surface/stroke evaluation. Refined parents are excluded. Unresolved/invalid volumes
are not counted as feasible. A center's feasibility does not prove its entire cell
is feasible, so this is a **sampling estimate, not a rigorous lower bound**. Grid
refinement can substantially change the volume estimate.

Cube search tests all overlapping represented leaf cells, including interiors,
cell extents, unresolved cells, and the configured margin. Candidates must remain
inside the evaluated domain. Qualifying candidates undergo additional sampled
interior and boundary checks; eight corners alone are insufficient. Candidate
search and refinement budgets are recorded. A cube's bounds constrain the cylinder
reference point, not its physical outer bounds. Enclosing workspace extents can
contain holes. A no-cube result means none was found at those search settings.

## Outputs

Each successful run includes `config_snapshot.json` and `run_summary.json`.
Selected analyses add:

- Torque: `torque.json`, `torque_joints.csv`, `torque_components.csv`.
- Workspace: `workspace.json`, `workspace_placements.csv`, `workspace_cells.csv`,
  three distinct 3D PNG views with equal spatial scale, and leaf-cell slice PNGs.
- When a cube is found: a separate cube-detail PNG.
- With `--latex`: `report.tex`, with relative image paths and selected analyses only.

CSV/JSON and images are generated independently of LaTeX. JSON includes the
configuration, provenance, units, definitions, surface solutions, stroke witnesses,
search settings, and limitations. LaTeX includes equations, mass/parameter tables,
component substitutions, maximizing configurations, and cube results. A TeX/PDF
compiler is not required and has not been installed. Existing unrelated output
files, including the original handoff plan, are preserved.

## Focused verification

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider test_analysis.py
.venv/bin/python -B -m pip check
.venv/bin/python -B verify_run.py outputs/front_region/runs/<run_folder>
```

Twelve focused checks cover the `mgr` special case, zero lever arms, vertical-axis gravity
torque, potential-energy gradients, FK/tool Jacobians, accepted/unresolved IK,
explicit attachments, independent IK-Geo FK/IK round trips, full-turn wrapping,
shoulder exclusion, capsule distances and folded-arm collision, fixed and per-placement cylinder
side with changing normals, cube interior
holes/unknown cells/missing coverage, empty workspace handling, and explicit
rejection of unsupported physics. Demonstrations and export/witness verification
are recorded in `outputs/verification/`. These are focused checks, not an
exhaustive certification of a robot.

Dynamics remain a prominent TODO in `torque.py`: actual component inertia tensors,
trajectories, velocity/acceleration effects, contact loads, and actuator speed and
continuous/peak ratings are needed before adding inverse dynamics.

## Reach versus drawing workspace investigation

The UR3e manufacturer lists 500 mm reach, which is not a guarantee that an entire
cylinder surface can be drawn at each center or orientation. Our configured tool
adds 40 mm past the flange, and the supplied dimensions remain unchanged.

`investigate_workspace.py` compares position-only center reach and the required
cylinder poses on the same 108 original coarse centers, with original full-turn
limits and no collisions, to isolate the IK solver effect. It found validated
position-only solutions at 71 centers. Both SciPy and IK-Geo passed all 15 cylinder
surface targets at only 4 centers. The first plots showed cylinder-placement
feasibility, not ordinary tip reach.

The latest default evaluates the requested front box, with the configured shoulder
sector, self-collision screening, and branch-graph stroke checks. See
[the front-box comparison](outputs/front_region/RESULTS.md) for both facing modes,
17.4% versus 26.5% sampled coverage, and the height-independent wrist geometry bound.
See [the investigation](outputs/IK_INVESTIGATION.md)
for comparison images, numerical evidence, the wrist-center exclusion example,
and current run outputs. Earlier outputs are preserved as historical runs.

Run the reproducible comparison with:

```bash
.venv/bin/python -B investigate_workspace.py
```

This diagnostic references the preserved first full run to compare historical
SciPy results without rerunning a local search. Accepted position-only results
are checked by FK; unsuccessful orientation searches remain uncertain. Its
point cloud is a sample, not proof of a solid reachable sphere.
