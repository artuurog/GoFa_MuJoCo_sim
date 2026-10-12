# Setup Mujoco simulations with the ABB CRB15000 GoFa
MuJoCo simulation environment with an ABB GoFa robot for tabletop manipulation tasks.

## Repo structure

```
GoFa_MuJoCo_sim/
├── gofa/                     # pick-and-place scenes
├── gofa_contact_force/       # peg-in-hole / insertion scenes with contact-force sensing
├── ik_new/                   # Python inverse kinematics + simulation
│   ├── gofa/                 # self-contained model used by the IK scripts
│   └── src/
│       ├── IK_solver.py
│       └── sim.py
└── README.md
```

### `gofa/`
- `assets/`: STL meshes of the robot (links, gripper body, fingers) and the robot model fragments (`gofa.xml`, `gofa_asset*.xml`, `gofa_body*.xml`, with and without hand)
- `robot_no_gripper.xml`: basic model of the robot, without the gripper
- `pick_place_scene.xml`: pick-and-place environment (table, colored blocks, container) with gravity compensation
- `visual_servo_scene.xml`: scene with an external camera, for visual servoing

### `gofa_contact_force/`
Peg-in-hole insertion tasks in which joint torques and end-effector contact forces are measured.
- `contact_forces_insertion.xml`: insertion of a cylinder into a box
- `contact_bead_segments.xml`: variant with a bead made of segments
- `contact_forces_insertion_sdf_bead.xml`: variant using the SDF tube plugin (see below)
- `tube_test.xml`: minimal model to check that the SDF tube plugin is installed correctly
- `tube_plugin.dll`: the SDF tube plugin
- `robot_no_gripper.xml` and `assets/`: robot model and meshes

### `ik_new/`
Inverse kinematics for the GoFa, applied to the simulation.
- `gofa/`: copy of the robot model and meshes, with `gofa.xml` (robot with gripper) as the entry model, `contact_forces_insertion.xml` and `robot_no_gripper.xml`
- `src/IK_solver.py`: `LevenbergMarquardtIK`, a Levenberg-Marquardt IK solver for position-only or full 6-DOF pose targets. It works directly on the simulation model, uses only the joints on the chain to the target site (default `ee_site`), respects joint limits, and supports random restarts. The convention for orientation targets is `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`
- `src/sim.py`: loads `ik_new/gofa/gofa.xml`, solves IK for a target pose (`TARGET_POS`, `TARGET_RPY_DEG`, set at the top of the file), and moves the robot to the solution with a smooth joint-space interpolation in the MuJoCo viewer. Run it from the repo root:
  ```
  python ik_new/src/sim.py
  ```

Requirements: Python with `numpy` and `mujoco`.


## How to use SDF tube plugin

SDF is a Mujoco plugin that allows to generate custom geometries defined as signed distance fields.
In order to use the tube_plugin.dll:
1. Clone the [official Mujoco repo](https://github.com/google-deepmind/mujoco)
1. Find the folder of Mujoco's simulate.exe
2. In the same folder, create folder mujoco_plugin/
3. Paste tube_plugin.dll in mujoco_plugin/

The simulate.exe will be able now to open the tube geometry and visualize it.
To verify the correct installation of the plugin, open the tube_test.xml model using simulate.exe. If the model loads correctly, then the installation was successful.

The parameters of the tube geometry are:
- External diameter
- Internal diameter
- Height

They can be set in the XML file when the tube is called for the first time (see tube_test.xml for a working example).