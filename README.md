# GoFa_MuJoCo_sim
MuJoCo simulation environment with an ABB GoFa robot for tabletop manipulation tasks.

## Repo structure

`gofa/assets` is the folder that contains the STL meshes of the robot model

`robot_no_gripper.xml` contains a basic model of the robot, without the gripper

`scene.xml` simulates a pick and place environment without gravity compensation

`pick_place_scene.xml` simulates the same pick and place environment, but also compensates gravity

`gofa_contact_force.xml` simulates a peg-in-hole task in which end-effector contact forces are also measured


## How to use SDF tube plugin

SDF is a Mujoco plugin that allows to generate custom geometries defined as signed distance fields.
In order to use the tube_plugin.dll:
1. Find the folder of Mujoco's simulate.exe
2. In the same folder, create folder mujoco_plugin/
3. Paste tube_plugin.dll in mujoco_plugin/

The simulate.exe will be able now to open the tube geometry and visualize it.
To verify the correct installation of the plugin, open the tube_text.xml model using simulate.exe. If the model loads correctly, then the installation was successful.

The parameters of the tube geometry are:
- External diameter
- Internal diameter
- Height

They can be set in the XML file when the tube is called for the first time (see tube_test.xml for a working example).