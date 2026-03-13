# GoFa_MuJoCo_sim
MuJoCo simulation environment with an ABB GoFa robot for tabletop manipulation tasks.

## Repo structure

`gofa/assets` is the folder that contains the STL meshes of the robot model

`robot_no_gripper.xml` contains a basic model of the robot, without the gripper

`scene.xml` simulates a pick and place environment without gravity compensation

`pick_place_scene.xml` simulates the same pick and place environment, but also compensates gravity
