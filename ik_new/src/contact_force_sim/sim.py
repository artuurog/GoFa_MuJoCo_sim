import numpy as np
import mujoco
import mujoco.viewer

from IK_solver import LevenbergMarquardtIK

# ---------------- user settings ----------------
MODEL_PATH = r"ik_new\gofa\gofa.xml"
Q_INIT = None

TARGET_POS     = [0.5, 0.2, 0.3]     # desired ee_site position, world frame [m]
TARGET_RPY_DEG = [0.0, 90.0, 0.0]    # desired ee_site orientation: roll, pitch, yaw [deg]
                                     # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)  (rotation_matrix_from_euler)
MOVE_TIME   = 3.0                    # duration of the motion to the IK solution [s]
IK_RESTARTS = 10                     # extra IK attempts from random configurations if one fails
# ------------------------------------------------

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Initial pose
if Q_INIT is not None:
    data.qpos[:len(Q_INIT)] = Q_INIT
elif model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# No gravity, so the position actuators track the commanded pose without sagging
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY

# ------------------------------------------------------------------
# Inverse kinematics: computed entirely by IK_solver (on its own MjData,
# so the simulation state is not modified). q_goal holds the robot joint
# values, ordered as ik.joint_ids / ik.qpos_idx.
# ------------------------------------------------------------------
ik = LevenbergMarquardtIK(model, step_size=0.25, tol=0.001,
                          damping=0.01, orientation_weight=1.0)
converged, q_goal = ik.solve(TARGET_POS, np.radians(TARGET_RPY_DEG),
                             site="ee_site", restarts=IK_RESTARTS)

# ------------------------------------------------------------------
# Move the robot to the IK solution in the (blocking) MuJoCo viewer.
# The control callback is called by the viewer at every mj_step.
# ------------------------------------------------------------------
act_idx = [np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                          & (model.actuator_trnid[:, 0] == j))[0]
           for j in ik.joint_ids]
q_start = data.qpos[ik.qpos_idx].copy()


def controller(model, data):
    """Smooth joint-space interpolation from q_start to q_goal (position actuators)."""
    s = np.clip(data.time / MOVE_TIME, 0.0, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * s)        # zero velocity at start and end
    data.ctrl[act_idx] = q_start + s * (q_goal - q_start)


mujoco.set_mjcb_control(controller)
mujoco.viewer.launch(model, data)
mujoco.set_mjcb_control(None)
