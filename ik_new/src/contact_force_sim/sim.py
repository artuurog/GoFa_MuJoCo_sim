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
# Inverse kinematics on the full model, using a separate MjData so the
# simulation state is not modified. Only the robot joints (chain from the
# world to ee_site) are used: fingers and other dofs are left untouched.
# ------------------------------------------------------------------
site_id = model.site("ee_site").id
ik_data = mujoco.MjData(model)
ik = LevenbergMarquardtIK(model, ik_data,
                          step_size=0.25, tol=0.001,
                          damping=0.01, orientation_weight=1.0)

R_goal = ik.rotation_matrix_from_euler(*np.radians(TARGET_RPY_DEG))
goal   = (np.array(TARGET_POS, dtype=float), R_goal)

robot_joints = ik.chain_joints(site_id)
qpos_idx = model.jnt_qposadr[robot_joints]
limited  = model.jnt_limited[robot_joints].astype(bool)
q_lo = np.where(limited, model.jnt_range[robot_joints, 0], -np.pi)
q_hi = np.where(limited, model.jnt_range[robot_joints, 1],  np.pi)

# First guess: middle of the joint ranges, then random configurations
rng = np.random.default_rng(0)
q_guess = data.qpos.copy()
q_guess[qpos_idx] = 0.5 * (q_lo + q_hi)

best_err, q_goal = np.inf, None
for attempt in range(IK_RESTARTS + 1):
    converged, n_iter = ik.calculate(goal, q_guess, site_id)

    pos_err = goal[0] - ik_data.site(site_id).xpos
    rot_err = ik.rotation_error(ik_data.site(site_id).xmat.reshape(3, 3), R_goal)
    err = np.linalg.norm(np.concatenate([pos_err, rot_err]))
    if err < best_err:
        best_err, q_goal = err, ik_data.qpos[qpos_idx].copy()
        best_pos_err, best_rot_err = np.linalg.norm(pos_err), np.linalg.norm(rot_err)

    if converged:
        break
    q_guess[qpos_idx] = rng.uniform(q_lo, q_hi)

print(f"IK converged: {converged}  |  attempts: {attempt + 1}  |  iterations (last): {n_iter}")
print(f"  joints   [deg] => {np.round(np.degrees(q_goal), 2)}")
print(f"  position error  => {best_pos_err * 1000:.2f} mm")
print(f"  rotation error  => {np.degrees(best_rot_err):.2f} deg")
if not converged:
    print("  WARNING: target not reached (out of workspace / joint limits?). "
          "Moving to the closest configuration found.")

# ------------------------------------------------------------------
# Move the robot to the IK solution in the (blocking) MuJoCo viewer.
# The control callback is called by the viewer at every mj_step.
# ------------------------------------------------------------------
act_idx = [np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                          & (model.actuator_trnid[:, 0] == j))[0]
           for j in robot_joints]
q_start = data.qpos[qpos_idx].copy()

# Position actuators clamp their target to ctrlrange, which can be narrower than
# the joint range (joint6: ±360 deg joint, ±180 deg actuator). For hinge joints
# use the equivalent angle (± 2*pi) that lies inside ctrlrange.
for k, (j, a) in enumerate(zip(robot_joints, act_idx)):
    if not model.actuator_ctrllimited[a]:
        continue
    lo, hi = model.actuator_ctrlrange[a]
    shifts = (0.0, 2 * np.pi, -2 * np.pi) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else (0.0,)
    for shift in shifts:
        if lo <= q_goal[k] + shift <= hi:
            q_goal[k] += shift
            break
    else:
        print(f"  WARNING: {model.joint(j).name} target {np.degrees(q_goal[k]):.1f} deg "
              f"is outside the actuator ctrlrange and will be clamped.")


def controller(model, data):
    """Smooth joint-space interpolation from q_start to q_goal (position actuators)."""
    s = np.clip(data.time / MOVE_TIME, 0.0, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * s)        # zero velocity at start and end
    data.ctrl[act_idx] = q_start + s * (q_goal - q_start)


mujoco.set_mjcb_control(controller)
mujoco.viewer.launch(model, data)
mujoco.set_mjcb_control(None)
