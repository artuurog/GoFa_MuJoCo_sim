"""
Motion sequence
---------------
1.  SETTLE       -- hold the starting pose for a moment so the weld-constrained
                     peg settles before anything moves.
2.  MOVE_RANDOM  -- smoothly move to a randomly-perturbed pose above the box
                     (coarse reach).
3.  REFINE       -- smoothly move from that random pose onto the exact
                     standoff target (fine positioning).
4.  HOLD         -- keep holding the final target pose.


Workflow (model split)
-----------------------
1.  IK is solved on the GRIPPERLESS model (robot_no_gripper.xml) using the
    provided LevenbergMarquardtIK solver (IK_solver.py, unmodified). Solving
    on the gripperless model keeps the kinematic chain identical to whatever
    robot_no_gripper.xml exposes and avoids the finger / free-peg DoFs
    interfering with the solve.
2.  The resulting arm joint angles (joint1 .. joint6) are copied by NAME onto
    the VISUALIZATION model (contact_forces_insertion.xml), which has the
    gripper, the peg ("cylinder_body") and the insertion box.
3.  The peg is rigidly attached to the end-effector via the <weld> equality
    constraint in that XML (ee_site <-> cylinder_site). Its free-joint qpos
    is pre-positioned so the weld doesn't have to yank the peg into place on
    the first physics step (this avoids a violent "snap" -- see notes below).
4.  The passive viewer is launched and each phase's ctrl target is streamed
    in via the quintic ramp described above.

Edit the variables in the "USER-EDITABLE TARGET POSE" and "MOTION TIMING"
sections below to change the target location, randomization range, or speed.

ASSUMPTION TO VERIFY
---------------------
This script assumes robot_no_gripper.xml (via its included
assets/gofa_body_nohand.xml) defines a site named "ee_site" at the flange,
matching the pattern already used in gofa_body.xml / contact_forces_insertion.xml.
That file wasn't available to inspect directly -- if it doesn't have this
site yet, add one at the same location the gripper attaches, e.g.:
    <site name="ee_site" pos="..." size="0.01"/>
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import mujoco
import mujoco.viewer

from IK_solver import LevenbergMarquardtIK

# ============================================================================
# USER-EDITABLE TARGET POSE
# ============================================================================
# Insertion box geometry, taken from contact_forces_insertion.xml
# (body "insertion_box"; wall geoms have z half-height 0.020).
BOX_POS         = np.array([-0.4, 0.2, 0.02])   # insertion_box body position
BOX_HALF_HEIGHT = 0.020                          # wall/hole half-height (z)
STANDOFF_HEIGHT = 0.05                           # <-- 50 mm above the box top

# Target position for "ee_site" in world coordinates (meters). Recomputed
# automatically from the geometry above -- change STANDOFF_HEIGHT (or set
# TARGET_POS directly) to move the target.
TARGET_POS = np.array([
    BOX_POS[0],
    BOX_POS[1],
    BOX_POS[2] + BOX_HALF_HEIGHT + STANDOFF_HEIGHT,
])

# Optional target orientation
USE_ORIENTATION = True

# ============================================================================
# MOTION TIMING  (seconds)
# ============================================================================
SETTLE_DURATION      = 0.5   # hold the start pose so the weld settles first
MOVE_RANDOM_DURATION = 3.0   # coarse move: start pose -> random pose above box
REFINE_DURATION      = 2.0   # fine move: random pose -> exact target pose

# Starting joint configuration (also the IK seed for the coarse solve).
# Edit this if you have a known-good home pose for the real arm.
# HOME_Q = np.zeros(len(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]))
HOME_Q = np.array((np.pi, -np.pi/4, np.pi/3, 0, np.pi/2, 0))

# ============================================================================
# IK solver / seed settings
# ============================================================================
IK_STEP_SIZE = 0.2
IK_TOL = 1e-3
IK_DAMPING = 0.01
IK_ORIENTATION_WEIGHT = 1.0 if USE_ORIENTATION else 0.0
IK_MAX_ITER = 1000

# Gripper finger position while holding the peg.
GRIPPER_CTRL_CLOSED = -0.012

# ============================================================================
# TORQUE LOGGING
# ============================================================================
# Names of the <jointactuatorfrc> sensors already defined in
# contact_forces_insertion.xml (one per arm joint) -- no XML changes needed.
TORQUE_SENSOR_NAMES = [f"torque_sensor_{i}" for i in range(1, 7)]

# Record every Nth physics step. timestep=0.0005s -> 2 kHz; logging every
# single step over a long HOLD phase piles up samples fast, so raise this
# (e.g. 4 or 10) if the plot gets too dense or memory becomes a concern.
LOG_EVERY_N_STEPS = 1

# ============================================================================
# Model paths
# ============================================================================
MODEL_DIR = Path("Mujoco_models")
IK_MODEL_PATH = MODEL_DIR / "robot_no_gripper.xml"
VIS_MODEL_PATH = MODEL_DIR / "contact_forces_insertion.xml"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def quintic_blend(s):
    """
    Minimum-jerk ('quintic') blend factor for s in [0, 1].
    Zero velocity AND zero acceleration at both s=0 and s=1
    """
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def solve_arm_ik(solver, ik_model, ik_data, target_pos, init_q, orientation_goal=None):
    """
    Run one IK solve on the gripperless model and return the arm joint
    angles as an ordered np.array (matching ARM_JOINTS), plus convergence
    info. `init_q` is the seed -- pass the previous solution to keep
    consecutive solves in the same "elbow" branch.
    """
    site_id = ik_model.site("ee_site").id

    if orientation_goal is not None:
        goal = (target_pos, orientation_goal)
    else:
        goal = target_pos

    converged, n_iter = solver.calculate(goal, init_q, site_id)

    final_pos = ik_data.site(site_id).xpos.copy()
    pos_err_mm = np.linalg.norm(final_pos - target_pos) * 1000.0
    print(f"[IK] target={np.round(target_pos, 4)}  converged={converged}  "
          f"iterations={n_iter}  error={pos_err_mm:.3f} mm")

    q_arm = np.array([ik_data.qpos[ik_model.joint(name).qposadr[0]] for name in ARM_JOINTS])
    return q_arm, converged


def preposition_peg(vis_model, vis_data):


    """
    Place the free peg ("cylinder_body") so its cylinder_site already
    coincides (position + orientation) with ee_site *before* the first
    physics step. Requires mj_forward to have been called already so that
    ee_site's world pose reflects the newly-copied arm qpos.

    Without this, the weld equality constraint has to pull the peg in from
    its XML-authored resting position (far from the robot) on step 1, which
    causes a violent snap.
    """
    ee_id = vis_model.site("ee_site").id
    cyl_site_id = vis_model.site("cylinder_site").id

    site_world_pos = vis_data.site(ee_id).xpos.copy()
    site_world_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_world_quat, vis_data.site(ee_id).xmat.copy())

    # Local offset of cylinder_site within cylinder_body, already compiled by
    # MuJoCo from the <site pos=".." euler=".."/> attributes -- no manual
    # euler-angle math needed.
    local_pos = vis_model.site_pos[cyl_site_id].copy()
    local_quat = vis_model.site_quat[cyl_site_id].copy()

    inv_pos, inv_quat = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(inv_pos, inv_quat, local_pos, local_quat)

    body_pos, body_quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(body_pos, body_quat, site_world_pos, site_world_quat, inv_pos, inv_quat)

    cyl_body_id = vis_model.site_bodyid[cyl_site_id]
    jnt_id = vis_model.body_jntadr[cyl_body_id]   # cylinder_free is the body's only joint
    qadr = vis_model.jnt_qposadr[jnt_id]

    vis_data.qpos[qadr:qadr + 3] = body_pos
    vis_data.qpos[qadr + 3:qadr + 7] = body_quat


def plot_joint_torques(time_arr, torque_arr, phase_log):
    """
    Plot the actuator-transmitted torque at each arm joint against simulated time, with the
    SETTLE / POSITIONING / REFINE / HOLD phase boundaries marked for context.

    Parameters
    ----------
    time_arr   : (N,) array of simulated timestamps [s]
    torque_arr : (N, 6) array, one column per joint1..joint6 [N*m]
    phase_log  : list of (sim_time, state_name) tuples marking phase entries
    """
    n_joints = torque_arr.shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(9, 1.8 * n_joints), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time_arr, torque_arr[:, i], linewidth=1.5, color="tab:blue")
        ax.set_ylabel(f"J{i + 1}\n[Nm]")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        for t_trans, _ in phase_log:
            ax.axvline(t_trans, color="gray", linestyle="--", linewidth=0.7)

    # Label each phase boundary once, on the top subplot only.
    ymax = axes[0].get_ylim()[1]
    for t_trans, label in phase_log:
        axes[0].text(t_trans, ymax, f" {label}", rotation=90,
                     va="top", ha="left", fontsize=7, color="dimgray")

    axes[-1].set_xlabel("simulated time [s]")
    fig.suptitle("Joint actuator torques (torque_sensor_1..6)")
    fig.tight_layout()
    plt.show()

def main():
    if not IK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find IK model file: {IK_MODEL_PATH.resolve()}")
    if not VIS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find visualization model file: {VIS_MODEL_PATH.resolve()}")

    # ---- Solve IK for both the coarse ("random") pose and the exact target --
    ik_model = mujoco.MjModel.from_xml_path(str(IK_MODEL_PATH))
    ik_data = mujoco.MjData(ik_model)

    solver = LevenbergMarquardtIK(
        ik_model, ik_data,
        step_size=IK_STEP_SIZE,
        tol=IK_TOL,
        damping=IK_DAMPING,
        orientation_weight=IK_ORIENTATION_WEIGHT,
        max_iter=IK_MAX_ITER,
    )
    orientation_goal = solver.rotation_matrix_from_euler(0, np.pi/2, 0) if USE_ORIENTATION else None

    rough_target_pos = TARGET_POS + np.array([
        0,
        0,
        0.05,
    ])

    print("[IK] --- coarse (random) pose above the box ---")
    q_rough, _ = solve_arm_ik(solver, ik_model, ik_data, rough_target_pos, HOME_Q, orientation_goal)
    print("[IK] --- refined (exact) target pose ---")
    q_target, _ = solve_arm_ik(solver, ik_model, ik_data, TARGET_POS, q_rough, orientation_goal)

    # ---- Set up the visualization model at the HOME_Q starting pose --------
    vis_model = mujoco.MjModel.from_xml_path(str(VIS_MODEL_PATH))
    vis_data = mujoco.MjData(vis_model)

    for i, name in enumerate(ARM_JOINTS):
        vis_data.qpos[vis_model.joint(name).qposadr[0]] = HOME_Q[i]

    # Gripper closed from the very first step -- this value is never changed
    vis_data.qpos[vis_model.joint("joint_finger1").qposadr[0]] = GRIPPER_CTRL_CLOSED
    vis_data.qpos[vis_model.joint("joint_finger2").qposadr[0]] = GRIPPER_CTRL_CLOSED

    mujoco.mj_forward(vis_model, vis_data)   # needed so ee_site.xpos/xmat reflect qpos
    preposition_peg(vis_model, vis_data)
    mujoco.mj_forward(vis_model, vis_data)   # refresh again after moving the peg

    arm_act_ids = [vis_model.actuator(f"{name}_act").id for name in ARM_JOINTS]
    finger1_act = vis_model.actuator("finger1_act").id
    finger2_act = vis_model.actuator("finger2_act").id

    # Ctrl starts pinned to HOME_Q (matches the qpos we just set) so the arm
    # doesn't jump the instant the sim starts.
    for act_id, q in zip(arm_act_ids, HOME_Q):
        vis_data.ctrl[act_id] = q
    vis_data.ctrl[finger1_act] = GRIPPER_CTRL_CLOSED
    vis_data.ctrl[finger2_act] = GRIPPER_CTRL_CLOSED

    # ---- Torque-sensor addresses, resolved once (never inside the loop) ----
    torque_sensor_adr = np.array(
        [vis_model.sensor(name).adr[0] for name in TORQUE_SENSOR_NAMES]
    )
    time_log = []
    torque_log = []
    phase_log = [(0.0, "SETTLE")]   # (sim_time, state_name) at every transition

    with mujoco.viewer.launch_passive(vis_model, vis_data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        vis_model.vis.scale.contactwidth = 0.05
        vis_model.vis.scale.contactheight = 0.02
        vis_model.vis.scale.forcewidth = 0.03

        # ---- Phase state machine, driven by simulated (not wall-clock) time
        sim_time = 0.0
        state = "SETTLE"
        state_start = 0.0
        step_idx = 0
        print(f"[state] entering SETTLE ({SETTLE_DURATION:.1f}s)")

        while viewer.is_running():
            step_start = time.time()
            t_in_state = sim_time - state_start

            if state == "SETTLE":
                q_cmd = HOME_Q
                if t_in_state >= SETTLE_DURATION:
                    state, state_start = "POSITIONING", sim_time
                    phase_log.append((sim_time, "POSITIONING"))
                    print(f"[state] entering POSITIONING ({MOVE_RANDOM_DURATION:.1f}s) "
                          f"-> {np.round(rough_target_pos, 4)}")

            elif state == "POSITIONING":
                s = t_in_state / MOVE_RANDOM_DURATION
                q_cmd = HOME_Q + (q_rough - HOME_Q) * quintic_blend(s)
                if t_in_state >= MOVE_RANDOM_DURATION:
                    state, state_start = "REFINE", sim_time
                    phase_log.append((sim_time, "REFINE"))
                    print(f"[state] entering REFINE ({REFINE_DURATION:.1f}s) "
                          f"-> {np.round(TARGET_POS, 4)}")

            elif state == "REFINE":
                s = t_in_state / REFINE_DURATION
                q_cmd = q_rough + (q_target - q_rough) * quintic_blend(s)
                if t_in_state >= REFINE_DURATION:
                    state, state_start = "HOLD", sim_time
                    phase_log.append((sim_time, "HOLD"))
                    print("[state] entering HOLD (final target reached)")

            else:  # HOLD
                q_cmd = q_target

            for act_id, q in zip(arm_act_ids, q_cmd):
                vis_data.ctrl[act_id] = q
            # Gripper: always closed, every step, every phase.
            vis_data.ctrl[finger1_act] = GRIPPER_CTRL_CLOSED
            vis_data.ctrl[finger2_act] = GRIPPER_CTRL_CLOSED

            mujoco.mj_step(vis_model, vis_data)
            sim_time += vis_model.opt.timestep
            step_idx += 1

             # ---- Torque logging (sensordata is a live view -> copy it)
            if step_idx % LOG_EVERY_N_STEPS == 0:
                time_log.append(sim_time)
                torque_log.append(vis_data.sensordata[torque_sensor_adr].copy())
            viewer.sync()

            time_until_next_step = vis_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


    # ---- Post-run torque plot ----------------------------------------------
    if len(time_log) > 1:
        time_arr = np.asarray(time_log)
        torque_arr = np.asarray(torque_log)   # shape (N_samples, 6)
        plot_joint_torques(time_arr, torque_arr, phase_log)
    else:
        print("[torque-plot] fewer than 2 samples recorded, skipping plot")


if __name__ == "__main__":
    main()