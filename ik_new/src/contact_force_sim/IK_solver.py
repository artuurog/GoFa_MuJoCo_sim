import numpy as np
import mujoco

# =============================================================================
# Levenberg-Marquardt IK  –  position + orientation (6-DOF task)
# =============================================================================

class LevenbergMarquardtIK:
    """
    Levenberg-Marquardt Inverse Kinematics solver.

    Supports two modes, selected by the 'orientation_weight' parameter:
      • position-only  : orientation_weight = 0  (backward compatible)
      • position + orientation : orientation_weight > 0  (default 1.0)

    Goal for position-only  : np.array([x, y, z])
    Goal for full pose      : (np.array([x, y, z]), R_goal_3x3)
                               where R_goal_3x3 is a 3×3 rotation matrix.
                               Helper rotation_matrix_from_euler() is provided.

    Works directly on the full simulation model (gripper, objects, ...):
    only the robot joints are used, i.e. the joints on the kinematic chain
    from the world to the target site (or the ones listed in 'joint_names').
    After calculate() / solve(), their indices are available as
      • self.joint_ids : joint ids
      • self.qpos_idx  : addresses in data.qpos
      • self.dof_idx   : columns of the site Jacobian / addresses in data.qvel

    If 'data' is None the solver uses its own MjData, so the IK never
    modifies the state of the simulation.

    Entry points:
      • solve()     : complete IK for a desired site pose (initial guess,
                      random restarts, best result) -> robot joint values
      • calculate() : a single Levenberg-Marquardt run from a given init_q
    """

    def __init__(self, model, data=None,
                 step_size=0.5,
                 tol=0.001,
                 alpha=0.5,
                 jacp=None,
                 jacr=None,
                 damping=0.01,
                 orientation_weight=1.0,
                 max_iter=1000,
                 joint_names=None):

        self.model = model
        self.data  = data if data is not None else mujoco.MjData(model)
        self.step_size         = step_size
        self.tol               = tol
        self.alpha             = alpha
        self.damping           = damping
        self.orientation_weight = orientation_weight
        self.max_iter          = max_iter

        # Jacobian buffers – allocate internally if not provided
        nv = model.nv
        self.jacp = jacp if jacp is not None else np.zeros((3, nv))
        self.jacr = jacr if jacr is not None else np.zeros((3, nv))

        # Robot joints: given explicitly, or found from the site in calculate()
        self.joint_ids = None
        if joint_names is not None:
            self._set_joints([model.joint(name).id for name in joint_names])

    # ------------------------------------------------------------------
    def _set_joints(self, joint_ids):
        """Store the robot joint ids and their qpos / dof addresses."""
        joint_ids = np.asarray(joint_ids, dtype=int)
        one_dof = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        for j in joint_ids:
            if self.model.jnt_type[j] not in one_dof:
                raise ValueError(f"[IK] joint '{self.model.joint(j).name}' is not "
                                 f"a hinge/slide joint")
        self.joint_ids = joint_ids
        self.qpos_idx  = self.model.jnt_qposadr[joint_ids]
        self.dof_idx   = self.model.jnt_dofadr[joint_ids]

    # ------------------------------------------------------------------
    def chain_joints(self, site_id):
        """
        Joint ids on the kinematic chain from the world to the site, base first.
        Joints that do not move the site (fingers, free objects, ...) are excluded.
        """
        joints = []
        body = self.model.site_bodyid[site_id]
        while body != 0:
            adr = self.model.body_jntadr[body]
            num = self.model.body_jntnum[body]
            joints = list(range(adr, adr + num)) + joints
            body = self.model.body_parentid[body]
        return joints

    # ------------------------------------------------------------------
    def check_joint_limits(self, q):
        """Clamp the robot joint values (entries self.qpos_idx of q) to their model limits."""
        for j, i in zip(self.joint_ids, self.qpos_idx):
            if self.model.jnt_limited[j]:
                q[i] = np.clip(q[i], *self.model.jnt_range[j])

    # ------------------------------------------------------------------
    def _joint_bounds(self):
        """Lower / upper bounds of the robot joints (±pi for unlimited joints)."""
        limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        q_lo = np.where(limited, self.model.jnt_range[self.joint_ids, 0], -np.pi)
        q_hi = np.where(limited, self.model.jnt_range[self.joint_ids, 1],  np.pi)
        return q_lo, q_hi

    # ------------------------------------------------------------------
    def _wrap_to_actuator_range(self, q):
        """
        Position actuators clamp their target to ctrlrange, which can be narrower
        than the joint range (GoFa joint6: ±360 deg joint, ±180 deg actuator).
        For hinge joints, shift q (robot joints) by ±2π – same pose – so that it
        lies inside the ctrlrange of the position actuator driving the joint.
        """
        m = self.model
        for k, j in enumerate(self.joint_ids):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            act = np.flatnonzero((m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                 & (m.actuator_trnid[:, 0] == j)
                                 & (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
                                 & m.actuator_ctrllimited.astype(bool))
            if act.size == 0:
                continue
            a = act[0]
            lo, hi = np.sort(m.actuator_ctrlrange[a] / m.actuator_gear[a, 0])
            for shift in (0.0, 2 * np.pi, -2 * np.pi):
                q_s = q[k] + shift
                in_joint_range = (not m.jnt_limited[j]) or \
                                 (m.jnt_range[j, 0] <= q_s <= m.jnt_range[j, 1])
                if lo <= q_s <= hi and in_joint_range:
                    q[k] = q_s
                    break
            else:
                print(f"[IK] WARNING: {m.joint(j).name} = {np.degrees(q[k]):.1f} deg is "
                      f"outside the ctrlrange of actuator '{m.actuator(a).name}'")
        return q

    # ------------------------------------------------------------------
    def pose_error(self, site_id, pos_goal, R_goal=None):
        """Position error norm [m] and orientation error angle [rad] of the site."""
        pos_err = np.linalg.norm(pos_goal - self.data.site(site_id).xpos)
        if R_goal is None:
            return pos_err, 0.0
        rot_err = np.linalg.norm(self.rotation_error(self._get_site_rotation(site_id), R_goal))
        return pos_err, rot_err

    # ------------------------------------------------------------------
    def _get_site_rotation(self, site_id):
        """Return the 3×3 rotation matrix of a site from xmat (row-major 9-vec)."""
        return self.data.site(site_id).xmat.reshape(3, 3).copy()
    
    # ------------------------------------------------------------------
    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        """
        Build a rotation matrix from ZYX Euler angles (in radians).
        Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        Rx = np.array([[1,           0,            0],
                       [0,  np.cos(roll), -np.sin(roll)],
                       [0,  np.sin(roll),  np.cos(roll)]])

        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                       [             0, 1,             0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [          0,            0, 1]])

        return Rx @ Ry @ Rz
    

    # def rotation_matrix_from_euler(self, roll, pitch, yaw, seq="zyx"):
    #     """

    #     Parameters
    #     ----------
    #     roll, pitch, yaw: euler angles in rad
    #     seq : The default is "zyx".

    #     Returns
    #     -------
    #     Rotation matrix

    #     """
        
    #     quat = np.zeros(4)
    #     euler = np.array([roll, pitch, yaw])
        
    #     mujoco.mju_euler2Quat(quat, euler, seq)
        
    #     mat = np.zeros(9)
    #     mujoco.mju_quat2Mat(mat, quat)
        
    #     return mat.reshape(3, 3)
    

    # ------------------------------------------------------------------
    def rotation_error(self, R_current, R_goal):
        """
        Compute the orientation error as an axis-angle vector (3D).

        e_R = axis * angle of the relative rotation  R_goal @ R_current.T
        (shortest rotation, angle in [0, pi]). Unlike the 'vee' of its
        skew-symmetric part, which is axis * sin(angle), it does not
        vanish for large errors close to 180 deg.

        Returns a 3-vector in the world frame (same frame as jacr) that
        drives R_current -> R_goal.
        """
        R_err = R_goal @ R_current.T          # relative rotation matrix
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_err.flatten())
        e_R = np.zeros(3)
        mujoco.mju_quat2Vel(e_R, quat, 1.0)   # axis * angle
        return e_R


    # ------------------------------------------------------------------
    def solve(self, target_pos, target_rpy=None, site="ee_site",
              init_q=None, restarts=10, seed=0, verbose=True):
        """
        Inverse kinematics of the robot for a desired pose of a site.

        Runs calculate() from 'init_q' (default: middle of the joint ranges)
        and, if it does not converge, again from up to 'restarts' random
        configurations within the joint limits. The best result is kept
        (also when no attempt converges, e.g. target out of reach).
        Hinge angles are then shifted by ±2π, if needed, into the ctrlrange
        of their position actuators, so the result can be used directly as
        data.ctrl targets.

        Parameters
        ----------
        target_pos : [x, y, z], desired site position in the world frame [m]
        target_rpy : [roll, pitch, yaw] desired site orientation [rad], same
                     convention as rotation_matrix_from_euler();
                     None -> position only
        site       : site name or id (default 'ee_site')
        init_q     : first guess, full qpos (model.nq) or robot joints only;
                     None -> middle of the joint ranges
        restarts   : max number of extra attempts from random configurations
        seed       : seed of the random restarts (reproducible results)
        verbose    : print a summary of the result

        Returns
        -------
        converged : bool       – True if the tolerance was reached
        q_sol     : np.ndarray – robot joint values, ordered as self.joint_ids
                                 (data.qpos[self.qpos_idx]); self.data holds
                                 the full solution configuration
        """
        site_id = self.model.site(site).id
        if self.joint_ids is None:
            self._set_joints(self.chain_joints(site_id))

        # ---- Goal --------------------------------------------------------
        pos_goal = np.asarray(target_pos, dtype=float)
        if target_rpy is None:
            R_goal, goal = None, pos_goal
        else:
            R_goal = self.rotation_matrix_from_euler(*target_rpy)
            goal   = (pos_goal, R_goal)

        # ---- Attempts: init_q / mid-range, then random configurations ----
        q_lo, q_hi = self._joint_bounds()
        rng     = np.random.default_rng(seed)
        q_guess = init_q if init_q is not None else 0.5 * (q_lo + q_hi)

        best_err = np.inf
        for attempt in range(restarts + 1):
            converged, n_iter = self.calculate(goal, q_guess, site_id, verbose=False)

            pos_err, rot_err = self.pose_error(site_id, pos_goal, R_goal)
            err = np.hypot(pos_err, self.orientation_weight * rot_err)
            if err < best_err:
                best_err, best_qpos = err, self.data.qpos.copy()
                best_pos_err, best_rot_err = pos_err, rot_err

            if converged:
                break
            q_guess = rng.uniform(q_lo, q_hi)

        # ---- Best solution, expressed inside the actuator ranges ----------
        self.data.qpos[:] = best_qpos
        q_sol = self._wrap_to_actuator_range(self.data.qpos[self.qpos_idx].copy())
        self.data.qpos[self.qpos_idx] = q_sol
        mujoco.mj_forward(self.model, self.data)

        if verbose:
            print(f"[IK] converged: {converged}  |  attempts: {attempt + 1}  |  "
                  f"iterations (last): {n_iter}")
            print(f"[IK]   joints   [deg] => {np.round(np.degrees(q_sol), 2)}")
            print(f"[IK]   position error  => {best_pos_err * 1000:.2f} mm")
            if R_goal is not None:
                print(f"[IK]   rotation error  => {np.degrees(best_rot_err):.2f} deg")
            if not converged:
                print("[IK]   WARNING: target not reached (out of workspace / joint "
                      "limits?). Returning the closest configuration found.")

        return converged, q_sol

    # ------------------------------------------------------------------
    def calculate(self, goal, init_q, body_id, verbose=True):
        """
        Run the IK solver.

        Parameters
        ----------
        goal     : array-like [x,y,z]          (position-only mode)
                   or tuple  ([x,y,z], R_3x3)  (full-pose mode)
        init_q   : array-like, initial configuration: either the full qpos
                   (length model.nq) or the robot joints only (length len(qpos_idx))
        body_id  : int, site id (use model.site('ee_site').id)
        verbose  : print a warning if max_iter is reached

        Returns
        -------
        converged : bool  – True if the solver reached tolerance
        n_iter    : int   – number of iterations performed
        """

        # ---- Parse goal --------------------------------------------------
        if isinstance(goal, (tuple, list)) and len(goal) == 2 \
                and isinstance(goal[1], np.ndarray) and goal[1].shape == (3, 3):
            pos_goal = np.asarray(goal[0], dtype=float)
            R_goal   = goal[1]
            use_orientation = (self.orientation_weight > 0.0)
        else:
            pos_goal = np.asarray(goal, dtype=float)
            R_goal   = None
            use_orientation = False

        # ---- Robot joints ------------------------------------------------
        if self.joint_ids is None:
            self._set_joints(self.chain_joints(body_id))

        # ---- Initialise --------------------------------------------------
        init_q = np.asarray(init_q, dtype=float)
        if init_q.size == self.model.nq:
            self.data.qpos[:] = init_q
        elif init_q.size == len(self.qpos_idx):
            self.data.qpos[self.qpos_idx] = init_q
        else:
            raise ValueError(f"[IK] init_q has {init_q.size} entries, expected "
                             f"{self.model.nq} (nq) or {len(self.qpos_idx)} (robot joints)")
        self.check_joint_limits(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

        # ---- Build the combined error vector -----------------------------
        def compute_error():
            pos_err = pos_goal - self.data.site(body_id).xpos
            if use_orientation:
                R_cur   = self._get_site_rotation(body_id)
                rot_err = self.orientation_weight * self.rotation_error(R_cur, R_goal)
                return np.concatenate([pos_err, rot_err])   # shape (6,)
            return pos_err                                   # shape (3,)

        error = compute_error()

        converged = False
        n_iter    = 0

        while np.linalg.norm(error) >= self.tol:
            if n_iter >= self.max_iter:
                if verbose:
                    print(f"[IK] WARNING: max iterations ({self.max_iter}) reached. "
                          f"Residual error = {np.linalg.norm(error):.6f}")
                break

            # ---- Compute Jacobians ---------------------------------------
            mujoco.mj_jacSite(self.model, self.data,
                               self.jacp, self.jacr, body_id)

            # ---- Build the task-space Jacobian J -------------------------
            # Keep only the columns of the robot dofs (n = number of robot joints)
            jacp = self.jacp[:, self.dof_idx]
            jacr = self.jacr[:, self.dof_idx]
            if use_orientation:
                # Stack translational (3×n) and rotational (3×n) parts → (6×n)
                J = np.vstack([jacp,
                               self.orientation_weight * jacr])
            else:
                J = jacp   # (3×n)

            # ---- Levenberg-Marquardt pseudo-inverse ----------------------
            n = J.shape[1]
            I = np.eye(n)
            A = J.T @ J + self.damping * I   # (n×n)  – always well-conditioned

            if np.isclose(np.linalg.det(A), 0):
                j_inv = np.linalg.pinv(A) @ J.T
            else:
                j_inv = np.linalg.solve(A, J.T)   # more stable than explicit inv

            delta_q = j_inv @ error

            # ---- Update joint angles -------------------------------------
            self.data.qpos[self.qpos_idx] += self.step_size * delta_q
            self.check_joint_limits(self.data.qpos)

            # ---- Forward kinematics & new error --------------------------
            mujoco.mj_forward(self.model, self.data)

            error = compute_error()

            n_iter += 1

        else:
            converged = True

        return converged, n_iter