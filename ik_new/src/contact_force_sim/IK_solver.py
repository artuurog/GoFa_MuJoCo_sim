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
    """

    def __init__(self, model, data,
                 step_size=0.5,
                 tol=0.001,
                 alpha=0.5,
                 jacp=None,
                 jacr=None,
                 damping=0.01,
                 orientation_weight=1.0,
                 max_iter=1000):

        self.model = model
        self.data  = data
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

    # ------------------------------------------------------------------
    def check_joint_limits(self, q):
        """Clamp joint angles to their model limits."""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0],
                       min(q[i], self.model.jnt_range[i][1]))

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

        Uses the formula:  e_R = 0.5 * (R_goal - R_current)^skew_extract
        i.e., the 'vee' of the skew-symmetric part of  R_goal @ R_current.T

        Returns a 3-vector in the body frame that drives R_current -> R_goal.
        """
        R_err = R_goal @ R_current.T          # relative rotation matrix
        # Extract the axis-angle vector from the skew-symmetric part
        # rot_vec = [R32-R23, R13-R31, R21-R12] / 2
        e_R = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        return e_R


    # ------------------------------------------------------------------
    def calculate(self, goal, init_q, body_id):
        """
        Run the IK solver.

        Parameters
        ----------
        goal     : array-like [x,y,z]          (position-only mode)
                   or tuple  ([x,y,z], R_3x3)  (full-pose mode)
        init_q   : array-like, initial joint configuration
        body_id  : int, site id (use model.site('ee_site').id)

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

        # ---- Initialise --------------------------------------------------
        self.data.qpos[:] = init_q
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
                print(f"[IK] WARNING: max iterations ({self.max_iter}) reached. "
                      f"Residual error = {np.linalg.norm(error):.6f}")
                break

            # ---- Compute Jacobians ---------------------------------------
            mujoco.mj_jacSite(self.model, self.data,
                               self.jacp, self.jacr, body_id)

            # ---- Build the task-space Jacobian J -------------------------
            if use_orientation:
                # Stack translational (3×nv) and rotational (3×nv) parts → (6×nv)
                J = np.vstack([self.jacp,
                               self.orientation_weight * self.jacr])
            else:
                J = self.jacp   # (3×nv)

            # ---- Levenberg-Marquardt pseudo-inverse ----------------------
            n = J.shape[1]
            I = np.eye(n)
            A = J.T @ J + self.damping * I   # (nv×nv)  – always well-conditioned

            if np.isclose(np.linalg.det(A), 0):
                j_inv = np.linalg.pinv(A) @ J.T
            else:
                j_inv = np.linalg.solve(A, J.T)   # more stable than explicit inv

            delta_q = j_inv @ error

            # ---- Update joint angles -------------------------------------
            self.data.qpos[:] += self.step_size * delta_q
            self.check_joint_limits(self.data.qpos)

            # ---- Forward kinematics & new error --------------------------
            mujoco.mj_forward(self.model, self.data)

            error = compute_error()

            n_iter += 1

        else:
            converged = True

        return converged, n_iter