import numpy as np
import mujoco
import matplotlib.pyplot as plt
import src.contact_force_sim.IK_solver as IK_solver


if __name__ == "__main__":

    path = "C:/Users/user/Desktop/GoFa_MuJoCo_sim/gofa/robot_no_gripper.xml"
    model    = mujoco.MjModel.from_xml_path(path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 2

    # ------------------------------------------------------------------
    # Build a reference pose from a known joint configuration
    # ------------------------------------------------------------------
    data.qpos = [-3*np.pi/2, -np.pi/4, np.pi/3, 0, np.pi/2, 0] # target joint angles
    qpos0 = data.qpos.copy()
    mujoco.mj_forward(model, data)

    site_id = model.site('ee_site').id

    target_pos = data.site('ee_site').xpos.copy()
    target_rot = data.site(site_id).xmat.reshape(3, 3).copy()
    

    print("Target position  =>", target_pos)
    print("Target rotation  =>\n", target_rot)

    # ------------------------------------------------------------------
    # Test 1 – position only  (backward-compatible mode)
    # ------------------------------------------------------------------
    print("\n--- Test 1: position only ---")

    init_q = np.zeros(model.nq)
    ik = IK_solver.LevenbergMarquardtIK(model, data,
                               step_size=0.25, tol=0.001,
                               damping=0.01, orientation_weight=0.0)

    mujoco.mj_resetDataKeyframe(model, data, 1)
    converged, n_iter = ik.calculate(target_pos, init_q, site_id)

    result_pos_only = data.qpos.copy()
    mujoco.mj_forward(model, data)
    print(f"  Converged: {converged}  |  Iterations: {n_iter}")
    print(f"  EE position  => {data.site('ee_site').xpos}")

    # ------------------------------------------------------------------
    # Test 2 – position + orientation
    # ------------------------------------------------------------------
    print("\n--- Test 2: position + orientation ---")

    # Define a goal orientation via Euler angles (roll, pitch, yaw in radians).
    # Here we use the same rotation as the reference pose to verify convergence.
    R_goal = target_rot   # replace with rotation_matrix_from_euler(roll, pitch, yaw)

    goal_pose = (target_pos, R_goal)

    mujoco.mj_resetDataKeyframe(model, data, 1)
    # converged, n_iter = ik.calculate.__func__(
    #     LevenbergMarquardtIK(model, data,
    #                           step_size=0.25, tol=0.001,
    #                           damping=0.01, orientation_weight=1.0),
    #     goal_pose, init_q, site_id
    # )
    
    converged, n_iter = IK_solver.LevenbergMarquardtIK(model, data,
                          step_size=0.25, tol=0.001,
                          damping=0.01, orientation_weight=1).calculate(goal_pose, init_q, site_id)

    result_full_pose = data.qpos.copy()
    mujoco.mj_forward(model, data)
    R_result = data.site(site_id).xmat.reshape(3, 3)

    print(f"  Converged: {converged}  |  Iterations: {n_iter}")
    print(f"  EE position  => {data.site('ee_site').xpos}")
    print(f"  EE rotation  =>\n{R_result}")
    print(f"  Orientation error norm => "
          f"{np.linalg.norm(ik.rotation_error(R_result, R_goal)):.6f}")

    # ------------------------------------------------------------------
    # Visual comparison
    # ------------------------------------------------------------------
    renderer.update_scene(data, camera)
    result_plot = renderer.render()

    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera)
    target_plot = renderer.render()

    plt.figure(1); plt.title("Target pose");  plt.imshow(target_plot)
    plt.figure(2); plt.title("IK result");    plt.imshow(result_plot)
    plt.show()