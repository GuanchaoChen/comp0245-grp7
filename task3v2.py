import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
from sklearn.ensemble import RandomForestRegressor
import torch
from sklearn.preprocessing import StandardScaler
from task2_main import MLP, cfg  # import your model & config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final3.rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists


PRINT_PLOTS = True# Set to True to enable plotting
RECORDING = True # Set to True to enable data recording

# downsample rate needs to be bigger than one (is how much I steps I skip when i downsample the data)
downsample_rate = 2

# Function to get downsample rate from the user without blocking the simulation loop
def get_downsample_rate():
    try:
        rate = int(input("Enter downsample rate (integer >=1): "))
        if rate < 1:
            print("Invalid downsample rate. Must be >= 1.")
            return None
        return rate
    except ValueError:
        print("Please enter a valid integer.")
        return None




def main():
    # Load trained models and scalers
    model = MLP(10, 14, cfg.hidden, activation=cfg.activation)
    model.load_state_dict(torch.load("task2_3.pth", map_location="cpu"))
    model.eval()

    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("scalers.pkl", "rb") as f:
        scaler_X, scaler_Y = pickle.load(f)

    print("✅ Loaded model & scalers for Part3 control")

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    lower_limits, upper_limits = sim.GetBotJointsLimit()
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")

    kp_pos = 100
    kp_ori = 0
    kp = 1000
    kd = 100

    list_of_desired_cartesian_positions = [
        [0.5, 0.0, 0.1],
        [0.4, 0.2, 0.1],
        [0.4, -0.2, 0.1],
        [0.4, 0.1, 0.1]
    ]
    list_of_desired_cartesian_orientations = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
    list_of_type_of_control = ["pos", "pos", "pos", "pos"]
    list_of_duration_per_desired_cartesian_positions = [5.0, 5.0, 5.0, 5.0]
    list_of_initialjoint_positions = [init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles]

    q_mes_all, qd_mes_all, qdd_est_all, q_d_all, qd_d_all, qdd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], [], [], []
    all_trajectory_points = []
    position_errors = []  # ✅ 存储每个目标点的误差
    current_time = 0
    time_step = sim.GetTimeStep()

    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        init_position = list_of_initialjoint_positions[i]
        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position,
                                    desired_cartesian_pos, np.zeros(3), desired_cartesian_ori, np.zeros(3),
                                    time_step, type_of_control, kp_pos, kp_ori, np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos / time_step)

        sim.ResetPose()
        if init_position is not None:
            sim.SetjointPosition(init_position)

        trajectory_points = []
        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0), dtype=float)

            X_input = np.concatenate([q_mes, desired_cartesian_pos]).reshape(1, -1)
            X_s = scaler_X.transform(X_input)
            X_t = torch.tensor(X_s, dtype=torch.float32)

            with torch.no_grad():
               pred_s = model(X_t).numpy()
            pred = scaler_Y.inverse_transform(pred_s)[0]
            q_des = pred[:7]
            qd_des_clip = np.clip(pred[7:], -np.array(joint_vel_limits), np.array(joint_vel_limits))

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd = MotorCommands()
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")

            trajectory_points.append(cart_pos)
            time.sleep(time_step)
            current_time += time_step

        print(f"✅ Finished segment {i+1}/{len(list_of_desired_cartesian_positions)}")

        # ✅ 计算误差
        final_cart_pos = np.array(trajectory_points[-1])
        error = np.linalg.norm(final_cart_pos - desired_cartesian_pos)
        position_errors.append(error)
        print(f"🎯 Target {i+1} goal: {desired_cartesian_pos}, reached: {final_cart_pos}, error: {error:.6f} m")

        # ✅ 保存每一段轨迹
        all_trajectory_points.extend(trajectory_points)
        np.save(FINAL_DIR / f"trajectory_points_{i}.npy", np.array(trajectory_points))

    # ✅ 保存误差信息
    position_errors = np.array(position_errors)
    np.save(FINAL_DIR / "position_errors.npy", position_errors)
    print(f"\n💾 Saved position errors to position_errors.npy")
    for i, e in enumerate(position_errors):
        print(f"Segment {i+1}: position error = {e:.6f} m")
    print(f"📊 Mean position error: {np.mean(position_errors):.6f} m")

    # ✅ 绘制轨迹
    all_traj = np.array(all_trajectory_points)
    np.save(FINAL_DIR / "final_trajectory.npy", all_traj)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(all_traj[:, 0], all_traj[:, 1], all_traj[:, 2], 'r-', linewidth=2, label='End-effector path')

    # 绘制目标点
    targets = np.array(list_of_desired_cartesian_positions)
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='b', s=80, marker='o', label='Target Points')

    # 标注误差
    for i, (target, err) in enumerate(zip(targets, position_errors)):
        ax.text(target[0], target[1], target[2] + 0.02, f"E{i+1}={err:.3f}m", color='blue')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory and Target Errors")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(FINAL_DIR / "final_trajectory_with_errors.png", dpi=300)
    plt.close(fig)

    print("📈 Saved trajectory plot with target points and error labels.")
        # ============================================================
    # ✅ visualize each trajectory segment separately
    # ============================================================
    

    print("\n📊 visualizing trajectories...")

    for i in range(len(list_of_desired_cartesian_positions)):
        traj_path = FINAL_DIR / f"trajectory_points_{i}.npy"
        if not traj_path.exists():
            print(f"⚠️ 未找到 {traj_path.name}，跳过该段。")
            continue

        traj = np.load(traj_path)
        print(f"Loaded {traj_path.name}, shape = {traj.shape}")  # debug

        # sanity reshape/validate
        if traj.ndim == 1:
            traj = traj.reshape(-1, 3)
        if traj.ndim != 2 or traj.shape[1] != 3:
            print(f"⚠️ 文件 {traj_path.name} 形状异常: {traj.shape}, 跳过")
            continue

        # debug: value ranges and number of unique points (rounded to 6 decimals)
        x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
        y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
        z_min, z_max = traj[:, 2].min(), traj[:, 2].max()
        uniq = np.unique(np.round(traj, 6), axis=0).shape[0]
        print(f"  X range: {x_min:.6f} ~ {x_max:.6f}")
        print(f"  Y range: {y_min:.6f} ~ {y_max:.6f}")
        print(f"  Z range: {z_min:.6f} ~ {z_max:.6f}")
        print(f"  Unique points (rounded 1e-6): {uniq}")

        target = np.array(list_of_desired_cartesian_positions[i])
        error_val = position_errors[i] if i < len(position_errors) else None

        # ---- 3D traj with improved visibility ----
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot line and overlay small semi-transparent dots so dense trajectories are visible
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='red', linewidth=2.5, alpha=0.9, label='End-effector path')
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color='red', s=2, alpha=0.6)

        # start/end/target markers
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=50, marker='^', label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='orange', s=60, marker='x', label='End')
        ax.scatter(target[0], target[1], target[2], c='blue', s=80, marker='o', label='Target')

        if error_val is not None:
            ax.text(target[0], target[1], target[2] + 0.02, f"Err={error_val:.3f} m", color='black', fontsize=10)

        # expand axes by a small margin to avoid clipping / compression
        margin = 0.02  # 2 cm margin
        xa = (x_min - margin, x_max + margin)
        ya = (y_min - margin, y_max + margin)
        za = (z_min - margin, z_max + margin)

        # ensure non-zero range (if min==max, expand)
        def ensure_range(a):
            if abs(a[1] - a[0]) < 1e-6:
                return (a[0] - 0.01, a[1] + 0.01)
            return a

        xa = ensure_range(xa)
        ya = ensure_range(ya)
        za = ensure_range(za)

        ax.set_xlim(xa)
        ax.set_ylim(ya)
        ax.set_zlim(za)

        # optional: make 3D axes equal scale (approx)
        # compute center and max span
        cx = 0.5 * (xa[0] + xa[1])
        cy = 0.5 * (ya[0] + ya[1])
        cz = 0.5 * (za[0] + za[1])
        max_span = max(xa[1]-xa[0], ya[1]-ya[0], za[1]-za[0]) / 2.0
        ax.set_xlim(cx - max_span, cx + max_span)
        ax.set_ylim(cy - max_span, cy + max_span)
        ax.set_zlim(cz - max_span, cz + max_span)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"3D Trajectory for Target {i}")
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        save_3d = FINAL_DIR / f"trajectory_segment_{i}_3d.png"
        plt.savefig(save_3d, dpi=300)
        print(f"💾 saved: {save_3d} (3D)")
        plt.close(fig)

        # ---- Top-down XY view ----
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2.5, alpha=0.9, label='End-effector path')
        ax2.scatter(traj[:, 0], traj[:, 1], c='r', s=3, alpha=0.6)
        ax2.scatter(target[0], target[1], c='b', s=80, marker='o', label='Target')
        ax2.scatter(traj[-1, 0], traj[-1, 1], c='g', s=60, marker='x', label='End')
        if error_val is not None:
            ax2.text(target[0], target[1] + 0.01, f"Err={error_val:.3f} m", ha='center', fontsize=9)
        # set limits and equal aspect
        ax2.set_xlim(xa[0], xa[1])
        ax2.set_ylim(ya[0], ya[1])
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f"Top-Down View (Target {i})")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_2d = FINAL_DIR / f"trajectory_segment_{i}_xy.png"
        plt.savefig(save_2d, dpi=300)
        print(f"💾 saved: {save_2d} (XY)")
        plt.close(fig2)

    print("✅ All trajectory plots saved.")


    print("✅ Simulation and analysis complete.")


    
    

if __name__ == '__main__':
    main()
    # test rollout loader
    rls = load_rollouts(indices=[0,1,2,3], directory=FINAL_DIR)  # looks for ./data_1.pkl or ./1.pkl, up to 4
    print(f"Loaded {len(rls)} rollouts")
    print("First rollout keys lengths:",len(rls[0].time),len(rls[0].q_mes_all),len(rls[0].qd_mes_all))