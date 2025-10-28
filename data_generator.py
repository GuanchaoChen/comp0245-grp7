import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists

PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

downsample_rate = 2  # 下采样步长

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
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
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
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")

    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"joint vel limits: {joint_vel_limits}")

    q_des = init_joint_angles
    qd_des_clip = np.zeros(num_joints)
    current_time = 0
    cmd = MotorCommands()

    kp_pos = 100
    kp_ori = 0
    kp = 1000
    kd = 100

    # -------------------- 修改 1：生成 10 个随机目标点 --------------------
    np.random.seed(42)
    num_targets = 20
    x_min, x_max = 0.4, 0.6
    y_min, y_max = -0.2, 0.2
    z_fixed = 0.1

    list_of_desired_cartesian_positions = [
        [np.random.uniform(x_min, x_max),
         np.random.uniform(y_min, y_max),
         z_fixed] for _ in range(num_targets)
    ]

    list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0]] * num_targets
    list_of_type_of_control = ["pos"] * num_targets
    list_of_duration_per_desired_cartesian_positions = [5.0] * num_targets
    list_of_initialjoint_positions = [init_joint_angles] * num_targets
    # ---------------------------------------------------------------

    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []
    current_time = 0
    time_step = sim.GetTimeStep()

    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        init_position = list_of_initialjoint_positions[i] if list_of_initialjoint_positions[i] is not None else init_joint_angles

        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position,
                                    desired_cartesian_pos, np.zeros(3),
                                    desired_cartesian_ori, np.zeros(3),
                                    time_step, type_of_control, kp_pos, kp_ori,
                                    np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos / time_step)

        sim.ResetPose()
        sim.SetjointPosition(init_position)

        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0), dtype=float)

            pd_d = [0.0, 0.0, 0.0]
            ori_d_des = [0.0, 0.0, 0.0]

            q_des, qd_des_clip = CartesianDiffKin(dyn_model, controlled_frame_name, q_mes,
                                                  desired_cartesian_pos, pd_d,
                                                  desired_cartesian_ori, ori_d_des,
                                                  time_step, "pos", kp_pos, kp_ori,
                                                  np.array(joint_vel_limits))

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * num_joints)
            sim.Step(cmd, "torque")

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)

            time.sleep(time_step)
            current_time += time_step

        current_time = 0

        if len(q_mes_all) > 0:
            print("Preparing to save data...")

            q_mes_all_ds = q_mes_all[::downsample_rate]
            qd_mes_all_ds = qd_mes_all[::downsample_rate]
            q_d_all_ds = q_d_all[::downsample_rate]
            qd_d_all_ds = qd_d_all[::downsample_rate]
            tau_mes_all_ds = tau_mes_all[::downsample_rate]
            cart_pos_all_ds = cart_pos_all[::downsample_rate]
            cart_ori_all_ds = cart_ori_all[::downsample_rate]

            time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_ds))]

            filename = FINAL_DIR / f"data_{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump({
                    'time': time_array,
                    'q_mes_all': q_mes_all_ds,
                    'qd_mes_all': qd_mes_all_ds,
                    'q_des_all': q_d_all_ds,
                    'qd_des_all': qd_d_all_ds,
                    'tau_mes_all': tau_mes_all_ds,
                    'cart_pos_all': cart_pos_all_ds,
                    'cart_ori_all': cart_ori_all_ds,
                    'final_target_cart_pos': desired_cartesian_pos.tolist(),  # 新增
                    'final_target_cart_ori': desired_cartesian_ori.tolist(),   # 新增
                    'goal_position': [desired_cartesian_pos.tolist()] * len(q_mes_all_ds)
                }, f)
            print(f"Data saved to {filename}")

        # Reset storage lists
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []

        if PRINT_PLOTS:
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_ds[0])):
                joint_positions = [q[joint_idx] for q in q_mes_all_ds]
                plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Positions (rad)')
            plt.title('Downsampled Joint Positions')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_mes_all_ds[0])):
                joint_velocities = [qd[joint_idx] for qd in qd_mes_all_ds]
                plt.plot(time_array, joint_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == '__main__':
    main()
    # Test rollout loader
    rls = load_rollouts(indices=list(range(10)), directory=FINAL_DIR)  # Load all 10 rollouts
    print(f"Loaded {len(rls)} rollouts")
    print("First rollout keys lengths:", len(rls[0].time), len(rls[0].q_mes_all), len(rls[0].qd_mes_all))
