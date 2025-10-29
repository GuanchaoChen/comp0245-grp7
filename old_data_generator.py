import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final3.rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists


PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

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

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    # print init joint
    print(f"Initial joint angles: {init_joint_angles}")
    
    # check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")


    joint_vel_limits = sim.GetBotJointsVelLimit()
    # increase the joint vel limits to not trigger warning in the simulation
    #joint_vel_limits = [vel * 100 for vel in joint_vel_limits]
    
    print(f"joint vel limits: {joint_vel_limits}")
    
    # desired value for regulation
    q_des =  init_joint_angles
    qd_des_clip = np.zeros(num_joints)
    
    
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # P controller high level
    kp_pos = 100  # position
    kp_ori = 0    # orientation

    # PD controller gains low level (feedback gain)
    kp = 1000
    kd = 100

    # desired cartesian position
    list_of_desired_cartesian_positions = [[0.5,0.0,0.1], 
                                           [0.4,0.2,0.1], 
                                           [0.4,-0.2,0.1], 
                                           [0.5,0.0,0.1],
                                           [0.4,0.0,0.1],
                                           [0.5,0.2,0.1],
                                           [0.5,-0.2,0.1],
                                           [0.4,0.2,0.1],
                                           [0.4,-0.2,0.1],
                                           [0.5,0.2,0.1]]  # in meters

    # desired cartesian orientation in quaternion (XYZW)
    list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0]]  # in quaternion XYZW
    list_of_type_of_control = ["pos", "pos", "pos", "pos", "pos", "pos", "pos", "pos", "pos", "pos"] # "pos",  "ori" or "both"
    list_of_duration_per_desired_cartesian_positions = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0] # in seconds
    list_of_initialjoint_positions = [init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles]  # None or specific joint angles

    # Initialize data storage
    q_mes_all, qd_mes_all, qdd_est_all, q_d_all, qd_d_all, qdd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], [], [], []

    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()


    for i in range(len(list_of_desired_cartesian_positions)):

        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        if list_of_initialjoint_positions[i] is None:
            init_position = init_joint_angles
        else:
            init_position = list_of_initialjoint_positions[i]
        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position, desired_cartesian_pos, np.zeros(3), desired_cartesian_ori, np.zeros(3), time_step, type_of_control, kp_pos, kp_ori, np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos/time_step)

        # reinitialize the robot to the initial position
        sim.ResetPose()
        if init_position is not None:
            sim.SetjointPosition(init_position)
        # Data collection loop
        for t in range(steps):
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity
            # Compute desired joint positions and velocities using Cartesian differential kinematics
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command


            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            
            # Conditional data recording
            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                qdd_est_all.append(qdd_est) # new
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                # qdd_d_all.append(qdd_des) # new
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            #print("Current time in seconds:", current_time)
    
        current_time = 0  # Reset current time for potential future use

        if len(q_mes_all) > 0:    
            print("Preparing to save data...")
            # Downsample data
            # Plot the downsampled data
            
            q_mes_all_downsampled = q_mes_all[::downsample_rate]
            qd_mes_all_downsampled = qd_mes_all[::downsample_rate]
            qdd_est_all_downsampled = qdd_est_all[::downsample_rate] # new
            q_d_all_downsampled = q_d_all[::downsample_rate]  # new
            qd_d_all_downsampled = qd_d_all[::downsample_rate]  # new
            qdd_d_all_downsampled = qdd_d_all[::downsample_rate]  # new
            tau_mes_all_downsampled = tau_mes_all[::downsample_rate]
            cart_pos_all_downsampled = cart_pos_all[::downsample_rate]
            cart_ori_all_downsampled = cart_ori_all[::downsample_rate]

            time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_downsampled))]

            # Save data to pickle file and for name use the current iteration number
            filename = FINAL_DIR / f"data_{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump({
                    'time': time_array,
                    'q_mes_all': q_mes_all_downsampled,
                    'qd_mes_all': qd_mes_all_downsampled,
                    'qdd_est_all': qdd_est_all_downsampled,  # new
                    'q_des_all': q_d_all_downsampled,   # new
                    'qd_des_all': qd_d_all_downsampled,  # new
                    'qdd_des_all': qdd_d_all_downsampled,  # new
                    'tau_mes_all': tau_mes_all_downsampled,
                    'cart_pos_all': cart_pos_all_downsampled,
                    'cart_ori_all': cart_ori_all_downsampled,
                    'cart_pos_final': list_of_desired_cartesian_positions[i],  # new
                    'cart_ori_final': list_of_desired_cartesian_orientations[i]  # new
                }, f)
            print(f"Data saved to {filename}")

            # Reinitialize data storage lists
        q_mes_all, qd_mes_all, qdd_est_all, q_d_all, qd_d_all, qdd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], [], [], []

        if PRINT_PLOTS:
            print("Plotting downsampled data...")
            # Plot joint positions
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_downsampled[0])):
                joint_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Positions (rad)')
            plt.title('Downsampled Joint Positions')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_mes_all_downsampled[0])):
                joint_velocities = [qd[joint_idx] for qd in qd_mes_all_downsampled]
                plt.plot(time_array, joint_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    
    

if __name__ == '__main__':
    main()
    # test rollout loader
    rls = load_rollouts(indices=[0,1,2,3], directory=FINAL_DIR)  # looks for ./data_1.pkl or ./1.pkl, up to 4
    print(f"Loaded {len(rls)} rollouts")
    print("First rollout keys lengths:",len(rls[0].time),len(rls[0].q_mes_all),len(rls[0].qd_mes_all))