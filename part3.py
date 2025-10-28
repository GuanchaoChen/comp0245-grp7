import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, CartesianDiffKin
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models
from mpl_toolkits.mplot3d import Axes3D
# Choose model type
neural_network_or_random_forest = "neural_network"

# Updated MLP with input = 11 (time + goal + q_mes(7))

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(11, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         return self.model(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load time from dataset to determine test horizon
    time_list = []
    for i in range(10):
        filename = os.path.join(script_dir, f'data_{i}.pkl')
        if not os.path.isfile(filename):
            continue
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if 'time' in data:
            time_list.append(np.array(data['time']))

    if len(time_list) == 0:
        print("❌ No time data found!")
        return

    time_array = np.concatenate(time_list, axis=0)

    print(f"✅ Loaded time data → shape {time_array.shape}")

    # Load trained joint networks
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            model = MLP()
            model.load_state_dict(torch.load(os.path.join(script_dir, f"v2mlp{joint_idx+1}.pt")))
            model.eval()
            models.append(model)

    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            model = joblib.load(os.path.join(script_dir, f"rf_joint{joint_idx+1}.joblib"))
            models.append(model)

    else:
        raise ValueError("Invalid model type!")

    # Goal space (same as Part2 test)
    goal_positions = [
        [0.5, 0.0, 0.1],
        [0.4, 0.2, 0.1]
        
    ]
    # number_of_goal_positions_to_test = len(goal_positions)

    conf_file_name = "pandaconfig.json"
    root_dir = script_dir
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, ["pybullet"], False, 0, root_dir)

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()

    joint_vel_limits = sim.GetBotJointsVelLimit()
    time_step = sim.GetTimeStep()
    test_times = np.arange(time_array.min(), time_array.max(), time_step)

    cmd = MotorCommands()
    kp, kd = 1000, 100  # You can tune this
    errors = []

    for goal in goal_positions:
        print(f"\n🎯 New Goal: {goal}")
        sim.ResetPose()

        # initialize state
        predicted_q = np.zeros((len(test_times), 7))
        predicted_q[0, :] = init_joint_angles.copy()

        for t in range(len(test_times)):
            current_t = test_times[t]
            current_q = predicted_q[t, :].copy()

            # Model input vector = [time, goal(3), q_mes(7)]
            x = np.hstack(([current_t], goal, current_q)).reshape(1, -1)

            pred_val = np.zeros(7)
            if neural_network_or_random_forest == "neural_network":
                with torch.no_grad():
                    xtensor = torch.from_numpy(x).float()
                    for j in range(7):
                        pred_val[j] = models[j](xtensor).numpy().flatten()[0]
            else:
                for j in range(7):
                    pred_val[j] = models[j].predict(x)[0]

            predicted_q[t, :] = pred_val

            if t < len(test_times) - 1:
                predicted_q[t+1, :] = pred_val

            # Simulation feedback control
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)

            if t < len(test_times) - 1:
                qd_des = (predicted_q[t+1] - predicted_q[t]) / time_step
            else:
                qd_des = np.zeros(7)

            qd_des = np.clip(qd_des, -np.array(joint_vel_limits), np.array(joint_vel_limits))

            tau = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, predicted_q[t], qd_des, kp, kd)
            cmd.SetControlCmd(tau, ["torque"] * 7)
            sim.Step(cmd, "torque")

            time.sleep(time_step)

        final_pos, _ = dyn_model.ComputeFK(predicted_q[-1], controlled_frame_name)
        err = np.linalg.norm(final_pos - np.array(goal))
        errors.append(err)
        print(f"📌 Final Pos: {final_pos} | Error: {err:.4f} m ✅")

    print("\n=== ✅ TEST SUMMARY ✅ ===")
    print(f"Mean Cartesian Error: {np.mean(errors):.4f} m")
    print(f"Max Error: {max(errors):.4f} m")

        # --- 绘制末端执行器轨迹 ---
    

    for idx, goal in enumerate(goal_positions):
        # 记录整个轨迹的末端位置
        cartesian_traj = np.zeros((len(test_times), 3))
        for t in range(len(test_times)):
            cartesian_traj[t], _ = dyn_model.ComputeFK(predicted_q[t], controlled_frame_name)

        # 2D轨迹随时间
        plt.figure(figsize=(10,5))
        plt.plot(test_times, cartesian_traj[:,0], label='X')
        plt.plot(test_times, cartesian_traj[:,1], label='Y')
        plt.plot(test_times, cartesian_traj[:,2], label='Z')
        plt.scatter(test_times[-1], cartesian_traj[-1,0], color='green', label='End')
        plt.xlabel("Time [s]")
        plt.ylabel("Cartesian Position [m]")
        plt.title(f"Goal {idx+1} Cartesian Position vs Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3D空间轨迹
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(cartesian_traj[:,0], cartesian_traj[:,1], cartesian_traj[:,2], label='Trajectory', color='blue')
        ax.scatter(goal[0], goal[1], goal[2], color='red', s=60, label='Goal')
        ax.scatter(cartesian_traj[-1,0], cartesian_traj[-1,1], cartesian_traj[-1,2], color='green', s=60, label='End')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f"Goal {idx+1} Trajectory in 3D Space")
        ax.legend()
        plt.show()

        # 打印末端误差
        position_error = np.linalg.norm(cartesian_traj[-1] - np.array(goal))
        print(f"Goal {idx+1} position error: {position_error:.4f} m")

    

if __name__ == '__main__':
    main()
