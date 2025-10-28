from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rollout_loader import load_rollouts

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("-" * 50) 

rollouts = load_rollouts(indices=range(50)) 

X_list = []
y_list = []

for rollout in rollouts:
    q_des_traj = np.array(rollout.q_d_all)
    q_mes_traj = np.array(rollout.q_mes_all)
    tau_traj = np.array(rollout.tau_mes_all)
    
    error = q_des_traj - q_mes_traj
    
    X_list.append(error)
    y_list.append(tau_traj)

X = np.vstack(X_list)
y = np.vstack(y_list)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)  #standardize the data
y_train_scaled = scaler_y.fit_transform(y_train)

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

class RobotTorqueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = RobotTorqueDataset(X_train_scaled, y_train_scaled)
val_dataset = RobotTorqueDataset(X_val_scaled, y_val_scaled)
test_dataset = RobotTorqueDataset(X_test_scaled, y_test_scaled)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class MLPNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class MLPNet2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet2, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet5(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet5, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 512)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet3(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet3, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(256, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(128, 64)
        self.activation = nn.ReLU()
        self.hidden_layer_4 = nn.Linear(64, 32)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.hidden_layer_3(x)
        x = self.activation(x)
        x = self.hidden_layer_4(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MLPNet4(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNet4, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, 512)
        self.activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(256, 128)
        self.activation = nn.ReLU()
        self.hidden_layer_4 = nn.Linear(128, 64)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.hidden_layer_3(x)
        x = self.activation(x)
        x = self.hidden_layer_4(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

input_dim = X.shape[1] 
output_dim = y.shape[1]
print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

model = MLPNet4(input_size=input_dim, output_size=output_dim).to(device)

print("\n 开始训练模型...")
learning_rate = 0.0003   
num_epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练和验证损失
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
print("模型训练完成！")

model.eval()
all_predictions, all_targets = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
y_pred_scaled = np.vstack(all_predictions)
y_test_scaled_from_loader = np.vstack(all_targets)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test_scaled_from_loader)
mse = mean_squared_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)
print(f"\n模型在测试集上的最终表现:")
print(f"  均方误差 (MSE): {mse:.6f}")
print(f"  决定系数 (R-squared): {r2:.4f}")

# 绘制训练和验证损失曲线
plt.figure(figsize=(15, 5))

# 子图1：训练和验证损失曲线
plt.subplot(1, 2, 1)
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
plt.plot(epochs_range, val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：关节力矩预测对比
joint_index = 2
num_points_to_plot = 300
plt.subplot(1, 2, 2)
plt.plot(y_test_real[:num_points_to_plot, joint_index], label='Ground Truth', color='b', linewidth=2)
plt.plot(y_pred[:num_points_to_plot, joint_index], label='Predicted', color='r', linestyle='--', linewidth=2)
plt.title(f'Joint {joint_index+1} Torque Prediction vs Ground Truth')
plt.xlabel('Time Step')
plt.ylabel('Torque')
plt.legend()
plt.grid(True)

# 在图上添加MSE和R²数值
textstr = f'MSE: {mse:.6f}\nR²: {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()