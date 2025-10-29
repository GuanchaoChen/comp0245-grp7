import pickle
from typing_extensions import Final
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

# approximate the behaviour of CartesianDiffKin
# Configuration 
class Config:
    mode = 1     # mlp                                
    train_flag = 1                               # re-train model
    test_flag = 1                                # re-select test set
    data_dir = Path(__file__).resolve().parent   # same dir as data_0.pkl etc.
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]                       # data generator
    n_joints = 7
    hidden = [128, 64, 32, 16]
    activation = "gelu" # Smooth version of ReLU, often slightly better for continuous outputs
    batch_size = 32
    lr = 3e-4
    weight_decay = 5e-5
    dropout = 0.0
    epochs = 200
    patience = 15
    scheduler_patience = 8
    val_frac = 0.15
    test_frac = 0.15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "task2_3.pth"

cfg = Config()

# Dataset loader 
def load_rollouts_for_cartesian_diffkin(data_dir, indices):
    X_list, Y_list = [], []
    for idx in indices:
        file_path = Path(data_dir) / f"data_{idx}.pkl"
        if not file_path.exists():
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        cart_pos_final = np.array(data["cart_pos_final"])  
        cart_ori_final = np.array(data["cart_ori_final"])  
        q_mes = np.array(data["q_mes_all"])  # current joint positions
        q_des = np.array(data["q_des_all"])  # next desired joint positions (ground truth)
        qd_des = np.array(data["qd_des_all"])  # next desired joint velocity (ground truth)
        x_target = np.array(data["cart_pos_final"])  # final target Cartesian (x, y, z)
        
        # feature
        X_feat = []
        for i in q_mes:
            X_feat.append(np.concatenate([i, x_target]))
        X_feat = np.array(X_feat)
        Y_feat = np.concatenate([q_des, qd_des], axis=1)
        X_list.append(X_feat)
        Y_list.append(Y_feat)

        print(f"Loaded {file_path.name}: {len(q_mes)} samples")

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    print(f"\nDataset built: X shape {X.shape}, Y shape {Y.shape}")
    return X, Y


# Torch Dataset
class CartesianDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# MLP Model
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:  # set layers iteratively
            layers.append(nn.Linear(last, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# Train function (returns losses)
def train_model(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.scheduler_patience, factor=0.5)
    # automatically reduces the learning rate when the model’s performance stops improving
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        # Train 
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(cfg.device), Yb.to(cfg.device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, Yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(cfg.device), Yb.to(cfg.device)
                preds = model(Xb)
                loss = criterion(preds, Yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.model_path) # mlp_torque_model_1.pth
        else:
            patience_counter += 1
            if patience_counter > cfg.patience:
                print("Early stopping triggered.")
                break

    return model, train_losses, val_losses


# Main
def main():

    # 1. Load dataset
    X, Y = load_rollouts_for_cartesian_diffkin(cfg.data_dir, cfg.indices)

    # 2. Split
    N = len(X)
    n_val = int(cfg.val_frac * N)
    n_test = int(cfg.test_frac * N)
    n_train = N - n_val - n_test
    idxs = np.random.permutation(N)  # creates a shuffled array of integers from 0 to N-1
    X_train, X_val, X_test = X[idxs[:n_train]], X[idxs[n_train:n_train+n_val]], X[idxs[n_train+n_val:]] # slice from random set
    Y_train, Y_val, Y_test = Y[idxs[:n_train]], Y[idxs[n_train:n_train+n_val]], Y[idxs[n_train+n_val:]]

    # 3. Normalize --- standard normal distribution, mean 0, std 1
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_val_s = scaler_Y.transform(Y_val)
    Y_test_s = scaler_Y.transform(Y_test)

    # 4. Loaders
    train_ds = CartesianDataset(X_train_s, Y_train_s)
    val_ds = CartesianDataset(X_val_s, Y_val_s)
    test_ds = CartesianDataset(X_test_s, Y_test_s)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # 5. Model
    model = MLP(X_train_s.shape[1], Y_train_s.shape[1], cfg.hidden,
                activation=cfg.activation, dropout=cfg.dropout).to(cfg.device)
    
    if cfg.train_flag == 1:
        # 6. Train
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, cfg)

    if cfg.test_flag == 1:
        # 7. Test evaluation
        model.load_state_dict(torch.load(cfg.model_path))
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        preds_orig = scaler_Y.inverse_transform(preds)
        trues_orig = scaler_Y.inverse_transform(trues)

        metrics = compute_metrics(trues_orig, preds_orig)
        print("\nFinal test metrics:")
        for k, v in metrics.items(): # key,value
            print(f"{k:>8}: {v:.6f}")

        # Random Forest comparison
        print("\nTraining Random Forest...")
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, Y_train_s)
        preds_rf = rf.predict(X_test_s)
        metrics_rf = compute_metrics(scaler_Y.inverse_transform(preds_rf), trues_orig)
        print("\nRandom Forest Performance:")
        for k, v in metrics_rf.items(): # key,value
            print(f"{k:>8}: {v:.6f}")

        with open("rf_model.pkl", "wb") as f:
            pickle.dump(rf, f)
        print("Random Forest model saved to rf_model.pkl")    

    if cfg.train_flag == 1:
        # 8. Plot training/validation losses
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("task2_loss", dpi=200)
        plt.show()

        print(f"\nModel saved to: {cfg.model_path}")
    with open("scalers.pkl", "wb") as f:
        pickle.dump((scaler_X, scaler_Y), f)

    print("Scalers saved to scalers.pkl")

if __name__ == "__main__":
    main()

# Final test metrics:
#      MSE: 0.001945
#     RMSE: 0.044099
#      MAE: 0.011398
#       R2: 0.925409

# Training Random Forest...

# Random Forest Performance:
#      MSE: 0.000311
#     RMSE: 0.017629
#      MAE: 0.001023
#       R2: 0.999309