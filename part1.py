# part1_improved.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from rollout_loader import load_rollouts
import random

# ------------------------
# Config
# ------------------------
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 800
PATIENCE = 30     # early stopping patience on val loss
HIDDEN_SIZES = (256, 128, 64)
MODEL_SAVE_PATH = "part1_mlp.pt"
FINAL_DIR = Path(__file__).resolve().parent  # adjust if needed

# ------------------------
# Reproducibility
# ------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ------------------------
# Load data via your rollout loader
# ------------------------
def load_dataset_from_rollouts(directory: str | Path):
    rollouts = load_rollouts(directory=directory)
    X_list = []
    y_list = []
    for r in rollouts:
        # r.q_mes_all, r.qd_mes_all, r.q_des_all, r.qd_des_all, r.tau_mes_all expected
        # Validate presence
        if not hasattr(r, "path"):
            pass
        # convert to numpy arrays
        q_mes = np.array(r.q_mes_all)        # (T, 7)
        qd_mes = np.array(r.qd_mes_all)      # (T, 7)
        tau = np.array(r.tau_mes_all)        # (T, 7)

        # Expect q_des_all & qd_des_all to be present in the saved dict
        # rollouts' from_dict will only include keys that were saved in pickle,
        # but r object fields are created from keys in from_dict; we check attributes
        # If not present, try to read via re-loading file to check keys.
        # Here we check attribute 'q_d_all' fallback
        has_qdes = hasattr(r, "q_des_all") or hasattr(r, "q_d_all") or False
        has_qddes = hasattr(r, "qd_des_all") or hasattr(r, "qd_d_all") or False

        if not (has_qdes and has_qddes):
            raise RuntimeError(
                "Rollouts do not contain q_des_all / qd_des_all. "
                "Please modify data_generator.py to save 'q_des_all' and 'qd_des_all' into the pickle files."
            )

        # handle possible naming differences: q_des_all vs q_d_all
        if hasattr(r, "q_des_all"):
            q_des = np.array(r.q_des_all)
        else:
            q_des = np.array(getattr(r, "q_d_all"))

        if hasattr(r, "qd_des_all"):
            qd_des = np.array(r.qd_des_all)
        else:
            qd_des = np.array(getattr(r, "qd_d_all"))

        # sanity shapes
        T = q_mes.shape[0]
        assert qd_mes.shape[0] == T and tau.shape[0] == T and q_des.shape[0] == T and qd_des.shape[0] == T

        # Build features per timestep:
        # features = [q_error (7), qd_error (7), q_mes (7), qd_mes (7)] -> 28-dim
        q_error = (q_des - q_mes)
        qd_error = (qd_des - qd_mes)
        features = np.concatenate([q_error, qd_error, q_mes, qd_mes], axis=1)  # (T,28)

        X_list.append(features)
        y_list.append(tau)

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y

# ------------------------
# Simple MLP model (flexible)
# ------------------------
class TorqueMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=HIDDEN_SIZES, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------
# Training loop with early stopping
# ------------------------
def train_model(model, train_loader, val_loader, epochs=MAX_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = running / count
        # val
        model.eval()
        with torch.no_grad():
            vrunning = 0.0
            vcount = 0
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                vrunning += loss.item() * xb.size(0)
                vcount += xb.size(0)
            val_loss = vrunning / vcount

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{epochs}  TrainLoss={train_loss:.6e}  ValLoss={val_loss:.6e}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_state, MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.6e}")
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses

# ------------------------
# Utils: metrics & plotting
# ------------------------
def evaluate_and_report(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred = model(X_tensor).cpu().numpy()
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    # r2 per-joint average
    try:
        r2 = r2_score(y_test, pred, multioutput='uniform_average')
    except Exception:
        r2 = float('nan')
    print("Test MSE: ", mse)
    print("Test MAE: ", mae)
    print("Test R2: ", r2)
    return pred, {'mse': mse, 'mae': mae, 'r2': r2}

def plot_loss(train_losses, val_losses, outpath="loss.png"):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log)')
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()

def plot_torque_sample(y_true, y_pred, n_examples=3, outdir="pred_samples"):
    os.makedirs(outdir, exist_ok=True)
    T = y_true.shape[0]
    idxs = np.random.choice(T, size=n_examples, replace=False)
    for i, idx in enumerate(idxs):
        plt.figure(figsize=(8,4))
        plt.plot(y_true[idx], label="true (7 joints)", marker='o')
        plt.plot(y_pred[idx], label="pred (7 joints)", marker='x')
        plt.title(f"Torque sample idx={idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"torque_sample_{i}.png"))
        plt.close()

# ------------------------
# Main
# ------------------------
def main():
    print("Loading data from rollouts...")
    X, y = load_dataset_from_rollouts(FINAL_DIR)
    print("Raw shapes:", X.shape, y.shape)  # expect (N,28) & (N,7)

    # train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=SEED, shuffle=True)
    # above yields roughly 70/15/15

    # to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    in_dim = X.shape[1]
    out_dim = y.shape[1]
    model = TorqueMLP(in_dim=in_dim, out_dim=out_dim).to(device)
    print(model)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader)

    # save final model (already saved best in training)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)

    # evaluate
    y_pred, metrics = evaluate_and_report(model, X_test, y_test)
    print(metrics)

    # plots
    plot_loss(train_losses, val_losses, outpath="part1_loss.png")
    plot_torque_sample(y_test, y_pred, n_examples=5, outdir="part1_samples")

if __name__ == "__main__":
    main()


# Early stopping at epoch 281. Best val loss: 1.665780e+01
# Model saved to part1_mlp.pt
# Test MSE:  12.906196478219885
# Test MAE:  0.4520408338442631
# Test R2:  0.9917079015490604
# Metrics: {'mse': 12.906196478219885, 'mae': 0.4520408338442631, 'r2': 0.9917079015490604}