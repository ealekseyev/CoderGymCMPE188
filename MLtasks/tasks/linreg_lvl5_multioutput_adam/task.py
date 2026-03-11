import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_id': 'linreg_lvl5_multioutput_adam',
        'algorithm': 'Linear Regression (Multi-Output + Adam)',
        'dataset': 'synthetic',
        'n_features': 5,
        'n_outputs': 3,
        'optimizer': 'Adam',
        'output_dir': OUTPUT_DIR,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=32):
    set_seed(42)

    W_true = np.array([
        [2.0,  0.0, -1.0,  0.0,  0.5],
        [0.0, -1.0,  0.0,  3.0,  0.0],
        [1.0,  1.0, -1.0,  1.0, -1.0],
    ])

    X = np.random.randn(500, 5)
    noise = np.random.randn(500, 3) * 0.5
    Y = X @ W_true.T + noise

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model():
    model = nn.Linear(5, 3)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def train(model, train_loader, val_loader, epochs=500, lr=0.01):
    device = get_device()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), Y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                val_loss += criterion(model(X_b), Y_b).item()
            val_losses.append(val_loss / len(val_loader))

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

    return {'train_losses': train_losses, 'val_losses': val_losses}


def evaluate(model, data_loader):
    device = get_device()
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for X_b, Y_b in data_loader:
            X_b = X_b.to(device)
            preds_list.append(model(X_b).cpu().numpy())
            targets_list.append(Y_b.numpy())

    Y_pred = np.concatenate(preds_list, axis=0)
    Y_true = np.concatenate(targets_list, axis=0)

    mse = float(np.mean((Y_pred - Y_true) ** 2))

    r2_list = []
    for k in range(3):
        ss_res = float(np.sum((Y_true[:, k] - Y_pred[:, k]) ** 2))
        ss_tot = float(np.sum((Y_true[:, k] - Y_true[:, k].mean()) ** 2))
        r2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    r2_mean = float(np.mean(r2_list))

    return {
        'mse': mse,
        'r2': r2_mean,
        'r2_output_0': float(r2_list[0]),
        'r2_output_1': float(r2_list[1]),
        'r2_output_2': float(r2_list[2]),
        'r2_mean': r2_mean,
        '_Y_pred': Y_pred,
        '_Y_true': Y_true,
    }


def predict(model, X):
    device = get_device()
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        return model(X.to(device)).cpu().numpy()


def save_artifacts(model, history, train_metrics, val_metrics):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'linreg_lvl5_model.pth'))

    metrics_json = {
        'train': {k: v for k, v in train_metrics.items() if not k.startswith('_')},
        'val': {k: v for k, v in val_metrics.items() if not k.startswith('_')},
    }
    with open(os.path.join(OUTPUT_DIR, 'linreg_lvl5_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Multi-Output Linear Regression - Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl5_loss.png'), dpi=150)
    plt.close()

    Y_pred = val_metrics['_Y_pred']
    Y_true = val_metrics['_Y_true']
    output_labels = ['Output 0', 'Output 1', 'Output 2']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k in range(3):
        r2_k = val_metrics[f'r2_output_{k}']
        ax = axes[k]
        ax.scatter(Y_true[:, k], Y_pred[:, k], alpha=0.5, s=20)
        lo = min(Y_true[:, k].min(), Y_pred[:, k].min())
        hi = max(Y_true[:, k].max(), Y_pred[:, k].max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{output_labels[k]}\nR2 = {r2_k:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predicted vs Actual - Validation Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl5_scatter.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    print("=" * 60)
    print("linreg_lvl5: Multi-Output Linear Regression + Adam")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    print("\n[1] Loading synthetic multi-output data...")
    train_loader, val_loader = make_dataloaders(batch_size=32)
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    print("\n[2] Building model: nn.Linear(5, 3) + Xavier init")
    model = build_model().to(device)
    print(f"  {model}")

    print("\n[3] Training 500 epochs with Adam(lr=0.01)...")
    history = train(model, train_loader, val_loader, epochs=500, lr=0.01)

    print("\n[4] Evaluating on train and val sets...")
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    print(f"\n  TRAIN  MSE={train_metrics['mse']:.4f}  R2_mean={train_metrics['r2_mean']:.4f}")
    print(f"         R2_0={train_metrics['r2_output_0']:.4f}  R2_1={train_metrics['r2_output_1']:.4f}  R2_2={train_metrics['r2_output_2']:.4f}")
    print(f"\n  VAL    MSE={val_metrics['mse']:.4f}  R2_mean={val_metrics['r2_mean']:.4f}")
    print(f"         R2_0={val_metrics['r2_output_0']:.4f}  R2_1={val_metrics['r2_output_1']:.4f}  R2_2={val_metrics['r2_output_2']:.4f}")

    print("\n[5] Saving artifacts...")
    save_artifacts(model, history, train_metrics, val_metrics)
    print(f"  Saved to: {OUTPUT_DIR}")

    print("\n[6] Quality assertions...")
    exit_code = 0
    checks = [
        ('val r2_output_0 > 0.85', val_metrics['r2_output_0'] > 0.85),
        ('val r2_output_1 > 0.85', val_metrics['r2_output_1'] > 0.85),
        ('val r2_output_2 > 0.85', val_metrics['r2_output_2'] > 0.85),
        ('val r2_mean > 0.88', val_metrics['r2_mean'] > 0.88),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            exit_code = 1

    if exit_code == 0:
        print("\nPASS: All quality thresholds met!")
    else:
        print("\nFAIL: One or more quality thresholds not met.")

    sys.exit(exit_code)
