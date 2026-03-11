import os
import sys
import json
import numpy as np
import torch
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
        'task_id': 'linreg_lvl6_lasso_proximal',
        'algorithm': 'LASSO Regression (Proximal Gradient Descent)',
        'dataset': 'synthetic_sparse',
        'n_features': 10,
        'lambda': 0.1,
        'lr': 0.01,
        'epochs': 1000,
        'output_dir': OUTPUT_DIR,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=None):
    set_seed(42)

    w_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    X = np.random.randn(300, 10)
    noise = np.random.randn(300) * 0.3
    y = X @ w_true + noise

    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)

    return {
        'X_train': torch.FloatTensor(X_train_np),
        'y_train': torch.FloatTensor(y_train_np),
        'X_val': torch.FloatTensor(X_val_np),
        'y_val': torch.FloatTensor(y_val_np),
        'X_train_np': X_train_np,
        'y_train_np': y_train_np,
        'X_val_np': X_val_np,
        'y_val_np': y_val_np,
        'w_true': w_true,
    }


class LassoModel:
    def __init__(self, n_features=10):
        self.w = torch.zeros(n_features, requires_grad=True, dtype=torch.float32)
        self.b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
        self.n_features = n_features


def build_model(n_features=10):
    return LassoModel(n_features)


def train(model, data, lambda_val=0.1, lr=0.01, epochs=1000):
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    m = float(X_train.shape[0])

    train_losses = []
    val_losses = []
    sparsity_history = []

    for epoch in range(epochs):
        if model.w.grad is not None:
            model.w.grad.zero_()
        if model.b.grad is not None:
            model.b.grad.zero_()

        y_pred = X_train @ model.w + model.b
        loss = (1.0 / (2.0 * m)) * torch.sum((y_pred - y_train) ** 2)
        loss.backward()

        with torch.no_grad():
            w_temp = model.w - lr * model.w.grad
            model.w.data = torch.sign(w_temp) * torch.clamp(torch.abs(w_temp) - lr * lambda_val, min=0.0)
            model.b.data = model.b - lr * model.b.grad

        train_losses.append(float(loss.item()))

        with torch.no_grad():
            y_val_pred = X_val @ model.w + model.b
            val_loss = (1.0 / (2.0 * float(X_val.shape[0]))) * torch.sum((y_val_pred - y_val) ** 2)
            val_losses.append(float(val_loss.item()))

        w_np = model.w.detach().numpy()
        sparsity_history.append(float(np.mean(np.abs(w_np) < 1e-4)))

        if (epoch + 1) % 200 == 0:
            lasso_obj = float(loss.item()) + lambda_val * float(torch.sum(torch.abs(model.w)).item())
            print(f"  Epoch [{epoch+1}/{epochs}]  smooth_loss={loss.item():.4f}  lasso_obj={lasso_obj:.4f}  sparsity={sparsity_history[-1]:.2f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'sparsity_history': sparsity_history,
    }


def evaluate(model, X, y):
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(y, np.ndarray):
        y_np = y.flatten()
    else:
        y_np = y.numpy().flatten()

    with torch.no_grad():
        y_pred_t = X @ model.w + model.b
    y_pred = y_pred_t.numpy().flatten()

    mse = float(np.mean((y_pred - y_np) ** 2))
    ss_res = float(np.sum((y_np - y_pred) ** 2))
    ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    w_vals = model.w.detach().numpy()
    sparsity_ratio = float(np.mean(np.abs(w_vals) < 1e-4))
    n_zero_weights = int(np.sum(np.abs(w_vals) < 1e-4))

    return {
        'mse': mse,
        'r2': float(r2),
        'sparsity_ratio': sparsity_ratio,
        'n_zero_weights': n_zero_weights,
        'weights': w_vals.tolist(),
    }


def predict(model, X):
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    with torch.no_grad():
        return (X @ model.w + model.b).numpy().flatten()


def save_artifacts(model, data, history, train_metrics, val_metrics):
    w_vals = model.w.detach().numpy()
    np.save(os.path.join(OUTPUT_DIR, 'linreg_lvl6_weights.npy'), w_vals)

    metrics_json = {
        'train': {k: v for k, v in train_metrics.items()},
        'val': {k: v for k, v in val_metrics.items()},
    }
    with open(os.path.join(OUTPUT_DIR, 'linreg_lvl6_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['train_losses'], label='Train loss')
    ax1.plot(history['val_losses'], label='Val MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('LASSO Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['sparsity_history'], color='orange', label='Sparsity ratio')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Fraction |w_i| < 1e-4')
    ax2.set_title('Weight Sparsity vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl6_loss.png'), dpi=150)
    plt.close()

    w_true = data['w_true']
    n = len(w_true)
    idx = np.arange(n)
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(idx - width / 2, w_true, width, label='True weights', alpha=0.8)
    plt.bar(idx + width / 2, w_vals, width, label='Learned weights', alpha=0.8)
    plt.xlabel('Feature index')
    plt.ylabel('Weight value')
    plt.title('True vs Learned Weights - LASSO Proximal GD')
    plt.xticks(idx, [f'w{i}' for i in range(n)])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl6_weights.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    print("=" * 60)
    print("linreg_lvl6: LASSO Regression + Proximal Gradient Descent")
    print("=" * 60)

    print("\n[1] Generating sparse synthetic data (300 samples, 10 features)...")
    data = make_dataloaders()
    print(f"  Train: {data['X_train'].shape}   Val: {data['X_val'].shape}")
    print(f"  True weights: {data['w_true']}")

    print("\n[2] Building LassoModel (raw tensors, no nn.Module)...")
    model = build_model(n_features=10)

    print("\n[3] Proximal GD training (lambda=0.1, lr=0.01, 1000 epochs)...")
    history = train(model, data, lambda_val=0.1, lr=0.01, epochs=1000)

    print("\n[4] Evaluating on train and val sets...")
    train_metrics = evaluate(model, data['X_train'], data['y_train'])
    val_metrics = evaluate(model, data['X_val'], data['y_val'])

    print(f"\n  Learned weights: {[f'{w:.4f}' for w in val_metrics['weights']]}")
    print(f"  True weights:    {data['w_true'].tolist()}")
    print(f"\n  TRAIN  MSE={train_metrics['mse']:.4f}  R2={train_metrics['r2']:.4f}  Sparsity={train_metrics['sparsity_ratio']:.2f}  n_zeros={train_metrics['n_zero_weights']}")
    print(f"  VAL    MSE={val_metrics['mse']:.4f}  R2={val_metrics['r2']:.4f}  Sparsity={val_metrics['sparsity_ratio']:.2f}  n_zeros={val_metrics['n_zero_weights']}")

    print("\n[5] Saving artifacts...")
    save_artifacts(model, data, history, train_metrics, val_metrics)
    print(f"  Saved to: {OUTPUT_DIR}")

    print("\n[6] Quality assertions...")
    exit_code = 0
    checks = [
        ('val r2 > 0.85', val_metrics['r2'] > 0.85),
        ('val sparsity_ratio > 0.50', val_metrics['sparsity_ratio'] > 0.50),
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
