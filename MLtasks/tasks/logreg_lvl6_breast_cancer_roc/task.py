import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
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
        'task_id': 'logreg_lvl6_breast_cancer_roc',
        'algorithm': 'Binary Logistic Regression (Breast Cancer + RMSprop + AUC)',
        'dataset': 'breast_cancer_wisconsin',
        'n_features': 30,
        'n_classes': 2,
        'optimizer': 'RMSprop(lr=0.001, alpha=0.99)',
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

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)

    X_train_t = torch.FloatTensor(X_train_np)
    y_train_t = torch.FloatTensor(y_train_np).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_np)
    y_val_t = torch.FloatTensor(y_val_np).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model():
    model = nn.Linear(30, 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def train(model, train_loader, val_loader, epochs=200, lr=0.001):
    device = get_device()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            probs = torch.sigmoid(model(X_b))
            loss = criterion(probs, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = (probs >= 0.5).float()
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(correct / total)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                probs = torch.sigmoid(model(X_b))
                val_loss += criterion(probs, y_b).item()
                preds = (probs >= 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_correct / val_total)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  loss={train_losses[-1]:.4f}  val_acc={val_accuracies[-1]:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }


def compute_manual_auc(probs, targets):
    probs = np.array(probs, dtype=float)
    targets = np.array(targets, dtype=float)

    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))

    thresholds = np.sort(np.unique(probs))[::-1]

    fpr_list = [0.0]
    tpr_list = [0.0]
    optimal_threshold = float(thresholds[0]) if len(thresholds) > 0 else 0.5
    max_j = -np.inf

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (targets == 1)))
        fp = int(np.sum((preds == 1) & (targets == 0)))
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        j = tpr - fpr
        if j > max_j:
            max_j = j
            optimal_threshold = float(tau)

    fpr_list.append(1.0)
    tpr_list.append(1.0)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    tpr_arr = tpr_arr[sort_idx]

    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    auc = float(trapz_fn(tpr_arr, fpr_arr))
    return auc, optimal_threshold, fpr_arr, tpr_arr


def evaluate(model, data_loader):
    device = get_device()
    model.eval()
    criterion = nn.BCELoss()

    all_probs = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            probs = torch.sigmoid(model(X_b))
            total_loss += criterion(probs, y_b).item()
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y_b.cpu().numpy().flatten())

    probs = np.array(all_probs)
    targets = np.array(all_targets)

    preds = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(preds == targets))

    auc, optimal_threshold, fpr_arr, tpr_arr = compute_manual_auc(probs, targets)

    preds_opt = (probs >= optimal_threshold).astype(int)
    acc_at_opt = float(np.mean(preds_opt == targets))

    tp = int(np.sum((preds_opt == 1) & (targets == 1)))
    fp = int(np.sum((preds_opt == 1) & (targets == 0)))
    fn = int(np.sum((preds_opt == 0) & (targets == 1)))
    tn = int(np.sum((preds_opt == 0) & (targets == 0)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    mse = float(np.mean((probs - targets) ** 2))
    ss_res = float(np.sum((targets - probs) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'accuracy_at_optimal_threshold': acc_at_opt,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'mse': mse,
        'r2': r2,
        'probs': probs,
        'targets': targets,
        'fpr_arr': fpr_arr,
        'tpr_arr': tpr_arr,
    }


def predict(model, X):
    device = get_device()
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        return torch.sigmoid(model(X.to(device))).cpu().numpy().flatten()


def save_artifacts(model, history, train_metrics, val_metrics):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'logreg_lvl6_model.pth'))

    _skip = {'probs', 'targets', 'fpr_arr', 'tpr_arr'}
    metrics_json = {
        'train': {k: v for k, v in train_metrics.items() if k not in _skip},
        'val': {k: v for k, v in val_metrics.items() if k not in _skip},
    }
    with open(os.path.join(OUTPUT_DIR, 'logreg_lvl6_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history['train_losses'], label='Train BCE Loss')
    axes[0].plot(history['val_losses'], label='Val BCE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_accuracies'], label='Train Accuracy')
    axes[1].plot(history['val_accuracies'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('logreg_lvl6: Breast Cancer + RMSprop', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl6_loss.png'), dpi=150)
    plt.close()

    fpr_arr = val_metrics['fpr_arr']
    tpr_arr = val_metrics['tpr_arr']
    auc = val_metrics['auc']
    tau = val_metrics['optimal_threshold']

    probs = val_metrics['probs']
    targets = val_metrics['targets']
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    preds_opt = (probs >= tau).astype(int)
    tp_opt = int(np.sum((preds_opt == 1) & (targets == 1)))
    fp_opt = int(np.sum((preds_opt == 1) & (targets == 0)))
    fpr_opt = fp_opt / n_neg if n_neg > 0 else 0.0
    tpr_opt = tp_opt / n_pos if n_pos > 0 else 0.0

    plt.figure(figsize=(8, 7))
    plt.plot(fpr_arr, tpr_arr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.scatter([fpr_opt], [tpr_opt], color='red', zorder=5, s=100,
                label=f'Optimal threshold t={tau:.3f}\n(FPR={fpr_opt:.3f}, TPR={tpr_opt:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Breast Cancer Val Set')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl6_roc.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    print("=" * 60)
    print("logreg_lvl6: Breast Cancer + RMSprop + Manual AUC")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    print("\n[1] Loading Breast Cancer dataset (stratified 80/20 split)...")
    train_loader, val_loader = make_dataloaders(batch_size=32)
    print(f"  Train samples: {len(train_loader.dataset)}  Val samples: {len(val_loader.dataset)}")

    print("\n[2] Building model: nn.Linear(30, 1) + sigmoid")
    model = build_model().to(device)
    print(f"  {model}")

    print("\n[3] Training 200 epochs with RMSprop(lr=0.001, alpha=0.99)...")
    history = train(model, train_loader, val_loader, epochs=200, lr=0.001)

    print("\n[4] Evaluating on train and val sets...")
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    print(f"\n  TRAIN  Loss={train_metrics['loss']:.4f}  Acc={train_metrics['accuracy']:.4f}  AUC={train_metrics['auc']:.4f}")
    print(f"  VAL    Loss={val_metrics['loss']:.4f}  Acc={val_metrics['accuracy']:.4f}  AUC={val_metrics['auc']:.4f}")
    print(f"  Optimal threshold (Youden's J): {val_metrics['optimal_threshold']:.4f}")
    print(f"  Acc @ optimal threshold: {val_metrics['accuracy_at_optimal_threshold']:.4f}")
    print(f"  Sensitivity={val_metrics['sensitivity']:.4f}  Specificity={val_metrics['specificity']:.4f}")
    print(f"  TP={val_metrics['tp']}  FP={val_metrics['fp']}  FN={val_metrics['fn']}  TN={val_metrics['tn']}")

    print("\n[5] Saving artifacts...")
    save_artifacts(model, history, train_metrics, val_metrics)
    print(f"  Saved to: {OUTPUT_DIR}")

    print("\n[6] Quality assertions...")
    exit_code = 0
    checks = [
        ('val auc > 0.95', val_metrics['auc'] > 0.95),
        ('val accuracy_at_optimal_threshold > 0.92', val_metrics['accuracy_at_optimal_threshold'] > 0.92),
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
