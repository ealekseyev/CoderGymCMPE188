import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']


def get_task_metadata():
    return {
        'task_id': 'logreg_lvl5_iris_multiclass',
        'algorithm': 'Multinomial Logistic Regression (Iris + StepLR)',
        'dataset': 'iris',
        'n_features': 4,
        'n_classes': 3,
        'optimizer': 'Adam + StepLR(step_size=50, gamma=0.5)',
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

    data = load_iris()
    X, y = data.data, data.target

    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)

    X_train_t = torch.FloatTensor(X_train_np)
    y_train_t = torch.LongTensor(y_train_np)
    X_val_t = torch.FloatTensor(X_val_np)
    y_val_t = torch.LongTensor(y_val_np)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model():
    model = nn.Linear(4, 3)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def train(model, train_loader, val_loader, epochs=300, lr=0.05):
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_losses = []
    val_losses = []
    val_accuracies = []
    lr_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(correct / total)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  train_loss={train_losses[-1]:.4f}  val_acc={val_accuracies[-1]:.4f}  lr={lr_history[-1]:.6f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'lr_history': lr_history,
    }


def compute_macro_f1(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_scores = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1_scores.append(float(2 * tp / denom) if denom > 0 else 0.0)

    return float(np.mean(f1_scores)), f1_scores, cm.tolist()


def evaluate(model, data_loader):
    device = get_device()
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())

    all_preds = np.array(all_preds, dtype=int)
    all_targets = np.array(all_targets, dtype=int)

    accuracy = float(np.mean(all_preds == all_targets))
    macro_f1, f1_per_class, cm = compute_macro_f1(all_targets, all_preds, n_classes=3)

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_setosa': f1_per_class[0],
        'f1_versicolor': f1_per_class[1],
        'f1_virginica': f1_per_class[2],
        'confusion_matrix': cm,
        'mse': 0.0,
        'r2': 0.0,
    }


def predict(model, X):
    device = get_device()
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        logits = model(X.to(device))
        return logits.argmax(dim=1).cpu().numpy()


def save_artifacts(model, history, train_metrics, val_metrics):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'logreg_lvl5_model.pth'))

    metrics_json = {
        'train': train_metrics,
        'val': val_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, 'logreg_lvl5_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_accuracies'], color='green', label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['lr_history'], color='purple', label='Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_title('StepLR Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('logreg_lvl5: Iris Multiclass + Adam + StepLR', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl5_curves.png'), dpi=150)
    plt.close()

    cm = np.array(val_metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - Val Set\nAccuracy={val_metrics["accuracy"]:.4f}  Macro-F1={val_metrics["macro_f1"]:.4f}')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl5_confusion.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    print("=" * 60)
    print("logreg_lvl5: Iris Multiclass + Adam + StepLR")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    print("\n[1] Loading Iris dataset (stratified 80/20 split)...")
    train_loader, val_loader = make_dataloaders(batch_size=32)
    print(f"  Train samples: {len(train_loader.dataset)}  Val samples: {len(val_loader.dataset)}")

    print("\n[2] Building model: nn.Linear(4, 3) + Xavier init")
    model = build_model().to(device)
    print(f"  {model}")

    print("\n[3] Training 300 epochs (Adam lr=0.05, StepLR step=50 gamma=0.5)...")
    history = train(model, train_loader, val_loader, epochs=300, lr=0.05)

    print("\n[4] Evaluating on train and val sets...")
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    print(f"\n  TRAIN  Loss={train_metrics['loss']:.4f}  Acc={train_metrics['accuracy']:.4f}  macro_F1={train_metrics['macro_f1']:.4f}")
    print(f"  VAL    Loss={val_metrics['loss']:.4f}  Acc={val_metrics['accuracy']:.4f}  macro_F1={val_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1 (val): Setosa={val_metrics['f1_setosa']:.4f}  Versicolor={val_metrics['f1_versicolor']:.4f}  Virginica={val_metrics['f1_virginica']:.4f}")
    print(f"  Confusion matrix (val):")
    for row in val_metrics['confusion_matrix']:
        print(f"    {row}")

    print("\n[5] Saving artifacts...")
    save_artifacts(model, history, train_metrics, val_metrics)
    print(f"  Saved to: {OUTPUT_DIR}")

    print("\n[6] Quality assertions...")
    exit_code = 0
    checks = [
        ('val accuracy > 0.90', val_metrics['accuracy'] > 0.90),
        ('val macro_f1 > 0.85', val_metrics['macro_f1'] > 0.85),
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
