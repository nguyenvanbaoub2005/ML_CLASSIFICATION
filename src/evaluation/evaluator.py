import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from config import CLASSES, CLASS_INFO, PATHS

def evaluate_model(model: keras.Model, val_generator, model_name: str = 'model') -> dict:
    print(f"\n🔎 Đánh giá model [{model_name}]...")
    loss, acc = model.evaluate(val_generator, verbose=0)
    print(f"   Val Loss    : {loss:.4f}\n   Val Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return {'model': model_name, 'val_loss': round(loss, 4), 'val_accuracy': round(acc, 4)}

def plot_confusion_matrix(model: keras.Model, val_generator, model_name: str = 'model', save_path: str = None):
    print(f"\n📊 Đang tạo Confusion Matrix [{model_name}]...")
    val_generator.reset()
    y_pred = np.argmax(model.predict(val_generator, verbose=0), axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{CLASS_INFO[c]['icon']} {c}" for c in CLASSES],
                yticklabels=[f"{CLASS_INFO[c]['icon']} {c}" for c in CLASSES], ax=ax)
    
    ax.set_title(f'Confusion Matrix – {model_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Thực tế (True)'); ax.set_xlabel('Dự đoán (Predicted)')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()

    save_path = save_path or PATHS['confusion_matrix'].replace('.png', f'_{model_name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def print_classification_report(model: keras.Model, val_generator, model_name: str = 'model', save_path: str = None):
    val_generator.reset()
    y_pred = np.argmax(model.predict(val_generator, verbose=0), axis=1)
    report = classification_report(val_generator.classes, y_pred, target_names=CLASSES, digits=4)
    print(f"\n📋 Classification Report [{model_name}]\n{'='*65}\n{report}\n{'='*65}")

    save_path = save_path or PATHS['class_report'].replace('.txt', f'_{model_name}.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report – {model_name}\n{'='*65}\n{report}")

def compare_models(results: list, save_path: str = None):
    if not results: return
    names, accuracies, losses = [r['model'] for r in results], [r['val_accuracy'] for r in results], [r['val_loss'] for r in results]
    
    best_idx = accuracies.index(max(accuracies))
    print("\n" + "=" * 55 + "\n🏆 SO SÁNH CÁC MODEL\n" + "=" * 55)
    print(f"{'Model':<20} {'Val Accuracy':>14} {'Val Loss':>12}\n{'-'*55}")
    for i, r in enumerate(results):
        print(f"{r['model']:<20} {r['val_accuracy']:>13.4f}  {r['val_loss']:>11.4f}{' ⭐' if i == best_idx else ''}")
    print("=" * 55)

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars1 = ax1.bar(x, accuracies, color='steelblue', alpha=0.85)
    ax1.set_title('Validation Accuracy'); ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylim([0, 1]); ax1.grid(axis='y', alpha=0.3)
    for b, v in zip(bars1, accuracies): ax1.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.4f}', ha='center')

    bars2 = ax2.bar(x, losses, color='coral', alpha=0.85)
    ax2.set_title('Validation Loss'); ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    for b, v in zip(bars2, losses): ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.4f}', ha='center')

    plt.tight_layout()
    save_path = save_path or PATHS['comparison_plot']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()