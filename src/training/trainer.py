# src/training/trainer.py
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from config import MODEL_CONFIG, PATHS

def get_callbacks(best_model_path: str = None) -> list:
    best_model_path = best_model_path or PATHS['best_model']
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True),
    ]

def train_model(model, train_generator, val_generator, epochs: int = None, model_name: str = 'model'):
    epochs = epochs or MODEL_CONFIG['epochs']
    best_path = PATHS['best_model'].replace('.keras', f'_{model_name}.keras')
    final_path = PATHS['model_save'].replace('.keras', f'_{model_name}.keras')
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    print(f"\n🚀 Bắt đầu training [{model_name}]\n   Epochs: {epochs}\n   Batch size: {MODEL_CONFIG['batch_size']}")
    
    history = model.fit(
        train_generator, epochs=epochs, validation_data=val_generator,
        callbacks=get_callbacks(best_path), verbose=1
    )

    model.save(final_path)
    print(f"\n✅ Model cuối đã lưu: {final_path}\n   Model tốt nhất: {best_path}\n")
    return history

def plot_training_history(history, model_name: str = 'model', save_path: str = None):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History – {model_name}', fontsize=14, fontweight='bold')

    axes[0].plot(h['accuracy'], label='Train'); axes[0].plot(h['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(h['loss'], label='Train'); axes[1].plot(h['val_loss'], label='Val')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

    save_path = save_path or PATHS['training_plot'].replace('.png', f'_{model_name}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()