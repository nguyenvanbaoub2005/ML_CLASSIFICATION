# main_pipeline.py
import os
import sys

from src.data.eda import run_eda
from src.data.preprocessing import create_data_generators
from src.models.architectures import create_cnn_model, create_transfer_model, get_model_summary
from src.training.trainer import train_model, plot_training_history
from src.evaluation.evaluator import evaluate_model, plot_confusion_matrix, print_classification_report, compare_models

def main():
    print("🤖 HỆ THỐNG PHÂN LOẠI RÁC THẢI – ML PIPELINE")
    train_dir = input("📁 Đường dẫn thư mục training  : ").strip()
    val_dir   = input("📁 Đường dẫn thư mục validation: ").strip()

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print("❌ Không tìm thấy thư mục dữ liệu!")
        sys.exit(1)

    epochs = int(input("⏱️  Số epochs (Enter = 50): ") or 50)

    # Bước 1-2: EDA
    run_eda(train_dir, val_dir)

    # Bước 3: Preprocessing
    train_gen, val_gen = create_data_generators(train_dir, val_dir)

    # Bước 4-5: Build & Train
    models = {
        'CNN_Custom': create_cnn_model(),
        'MobileNetV2': create_transfer_model('MobileNetV2')
        # Tạm thời tắt ResNet50 để chạy nhanh hơn. Bạn có thể mở comment lại nếu muốn test
        # 'ResNet50': create_transfer_model('ResNet50') 
    }

    eval_results = []
    for name, model in models.items():
        get_model_summary(model)
        history = train_model(model, train_gen, val_gen, epochs=epochs, model_name=name)
        plot_training_history(history, model_name=name)

        # Bước 6: Evaluate
        result = evaluate_model(model, val_gen, model_name=name)
        eval_results.append(result)
        plot_confusion_matrix(model, val_gen, model_name=name)
        print_classification_report(model, val_gen, model_name=name)

    # Bước 7: Compare
    compare_models(eval_results)

if __name__ == '__main__':
    main()