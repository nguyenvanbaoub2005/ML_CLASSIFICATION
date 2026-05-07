import os
import numpy as np
from tensorflow import keras
from config import CLASSES, CLASS_INFO, CONFIDENCE_THRESHOLD, COLORS, PATHS
from src.data.preprocessing import preprocess_single_image

class WasteClassifier:
    def __init__(self, model_path: str = None):
        self.classes = CLASSES
        self.class_info = CLASS_INFO
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        model_path = model_path or PATHS['best_model']
        if os.path.exists(model_path):
            print(f"📂 Đang load model: {model_path}")
            self.model = keras.models.load_model(model_path)
            print("✅ Load model thành công!\n")
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy model tại: {model_path}")

    def predict(self, image_input, return_all: bool = True) -> dict:
        arr = preprocess_single_image(image_input)
        preds = self.model.predict(arr, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx]) * 100
        pred_class = self.classes[class_idx]

        result = {
            'class': pred_class,
            'class_name_vi': self.class_info[pred_class]['name_vi'],
            'confidence': confidence,
            'is_confident': confidence >= self.confidence_threshold,
        }
        if return_all:
            result['all_predictions'] = {
                self.classes[i]: float(preds[i]) * 100
                for i in range(len(self.classes))
            }
        return result

    def predict_batch(self, image_paths: list) -> list:
        results = []
        for path in image_paths:
            try:
                results.append({'image': path, 'result': self.predict(path)})
            except Exception as e:
                print(f"❌ Lỗi {path}: {e}")
        return results