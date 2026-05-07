# gui_main.py
import tkinter as tk
from tkinter import messagebox
from src.gui.camera_panel import CameraPanel
from src.gui.result_panel import ResultPanel
from src.inference.classifier import WasteClassifier
from src.gui.widgets import COLORS
from config import PATHS

class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Loại Rác Thải Thông Minh")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLORS['bg'])

        # Load AI Model
        try:
            self.classifier = WasteClassifier(PATHS['best_model'])
        except Exception as e:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy file model. Hãy chạy main_pipeline.py trước!")
            self.classifier = None

        # Layout chính
        self.main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        self.main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # Cột phải: Hiển thị Kết quả
        self.result_panel = ResultPanel(self.main_frame, width=400)
        self.result_panel.pack(side='right', fill='y', padx=(10, 0))

        # Cột trái: Camera
        self.camera_panel = CameraPanel(
            self.main_frame, 
            on_scan_result=self.process_image
        )
        self.camera_panel.pack(side='left', fill='both', expand=True)

    def process_image(self, image_path, cropped_frame):
        """Hàm nhận ảnh từ Camera và đưa cho AI phân tích"""
        if self.classifier:
            try:
                # Phân loại bằng AI
                result = self.classifier.predict(image_path, return_all=True)
                # Đẩy kết quả qua màn hình bên phải
                self.result_panel.show_result(result, image_path)
            except Exception as e:
                print(f"Lỗi khi dự đoán: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.mainloop()