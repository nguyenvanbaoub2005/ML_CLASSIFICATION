# config.py
"""
File cấu hình trung tâm cho toàn bộ hệ thống phân loại rác thải.
Tất cả các module khác đều import từ đây.
"""

# ─── 1. CÁC LOẠI RÁC ────────────────────────────────────────────────────────
CLASSES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']

CLASS_INFO = {
    'plastic':   {'name_vi': 'Nhựa',              'icon': '🥤', 'color': '\033[94m', 'disposal': 'Tái chế - Rửa sạch và bỏ vào thùng nhựa',                          'examples': ['Chai nước', 'Túi nilon', 'Hộp nhựa'],        'recycling_value': 'Cao'},
    'paper':     {'name_vi': 'Giấy',              'icon': '📄', 'color': '\033[93m', 'disposal': 'Tái chế - Bỏ vào thùng giấy',                                        'examples': ['Báo cũ', 'Hộp giấy', 'Sách vở'],            'recycling_value': 'Trung bình'},
    'glass':     {'name_vi': 'Thủy tinh',         'icon': '🍾', 'color': '\033[92m', 'disposal': 'Tái chế - Cẩn thận khi xử lý',                                       'examples': ['Chai thủy tinh', 'Lọ', 'Cốc'],              'recycling_value': 'Cao'},
    'metal':     {'name_vi': 'Kim loại',          'icon': '🥫', 'color': '\033[90m', 'disposal': 'Tái chế - Bỏ vào thùng kim loại',                                     'examples': ['Lon nước ngọt', 'Hộp thiếc', 'Vỏ lon'],     'recycling_value': 'Rất cao'},
    'cardboard': {'name_vi': 'Bìa cứng',          'icon': '📦', 'color': '\033[33m', 'disposal': 'Tái chế - Gấp gọn trước khi bỏ',                                     'examples': ['Hộp carton', 'Thùng giấy', 'Bìa đóng gói'], 'recycling_value': 'Trung bình'},
    'trash':     {'name_vi': 'Rác thông thường',  'icon': '🗑️', 'color': '\033[91m', 'disposal': 'Rác thông thường - Bỏ vào thùng rác',                                 'examples': ['Rác không tái chế', 'Rác bẩn'],             'recycling_value': 'Không'},
    'organic':   {'name_vi': 'Rác hữu cơ',        'icon': '🍃', 'color': '\033[32m', 'disposal': 'Phân hủy sinh học - Bỏ vào thùng rác hữu cơ hoặc ủ compost',         'examples': ['Thức ăn thừa', 'Vỏ trái cây', 'Lá cây'],    'recycling_value': 'Cao (Compost)'},
}

# ─── 2. CẤU HÌNH MODEL ──────────────────────────────────────────────────────
MODEL_CONFIG = {
    'input_shape':    (224, 224, 3),
    'num_classes':    len(CLASSES),
    'batch_size':     32,
    'epochs':         50,
    'learning_rate':  0.001,
}

# ─── 3. CẤU HÌNH DATA AUGMENTATION ─────────────────────────────────────────
AUGMENTATION_CONFIG = {
    'rotation_range':     20,
    'width_shift_range':  0.2,
    'height_shift_range': 0.2,
    'horizontal_flip':    True,
    'zoom_range':         0.2,
    'shear_range':        0.2,
    'fill_mode':          'nearest',
}

# ─── 4. ĐƯỜNG DẪN FILE ──────────────────────────────────────────────────────
PATHS = {
    # Models (Đã đổi sang .keras cho chuẩn mới)
    'model_save':        'outputs/models/waste_classifier_final.keras',
    'best_model':        'outputs/models/waste_classifier_best.keras',
    'temp_image':        'outputs/temp_capture.jpg',
    # Plots
    'training_plot':     'outputs/plots/training_history.png',
    'eda_plot':          'outputs/plots/eda_analysis.png',
    'confusion_matrix':  'outputs/plots/confusion_matrix.png',
    'class_report':      'outputs/reports/classification_report.txt',
    'comparison_plot':   'outputs/plots/model_comparison.png',
}

# ─── 5. NGƯỠNG & MÀU SẮC ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 70.0

COLORS = {
    'reset':  '\033[0m',
    'blue':   '\033[94m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'gray':   '\033[90m',
    'orange': '\033[33m',
}