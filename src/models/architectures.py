from tensorflow import keras
from tensorflow.keras import layers
from config import MODEL_CONFIG

SUPPORTED_BACKBONES = ('MobileNetV2', 'VGG16', 'ResNet50')

def create_cnn_model(input_shape=None, num_classes=None) -> keras.Model:
    input_shape = input_shape or MODEL_CONFIG['input_shape']
    num_classes = num_classes or MODEL_CONFIG['num_classes']

    model = keras.Sequential(name='WasteCNN', layers=[
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    _compile(model)
    return model

def create_transfer_model(backbone: str = 'MobileNetV2', input_shape=None, num_classes=None) -> keras.Model:
    input_shape = input_shape or MODEL_CONFIG['input_shape']
    num_classes = num_classes or MODEL_CONFIG['num_classes']

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone '{backbone}' không được hỗ trợ.")

    backbone_map = {
        'MobileNetV2': keras.applications.MobileNetV2,
        'VGG16':       keras.applications.VGG16,
        'ResNet50':    keras.applications.ResNet50,
    }

    base = backbone_map[backbone](input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False

    model = keras.Sequential(name=f'WasteTransfer_{backbone}', layers=[
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    _compile(model)
    return model

def _compile(model: keras.Model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

def get_model_summary(model: keras.Model) -> int:
    print("\n" + "=" * 70)
    print(f"📐 MODEL: {model.name}")
    print("=" * 70)
    model.summary()
    total = model.count_params()
    print(f"\nTổng parameters: {total:,}")
    print("=" * 70 + "\n")
    return total