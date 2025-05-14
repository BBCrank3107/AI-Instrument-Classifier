import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import json
import librosa

# Định nghĩa đường dẫn và hàm hỗ trợ
data_dir = {
    'train': r'D:\Python\AI-Instrument-Classifier\datasets\nsynth-train',
    'test': r'D:\Python\AI-Instrument-Classifier\datasets\nsynth-test',
    'valid': r'D:\Python\AI-Instrument-Classifier\datasets\nsynth-valid'
}

def load_metadata(split='train'):
    json_path = os.path.join(data_dir[split], 'examples.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {json_path}")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def preprocess_audio(audio, max_length=16000, n_mels=128, hop_length=512):
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def data_generator(metadata, split='train'):
    for note_str, info in metadata.items():
        audio_path = os.path.join(data_dir[split], 'audio', f'{note_str}.wav')
        audio, sr = librosa.load(audio_path, sr=16000)
        mel_spec = preprocess_audio(audio)
        label = info['instrument_family']
        yield mel_spec[..., np.newaxis], label

# Chuẩn bị dataset
train_metadata = load_metadata('train')
valid_metadata = load_metadata('valid')
test_metadata = load_metadata('test')

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_metadata, 'train'),
    output_types=(tf.float32, tf.int32),
    output_shapes=([128, 32, 1], [])
).batch(64).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(valid_metadata, 'valid'),
    output_types=(tf.float32, tf.int32),
    output_shapes=([128, 32, 1], [])
).batch(64).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_metadata, 'test'),
    output_types=(tf.float32, tf.int32),
    output_shapes=([128, 32, 1], [])
).batch(64).prefetch(tf.data.AUTOTUNE)

# Định nghĩa hàm residual_block
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

# Định nghĩa hàm build_resnet với kernel_size áp dụng cho cả tầng đầu tiên
def build_resnet(input_shape=(128, 32, 1), num_classes=11, kernel_size=3, filters=[32, 64, 128]):
    print(f"Building model with kernel_size={kernel_size}, filters={filters}")
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], (kernel_size, kernel_size), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(inputs)  # Dùng kernel_size linh hoạt
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, filters[0], kernel_size=kernel_size)
    x = residual_block(x, filters[1], kernel_size=kernel_size)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters[2], kernel_size=kernel_size)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Danh sách mô hình cần đánh giá
models_info = [
    {
        'name': 'resnet_mel_instrument_classifier_1x1.h5',
        'kernel_size': 1,
        'filters': [16, 32, 64],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_1x1.h5'
    },
    {
        'name': 'resnet_mel_instrument_classifier_3x3.h5',
        'kernel_size': 3,
        'filters': [32, 64, 128],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_3x3.h5'
    },
    {
        'name': 'resnet_mel_instrument_classifier_5x5.h5',
        'kernel_size': 5,
        'filters': [64, 128, 256],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_5x5.h5'
    }
]

# Đánh giá từng mô hình và lưu kết quả
results = []
for model_info in models_info:
    print(f"\nĐánh giá mô hình: {model_info['name']} (Kernel Size: {model_info['kernel_size']}, Filters: {model_info['filters']})")
    
    # Khởi tạo mô hình với kernel_size và filters tương ứng
    model = build_resnet(
        kernel_size=model_info['kernel_size'],
        filters=model_info['filters']
    )
    
    # Load trọng số
    model.load_weights(model_info['path'])
    
    # Compile mô hình
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Đánh giá trên các tập dữ liệu
    train_loss, train_accuracy = model.evaluate(train_dataset, verbose=1)
    valid_loss, valid_accuracy = model.evaluate(valid_dataset, verbose=1)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    
    # Lưu kết quả
    results.append({
        'Model': model_info['name'],
        'Train Loss': train_loss,
        'Train Accuracy': train_accuracy,
        'Validation Loss': valid_loss,
        'Validation Accuracy': valid_accuracy,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy
    })

# Trình bày kết quả dưới dạng bảng
results_df = pd.DataFrame(results)
print("\nSo sánh kết quả của 3 mô hình:")
print(results_df.to_string(index=False))

# (Tùy chọn) Vẽ biểu đồ so sánh
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Biểu đồ Accuracy
plt.subplot(1, 2, 1)
for i, model in enumerate(results):
    plt.bar([f"{model['Model'][:10]}\nTrain", f"{model['Model'][:10]}\nValid", f"{model['Model'][:10]}\nTest"],
            [model['Train Accuracy'], model['Validation Accuracy'], model['Test Accuracy']],
            label=model['Model'])
plt.title('So sánh Accuracy')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()

# Biểu đồ Loss
plt.subplot(1, 2, 2)
for i, model in enumerate(results):
    plt.bar([f"{model['Model'][:10]}\nTrain", f"{model['Model'][:10]}\nValid", f"{model['Model'][:10]}\nTest"],
            [model['Train Loss'], model['Validation Loss'], model['Test Loss']],
            label=model['Model'])
plt.title('So sánh Loss')
plt.ylabel('Loss')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()