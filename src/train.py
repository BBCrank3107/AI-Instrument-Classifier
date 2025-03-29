import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Định nghĩa đường dẫn đã xử lý
PROCESSED_PATH = "D:/Python/AI-Instrument-Classifier/processed_data/"

# Load dữ liệu đã xử lý
def load_processed_data(dataset_type):
    data_file = os.path.join(PROCESSED_PATH, f"{dataset_type}_data.npy")
    label_file = os.path.join(PROCESSED_PATH, f"{dataset_type}_labels.npy")

    if os.path.exists(data_file) and os.path.exists(label_file):
        print(f"📂 Đang load {data_file} và {label_file}...")
        X = np.load(data_file)
        y = np.load(label_file)
        return X, y
    else:
        raise ValueError(f"Không tìm thấy file {dataset_type} trong {PROCESSED_PATH}!")

# Load dữ liệu train và validation
X_train, y_train = load_processed_data("train")
X_valid, y_valid = load_processed_data("valid")

# Debug: Kiểm tra shape của dữ liệu
print("📊 Shape của X_train:", X_train.shape)  # (số mẫu, chiều cao, chiều rộng, kênh)
print("📊 Shape của y_train:", y_train.shape)
print("📊 Shape của X_valid:", X_valid.shape)
print("📊 Shape của y_valid:", y_valid.shape)

# Kiểm tra kích thước ảnh thực tế
image_height, image_width = X_train.shape[1], X_train.shape[2]

# Mã hóa nhãn thành số nguyên
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_valid = label_encoder.transform(y_valid)

# Xác định số lớp đầu ra (số loại nhạc cụ)
num_classes = len(np.unique(y_train))

# Chuyển nhãn sang one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

# Debug: Kiểm tra lại nhãn
print("🔍 Shape mới của y_train:", y_train.shape)
print("🔍 Shape mới của y_valid:", y_valid.shape)

# Định nghĩa model CNN với kích thước ảnh động
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(image_height, image_width, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model với validation
print("🚀 Bắt đầu huấn luyện...")
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=32)

# Lưu model
model_dir = "D:/Python/AI-Instrument-Classifier/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_with_valid2.h5")
model.save(model_path)
print(f"✅ Training hoàn tất! Model đã lưu tại {model_path}")