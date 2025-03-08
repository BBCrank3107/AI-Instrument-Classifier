import tensorflow as tf
import numpy as np
import os
import argparse
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--batch_start", type=int, required=True, help="Batch bắt đầu")
parser.add_argument("--batch_end", type=int, required=True, help="Batch kết thúc")
args = parser.parse_args()

PROCESSED_PATH = "E:/AI-Instrument-Classifier/processed_data/"

# Load dữ liệu đã xử lý
def load_processed_data(batch_start, batch_end):
    X, y = [], []
    for i in range(batch_start, batch_end + 1):
        data_file = os.path.join(PROCESSED_PATH, f"batch_{i:02d}_data.npy")
        label_file = os.path.join(PROCESSED_PATH, f"batch_{i:02d}_labels.npy")

        if os.path.exists(data_file) and os.path.exists(label_file):
            print(f"📂 Đang load {data_file} và {label_file}...")
            X.append(np.load(data_file))
            y.append(np.load(label_file))
        else:
            print(f"⚠️ Không tìm thấy batch {i}, bỏ qua...")

    if len(X) == 0:
        raise ValueError("Không có batch nào hợp lệ!")

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

# Load dữ liệu
X_train, y_train = load_processed_data(args.batch_start, args.batch_end)

# Debug: Kiểm tra shape của dữ liệu
print("📊 Shape của X_train:", X_train.shape)  # (số mẫu, chiều cao, chiều rộng, kênh)
print("📊 Shape của y_train:", y_train.shape)

# Kiểm tra kích thước ảnh thực tế
image_height, image_width = X_train.shape[1], X_train.shape[2]

# Mã hóa nhãn thành số nguyên
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Xác định số lớp đầu ra (số loại nhạc cụ)
num_classes = len(np.unique(y_train))

# 🔥 Chuyển nhãn sang one-hot encoding để tránh lỗi softmax
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# Debug: Kiểm tra lại nhãn
print("🔍 Shape mới của y_train:", y_train.shape)

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

# Train model
print("🚀 Bắt đầu huấn luyện...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Lưu model
model_dir = "E:/AI-Instrument-Classifier/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"model_batch_{args.batch_start}_{args.batch_end}.h5")
model.save(model_path)
print(f"✅ Training hoàn tất! Model đã lưu tại {model_path}")