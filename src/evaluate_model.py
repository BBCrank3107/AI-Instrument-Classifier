import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Đường dẫn
MODEL_PATH = "D:/Python/AI-Instrument-Classifier/models/model_with_valid.h5"
TEST_DATA_PATH = "D:/Python/AI-Instrument-Classifier/processed_data/test_data.npy"
TEST_LABELS_PATH = "D:/Python/AI-Instrument-Classifier/processed_data/test_labels.npy"

# Load dữ liệu test
X_test, y_test = np.load(TEST_DATA_PATH), np.load(TEST_LABELS_PATH)

print("📊 Shape của X_test:", X_test.shape)
print("📊 Shape của y_test:", y_test.shape)

# Nhãn
train_labels = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth', 'vocal']

# Mã hóa nhãn
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(train_labels)
y_test_encoded = label_encoder.transform(y_test)
y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=11)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model đã được load từ:", MODEL_PATH)

# Dự đoán
print("🚀 Đang dự đoán trên tập test...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_onehot, axis=1)

# Đánh giá
accuracy = np.mean(y_pred_classes == y_test_classes)
print(f"📈 Accuracy trên tập test (11 lớp): {accuracy:.4f}")

print("\n📋 Báo cáo phân loại (11 lớp):")
print(classification_report(y_test_classes, y_pred_classes, target_names=train_labels))

print("\n📉 Confusion Matrix (11 lớp):")
print(confusion_matrix(y_test_classes, y_pred_classes))

# Lưu kết quả
results_dir = "D:/Python/AI-Instrument-Classifier/results"
os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, "y_pred_classes_test_split.npy"), y_pred_classes)
print("✅ Kết quả dự đoán đã được lưu.")