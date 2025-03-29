import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import os

# Đường dẫn
BASE_PATH = "D:/Python/AI-Instrument-Classifier/"
PRED_PATH = os.path.join(BASE_PATH, "results/y_pred_classes_test_split.npy")
TEST_LABELS_PATH = os.path.join(BASE_PATH, "processed_data/test_labels.npy")
RESULTS_DIR = os.path.join(BASE_PATH, "results/")

# Đảm bảo thư mục results tồn tại
os.makedirs(RESULTS_DIR, exist_ok=True)

# Danh sách nhãn (11 lớp)
LABELS = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth', 'vocal']

def load_data():
    """Load dự đoán và nhãn thật"""
    if not os.path.exists(PRED_PATH) or not os.path.exists(TEST_LABELS_PATH):
        raise FileNotFoundError("Không tìm thấy file dự đoán hoặc nhãn thật!")

    y_pred = np.load(PRED_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    # Mã hóa nhãn thật thành số nguyên
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(LABELS)
    y_test_encoded = label_encoder.transform(y_test)

    return y_pred, y_test_encoded, label_encoder

def analyze_predictions(y_pred, y_test, label_encoder):
    """Phân tích kết quả dự đoán"""
    # Kiểm tra shape
    print("📊 Shape của y_pred:", y_pred.shape)
    print("📊 Shape của y_test:", y_test.shape)

    # Tính accuracy
    correct = y_pred == y_test
    accuracy = np.mean(correct)
    print(f"📈 Accuracy: {accuracy:.4f}")

    # Liệt kê mẫu sai
    wrong_indices = np.where(correct == False)[0]
    print(f"📌 Số mẫu dự đoán sai: {len(wrong_indices)}")
    print("\n📋 Một số mẫu dự đoán sai (index, nhãn thật, nhãn dự đoán):")
    with open(os.path.join(RESULTS_DIR, "wrong_predictions.txt"), "w", encoding="utf-8") as f:
        f.write("Index,Nhãn thật,Nhãn dự đoán\n")
        for idx in wrong_indices[:10]:  # Hiển thị 10 mẫu đầu tiên
            true_label = label_encoder.classes_[y_test[idx]]
            pred_label = label_encoder.classes_[y_pred[idx]]
            print(f"Index {idx}: {true_label} -> {pred_label}")
            f.write(f"{idx},{true_label},{pred_label}\n")
    print(f"📝 Danh sách đầy đủ mẫu sai đã lưu tại: {RESULTS_DIR}wrong_predictions.txt")

    # Tính lỗi theo nhãn
    errors_per_label = {}
    for label_idx, label in enumerate(LABELS):
        label_mask = (y_test == label_idx)
        label_errors = np.sum(label_mask & ~correct)
        errors_per_label[label] = label_errors
    print("\n📌 Số lỗi theo nhãn:")
    for label, count in errors_per_label.items():
        print(f"  - {label}: {count} lỗi")

    return wrong_indices, errors_per_label

def plot_confusion_matrix(y_test, y_pred, label_encoder):
    """Vẽ và lưu ma trận nhầm lẫn"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"📉 Ma trận nhầm lẫn đã lưu tại: {RESULTS_DIR}confusion_matrix.png")

def plot_error_distribution(errors_per_label):
    """Vẽ biểu đồ phân bố lỗi theo nhãn"""
    labels = list(errors_per_label.keys())
    errors = list(errors_per_label.values())

    plt.figure(figsize=(12, 6))
    plt.bar(labels, errors, color="salmon")
    plt.title("Error Distribution by Label")
    plt.xlabel("Label")
    plt.ylabel("Number of Errors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
    plt.close()
    print(f"📊 Biểu đồ phân bố lỗi đã lưu tại: {RESULTS_DIR}error_distribution.png")

def main():
    # Load dữ liệu
    print("🚀 Đang load dữ liệu...")
    y_pred, y_test, label_encoder = load_data()

    # Phân tích dự đoán
    print("\n🔍 Phân tích kết quả dự đoán...")
    wrong_indices, errors_per_label = analyze_predictions(y_pred, y_test, label_encoder)

    # Vẽ biểu đồ
    print("\n🎨 Vẽ biểu đồ...")
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_error_distribution(errors_per_label)

if __name__ == "__main__":
    main()