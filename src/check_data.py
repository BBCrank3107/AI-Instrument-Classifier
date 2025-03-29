import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# Đường dẫn đến dữ liệu
BASE_PATH = "D:/Python/AI-Instrument-Classifier/processed_data/"

def analyze_labels(file_path, dataset_name):
    """Phân tích nhãn từ file .npy"""
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return

    labels = np.load(file_path)

    # Hiển thị thông tin tổng quan về dữ liệu
    print(f"\n🔍 **Phân tích Labels - {dataset_name}**")
    print("📌 Shape của labels:", labels.shape)
    print("📌 Kiểu dữ liệu:", labels.dtype)

    # Kiểm tra các lớp có trong dataset
    unique_labels = np.unique(labels)
    print("📌 Các lớp có trong dataset:", unique_labels)
    print("📌 Số lớp:", len(unique_labels))

    # Hiển thị một số nhãn đầu tiên
    print("📌 Một số nhãn đầu tiên:", labels[:10])

    # Đếm số lần xuất hiện của từng nhãn
    label_counts = Counter(labels)
    print("📌 Tần suất xuất hiện của mỗi nhãn:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count} mẫu")

def display_mel_spectrogram(file_path, dataset_name):
    """Hiển thị Mel-Spectrogram đầu tiên trong file"""
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return

    mel_spec = np.load(file_path)

    if len(mel_spec.shape) < 3:
        print(f"⚠️ Dữ liệu không đúng định dạng Mel-Spectrogram trong {dataset_name}!")
        return

    # Hiển thị Mel-Spectrogram đầu tiên
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0, :, :, 0], cmap="inferno", aspect="auto")
    plt.colorbar(label="Decibels")
    plt.title(f"Mel-Spectrogram Sample - {dataset_name}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency")
    plt.show()

def main():
    # Danh sách các tập dữ liệu cần kiểm tra
    datasets = ["train", "valid", "test"]

    # Duyệt qua từng tập dữ liệu
    for dataset in datasets:
        labels_path = os.path.join(BASE_PATH, f"{dataset}_labels.npy")
        data_path = os.path.join(BASE_PATH, f"{dataset}_data.npy")

        analyze_labels(labels_path, dataset)
        display_mel_spectrogram(data_path, dataset)

if __name__ == "__main__":
    main()