import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import argparse

# Đường dẫn đến dữ liệu, thay thế bằng phần biến
BASE_PATH = "D:/Python/AI-Instrument-Classifier/processed_data/"

def analyze_labels(file_path):
    """Phân tích nhãn từ file .npy"""
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return

    labels = np.load(file_path)

    # Hiển thị thông tin tổng quan về dữ liệu
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

    # Nếu dữ liệu là số, kiểm tra giá trị min/max
    if np.issubdtype(labels.dtype, np.number):
        print("📌 Giá trị nhỏ nhất:", labels.min())
        print("📌 Giá trị lớn nhất:", labels.max())

def display_mel_spectrogram(file_path):
    """Hiển thị Mel-Spectrogram đầu tiên trong batch"""
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} không tồn tại!")
        return

    mel_spec = np.load(file_path)

    if len(mel_spec.shape) < 3:
        print("⚠️ Dữ liệu không đúng định dạng Mel-Spectrogram!")
        return

    # Hiển thị Mel-Spectrogram đầu tiên
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0, :, :, 0], cmap="inferno", aspect="auto")
    plt.colorbar(label="Decibels")
    plt.title("Mel-Spectrogram Sample")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency")
    plt.show()

def main(batch_start, batch_end):
    # Duyệt qua các batch từ batch_start đến batch_end
    for i in range(batch_start, batch_end + 1):
        batch_num = f"{i:02d}"
        labels_path = os.path.join(BASE_PATH, f"batch_{batch_num}_labels.npy")
        data_path = os.path.join(BASE_PATH, f"batch_{batch_num}_data.npy")

        print(f"\n🔍 **Phân tích Labels - batch_{batch_num}**")
        analyze_labels(labels_path)

        print(f"\n🎵 **Hiển thị Mel-Spectrogram - batch_{batch_num}**")
        display_mel_spectrogram(data_path)

if __name__ == "__main__":
    # Thiết lập parser để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Analyze and display Mel-Spectrograms from batches.")
    parser.add_argument('--batch_start', type=int, required=True, help="The starting batch number")
    parser.add_argument('--batch_end', type=int, required=True, help="The ending batch number")

    # Parse các tham số
    args = parser.parse_args()

    # Chạy chính hàm main với các tham số từ dòng lệnh
    main(args.batch_start, args.batch_end)