import librosa
import numpy as np
import os

# Đường dẫn dữ liệu
DATASET_PATH = "../datasets/nsynth-train/batches/"
PROCESSED_PATH = "../processed_data/"

def extract_mel_spectrogram(file_path):
    """Trích xuất Mel-Spectrogram từ file âm thanh"""
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def get_label_from_filename(filename):
    """Trích xuất nhãn từ tên file theo định dạng `name_type_000-000-000.wav`"""
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0]
    return "unknown"

def process_batch(batch_start, batch_end):
    """Xử lý dữ liệu từ batch_00 đến batch_end và lưu thành numpy arrays"""
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    for batch_id in range(batch_start, batch_end + 1):
        batch_folder = os.path.join(DATASET_PATH, f"batch_{str(batch_id).zfill(2)}")
        batch_data = []
        batch_labels = []

        if not os.path.exists(batch_folder):
            print(f"⚠️ Batch {batch_folder} không tồn tại, bỏ qua...")
            continue
        
        print(f"📂 Đang xử lý {batch_folder}...")

        for file in os.listdir(batch_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(batch_folder, file)

                # Trích xuất Mel-Spectrogram
                mel_spec = extract_mel_spectrogram(file_path)

                # Chuẩn hóa kích thước (128x128) và thêm kênh (1)
                mel_spec = np.expand_dims(mel_spec, axis=-1)

                # Lấy nhãn từ tên file
                label = get_label_from_filename(file)

                batch_data.append(mel_spec)
                batch_labels.append(label)

        # Chuyển sang numpy array
        batch_data = np.array(batch_data, dtype=np.float32)
        batch_labels = np.array(batch_labels)

        # Lưu file
        np.save(os.path.join(PROCESSED_PATH, f"batch_{str(batch_id).zfill(2)}_data.npy"), batch_data)
        np.save(os.path.join(PROCESSED_PATH, f"batch_{str(batch_id).zfill(2)}_labels.npy"), batch_labels)

        print(f"✅ Lưu batch_{str(batch_id).zfill(2)} xong! ({len(batch_data)} samples)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_start", type=int, required=True, help="Batch bắt đầu")
    parser.add_argument("--batch_end", type=int, required=True, help="Batch kết thúc")

    args = parser.parse_args()
    process_batch(args.batch_start, args.batch_end)