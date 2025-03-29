import librosa
import numpy as np
import os

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def get_label_from_filename(filename):
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0]
    return "unknown"

def process_data(src_folder, output_path, name):
    os.makedirs(output_path, exist_ok=True)
    all_files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
    print(f"📊 Tổng số tệp cần xử lý cho {name}: {len(all_files)}")

    data = []
    labels = []

    print(f"📂 Đang xử lý {name}...")
    for file in all_files:
        file_path = os.path.join(src_folder, file)
        mel_spec = extract_mel_spectrogram(file_path)
        mel_spec = np.expand_dims(mel_spec, axis=-1)
        label = get_label_from_filename(file)

        data.append(mel_spec)
        labels.append(label)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)

    np.save(os.path.join(output_path, f"{name}_data.npy"), data)
    np.save(os.path.join(output_path, f"{name}_labels.npy"), labels)
    print(f"✅ Đã lưu {name} tại {output_path} ({len(data)} samples)")

if __name__ == "__main__":
    BASE_PATH = "D:/Python/AI-Instrument-Classifier/"
    PROCESSED_PATH = BASE_PATH + "processed_data/"

    process_data(BASE_PATH + "datasets/nsynth-train/train_split/", PROCESSED_PATH, "train")
    process_data(BASE_PATH + "datasets/nsynth-train/valid_split/", PROCESSED_PATH, "valid")
    process_data(BASE_PATH + "datasets/nsynth-train/test_split/", PROCESSED_PATH, "test")