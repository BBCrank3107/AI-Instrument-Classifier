import os
import shutil

# Đường dẫn thư mục batches và audio
batches_dir = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/batches"
audio_dir = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/audio"

# Tạo thư mục audio nếu chưa tồn tại
os.makedirs(audio_dir, exist_ok=True)

# Duyệt qua từng batch
for batch_folder in os.listdir(batches_dir):
    batch_path = os.path.join(batches_dir, batch_folder)
    if os.path.isdir(batch_path):
        # Duyệt qua các file trong batch
        for file in os.listdir(batch_path):
            if file.endswith(".wav"):
                src_path = os.path.join(batch_path, file)
                dst_path = os.path.join(audio_dir, file)
                shutil.move(src_path, dst_path)
                print(f"✅ Đã chuyển: {file}")

print("🎉 Hoàn thành!")