import os
import shutil
import random
from collections import defaultdict

# Thư mục nguồn và đích
src_folder = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/audio"
dst_folder = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/train_balanced"
os.makedirs(dst_folder, exist_ok=True)

# Số lượng tệp tối đa cho mỗi nhãn
files_per_label = 5501

# Bước 1: Nhóm file theo loại nhạc cụ
instrument_groups = defaultdict(list)
for file in os.listdir(src_folder):
    if file.endswith(".wav"):
        instrument = file.split("_")[0]
        instrument_groups[instrument].append(file)

# Bước 2: Lấy ngẫu nhiên 5,501 tệp từ mỗi nhãn
for instrument, files in instrument_groups.items():
    random.shuffle(files)  # Xáo trộn để lấy ngẫu nhiên
    selected_files = files[:files_per_label]  # Lấy 5,501 tệp
    
    # Di chuyển file sang thư mục đích
    for file in selected_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        shutil.copy(src_path, dst_path)

# Bước 3: Kiểm tra kết quả
print(f"Đã tạo dataset train cân bằng tại: {dst_folder}")
print(f"Mỗi nhãn có {files_per_label} tệp, tổng cộng {files_per_label * len(instrument_groups)} tệp")