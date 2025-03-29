import os
import shutil
import random
from collections import defaultdict

# Đường dẫn
SRC_FOLDER = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/train_balanced/"
TRAIN_FOLDER = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/train_split/"
VALID_FOLDER = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/valid_split/"
TEST_FOLDER = "D:/Python/AI-Instrument-Classifier/datasets/nsynth-train/test_split/"

# Số tệp mỗi nhãn cho từng tập
TRAIN_SIZE_PER_LABEL = 3850  # ~70%
VALID_SIZE_PER_LABEL = 825   # ~15%
TEST_SIZE_PER_LABEL = 825    # ~15%

# Tạo thư mục mới
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Nhóm file theo nhãn
instrument_groups = defaultdict(list)
for file in os.listdir(SRC_FOLDER):
    if file.endswith(".wav"):
        instrument = file.split("_")[0]
        instrument_groups[instrument].append(file)

# Chia train/valid/test cho mỗi nhãn
for instrument, files in instrument_groups.items():
    random.shuffle(files)  # Xáo trộn để chọn ngẫu nhiên
    train_files = files[:TRAIN_SIZE_PER_LABEL]
    valid_files = files[TRAIN_SIZE_PER_LABEL:TRAIN_SIZE_PER_LABEL + VALID_SIZE_PER_LABEL]
    test_files = files[TRAIN_SIZE_PER_LABEL + VALID_SIZE_PER_LABEL:TRAIN_SIZE_PER_LABEL + VALID_SIZE_PER_LABEL + TEST_SIZE_PER_LABEL]

    # Di chuyển file vào thư mục train
    for file in train_files:
        src_path = os.path.join(SRC_FOLDER, file)
        dst_path = os.path.join(TRAIN_FOLDER, file)
        shutil.copy(src_path, dst_path)

    # Di chuyển file vào thư mục valid
    for file in valid_files:
        src_path = os.path.join(SRC_FOLDER, file)
        dst_path = os.path.join(VALID_FOLDER, file)
        shutil.copy(src_path, dst_path)

    # Di chuyển file vào thư mục test
    for file in test_files:
        src_path = os.path.join(SRC_FOLDER, file)
        dst_path = os.path.join(TEST_FOLDER, file)
        shutil.copy(src_path, dst_path)

print(f"✅ Train: {TRAIN_SIZE_PER_LABEL * len(instrument_groups)} tệp ({TRAIN_SIZE_PER_LABEL} mỗi nhãn).")
print(f"✅ Validation: {VALID_SIZE_PER_LABEL * len(instrument_groups)} tệp ({VALID_SIZE_PER_LABEL} mỗi nhãn).")
print(f"✅ Test: {TEST_SIZE_PER_LABEL * len(instrument_groups)} tệp ({TEST_SIZE_PER_LABEL} mỗi nhãn).")