# AI Instrument Classifier
- Dự án này xây dựng một mô hình học sâu (deep learning) dựa trên kiến trúc ResNet để phân loại nhạc cụ từ dữ liệu âm thanh (Mel-spectrogram). Dự án sử dụng bộ dữ liệu NSynth và bao gồm các mô hình với các kích thước kernel khác nhau để thử nghiệm hiệu suất.

## Mục tiêu
- Huấn luyện mô hình ResNet để nhận diện 11 loại nhạc cụ khác nhau.
- Thử nghiệm với các kích thước kernel khác nhau (1x1, 3x3, 5x5) trong residual block để so sánh hiệu quả.

## Yêu cầu hệ thống
- Python 3.8 hoặc cao hơn
- GPU (khuyến nghị, nhưng có thể chạy trên CPU)
- Hệ điều hành: Windows

## Cài đặt
### 2. Cài đặt các thư viện
- Cài đặt các thư viện cần thiết từ file requirements.txt:
    pip install -r requirements.txt

### 3. Tải dữ liệu
- Tải bộ dữ liệu NSynth từ đây: https://magenta.tensorflow.org/datasets/nsynth
- Giải nén và đặt vào thư mục datasets/ với cấu trúc:
    datasets/

        nsynth-train/

        nsynth-test/

        nsynth-valid/

## Cấu trúc thư mục
AI-Instrument-Classifier/

├── datasets/                       # Dữ liệu NSynth

│   ├── nsynth-train/

│   ├── nsynth-test/

│   └── nsynth-valid/

├── models/                         # Mô hình đã huấn luyện

│   ├── resnet_mel_instrument_classifier_1x1.h5

│   ├── resnet_mel_instrument_classifier_3x3.h5

│   └── resnet_mel_instrument_classifier_5x5.h5

├── music/                          # File âm thanh để thử nghiệm 

│   └── audio2.wav

├── src/                            # Mã nguồn

│   ├── analyze_model.py            # Phân tích chi tiết mô hình

│   ├── confusion_matrix.py         # Tạo ma trận nhầm lẫn

│   ├── evaluate_model.py           # Tính toán model dựa trên tập test

│   ├── test_classifier.py          # Nhận diện và phân loại nhạc cụ

│   ├── train_large_kernel.ipynb    # Huấn luyện mô hình kernel 5x5

│   ├── train_small_kernel.ipynb    # Huấn luyện mô hình kernel 1x1

│   └── train.ipynb                 # Huấn luyện mô hình gốc (kernel 3x3)

├── README.md                       # File này

└── requirements.txt                # Danh sách thư viện

## Cách sử dụng
### 1. Huấn luyện mô hình
- Mở file .ipynb và chạy từng cell để huấn luyện:
    train.ipynb: Mô hình gốc (kernel 3x3).
    train_large_kernel.ipynb: Mô hình với kernel 5x5.
    train_small_kernel.ipynb: Mô hình với kernel 1x1.

### 2. Tính toán Accuracy của model dựa trên tập test
- Chạy file evaluate_model.py để tính toán Accuracy của model trên tập test:
    python src/evaluate_model.py

### 3. Kiểm tra mô hình
- Chạy file confusion_matrix.py để tạo ma trận nhầm lẫn:
    python src/confusion_matrix.py

### 4. Phân tích mô hình
- Chạy file analyze_model.py để xem chi tiết lớp và trọng số:
    python src/analyze_model.py

### 5. Thử nghiệm
- Chạy file test_classifier.py để thử nghiệm model với nhạc thực tế:
    python src/test_classifier.py