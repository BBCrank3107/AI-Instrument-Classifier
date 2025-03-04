1. Xóa các file example.txt trong các thư mục
    - datasets/nsynth-test
    - datasets/nsynth-train
    - datasets/nsynth-valid
    - models
    - output
    - processed_data
    - spectrograms
    - spleeter_output
2. Tải thư viện theo cú pháp trong requirements.txt
3. Tải datasets sau đó di chuyển vào các thư mục tương ứng trong folder datasets
    Lưu ý: tạo folder batches trong nsynth-train
4. Run code
    - Bước 1: Chạy file preprocess.py bằng lệnh:
        python src/preprocess.py --batch_start 6 --batch_end 11
        Lưu ý: Số 6 và 11 là số của batch được đánh dấu. Ví dụ Bình làm từ batch_00 đến batch_05 thì lệnh sẽ là:
            python src/preprocess.py --batch_start 0 --batch_end 5
    - Bước 2: Sau khi chạy preprocess.py xong thì kiểm tra xem folder processed_data đã có dữ liệu chưa. Nếu processed_data chưa có dữ liệu thì kiểm tra lỗi và chạy lại Bước 11, nếu có rồi thì chạy file train.py bằng lệnh:
        python src/train.py --batch_start 6 --batch_end 11
        Lưu ý: Số 6 và 11 là số của batch được đánh dấu. Ví dụ Bình làm từ batch_00 đến batch_05 thì lệnh sẽ là:
            python src/train.py --batch_start 0 --batch_end 5
5. Còn lại Bình sẽ cập nhật sau
