import json
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score

# Đường dẫn đến dataset và mô hình
data_dir = r'D:\Python\AI-Instrument-Classifier\datasets\nsynth-test'
model_path = r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_89.h5'  # Cập nhật tên file

# Định nghĩa ResNet
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_resnet(input_shape=(128, 32, 1), num_classes=11):
    inputs = layers.Input(shape=input_shape)  # Dùng shape thay vì batch_shape
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, 32)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 128)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Hàm tiền xử lý âm thanh với Mel-spectrogram
def preprocess_audio(audio, max_length=16000, n_mels=128, hop_length=512):
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Đọc metadata từ examples.json
def load_test_metadata():
    json_path = os.path.join(data_dir, 'examples.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {json_path}")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata

# Hàm dự đoán nhãn cho một mẫu âm thanh
def predict_instrument(audio_file, model):
    try:
        audio, sr = librosa.load(audio_file, sr=16000)
        mel_spec = preprocess_audio(audio)
        mel_spec = mel_spec[..., np.newaxis]
        mel_spec = np.expand_dims(mel_spec, axis=0)
        pred = model.predict(mel_spec, verbose=0)[0]
        predicted_label = np.argmax(pred)
        return predicted_label
    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_file}: {e}")
        return None

# Đánh giá accuracy trên tập test
def evaluate_model():
    # Tải mô hình
    try:
        model = build_resnet()  # Định nghĩa lại mô hình
        model.load_weights(model_path)  # Tải trọng số
    except Exception as e:
        raise ValueError(f"Không thể tải trọng số từ {model_path}: {e}")
    
    # Tải metadata tập test
    test_metadata = load_test_metadata()
    
    true_labels = []
    predicted_labels = []
    
    # Duyệt qua từng mẫu trong tập test
    for note_str, info in test_metadata.items():
        audio_file = os.path.join(data_dir, 'audio', f'{note_str}.wav')
        if not os.path.isfile(audio_file):
            print(f"Không tìm thấy file âm thanh: {audio_file}")
            continue
        
        true_label = info['instrument_family']
        true_labels.append(true_label)
        
        predicted_label = predict_instrument(audio_file, model)
        if predicted_label is not None:
            predicted_labels.append(predicted_label)
        else:
            true_labels.pop()
    
    # Tính accuracy
    if len(true_labels) == 0:
        raise ValueError("Không có mẫu nào được xử lý thành công để đánh giá.")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Số mẫu test: {len(true_labels)}")
    print(f"Accuracy trên tập test: {accuracy * 100:.2f}%")
    
    return accuracy

# Chạy đánh giá
def main():
    try:
        accuracy = evaluate_model()
    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình: {e}")

if __name__ == "__main__":
    main()