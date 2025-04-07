import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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
    inputs = layers.Input(shape=input_shape)
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

# Hàm dự đoán nhạc cụ
def predict_instruments(audio_file, model_path, sr=16000, segment_length=16000, threshold=0.3):
    # Tải mô hình với custom_objects nếu cần
    try:
        model = tf.keras.models.load_model(model_path, compile=False)  # Không biên dịch lại
    except Exception as e:
        print(f"Không thể tải trực tiếp mô hình: {e}")
        print("Đang thử định nghĩa lại mô hình...")
        model = build_resnet()  # Định nghĩa lại kiến trúc
        model.load_weights(model_path)  # Tải trọng số
        
    # Tải file âm thanh
    try:
        audio, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        raise ValueError(f"Không thể tải file âm thanh {audio_file}: {e}")
    
    # Chia thành các đoạn
    segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]
    instrument_labels = [
        'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 
        'organ', 'reed', 'string', 'synth_lead', 'vocal'
    ]
    all_predictions = []
    
    # Dự đoán cho từng đoạn
    for segment in segments:
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        mel_spec = preprocess_audio(segment)
        mel_spec = mel_spec[..., np.newaxis]
        mel_spec = np.expand_dims(mel_spec, axis=0)
        pred = model.predict(mel_spec, verbose=0)[0]
        
        detected_instruments = [instrument_labels[i] for i in range(len(pred)) if pred[i] > threshold]
        all_predictions.extend(detected_instruments)
    
    unique_instruments = list(set(all_predictions))
    return unique_instruments

# Thử nghiệm
def main():
    # Đường dẫn đến file mô hình và file âm thanh
    model_path = r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_89.h5'
    audio_file = r'D:\Python\AI-Instrument-Classifier\music\audio.wav'
    
    try:
        instruments = predict_instruments(audio_file, model_path, threshold=0.3)
        print("Nhạc cụ được nhận diện:", instruments)
        print("Số lượng nhạc cụ nhận diện được:", len(instruments))
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()