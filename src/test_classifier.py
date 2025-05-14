import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Định nghĩa residual_block
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

# Định nghĩa build_resnet
def build_resnet(input_shape=(128, 32, 1), num_classes=11, kernel_size=3, filters=[32, 64, 128]):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], (kernel_size, kernel_size), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, filters[0], kernel_size=kernel_size)
    x = residual_block(x, filters[1], kernel_size=kernel_size)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters[2], kernel_size=kernel_size)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Hàm tiền xử lý âm thanh
def preprocess_audio(audio, max_length=16000, n_mels=128, hop_length=512):
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Hàm dự đoán nhạc cụ
def predict_instruments(audio_file, model, model_info, sr=16000, segment_length=16000, threshold=0.3):
    try:
        audio, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        raise ValueError(f"Không thể tải file âm thanh {audio_file}: {e}")
    
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length // 2)]
    instrument_labels = [
        'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
        'organ', 'reed', 'string', 'synth_lead', 'vocal'
    ]
    all_predictions = []
    timestamps = []
    
    for i, segment in enumerate(segments):
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        mel_spec = preprocess_audio(segment)
        mel_spec = mel_spec[..., np.newaxis]
        mel_spec = np.expand_dims(mel_spec, axis=0)
        pred = model.predict(mel_spec, verbose=0)[0]
        
        detected_instruments = [instrument_labels[i] for i in range(len(pred)) if pred[i] > threshold]
        if detected_instruments:
            all_predictions.append(detected_instruments)
            timestamps.append((i * segment_length // 2) / sr)
    
    instrument_counts = Counter([instr for segment in all_predictions for instr in segment])
    detected_instruments = [
        (instr, count, count / sum(instrument_counts.values()) * 100 if instrument_counts else 0)
        for instr, count in instrument_counts.items()
    ]
    
    return detected_instruments, timestamps, all_predictions

# Thông tin 3 mô hình
models_info = [
    {
        'name': 'ResNet 1x1',
        'kernel_size': 1,
        'filters': [16, 32, 64],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_1x1.h5'
    },
    {
        'name': 'ResNet 3x3',
        'kernel_size': 3,
        'filters': [32, 64, 128],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_3x3.h5'
    },
    {
        'name': 'ResNet 5x5',
        'kernel_size': 5,
        'filters': [64, 128, 256],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_5x5.h5'
    }
]

# Danh sách nhạc cụ
INSTRUMENT_CLASSES = [
    'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet',
    'organ', 'reed', 'string', 'synth_lead', 'vocal'
]

# Gán màu cố định cho từng nhạc cụ
COLOR_MAP = {
    'bass': 'blue',
    'brass': 'orange',
    'flute': 'green',
    'guitar': 'red',
    'keyboard': 'purple',
    'mallet': 'brown',
    'organ': 'pink',
    'reed': 'gray',
    'string': 'yellow',
    'synth_lead': 'cyan',
    'vocal': 'magenta'
}

# Trực quan hóa timeline với màu cố định
def plot_instrument_timelines(audio_file, models_info, results):
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_duration = len(audio) / sr
    
    plt.figure(figsize=(15, 10))
    for i, (model_info, result) in enumerate(zip(models_info, results), 1):
        timestamps, predictions = result['timestamps'], result['all_predictions']
        plt.subplot(3, 1, i)
        
        # Vẽ từng đoạn với màu cố định cho từng nhạc cụ
        for t, segment_preds in zip(timestamps, predictions):
            for pred in segment_preds:
                idx = INSTRUMENT_CLASSES.index(pred)
                plt.plot([t, t + 1.0], [idx, idx], color=COLOR_MAP[pred], lw=2, label=pred if t == timestamps[0] else "")
        
        plt.title(f"Instrument Timeline - {model_info['name']}")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Detected Instrument')
        plt.yticks(range(len(INSTRUMENT_CLASSES)), INSTRUMENT_CLASSES)
        plt.xlim(0, audio_duration)
        plt.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

# Hàm so sánh 3 mô hình
def compare_models(audio_file, models_info, threshold=0.3):
    results = []
    
    for model_info in models_info:
        print(f"\nProcessing with {model_info['name']}...")
        try:
            model = build_resnet(
                kernel_size=model_info['kernel_size'],
                filters=model_info['filters']
            )
            model.load_weights(model_info['path'])
        except Exception as e:
            print(f"Không thể tải mô hình {model_info['name']}: {e}")
            continue
        
        detected_instruments, timestamps, all_predictions = predict_instruments(
            audio_file, model, model_info, threshold=threshold
        )
        
        results.append({
            'name': model_info['name'],
            'detected_instruments': detected_instruments,
            'timestamps': timestamps,
            'all_predictions': all_predictions
        })
    
    # Hiển thị kết quả
    print("\nComparison of Detected Instruments Across Models:")
    for result in results:
        print(f"\n{result['name']}:")
        df = pd.DataFrame(
            result['detected_instruments'],
            columns=['Instrument', 'Count', 'Percentage (%)']
        ).sort_values(by='Percentage (%)', ascending=False)
        print(df.to_string(index=False))
    
    # Trực quan hóa timeline
    plot_instrument_timelines(audio_file, models_info, results)
    
    # Bảng so sánh
    print("\nSummary of Differences:")
    all_instruments = set()
    for result in results:
        instruments = {instr[0] for instr in result['detected_instruments']}
        all_instruments.update(instruments)
    
    comparison_df = pd.DataFrame(index=list(all_instruments), columns=[m['name'] for m in models_info])
    for result in results:
        instr_dict = {instr[0]: instr[2] for instr in result['detected_instruments']}
        for instr in all_instruments:
            comparison_df.loc[instr, result['name']] = instr_dict.get(instr, 0.0)
    print(comparison_df.to_string())

if __name__ == "__main__":
    AUDIO_FILE = r'D:\Python\AI-Instrument-Classifier\music\audio2.wav'
    compare_models(AUDIO_FILE, models_info, threshold=0.3)