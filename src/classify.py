import tensorflow as tf
import numpy as np
from preprocess import extract_mel_spectrogram

model = tf.keras.models.load_model("../models/instrument_cnn.h5")

def predict_instrument(file_path):
    spectrogram = extract_mel_spectrogram(file_path, None)
    spectrogram = np.expand_dims(spectrogram, axis=[0, -1])
    prediction = model.predict(spectrogram)
    instrument = np.argmax(prediction)
    return instrument

file = "../datasets/nsynth-test/example.wav"
result = predict_instrument(file)
print(f"Nhạc cụ nhận diện: {result}")