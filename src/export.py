import shutil
from classify import predict_instrument

instrument_mapping = {
    "bass": "bass.wav",
    "flute": "flute.wav",
    "brass": "brass.wav",
    "guitar": "guitar.wav",
    "keyboard": "keyboard.wav",
    "mallet": "mallet.wav",
    "organ": "organ.wav",
    "reed": "reed.wav",
    "string": "string.wav",
    "synth_lead": "synth_lead.wav",
    "vocal": "vocal.wav",
}

file = "../datasets/nsynth-test/example.wav"
detected_instruments = [predict_instrument(file)]

for instrument in detected_instruments:
    if instrument in instrument_mapping:
        source_file = f"../spleeter_output/{instrument_mapping[instrument]}"
        target_file = f"../output/{instrument}.wav"
        shutil.move(source_file, target_file)
        print(f"Đã lưu file: {target_file}")
