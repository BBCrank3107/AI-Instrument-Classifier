import os

def separate_audio(file_path, output_dir):
    command = f"spleeter separate -p spleeter:4stems -o {output_dir} {file_path}"
    os.system(command)

file = "../datasets/nsynth-test/example.wav"
separate_audio(file, "../spleeter_output")