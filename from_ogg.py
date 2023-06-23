from moviepy.editor import *
from pydub import AudioSegment
def split_audio(input_file, output_dir, segment_duration=5*1000):
    audio = AudioSegment.from_ogg(input_file)
    total_duration = len(audio)
    num_segments = total_duration // segment_duration

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"segment_{i + 1}.wav"), format="wav")


def ogg_to_wav(input_file, output_dir, segment_duration=10*1000):
    audio = AudioSegment.from_ogg(input_file)
    start_time = 0
    end_time = segment_duration
    segment = audio[start_time:end_time]
    output_file = os.path.join(output_dir, f"test_{os.path.splitext(os.path.basename(input_file))[0]}.wav")
    segment.export(output_file, format="wav")
    return output_file


def mp3_to_wav(input_file, output_dir, segment_duration=10*1000):
    audio = AudioSegment.from_mp3(input_file)
    start_time = 0
    end_time = segment_duration
    segment = audio[start_time:end_time]
    output_file = os.path.join(output_dir, f"test_{os.path.splitext(os.path.basename(input_file))[0]}.wav")
    segment.export(output_file, format="wav")
    return output_file

# ogg_to_wav("test_data/Baxtibek.ogg","test_data",60000)

name = "Bahtibek_Anvarov"
input_file = f"test_data/test_Baxtibek.wav"
output_dir = f"dataset/{name}"
os.makedirs(output_dir, exist_ok=True)
split_audio(input_file, output_dir)