import os
import subprocess
from moviepy.editor import *
from pydub import AudioSegment

famous_people = {
    # "Alijon_qori": "https://www.youtube.com/watch?v=rNN_4_voleU",
    # "Abror_Muxtor_Ali": "https://www.youtube.com/watch?v=ruaiX-SKfNo",
    # "Sardor_domla_test": "https://www.youtube.com/watch?v=aDNTJMq6deY",
}
for name, url in famous_people.items():
    try:
        # Download the video
        video_filename = f"{name}_video.mp4"
        subprocess.run(["yt-dlp", "-o", video_filename, "-f", "best", url], check=True)

        # Extract audio
        audio_filename = f"{name}_audio.wav"
        video = VideoFileClip(video_filename)
        video.audio.write_audiofile(audio_filename)

        # Delete the video file
        os.remove(video_filename)
    except Exception as e:
        print(f"An error occurred while processing {name}: {e}")


def split_audio(input_file, output_dir, segment_duration=5*1000):
    audio = AudioSegment.from_wav(input_file)
    total_duration = len(audio)
    num_segments = total_duration // segment_duration

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"segment_{i + 1}.wav"), format="wav")


for name in famous_people:
    input_file = f"{name}_audio.wav"
    output_dir = f"dataset/{name}"
    os.makedirs(output_dir, exist_ok=True)
    split_audio(input_file, output_dir)
