import os
import torch
import librosa
import numpy as np
from train_model import SpeakerDataset
from train_model import CustomResNet34
from fastapi import FastAPI, File, UploadFile
dataset = SpeakerDataset('dataset/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def identify_speaker(model, audio_file, dataset):
    samples, sample_rate = librosa.load(audio_file, sr=None)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128)

    # Convert the Mel spectrogram to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize the Mel spectrogram to the range [0, 1]
    mel_spectrogram_db_normalized = (mel_spectrogram_db + 80) / 80

    # Convert the NumPy array to a PyTorch tensor and add a channel dimension
    mel_spectrogram_tensor = torch.tensor(mel_spectrogram_db_normalized).unsqueeze(0)

    input_tensor = mel_spectrogram_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        _, predicted_label = torch.max(output, 1)
        predicted_speaker = dataset.speakers[predicted_label]

    return predicted_speaker


# Test the speaker identification function with a new audio file
# test_audio_file = 'test_data/Shukrulloh_domla_test_audio.wav'
test_audio_file = 'test_data/Sardor_domla_test_audio.wav'
model_path = 'speaker_identification_model.pth'
num_classes = len(dataset.speakers)
model = CustomResNet34(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
# identified_speaker = identify_speaker(model, test_audio_file, dataset)
# print(f"Ovoz egasi: {identified_speaker}")

app = FastAPI()


@app.post("/detect/")
async def speaker_identification(audio_file: UploadFile = File(...)):
    with open("temp_audio.wav", "wb") as f:
        f.write(await audio_file.read())
    identified_speaker = identify_speaker(model, "temp_audio.wav", dataset)
    return {"speaker": identified_speaker}

# uvicorn detector_api:app --reload


