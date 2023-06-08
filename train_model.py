import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import ResNet, BasicBlock


class SpeakerDataset(Dataset):
    def __init__(self, root_dir,T=5.0, transform=None):
        self.root_dir = root_dir
        self.speakers = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]  # Only include directories
        self.filepaths = []
        self.labels = []
        self.transform = transform
        self.T = T  # Target duration in seconds


        for i, speaker in enumerate(self.speakers):
            speaker_dir = os.path.join(root_dir, speaker)
            for filename in os.listdir(speaker_dir):
                self.filepaths.append(os.path.join(speaker_dir, filename))
                self.labels.append(i)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        samples, sample_rate = librosa.load(self.filepaths[idx], sr=16000)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=128)

        # Convert the Mel spectrogram to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize the Mel spectrogram to the range [0, 1]
        mel_spectrogram_db_normalized = (mel_spectrogram_db + 80) / 80

        ###
        # Calculate target number of frames
        hop_length = 512  # Default hop length in librosa
        N = int(self.T * sample_rate // hop_length)

        # Pad or truncate to target length
        if mel_spectrogram_db_normalized.shape[1] < N:
            # Zero pad
            pad_width = N - mel_spectrogram_db_normalized.shape[1]
            mel_spectrogram_db_normalized = np.pad(mel_spectrogram_db_normalized, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mel_spectrogram_db_normalized = mel_spectrogram_db_normalized[:, :N]

        ###

        # Convert the NumPy array to a PyTorch tensor and add a channel dimension
        mel_spectrogram_tensor = torch.tensor(mel_spectrogram_db_normalized).unsqueeze(0)

        if self.transform:
            mel_spectrogram_tensor = self.transform(mel_spectrogram_tensor)

        label = self.labels[idx]
        return mel_spectrogram_tensor, label


class CustomResNet34(ResNet):
    def __init__(self, num_classes):
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


def main():
    dataset = SpeakerDataset('dataset/')
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    # # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_classes = len(dataset.speakers)
    model = CustomResNet34(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item()}')

    # Save the trained model

    torch.save(model.state_dict(), 'speaker_identification_model.pth')


if __name__ == "__main__":
    main()