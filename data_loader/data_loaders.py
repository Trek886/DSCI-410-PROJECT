import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample

torchaudio.set_audio_backend("soundfile")

class InstrumentDataset(Dataset):
    def __init__(self, directory, metadata_file, sample_rate=22050, n_mels=128, fixed_length=12):
        """
        Args:
            data_dir (str): Path to the directory containing wav files.
            metadata_file (str): Path to the CSV metadata file.
            sample_rate (int): Target sample rate for audio.
            n_mels (int): Number of mel filterbanks.
            fixed_length (int): Length of audio clips in seconds.
        """
        self.data_dir = directory
        self.metadata = pd.read_csv(metadata_file)
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length
        self.target_samples = sample_rate * fixed_length  # Total samples for 12 seconds

        self.mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db_transform = AmplitudeToDB()

        # Convert class labels to numerical values
        self.class_map = {label: idx for idx, label in enumerate(self.metadata['Class'].unique())}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.data_dir, row['FileName'])

        # Load audio and force mono
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

        # Resample if needed
        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Ensure fixed length (12 seconds)
        num_samples = waveform.shape[1]
        if num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]  # Trim
        else:
            pad_length = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad with zeros

        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)  # Convert to decibels

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        # Get class label
        label = self.class_map[row['Class']]

        return mel_spec, label

class RandomPitch(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_steps=(-2, 2)):
        super(RandomPitch, self).__init__()
        self.sample_rate = sample_rate
        self.n_steps_range = n_steps

    def forward(self, waveform):
        steps = random.uniform(*self.n_steps_range)
        pitch_shift = T.PitchShift(self.sample_rate, n_steps=steps)
        return pitch_shift(waveform)

def get_data_loaders(base_path='Data Science\DSCI 410 project\dataset', batch_size=32, duration=5, num_workers=4):
    """
    Create data loaders for train, test, and validation sets for audio classification.
    
    Args:
        base_path (str): Base path to the dataset.
        batch_size (int): Batch size for the data loaders.
        duration (float): Duration (in seconds) to which each audio clip will be trimmed or padded.
        num_workers (int): Number of subprocesses to use for data loading.
    """
    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64
    )

    # Altered pitch for extra validation set.
    alter_transform = nn.Sequential(
        RandomPitch(sample_rate=16000, n_steps=(-2, 2)),
        audio_transform
    )
    
    # Create the training dataset.
    train_dataset = InstrumentDataset(
        directory=os.path.join(base_path, 'Train_submission'),
        metadata_file=os.path.join(base_path, 'Metadata_Train.csv'),
    )
    
    # Create the testing dataset.
    test_dataset = InstrumentDataset(
        directory=os.path.join(base_path, 'Test_submission'),
        metadata_file=os.path.join(base_path, 'Metadata_Test.csv'),
    )

    # Create validation dataset (using the training data with altered pitch).
    val_dataset = InstrumentDataset(
        directory=os.path.join(base_path, 'Train_submission'),
        metadata_file=os.path.join(base_path, 'Metadata_Train.csv'),
    )

    # Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, val_loader