import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        audio_dir,
        folds=[1],
        augmentations=None,
        mel_augmentations=None,
        augmentation_prob=0.8,
        resample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        mel_size=(64, 128),
        mean=None,
        std=None,
    ):
        """
        Custom Dataset for the ESC-50 audio data.

        Args:
            csv_path (str): Path to the 'esc50.csv' file.
            audio_dir (str): Directory containing the audio files.
            folds (list): List of folds to use (values from 1 to 5).
            transform (callable, optional): Optional transform to apply to the Mel-spectrogram.
            target_transform (callable, optional): Optional transform to apply to the target labels.
            resample_rate (int): Target sampling rate for audio files.
            n_mels (int): Number of Mel bands to generate.
            n_fft (int): FFT window size.
            hop_length (int): Number of samples between successive frames.
            mel_size (tuple): Desired output size of Mel-spectrogram (height, width).
            mean (float, optional): Mean value for normalization.
            std (float, optional): Standard deviation for normalization.
        """
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.folds = folds
        self.augmentations = augmentations
        self.mel_augmentations = mel_augmentations
        self.augmentation_prob = augmentation_prob
        self.resample_rate = resample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_size = mel_size

        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["fold"].isin(folds)].reset_index(drop=True)
        self.labels = self.data["target"].tolist()
        self.file_paths = (
            self.data["filename"].apply(lambda x: os.path.join(audio_dir, x)).tolist()
        )

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.resample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        if mean is None or std is None:
            self.mean, self.std = self._compute_mean_std()
        else:
            self.mean = mean
            self.std = std
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.resample_rate, new_freq=self.resample_rate
        )

    def _compute_mean_std(self):
        # Compute mean and std over the training data
        sum_ = 0.0
        sum_sq = 0.0
        count = 0

        for idx in range(len(self.data)):
            file_path = self.file_paths[idx]
            waveform, sample_rate = torchaudio.load(file_path)

            if sample_rate != self.resample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.resample_rate
                )
                waveform = resampler(waveform)

            mel_spec = self.mel_spectrogram(waveform)
            mel_spec = self.amplitude_to_db(mel_spec)

            sum_ += mel_spec.sum()
            sum_sq += (mel_spec**2).sum()
            count += mel_spec.numel()

        mean = sum_ / count
        std = torch.sqrt((sum_sq / count) - (mean**2))

        return mean.item(), std.item()

    def _mixup(self, idx: int, waveform, label, alpha=0.4, beta=0.4):
        """
        Mixup two waveforms

        Args:
            idx: int, index of the first waveform
            waveform: torch.Tensor, waveform to mixup
            label: torch.Tensor, label of the first waveform
            alpha: float, alpha parameter for the beta distribution
            beta: float, beta parameter for the beta distribution
        """
        idx2 = random.randint(0, len(self.data) - 1)
        if idx == idx2:
            return waveform, label
        label2 = torch.tensor(self.labels[idx2])
        label2 = F.one_hot(label2, num_classes=50).float()
        waveform2, _ = torchaudio.load(self.file_paths[idx2])
        lam = np.random.beta(alpha, beta)
        waveform = (1 - lam) * waveform + lam * waveform2
        label = (1 - lam) * label + lam * label2
        return waveform, label
  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        label = torch.tensor(self.labels[idx])
        label = F.one_hot(label, num_classes=50).float()

        if self.augmentation_prob > 0:
            if random.random() < self.augmentation_prob:
                waveform, label = self._mixup(idx, waveform, label)


        if sample_rate != self.resample_rate:
            waveform = self.resampler(waveform)

        if self.augmentations:
            if random.random() < self.augmentation_prob:
                for aug in self.augmentations:
                    waveform = aug(waveform, self.resample_rate)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)

        mel_spec = (mel_spec - self.mean) / self.std

        mel_spec = F.interpolate(
            mel_spec.unsqueeze(0),
            size=self.mel_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.mel_augmentations:
            if random.random() < self.augmentation_prob:
                for aug in self.mel_augmentations:
                    mel_spec = aug(mel_spec)

        return mel_spec, label
