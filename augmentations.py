import random

import torch
import torchaudio
import torchaudio.transforms as T


def time_shift(
    waveform: torch.Tensor, shift_limit: float
) -> torch.Tensor:
    """Randomly shifts the audio waveform along the time axis."""
    _, num_frames = waveform.shape
    shift_amount = int(random.uniform(0, shift_limit) * num_frames)
    if shift_amount > 0:
        waveform = torch.cat(
            (torch.zeros((1, shift_amount)), waveform[:, :-shift_amount]), dim=1
        )
    return waveform

def add_noise(waveform: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Adds noise to the audio waveform."""
    max_noise = waveform.mean() * noise_level
    noise = torch.randn_like(waveform) * max_noise
    return waveform + noise


def random_time_shift(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return time_shift(waveform, shift_limit=0.20)


def random_noise(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return add_noise(waveform, noise_level=0.20)


def random_frequency_masking(mel_spec: torch.Tensor) -> torch.Tensor:
    """Mel spectrogram frequency masking"""
    # mask max. 3*15% of the frequency axis 
    fm_param = int(mel_spec.shape[0] * 0.15)
    n_times = random.randint(1, 3)
    for _ in range(n_times):
        freq_mask = T.FrequencyMasking(freq_mask_param=fm_param, iid_masks=True)
        mel_spec = freq_mask(mel_spec)
    return mel_spec


def random_time_masking(mel_spec: torch.Tensor) -> torch.Tensor:
    """Mel spectrogram time masking"""
    # mask max. 3*15% of the time axis
    tm_param = int(mel_spec.shape[1] * 0.15)
    n_times = random.randint(1, 3)
    for _ in range(n_times):
        time_mask = T.TimeMasking(time_mask_param=tm_param, iid_masks=True)
        mel_spec = time_mask(mel_spec)
    return mel_spec
