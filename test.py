from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torchaudio.functional import angle, istft

from dataloader.wav_dataset import WAVDataset
from model.nsnet2 import NSNet2

model = NSNet2.load_from_checkpoint('nsnet2.ckpt')

test_dir = './WAVs/dataset/testing'
n_fft = 320
dataset = WAVDataset(dir=test_dir, n_fft=n_fft, test=True)
dataloader = DataLoader(dataset, batch_size=16, drop_last=False, shuffle=True)
noisy_waveform, clean_waveform, x_stft, _, x_lps, x_ms, _, _ = next(iter(dataloader))

#  enable eval mode
model.zero_grad()
model.eval()
model.freeze()

# disable gradients to save memory
torch.set_grad_enabled(False)

gain_mask = model(x_lps)
y_spectrogram_hat = x_ms * gain_mask

y_stft_hat = torch.stack([
    y_spectrogram_hat * torch.cos(angle(x_stft)),
    y_spectrogram_hat * torch.sin(angle(x_stft))
    ], dim=-1)

window = torch.hamming_window(n_fft)
y_waveform_hat = istft(
    y_stft_hat,
    n_fft=n_fft,
    hop_length=n_fft // 4,
    win_length=n_fft,
    window=window,
    length=clean_waveform.shape[-1],
)

for i, waveform in enumerate(y_waveform_hat.numpy()):
    sf.write('denoised' + str(i) + '.wav', waveform, 16000)
