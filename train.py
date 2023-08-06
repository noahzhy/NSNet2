import os
import torch

from model.nsnet2 import NSNet2


train_dir = './WAVs/dataset/training'
val_dir = './WAVs/dataset/validation'

train_cfg = {
    'train_dir': train_dir,
    'val_dir': val_dir,
    'batch_size': 64,
    'alpha': 0.35,
}

model_cfg = {
    'n_fft': 320,
    'hop_len': 160,
    'win_len': 320,
}

model = NSNet2(model_cfg)
# adamw
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# MSE
criterion = torch.nn.MSELoss()

# train

