import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NSNet2(nn.Module):
    def __init__(self, cfg={
            'n_fft': 320,
            'hop_len': 160,
            'win_len': 320,
        }):
        super(NSNet2, self).__init__()
        self.n_fft = cfg['n_fft']
        self.n_freq_bins = self.n_fft // 2 + 1
        self.n_gru = 2
        self.gru_dropout = 0.2
        # build model
        self.fc_input = nn.Linear(self.n_freq_bins, 400)
        self.gru = nn.GRU(
            batch_first=True,
            input_size=400,
            hidden_size=400,
            num_layers=self.n_gru,
            dropout=self.gru_dropout,
        )
        self.dense0 = nn.Linear(400, 600)
        self.dense1 = nn.Linear(600, 600)
        self.dense2 = nn.Linear(600, self.n_freq_bins)

    def forward(self, x):
        x = torch.relu(self.fc_input(x))
        x, _ = self.gru(x)
        x = torch.relu(self.dense0(x))
        x = torch.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x


# main
if __name__ == '__main__':
    x = torch.randn(1, 100, 161)
    model = NSNet2()
    y = model(x)
    print(y.shape)
    # export to onnx
    torch.onnx.export(
        model, x,
        'save/nsnet2.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        verbose=False,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch', 1: 'time', 2: 'freq'},
            'output': {0: 'batch', 1: 'time', 2: 'freq'},
        }
    )
    os.system('python3 -m onnxsim save/nsnet2.onnx save/nsnet2_sim.onnx')
