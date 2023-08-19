import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CalcFeat(torch.nn.Module):
    def __init__(self):
        super(CalcFeat, self).__init__()

    def forward(self, Spec):
        powSpec = torch.abs(Spec)**2
        inpFeat = torch.log10(torch.max(powSpec, torch.tensor([10**(-12)], dtype=torch.float32)))
        inpFeat = torch.transpose(inpFeat, 0, 1)
        return inpFeat


class LimitGain(torch.nn.Module):
    def __init__(self, mingain=-80):
        super(LimitGain, self).__init__()
        self.mingain = 10**(mingain/20)

    def forward(self, out, spec):
        gain = torch.transpose(out, 0, 1)
        gain = torch.clamp(gain, min=self.mingain, max=1.0)
        return spec * gain


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
        self.feat = CalcFeat()
        self.fc_input = nn.Linear(self.n_freq_bins, 200)
        self.gru = nn.GRU(
            input_size=200,
            hidden_size=200,
            num_layers=self.n_gru,
            dropout=self.gru_dropout,
        )
        self.dense0 = nn.Linear(200, 300)
        self.dense1 = nn.Linear(300, 300)
        self.dense2 = nn.Linear(300, self.n_freq_bins)
        self.gain = LimitGain(-80)

    def forward(self, x):
        shortcut = x
        print(x.shape)
        x = self.feat(x)
        print(x.shape)

        x = torch.relu(self.fc_input(x))
        print(x.shape)
        x, _ = self.gru(x)
        print(x.shape)
        x = torch.relu(self.dense0(x))
        x = torch.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))

        x = self.gain(x, shortcut)
        return x


# main
if __name__ == '__main__':
    # ms
    t = 20
    win = 16000 * t // 1000
    x = torch.randn(win,)

    x = np.fft.rfft(x.numpy())
    x = np.expand_dims(x, axis=-1)
    # to torch
    x = torch.from_numpy(x).type(torch.float32)
    print(x.shape)

    model = NSNet2()
    # export to onnx
    torch.onnx.export(
        model, x,
        'save/nsnet2_tiny.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
        verbose=False,
        # dynamic_axes={
        #     'input': {0: 'freq', 1: 'time'},
        #     'output': {0: 'freq', 1: 'time'},
        # }
    )
    # simple onnx model
    import os
    os.system('python3 -m onnxsim save/nsnet2_tiny.onnx save/nsnet2_tiny_sim.onnx')

    y = model(x)
    y = y.detach().numpy()
    # squeeze last dim
    y = np.squeeze(y, axis=-1)
    # irfft
    y = np.fft.irfft(y)
    print("after irfft:\t", y.shape)

    # calc flops
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    # to Gflops
    flops = flops / 1e9
    print("FLOPs: %.2f G" % flops)
