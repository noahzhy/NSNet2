import os
import torch

from model.nsnet2 import NSNet2


cfg = {
    'n_fft': 320,
    'hop_len': 160,
    'win_len': 320,
}

model = NSNet2(cfg=cfg)
# # save checkpoint
# torch.save(model.state_dict(), 'nsnet2.ckpt')
# load checkpoint
model.load_state_dict(torch.load('nsnet2.ckpt'))
model.eval()

# 20ms window, 10ms hop
x = torch.randn(cfg['n_fft'],)
n = torch.stft(x, 
    n_fft=cfg['n_fft'],
    hop_length=cfg['hop_len'],
    win_length=cfg['win_len'],
    window=torch.hann_window(cfg['win_len']),
    return_complex=True,
)
n_freq, n_frames = n.shape[-2:]
x = torch.randn(1, n_frames, n_freq)

# save as ONNX model
torch.onnx.export(
    model, x,
    "nsnet2.onnx",
    do_constant_folding=True,
    opset_version=16,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'frames', 2: 'freq_bins'},
        'output': {0: 'batch_size', 2: 'frames', 3: 'freq_bins'},
    }
)

# simplify ONNX model
os.system('python3 -m onnxsim nsnet2.onnx nsnet2_simplified.onnx')
