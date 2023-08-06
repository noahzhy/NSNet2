import os
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from model.nsnet2_ex import NSNet2


class NSNet2Demo:
    def __init__(self, model_ckpt, cfg=None):
        self.cfg = cfg
        self.model = NSNet2(cfg)
        self.model.load_state_dict(torch.load(model_ckpt))

    def enhance(self, audio_file):
        # check extension of audio file if it is not a wav file
        if Path(audio_file).suffix != '.wav':
            print("Please provide a wav file.")
            return

        sigIn, fs = sf.read(audio_file)
        assert fs == 16000

        if len(sigIn.shape) > 1:
            sigIn = sigIn[:, 0]

        # convert to torch
        sigIn = torch.from_numpy(sigIn).type(torch.float32)

        spec = torch.stft(
            sigIn,
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_len'],
            win_length=self.cfg['win_len'],
            window=torch.hann_window(self.cfg['win_len']),
            return_complex=True,
        )

        sigOut = self.model(spec)

        sigOut = torch.istft(
            sigOut,
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_len'],
            win_length=self.cfg['win_len'],
            window=torch.hann_window(self.cfg['win_len']),
        )
        # convert to numpy
        sigOut = sigOut.detach().numpy()
        # write file
        sf.write('denoised.wav', sigOut, fs)


# main
if __name__ == '__main__':
    cfg = {
        'n_fft': 320,
        'hop_len': 160,
        'win_len': 320,
        'minGain': -80,
    }

    model = NSNet2Demo(model_ckpt='nsnet2.ckpt', cfg=cfg)
    model.enhance('test.wav')
