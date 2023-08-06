import os
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from model.nsnet2 import NSNet2


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
        print("sigIn shape:", sigIn.shape, "fs:", fs)
        assert fs == 16000

        if len(sigIn.shape) > 1:
            sigIn = sigIn[:, 0]

        spec, inpFeat = self.preProcessing(sigIn)

        # add a batch dimension
        inpFeat = inpFeat.unsqueeze(0).type(torch.float32)

        sigOut = self.model(inpFeat)
        sigOut = self.afterProcessing(sigOut, spec)
        # convert to numpy
        sigOut = sigOut.detach().numpy()
        # write file
        sf.write('denoised.wav', sigOut, fs)

    def preProcessing(self, sigIn):
        spec = torch.stft(
            torch.from_numpy(sigIn),
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_len'],
            win_length=self.cfg['win_len'],
            window=torch.hann_window(self.cfg['win_len']),
            return_complex=True,
        )
        powSpec = torch.abs(spec)**2
        inpFeat = torch.log10(torch.max(powSpec, torch.tensor([10**(-12)], dtype=torch.float32)))
        inpFeat = torch.transpose(inpFeat, 0, 1)
        return spec, inpFeat

    def afterProcessing(self, sigOut, spec):
        # limit suppression gain
        minGain = 10**(self.cfg['minGain'] / 20)
        out = torch.squeeze(sigOut)
        gain = torch.transpose(out, 0, 1)
        gain = torch.clamp(gain, min=minGain, max=1.0)
        outSpec = spec * gain
        # istft
        sigOut = torch.istft(
            outSpec,
            n_fft=self.cfg['n_fft'],
            hop_length=self.cfg['hop_len'],
            win_length=self.cfg['win_len'],
            window=torch.hann_window(self.cfg['win_len']),
        )
        return sigOut


# main
if __name__ == '__main__':
    cfg = {
        'n_fft': 320,
        'hop_len': 160,
        'win_len': 320,
        'minGain': -80,
    }

    model = NSNet2Demo(
        model_ckpt='nsnet2.ckpt',
        cfg=cfg,
    )
    import time
    start = time.time()
    for i in range(10):
        model.enhance('test.wav')
    print("time:", time.time() - start)
