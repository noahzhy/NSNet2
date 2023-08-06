import os
import glob
import random

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from PIL import Image


# show audio as wave and spectrogram
def show_audio(y, title='Raw', sr=16000):
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('{} audio signal'.format(title))

    plt.subplot(2, 1, 2)
    D = librosa.stft(y, hop_length=160, win_length=320)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='s', y_axis='linear')
    plt.title('Linear-frequency power spectrogram')

    plt.colorbar(img, format="%+2.f dB")
    plt.tight_layout()
    # return as PIL image
    fig = plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


# main
if __name__ == '__main__':
    y, sr = librosa.load('test/test.wav', sr=16000)
    raw = show_audio(y, title='Raw')

    y, sr = librosa.load('test/test_nsnet2-20ms-baseline.wav', sr=16000)
    after_p = show_audio(y, title='After Processing')

    # hstack two images
    c = np.hstack((raw, after_p))
    img = Image.fromarray(c)
    # show and save
    img.show()
    img.save('compare.png')
