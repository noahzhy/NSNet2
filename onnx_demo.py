import time
import numpy as np
import soundfile as sf
import onnxruntime as rt


"""
Using onnxruntime to inference onnx model,
numpy to take fft and ifft,
soundfile to read and write wav file.
"""
cfg = {
    'n_fft': 320,
    'hop_len': 160,
    'win_len': 320,
}

# load onnx model
sess = rt.InferenceSession('save/nsnet2_ex_sim.onnx')
sess_input = sess.get_inputs()[0]
# get input shape and name
input_name, input_shape = sess_input.name, sess_input.shape
# random input
x = np.random.randn(*input_shape).astype(np.float32)

# load wav file
audio_path = 'test/test.wav'
audio, fs = sf.read(audio_path)
if len(audio.shape) == 2:
    audio = audio[:, 0]

n_window = len(audio) // cfg['hop_len'] - 1
windows = []
for i in range(n_window):
    _win = audio[i * cfg['hop_len']:i * cfg['hop_len'] + cfg['win_len']]
    # fft
    _win = np.fft.rfft(_win).astype(np.float32)
    # reshape to [n_freq, 1]
    _win = np.expand_dims(_win, axis=-1)
    print(_win.shape)
    # onnx inference
    _win = sess.run(None, {input_name: _win})[0]
    print(_win.shape)
    # squeeze to [n_freq]
    _win = np.squeeze(_win)
    # ifft
    _win = np.fft.irfft(_win).astype(np.float32)
    windows.append(_win)

print('Done.')

# overlap and add
sigOut = np.zeros(len(audio))
for i in range(n_window):
    sigOut[i * cfg['hop_len']:i * cfg['hop_len'] + cfg['win_len']] += windows[i]

# write file
sf.write('onnx_output.wav', sigOut, fs)
