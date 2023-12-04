import time

import numpy as np
import soundfile as sf
import tensorflow as tf


# load model from tflite
interpreter = tf.lite.Interpreter(model_path='save/tinySenet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


def model(x):
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


audio, sr = sf.read('test/test.wav')
if len(audio.shape) == 2:
    audio = audio[:, 0]
print('audio shape: {}'.format(audio.shape))

n_window = len(audio) // 160 - 1
print('n_window: {}'.format(n_window))


# warm up 50 times
for i in range(50):
    x = np.random.randn(1, 161, 1).astype(np.float32)
    model(x)

windows = []
avg_time = []
for i in range(n_window):
    _win = audio[i*160:(i+2)*160]
    _win = np.fft.rfft(_win).astype(np.float32)
    _win = np.reshape(_win, (1, 161, 1))
    t_start = time.process_time()
    _win = model(_win)
    t_end = time.process_time()
    avg_time.append((t_end - t_start) * 1000)   # ms
    _win = np.squeeze(_win)
    _win = np.fft.irfft(_win).astype(np.float32)
    windows.append(_win)

print('avg time: {:.4f} ms'.format(np.mean(avg_time)))
# overlap and add
out = np.zeros_like(audio)
for i in range(n_window):
    out[i*160:(i+2)*160] += windows[i]

sf.write('test/test_tflite.wav', out, sr)
