import numpy as np
import soundfile as sf
import tensorflow as tf


# load model from tflite
interpreter = tf.lite.Interpreter(model_path='tinySenet.tflite')
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
windows = []
for i in range(n_window):
    _win = audio[i*160:(i+2)*160]
    _win = np.fft.rfft(_win).astype(np.float32)
    _win = np.reshape(_win, (1, 161, 1))
    _win = model(_win)
    _win = np.squeeze(_win)
    _win = np.fft.irfft(_win).astype(np.float32)
    windows.append(_win)

    
# overlap and add
out = np.zeros_like(audio)
for i in range(n_window):
    out[i*160:(i+2)*160] += windows[i]

sf.write('test/test_tflite.wav', out, sr)
