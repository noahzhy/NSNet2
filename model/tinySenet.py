import os
import sys

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models, optimizers
from keras_flops import get_flops

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from model.rnn import *
from get_flops import try_count_flops


# using cpu only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

hidden_size = 320


class TinySenet(models.Model):
    def __init__(self, n_freq_bins=161, hidden_size = 320, rnn_dropout=0.2):
        super(TinySenet, self).__init__()
        self.n_freq_bins = n_freq_bins
        self.dense0 = Dense(hidden_size, activation='relu')
        self.rnn = models.Sequential([], name='rnn')

        # rnn cell stack
        fast_grnn = [
            FastGRNNCell(hidden_size),
            FastGRNNCell(hidden_size),
        ]

        for idx, cell in enumerate(fast_grnn):
            self.rnn.add(RNN(cell, return_sequences=True, name='fast_grnn_{}'.format(idx)))
            self.rnn.add(Dropout(rnn_dropout))

        self.dense1 = Dense(320, activation='relu')
        self.dense2 = Dense(320, activation='relu')
        self.dense3 = Dense(self.n_freq_bins, activation='sigmoid')

    def calcFeat(self, x):
        x = tf.math.pow(tf.abs(x), 2, name='power')
        x = tf.math.log(tf.math.maximum(x, 1e-12, name='log')) / tf.math.log(10.0)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

    def limitGain(self, x, spec, mingain=-80):
        mingain = 10**(mingain/20)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.clip_by_value(x, mingain, 1.0, name='clip')
        return tf.multiply(x, spec, name='mul')

    def build(self, input_shape):
        inputs = Input(shape=input_shape, name='input')
        print('input_shape:', input_shape)
        x = self.calcFeat(inputs)
        x = self.dense0(x)
        x = self.rnn(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.limitGain(x, inputs)
        return models.Model(inputs=inputs, outputs=x, name='tinySenet')

    def call(self, x):
        return self.build(x.shape)


# main
if __name__ == '__main__':
    model = TinySenet().build((161, 1))
    model.save('save/tinySenet.h5')
    model.summary()

    # count flops
    flops = get_flops(model, batch_size=1)
    print('flops: {:.2f}M'.format(flops / 1e6))

    flops = try_count_flops(model)
    print('flops: {:.2f}M'.format(flops / 1e6))

    # save as .tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
    ]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    with open('save/tinySenet.tflite', 'wb') as f:
        f.write(tflite_model)
