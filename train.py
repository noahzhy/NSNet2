import os, sys, time, glob, random

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *


from model.tinyNSNet import TinyNSNet



# main
if __name__ == '__main__':
    train_dir = './WAVs/dataset/training'
    val_dir = './WAVs/dataset/validation'

    train_cfg = {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'batch_size': 64,
        'alpha': 0.35,
    }

    model_cfg = {
        'n_fft': 320,
        'hop_len': 160,
        'win_len': 320,
    }

    model = TinyNSNet().build(input_shape=(161, 1))
    # summary
    model.summary()
    quit()

    # adamw
    optimizer = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # MSE
    criterion = MSELoss()

    # train
    train(model, optimizer, criterion, train_cfg, model_cfg)
