import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

class LNSimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, activation = 'tanh',**kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.activation = tf.keras.activations.get(activation)
        self.simple_rnn_cell  = tf.keras.layers.SimpleRNNCell(self.output_size , activation = None)
        self.layer_norm = tf.keras.layers.Normalization()

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs , [norm_outputs]
    

def to_dataset(sequence, length, shuffle = False, seed = None, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length +1 , shift = 1, drop_remainder= True)
    ds = ds.flat_map(lambda window_ds : window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100000, seed = seed)
    ds =ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1] , window[:,1:])).prefetch(1)