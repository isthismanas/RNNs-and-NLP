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