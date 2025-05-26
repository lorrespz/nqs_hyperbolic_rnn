#Code written by HLD - This is the TF2 rewritten and adapted version of the following TF1 code:
#https://github.com/mhibatallah/RNNWavefunctions/blob/master/2DTFIM_2DRNN/MDRNNcell.py

import numpy as np
import tensorflow as tf

###################################################################################################333

class MDRNNcell(tf.keras.Layer):
    """An implementation of the most basic 2DRNN Vanilla RNN cell.
    Args:
        num_units (int): The number of units in the RNN cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in):
        super(MDRNNcell, self).__init__()

        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

    #def build(self):

        self.Wh = self.add_weight(shape=[self._num_units, self._num_units],
                                    initializer=tf.keras.initializers.glorot_normal(), 
                                    trainable=True, name ="Wh",)

        self.Uh = self.add_weight(shape=[self._num_in,self._num_units],
                                    initializer=tf.keras.initializers.glorot_normal(), 
                                    trainable=True, name ="Uh",)

        self.Wv = self.add_weight(shape=[self._num_units, self._num_units],
                                   initializer=tf.keras.initializers.glorot_normal(), 
                                    trainable=True, name ="Wv",)

        self.Uv = self.add_weight(shape=[self._num_in,self._num_units],
                                   initializer=tf.keras.initializers.glorot_normal(), 
                                    trainable=True, name ="Uv", )


        self.b = self.add_weight(shape=[self._num_units],
                                initializer=tf.keras.initializers.glorot_normal(), 
                                trainable=True, name ="b", dtype = self.dtype)


    # needed properties
    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, states):

        # prepare input linear combination
        input_mul_left = tf.linalg.matmul(inputs[0], self.Uh) # [batch_sz, num_units] #Horizontal
        input_mul_up = tf.linalg.matmul(inputs[1], self.Uv) # [batch_sz, num_units] #Vectical

        state_mul_left = tf.linalg.matmul(states[0], self.Wh)  # [batch_sz, num_units] #Horizontal
        state_mul_up = tf.linalg.matmul(states[1], self.Wv) # [batch_sz, num_units] #Vectical

        preact = input_mul_left + state_mul_left + input_mul_up + state_mul_up  + self.b #Calculating the preactivation

        output = tf.nn.elu(preact) # [batch_sz, num_units]

        new_state = output

        return output, new_state
