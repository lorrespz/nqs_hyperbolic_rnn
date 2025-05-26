# Code written by HLD based on/ adapted from the following TF1 code:
# https://github.com/mhibatallah/RNNWavefunctions/blob/master/2DTFIM_1DRNN/RNNwavefunction.py

import tensorflow as tf
import keras
import numpy as np
import random
from hyprnn_impl import *
from hyp_util import *

def sqsoftmax(inputs):
    return tf.sqrt(tf.nn.softmax(inputs))

def softsign_(inputs):
    return np.pi*(tf.nn.softsign(inputs))

def heavyside(inputs):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) 
    return 0.5*(sign+1.0)

class wfModel(keras.Model):
    def __init__(self, rnn_layer, dense_amp, dense_phase, is_hyp):
        super().__init__()
        self.rnn = rnn_layer
        self.dense_a = dense_amp
        self.dense_p = dense_phase
        self.hyp = is_hyp

    def call(self, inputs, rnn_state, compute_phase):
        if not self.hyp: 
            rnn_output, rnn_state = self.rnn(inputs, rnn_state) 
        if self.hyp:
            rnn_outp, rnn_state = self.rnn(inputs, rnn_state)
            rnn_output = tf_log_map_zero(rnn_outp, 1.0)

        output_a=self.dense_a(rnn_output) 
        if not compute_phase:   
            return output_a, rnn_state
        if compute_phase:
            output_p = self.dense_p(rnn_output)
            return output_a, output_p, rnn_state

class rnn_eucl_wf(object):
    def __init__(self,systemsize,cell_type, units, seed=111):
        """
            systemsize:  int, size of the lattice
            cell_type:    Euclidean or hyperbolic cell 
            hyp_non_lin: nonlinearity activation of the hyperbolic cell ('tanh'/'id')
            units:       list of int
                         number of units per RNN layer
            seed:       pseudo-random number generator
        """
        self.N=systemsize #Number of sites of the 1D chain
        self.dtype = tf.float32
        self.units = units
        self.inputs_geom = 'eucl'
        
        #Seeding
        random.seed(seed) 
        np.random.seed(seed) 

        #Defining the neural network
        tf.random.set_seed(seed)  

        if cell_type == 'EuclRNN':
            self.rnn = EuclRNN(self.units, dtype = self.dtype)
            self.name = f'{cell_type}_{self.units}'
        if cell_type == 'EuclGRU':
            self.rnn = EuclGRU(self.units,dtype = self.dtype)
            self.name = f'{cell_type}_{self.units}'
        #Define the Fully-Connected layer followed by a square root of Softmax
        self.dense_ampl = tf.keras.layers.Dense(2,activation=sqsoftmax,name='wf_dense_ampl') 
        #Define the Fully-Connected layer followed by a Softsign*pi
        self.dense_phase = tf.keras.layers.Dense(2,activation=softsign_,name='wf_dense_phase')
        self.model = wfModel(self.rnn, self.dense_ampl, self.dense_phase, is_hyp = False)

    def sample(self,numsamples,inputdim):
        """
            Generate samples from a probability distribution parametrized by a recurrent network
            We also impose zero magnetization (U(1) symmetry)
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        a=tf.zeros(numsamples, dtype=tf.float32)
        b=tf.zeros(numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)
        #Initial input sigma_0 to feed to the cRNN

        self.inputdim=inputs.shape[1]
        self.outputdim=self.inputdim
        self.numsamples=inputs.shape[0]

        samples=[]
        #Define the initial hidden state of the RNN
        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) 
        inputs_ampl = inputs

        #Need to call sample once to build the rnn layers
        o2, rs2 = self.rnn(inputs_ampl, rnn_state)

        for n in range(self.N):
            output_ampl, rnn_state = self.model.call(inputs_ampl, rnn_state, compute_phase=False)

            if n>=self.N/2: #Enforcing zero magnetization
                num_up = tf.cast(tf.reduce_sum(tf.stack(values=samples,axis=1), axis = 1), tf.float32)
                baseline = (self.N//2-1)*tf.ones(shape = [self.numsamples], dtype = tf.float32)
                num_down = n*tf.ones(shape = [self.numsamples], dtype = tf.float32) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float32)
                output_ampl = tf.math.l2_normalize(output_ampl, axis = 1, epsilon = 1e-30) #l2 normalizing
            #tf.categorical(logits [batchsize, numclasses], numsamples) produces samples of shape [batchsize, numsamples]
            #with integer value ranging from 0 to numclasses.
            sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output_ampl**2),num_samples=1),[-1,])
            samples.append(sample_temp)
            # 1-> (0,1), 0 -> (1,0)
            inputs=tf.one_hot(sample_temp,depth=self.outputdim)

            inputs_ampl = inputs

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_amplitude(self,samples,inputdim):
        """
            calculate the log-ampliturdes of ```samples`` while imposing zero magnetization
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-amps      tf.Tensor of shape (number of samples,)
                             the log-amplitude of each sample
            """

        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=tf.shape(samples)[0]
        a=tf.zeros(self.numsamples, dtype=tf.float32)
        b=tf.zeros(self.numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)
        amplitudes=[]

        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) 
        inputs_ampl = inputs

        #Need to call sample once to build the hyp-rnn layers
        o2, rs2 = self.rnn(inputs_ampl, rnn_state)

        for n in range(self.N):
            output_ampl, output_phase, rnn_state = self.model.call(inputs_ampl, rnn_state, compute_phase=True)

            if n>=self.N/2: #Enforcing zero magnetization
                num_up = tf.cast(tf.reduce_sum(tf.slice(samples,begin=[np.int32(0),np.int32(0)],size=[np.int32(-1),np.int32(n)]),axis=1), tf.float32)
                baseline = (self.N//2-1)*tf.ones(shape = [self.numsamples], dtype = tf.float32)
                num_down = n*tf.ones(shape = [self.numsamples], dtype = tf.float32) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float32)
                output_ampl = tf.math.l2_normalize(output_ampl, axis = 1, epsilon = 1e-30) #l2 normalizing

            amplitude = tf.complex(output_ampl,0.0)*tf.exp(tf.complex(0.0,output_phase)) #You can add a bias

            amplitudes.append(amplitude)

            inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
            inputs_ampl = inputs

        amplitudes=tf.stack(values=amplitudes,axis=1) # (self.N, num_samples,2) to (num_samples, self.N, 2): Generate self.numsamples vectors of size (self.N, 2) spin containing the log_amplitudes of each sample
        one_hot_samples=tf.one_hot(samples,depth=self.inputdim)

        self.log_amplitudes = tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(amplitudes,tf.complex(one_hot_samples,tf.zeros_like(one_hot_samples))),axis=2)),axis=1) #To get the log amplitude of each sample

        return self.log_amplitudes

class rnn_hyp_wf(object):
    def __init__(self,systemsize,cell_type, bias_geom, hyp_non_lin, units, seed=111):
        """
            systemsize:  int, size of the lattice
            cell_type:    yperbolic cell (HypRNN, HypGRU only)
            bias_geom:   'eucl' or 'hyp' 
            hyp_non_lin: nonlinearity activation of the hyperbolic cell ('tanh'/'id')
            units:       list of int
                         number of units per RNN layer
            seed:       pseudo-random number generator
        """
        self.N=systemsize #Number of sites of the 1D chain
        self.dtype = tf.float32
        self.units = units
        self.inputs_geom = 'eucl'
        self.bias_geom = bias_geom
        self.h_non_lin = hyp_non_lin
        #Seeding
        random.seed(seed) 
        np.random.seed(seed) 

        #Defining the neural network
        tf.random.set_seed(seed)  

        if cell_type =='HypRNN':
            self.rnn = HypRNN(self.units, self.inputs_geom, self.bias_geom, 1.0, self.h_non_lin, dtype = self.dtype)
            self.name = f'{cell_type}_{self.units}_{self.h_non_lin}_{self.bias_geom}'
        if cell_type == 'HypGRU':
            self.rnn = HypGRU(self.units,self.inputs_geom, self.bias_geom, 1.0, self.h_non_lin, dtype = self.dtype)
            self.name = f'{cell_type}_{self.units}_{self.h_non_lin}_{self.bias_geom}'

        self.dense_ampl = tf.keras.layers.Dense(2,activation=sqsoftmax,name='wf_dense_ampl') 
        self.dense_phase = tf.keras.layers.Dense(2,activation=softsign_,name='wf_dense_phase')
        self.model = wfModel(self.rnn, self.dense_ampl, self.dense_phase, is_hyp = True)

    def sample(self,numsamples,inputdim):
        """
            Generate samples from a probability distribution parametrized by a recurrent network
            We also impose zero magnetization (U(1) symmetry)
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        a=tf.zeros(numsamples, dtype=tf.float32)
        b=tf.zeros(numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)
        #Initial input sigma_0 to feed to the cRNN

        self.inputdim=inputs.shape[1]
        self.outputdim=self.inputdim
        self.numsamples=inputs.shape[0]

        samples=[]

        #rnn_state = self.rnn.zero_state(self.numsamples,dtype=tf.float32) #Define the initial hidden state of the RNN
        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) 
        inputs_ampl = inputs

        #Need to call sample once to build the hyp-rnn layers
        o2, rs2 = self.rnn(inputs_ampl, rnn_state)

        for n in range(self.N):
            output_ampl, rnn_state = self.model.call(inputs_ampl, rnn_state, compute_phase=False)

            if n>=self.N/2: #Enforcing zero magnetization
                num_up = tf.cast(tf.reduce_sum(tf.stack(values=samples,axis=1), axis = 1), tf.float32)
                baseline = (self.N//2-1)*tf.ones(shape = [self.numsamples], dtype = tf.float32)
                num_down = n*tf.ones(shape = [self.numsamples], dtype = tf.float32) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float32)
                output_ampl = tf.math.l2_normalize(output_ampl, axis = 1, epsilon = 1e-30) #l2 normalizing
            #tf.categorical(logits [batchsize, numclasses], numsamples) produces samples of shape [batchsize, numsamples]
            #with integer value ranging from 0 to numclasses.
            sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output_ampl**2),num_samples=1),[-1,])
            samples.append(sample_temp)
            # 1-> (0,1), 0 -> (1,0)
            inputs=tf.one_hot(sample_temp,depth=self.outputdim)

            inputs_ampl = inputs

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_amplitude(self,samples,inputdim):
        """
            calculate the log-ampliturdes of ```samples`` while imposing zero magnetization
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-amps      tf.Tensor of shape (number of samples,)
                             the log-amplitude of each sample
            """

        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=tf.shape(samples)[0]
        a=tf.zeros(self.numsamples, dtype=tf.float32)
        b=tf.zeros(self.numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)
        amplitudes=[]

        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) 
        inputs_ampl = inputs

        #Need to call sample once to build the hyp-rnn layers
        o2, rs2 = self.rnn(inputs_ampl, rnn_state)

        for n in range(self.N):
            output_ampl, output_phase, rnn_state = self.model.call(inputs_ampl, rnn_state, compute_phase=True)

            if n>=self.N/2: #Enforcing zero magnetization
                num_up = tf.cast(tf.reduce_sum(tf.slice(samples,begin=[np.int32(0),np.int32(0)],size=[np.int32(-1),np.int32(n)]),axis=1), tf.float32)
                baseline = (self.N//2-1)*tf.ones(shape = [self.numsamples], dtype = tf.float32)
                num_down = n*tf.ones(shape = [self.numsamples], dtype = tf.float32) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float32)
                output_ampl = tf.math.l2_normalize(output_ampl, axis = 1, epsilon = 1e-30) #l2 normalizing

            amplitude = tf.complex(output_ampl,0.0)*tf.exp(tf.complex(0.0,output_phase)) #You can add a bias

            amplitudes.append(amplitude)

            inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
            inputs_ampl = inputs

        amplitudes=tf.stack(values=amplitudes,axis=1) # (self.N, num_samples,2) to (num_samples, self.N, 2): Generate self.numsamples vectors of size (self.N, 2) spin containing the log_amplitudes of each sample
        one_hot_samples=tf.one_hot(samples,depth=self.inputdim)

        self.log_amplitudes = tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(amplitudes,tf.complex(one_hot_samples,tf.zeros_like(one_hot_samples))),axis=2)),axis=1) #To get the log amplitude of each sample

        return self.log_amplitudes
