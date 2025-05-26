# Code written by HLD
# Adapted from https://github.com/mhibatallah/RNNWavefunctions/blob/master/1DTFIM/RNNwavefunction.py

import tensorflow as tf
import numpy as np
import random
from hyprnn_impl import *
from hyp_util import *

class wfModel(keras.Model):
    def __init__(self, rnn_layer, dense_layer, is_hyp):
        super().__init__()
        self.rnn = rnn_layer
        self.dense = dense_layer
        self.hyp = is_hyp

    def call(self, inputs, rnn_state):
        if not self.hyp:
            rnn_output, rnn_state = self.rnn.call(inputs, rnn_state)
        if self.hyp:
            rnn_outp, rnn_state = self.rnn.call(inputs, rnn_state)
            rnn_output = tf_log_map_zero(rnn_outp, 1.0)
        output=self.dense(rnn_output)
        return output, rnn_state

class rnn_eucl_wf(object):
    def __init__(self,systemsize,cell_type, units, seed = 111):
        """
            systemsize:  int, number of sites  

            cell_type:  EuclGRU or EuclRNN (Euclidean type only)
    
            units:       list of int
                         number of units per RNN layer
 
            seed:        pseudo-random number generator           
        """
        self.cell_type = cell_type
        self.units = units
        self.N=systemsize #Number of sites of the 1D chain
        self.dtype = tf.float32
        if cell_type == 'EuclRNN':
            self.cell = EuclRNN(self.units, dtype = self.dtype)
            self.name = f'N{self.N}_{cell_type}_{self.units}'
        if cell_type == 'EuclGRU':
            self.cell = EuclGRU(self.units,dtype = self.dtype)
            self.name = f'N{self.N}_{cell_type}_{self.units}'

        random.seed(seed) 
        np.random.seed(seed) 

        #Defining the neural network   
        tf.random.set_seed(seed) 
        self.rnn=self.cell
        self.dense = tf.keras.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense') 
        self.model = wfModel(self.rnn, self.dense, is_hyp = False)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
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
    
        samples = []

        b=np.zeros((numsamples,inputdim)).astype(np.float64)
        #b = state of sigma_0 for all the samples
        inputs=tf.constant(dtype=tf.float32,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
        #Initial input to feed to the rnn

        self.inputdim=inputs.shape[1]
        self.outputdim=self.inputdim
        self.numsamples=inputs.shape[0]

        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) #Initialize the RNN hidden state
        # Need to run the input & rnn_state to the rnn layer to build it
        o2, s2 = self.rnn(inputs, rnn_state)

        for n in range(self.N):

            output, rnn_state =self.model.call(inputs, rnn_state) 
            sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
            samples.append(sample_temp)
            inputs=tf.one_hot(sample_temp,depth=self.outputdim)

        self.samples=tf.stack(values=samples,axis=1) 

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
   
        self.inputdim=inputdim
        self.outputdim=self.inputdim

        self.numsamples=tf.shape(samples)[0]
        a=tf.zeros(self.numsamples, dtype=tf.float32)
        b=tf.zeros(self.numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)

            
        probs=[]
        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32)

        for n in range(self.N):  
            #rnn_output, rnn_state = self.rnn.call(inputs, rnn_state)
            #output=self.dense(rnn_output)
            output, rnn_state =self.model.call(inputs, rnn_state)
            probs.append(output)
            inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),
                                    np.int32(1)]),shape=[self.numsamples]),
                                    depth=self.outputdim),shape=[self.numsamples,self.inputdim])

        probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
        one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.log_probs

class rnn_hyp_wf(object):
    def __init__(self,systemsize,cell_type, bias_geo, hyp_non_lin, units, seed = 111):
        """
            systemsize:  int, number of sites  

            cell_type:  HypGRU and HypRNN _init_: (num_units, inputs_geom='eucl', bias_geom, c_val=1,
                        non_lin, fix_biases=False, fix_matrices=False, matrices_init_eye=False, dtype). 
                        See hyprnn_impl.py file. 

            hyp_non_lin: 'non_lin' argument in cell_type above - the nonlinearity function for the HNN
                         either 'id' for identity or 'tanh'.

            bias_geo: bias_geom argument in 'cell_type' , either 'hyp' or 'eucl'
                      If 'bias_geom' is chosen to be 'hyp', need to optimize separately using 
                     a Riemannian adaptation of conventional optimizers (e.g. RSGD)

            units:       list of int
                         number of units per RNN layer
 
            seed:        pseudo-random number generator           
        """
        self.cell_type = cell_type
        self.units = units
        self.N=systemsize 
        self.dtype = tf.float32
        self.h_non_lin = hyp_non_lin
        self.h_bias_geo = bias_geo
        if cell_type == 'HypRNN':
            self.cell = HypRNN(self.units,'eucl', self.h_bias_geo, 1, 
                                        self.h_non_lin,  dtype = self.dtype)
            self.name = f'N{self.N}_{cell_type}_{self.units}_{self.h_bias_geo}_{self.h_non_lin}'
    
        if cell_type == 'HypGRU':
            self.cell = HypGRU(self.units,'eucl', self.h_bias_geo, 1, 
                                       self.h_non_lin,  dtype = self.dtype)
            self.name = f'N{self.N}_{cell_type}_{self.units}_{self.h_bias_geo}_{self.h_non_lin}'
        random.seed(seed) 
        np.random.seed(seed) 
  
        tf.random.set_seed(seed) 
        self.rnn=self.cell
        self.dense = tf.keras.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense') 
        #Define the Fully-Connected layer followed by a Softmax
        self.model = wfModel(self.rnn, self.dense, is_hyp=True)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a hyperbolic recurrent network
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
    
        samples = []

        b=np.zeros((numsamples,inputdim)).astype(np.float64)
        #b = state of sigma_0 for all the samples
        inputs=tf.constant(dtype=tf.float32,value=b,shape=[numsamples,inputdim]) 
        #Initial input to feed to the rnn

        self.inputdim=inputs.shape[1]
        self.outputdim=self.inputdim
        self.numsamples=inputs.shape[0]

        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32) #Initialize the RNN hidden state
        # Need to run the input & rnn_state to the rnn layer to build it
        o2, s2 = self.rnn(inputs, rnn_state)

        for n in range(self.N):
            output, rnn_state =self.model.call(inputs, rnn_state) 
            sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
            samples.append(sample_temp)
            inputs=tf.one_hot(sample_temp,depth=self.outputdim)

        self.samples=tf.stack(values=samples,axis=1) 

        return self.samples

    def log_probability(self,samples,inputdim):
   
        self.inputdim=inputdim
        self.outputdim=self.inputdim

        self.numsamples=tf.shape(samples)[0]
        a=tf.zeros(self.numsamples, dtype=tf.float32)
        b=tf.zeros(self.numsamples, dtype=tf.float32)

        inputs=tf.stack([a,b], axis = 1)

            
        probs=[]
        rnn_state=tf.zeros((self.numsamples, self.units),dtype=tf.float32)

        for n in range(self.N):  
            output, rnn_state =self.model.call(inputs, rnn_state) 
            probs.append(output)
            inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])

        probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
        one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.log_probs
