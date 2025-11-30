#Code written by HLD for arXiv:2505.22083 
#[Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz]

#This is the TF2 rewritten and adapted version of the following TF1 code:
#https://github.com/mhibatallah/RNNWavefunctions/blob/master/2DTFIM_2DRNN/RNNwavefunction.py

import tensorflow as tf
import numpy as np
import random
import keras
from tfim2d_MDRNNcell import *

class wfModel(keras.Model):
    def __init__(self, rnn_layer, dense_layer):
        super().__init__()
        self.rnn = rnn_layer
        self.dense = dense_layer

    def call(self, inputs, rnn_state):
        rnn_output, rnn_state = self.rnn(inputs, rnn_state)
        output=self.dense(rnn_output)
        return output, rnn_state

class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y, _units, _num_in,  seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            units:       list of int
                         number of units per RNN layer

            seed:        pseudo-random number generator 
        """
        self.Nx=systemsize_x #size of x direction in the 2d model
        self.Ny=systemsize_y
        self.units = _units
        self.num_in = _num_in
        self.name = f'2DRNN_Nx={self.Nx}_Ny={self.Ny}_u={self.units}'

        np.random.seed(seed)  # numpy pseudo-random generator
        tf.random.set_seed(seed) 
        #Defining the neural network   
        self.rnn=MDRNNcell(self.units, self.num_in)
        self.dense = tf.keras.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense') 
        self.model = wfModel(self.rnn, self.dense)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int                 
                             samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
        """

        #Initial input to feed to the 2drnn

        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=numsamples

        samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        rnn_states = {}
        inputs = {}

        for ny in range(self.Ny): #Loop over the boundary
            if ny%2==0:
                nx = -1
                #print(nx,ny)
                rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples,self.units) ,dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64)
            if ny%2==1:
                nx = self.Nx
                #print(nx,ny)
                rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples,self.units),dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64)

            for nx in range(self.Nx): #Loop over the boundary
                ny = -1
                #print(nx, ny)
                rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples,self.units),dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 

        #print(rnn_states.keys())
        #print(inputs.keys())

        #Begin sampling
        for ny in range(self.Ny):
            if ny%2 == 0:
                for nx in range(self.Nx): #left to right
                    output, rnn_states[str(nx)+str(ny)] = self.model.call((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])
                    samples[nx][ny] = sample_temp
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

            if ny%2 == 1:
                for nx in range(self.Nx-1,-1,-1): #right to left
                    output, rnn_states[str(nx)+str(ny)] = self.model.call((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])
                    samples[nx][ny] = sample_temp
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

        self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,0,1])

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x,system_size_y)
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
        self.outputdim=self.inputdim


        samples_=tf.transpose(samples, perm = [1,2,0])
        rnn_states = {}
        inputs = {}

        for ny in range(self.Ny): #Loop over the boundary
            if ny%2==0:
                nx = -1
                rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples, self.units),dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 

            if ny%2==1:
                nx = self.Nx
                rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples, self.units),dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 


        for nx in range(self.Nx): #Loop over the boundary
            ny = -1
            rnn_states[str(nx)+str(ny)]=tf.zeros((self.numsamples, self.units),dtype=tf.float64)
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) 

        probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        #Begin estimation of log probs
        for ny in range(self.Ny):
            if ny%2 == 0:
                for nx in range(self.Nx): #left to right
                    output, rnn_states[str(nx)+str(ny)] = self.model.call((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]),
                                                                            (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])
                    probs[nx][ny] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

            if ny%2 == 1:
                for nx in range(self.Nx-1,-1,-1): #right to left
                    output, rnn_states[str(nx)+str(ny)] = self.model.call((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]),
                                                                        (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])
                    probs[nx][ny] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

        probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,0,1,3])
        probs = tf.cast(probs, dtype = tf.float64)
        one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)


        self.log_probs=tf.reduce_sum(tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)

        return self.log_probs
