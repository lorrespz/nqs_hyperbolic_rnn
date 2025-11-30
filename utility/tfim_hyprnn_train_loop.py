#Code written by HLD for arXiv:2505.22083 
#[Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz]

#This is the TF2 rewritten and adapted version of the following TF1 code:
#https://github.com/mhibatallah/RNNWavefunctions/blob/master/1DTFIM/TrainingRNN_1DTFIM.py

import tensorflow as tf
import keras
import numpy as np
import os
import time
import sys
import random
from math import ceil

from hyprnn_wf import *
from hyp_rsgd import RSGD

#Check GPU:
#https://www.tensorflow.org/guide/gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Loading Functions --------------------------
def Ising_local_energies(Jz, Bx, samples, wf):
    """ To get the local energies of 1D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, N)
    - Jz: (N) np array
    - Bx: float
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]

    queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) 
    #Array to store all the diagonal and non diagonal matrix elements 

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1): #diagonal elements
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += valuesT*(-Jz[i])

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip
            valuesT[:,i][samples[:,i]==0] = 1 #Flip

            queue_samples[i+1] = valuesT


    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/numsamples) 

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])

    log_probs = []
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs_i = wf.log_probability(queue_samples_reshaped[cut].astype(int), 2)
      log_probs.append(log_probs_i)

    log_probs=np.array(log_probs)
    local_energies += -Bx*np.sum(np.exp(0.5*log_probs[1:,:]-0.5*log_probs[0,:]), axis = 0) 

    return local_energies
#----------------------------------------------------------------------------------------------------
def cost_fn(Eloc, log_probs_):
        cost = tf.reduce_mean(tf.multiply(log_probs_,Eloc)) - tf.reduce_mean(Eloc)*tf.reduce_mean(log_probs_)
        return cost
#----------------------------------------------------------------------------------------------------
def train_step(wf, numsamples, input_dim, Jz, Bx, optimizer):
    with tf.GradientTape() as tape:
        wfmodel = wf.model
        tsamp = wf.sample(numsamples, input_dim)
        log_prob_tensor = wf.log_probability(tsamp,input_dim)
        total_vars = wfmodel.trainable_variables
        Eloc = Ising_local_energies(Jz, Bx, tsamp, wf)
        loss_value = cost_fn(Eloc, log_prob_tensor)

    grads = tape.gradient(loss_value, total_vars)
    optimizer.apply_gradients(zip(grads, total_vars))
    return loss_value, Eloc, wfmodel
#----------------------------------------------------------------------------------------------------
def train_step_with_hyp_vars(wf, numsamples, input_dim, Jz, Bx, opt_eucl, opt_hyp):
        with tf.GradientTape(persistent=True) as tape:
            wfmodel = wf.model
            tsamp = wf.sample(numsamples, input_dim)            
            log_prob_tensor = wf.log_probability(tsamp,input_dim)          
            Eloc = Ising_local_energies(Jz, Bx, tsamp, wf)
            loss_value = cost_fn(Eloc, log_prob_tensor)

            wf_hyp_vars = wf.model.rnn.hyp_vars
            wf_eucl_vars = []
            for var in wf.model.rnn.eucl_vars:
                wf_eucl_vars.append(var)
            for var in wf.model.dense.trainable_variables:
                wf_eucl_vars.append(var)          
   
        grads_eucl = tape.gradient(loss_value, wf_eucl_vars)
        grads_hyp = tape.gradient(loss_value, wf_hyp_vars)
        opt_eucl.apply_gradients(zip(grads_eucl, wf_eucl_vars))
        opt_hyp.apply_gradients(grads_hyp, wf_hyp_vars)
        return loss_value, Eloc, wfmodel

# ---------------- Running VMC with RNNs -------------------------------------
def run_1DTFIM(numsteps, wf,  systemsize, num_units, var_tol, Bx = 1, 
                numsamples = 50, learningrate=1e-2, seed = 111, fname = 'results'):

    with tf.device('/GPU:0'):

        random.seed(seed) 
        np.random.seed(seed) 
        tf.random.set_seed(seed)  
        N = systemsize
        Jz = +np.ones(systemsize) #Ferromagnetic coupling
        input_dim=2 #Dimension of the Hilbert space   
  
        learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learningrate,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer

        meanEnergy=[]
        varEnergy=[]
        best_E_list=[]
        for step in range(numsteps):
            cost, E, wfmodel = train_step(wf, numsamples, input_dim, Jz, Bx, optimizer)
            meanE = np.mean(E)
            varE = np.var(E)
            meanEnergy.append(meanE)
            varEnergy.append(varE)

            np.save(f'{fname}/{wf.name}_ns{numsamples}_meanE.npy',meanEnergy)
            np.save(f'{fname}/{wf.name}_ns{numsamples}_varE.npy',varEnergy)
            if step==0:
                best_E = meanE
                best_E_list.append(best_E)
            if step !=0:
                if np.real(meanE) < min(np.real(best_E_list)) and  np.abs(varE) < var_tol:
                    best_E = meanE
                    best_E_list.append(meanE)
                    #only save best models
                    wfmodel.save_weights(f'{fname}/{wf.name}_ns{numsamples}_checkpoint.weights.h5') 
                    print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
            if step %10 ==0:
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}')   
                
        return meanEnergy, varEnergy

   
# ---------------- Running VMC with hyperbolic RNNs -------------------------------------
def run_1DTFIM_hypvars(numsteps, wf, systemsize, num_units, var_tol, Bx = 1, 
                numsamples = 50, lr1=1e-2, lr2 =1e-2, seed = 111, fname = 'results'):

    random.seed(seed)  
    np.random.seed(seed)  
    tf.random.set_seed(seed) 
    N = systemsize
    Jz = +np.ones(systemsize) 
    input_dim=2 
  
    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr1,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    opt_eucl=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) 
    opt_hyp=RSGD('rsgd', learning_rate=lr2)

    meanEnergy=[]
    varEnergy=[]
    best_E_list=[]
    max_patience = 20
    patience = 0
    for step in range(numsteps):
        cost, E, wfmodel = train_step_with_hyp_vars(wf, numsamples, input_dim, Jz, Bx, opt_eucl, opt_hyp)
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)

        np.save(f'{fname}/{wf.name}_ns{numsamples}_meanE.npy',meanEnergy)
        np.save(f'{fname}/{wf.name}_ns{numsamples}_varE.npy',varEnergy)
        if step==0:
            best_E = meanE
            best_E_list.append(best_E)
        if step !=0:
            if np.real(meanE) < min(np.real(best_E_list)) and  np.abs(varE) < var_tol:
                best_E = meanE
                best_E_list.append(meanE)
                #only save best models
                wfmodel.save_weights(f'{fname}/{wf.name}_ns{numsamples}_checkpoint.weights.h5') 
                print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
        if step %10 ==0:
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}')
              
                
    return meanEnergy, varEnergy
    #----------------------------

