#Code written by HLD for arXiv:2505.22083 
#[Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz]

#This is the TF2 rewritten and adapted version of the following TF1 code:
#https://github.com/mhibatallah/RNNWavefunctions/tree/master/2DTFIM_1DRNN/Training1DRNN_2DTFIM.py

import tensorflow as tf
import numpy as np
import os
import time
import random
from math import ceil

from tfim2d_1drnn import *
from hyprnn_wf import *
from hyp_rsgd import RSGD


# Loading Functions --------------------------
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, wf):
    """ To get the local energies of 2D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx*Ny)
    - Jz: (Nx*Ny) np array
    - Bx: float
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    """
    numsamples = samples.shape[0]
    queue_samples=np.zeros(((Nx*Ny+1),numsamples, Nx*Ny), dtype = np.int32)
    samples_reshaped = np.reshape(samples, [numsamples, Nx, Ny])

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples_reshaped[:,i]+samples_reshaped[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice :)))
        values = samples_reshaped[:,:,i]+samples_reshaped[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i]), axis = 1)


    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip
            valuesT[:,i][samples[:,i]==0] = 1 #Flip

            queue_samples[i+1] = valuesT

    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/numsamples) #Get a maximum of 25000 configurations in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
    log_probs = []
    for i in range(steps):
        if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
        log_probs_i = wf.log_probability(queue_samples_reshaped[cut].astype(int),2)
        log_probs.append(log_probs_i)

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
    local_energies += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,:]-0.5*log_probs_reshaped[0,:]), axis = 0) #This is faster than previous loop, since it runs in parallel

    return local_energies
#--------------------------
def cost_fn(Eloc, log_probs_):
    #cost = tf.reduce_mean(tf.multiply(log_probs_,tf.stop_gradient(Eloc))) - tf.reduce_mean(tf.stop_gradient(Eloc))*tf.reduce_mean(log_probs_)
        cost = tf.reduce_mean(tf.multiply(log_probs_,Eloc)) - tf.reduce_mean(Eloc)*tf.reduce_mean(log_probs_)
        return cost

def train_step(wf, numsamples, input_dim, Jz, Nx, Ny, Bx, optimizer):
    with tf.GradientTape() as tape:
        wfmodel = wf.model
        tsamp = wf.sample(numsamples, input_dim)
        log_prob_tensor = wf.log_probability(tsamp,input_dim)
        total_vars = wfmodel.trainable_variables
        Eloc = Ising2D_local_energies(Jz, Bx, Nx, Ny, tsamp, wf)
        loss_value = cost_fn(Eloc, log_prob_tensor)

    grads = tape.gradient(loss_value, total_vars)
    optimizer.apply_gradients(zip(grads, total_vars))
    return loss_value, Eloc, wfmodel

def train_step_with_hyp_vars(wf, numsamples, input_dim, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp):
        with tf.GradientTape(persistent=True) as tape:
            wfmodel = wf.model
            tsamp = wf.sample(numsamples, input_dim)
            log_prob_tensor = wf.log_probability(tsamp,input_dim)
            Eloc = Ising2D_local_energies(Jz, Bx, Nx, Ny, tsamp, wf)
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

# ---------------- ----------Running VMC with Euclidean RNNs -------------------------------------
def run_2DTFIM(numsteps = 201, Nx = 5, Ny = 5, Bx = +3, cell = 'EuclGRU',
                num_units = 50, num_layers = 1, numsamples = 50, learningrate = 1e-3,
                var_tol=0.5, seed = 111, fname = 'results/'):
    #Seeding
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    Jz = +np.ones((Nx,Ny)) #Ferromagnetic couplings
    lr=np.float64(learningrate)
    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    wf=RNNwavefunction(Nx,Ny, cell, num_units, seed = seed)

    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learningrate,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer

    #--------------------------
    meanEnergy=[]
    varEnergy=[]
    
    best_E_list=[]
    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, input_dim, Jz, Nx, Ny, Bx, optimizer)
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

# ---------------- ----------Running VMC with hyperbolic RNNs -------------------------------------
def run_2DTFIM_hyp(numsteps = 201, Nx = 5, Ny = 5, Bx = +3, cell = 'HypGRU',
                num_units = 50, num_layers = 1, numsamples = 50, lr1=1e-2, lr2 =1e-2,
                var_tol=0.5, seed = 111, fname = 'results/'):
    #Seeding
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    Jz = +np.ones((Nx,Ny)) #Ferromagnetic couplings
    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    wf=RNNwavefunction_hyp(Nx,Ny, cell, 'hyp', 'id', num_units, seed = seed)

    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr1,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    opt_eucl=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay)
    opt_hyp=RSGD('rsgd', learning_rate=lr2)

    #--------------------------
    meanEnergy=[]
    varEnergy=[]
    best_E_list=[]
    for step in range(numsteps):
        cost, E, wfmodel = train_step_with_hyp_vars(wf, numsamples, input_dim, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp)
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


