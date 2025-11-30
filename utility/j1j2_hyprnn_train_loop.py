#Code written by HLD for arXiv:2505.22083 
#[Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz]

#This is the TF2 rewritten and adapted version of the TF1 code found at:
#https://github.com/mhibatallah/RNNWavefunctions/tree/master/J1J2

import tensorflow as tf
import keras
import numpy as np
import os
import time
import random
import sys
from math import ceil

from j1j2_hyprnn_wf import *
from hyp_rsgd import RSGD

#Check GPU:
#https://www.tensorflow.org/guide/gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Loading Functions --------------------------
def J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements, periodic = False, Marshall_sign = False):
    N=len(Bz)
    #the diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag=np.dot(np.float32(sigmap)-0.5,Bz)

    num = 0 #Number of basis elements
    if periodic:
        limit = N
    else:
        limit = N-1    
 
    if periodic:
        limit2 = N
    else:
        limit2 = N-2     

    for site in range(limit):
        if sigmap[site]!=sigmap[(site+1)%N]: #if the two neighouring spins are opposite
            diag-=0.25*J1[site] #add a negative energy contribution
        else:
            diag+=0.25*J1[site]
    
    for site in range(limit2):
        if J2[site] != 0.0:
            if sigmap[site]!=sigmap[(site+2)%N]: #if the two second neighouring spins are opposite
                diag-=0.25*J2[site] #add a negative energy contribution
            else:
                diag+=0.25*J2[site]

    matrixelements[num] = diag #add the diagonal part to the matrix elements
    sig = np.copy(sigmap)
    sigmaH[num] = sig

    num += 1
    #off-diagonal part:
    for site in range(limit):
        if J1[site] != 0.0:
          if sigmap[site]!=sigmap[(site+1)%N]:
              sig=np.copy(sigmap)
              sig[site]=sig[(site+1)%N] #Make the two neighbouring spins equal.
              sig[(site+1)%N]=sigmap[site]
              sigmaH[num] = sig #The last three lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)

              if Marshall_sign:
                  matrixelements[num] = -J1[site]/2
              else:
                  matrixelements[num] = +J1[site]/2

              num += 1

    for site in range(limit2):
      if J2[site] != 0.0:
        if sigmap[site]!=sigmap[(site+2)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+2)%N] #Make the two next-neighbouring spins equal.
            sig[(site+2)%N]=sigmap[site]
            sigmaH[num] = sig #The last three lines are meant to flip the two next-neighbouring spins (that the effect of applying J+ and J-)
            matrixelements[num] = +J2[site]/2
            num += 1
    return num

def J1J2Slices(J1, J2, Bz, sigmasp, sigmas, H, sigmaH, matrixelements, Marshall_sign):
    slices=[]
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements, Marshall_sign)
        #note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]
        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]
        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length

#---------------------------------------------------------------------------------------
def J1J2_local_energies(wf, N,J1,J2,Bz,numsamples, samples, Marshall_sign):
    local_energies = np.zeros(numsamples, dtype = np.complex64)
    #Array to store all the diagonal and non diagonal log_probabilities for all the samples
    log_amplitudes = np.zeros(2*N*numsamples, dtype=np.complex64) 
    #Array to store all the diagonal and non diagonal sigmas for all the samples 
    sigmas=np.zeros((2*N*numsamples,N), dtype=np.int32)
    #Array to store all the diagonal and non diagonal matrix elements for all the samples 
    H = np.zeros(2*N*numsamples, dtype=np.float32)
    #Array to store all the diagonal and non diagonal sigmas for each sample sigma
    sigmaH = np.zeros((2*N,N), dtype = np.int32) 
    #Array to store all the diagonal and non diagonal matrix elements for each sample sigma 
    #(the number of matrix elements is bounded by at most 2N)
    matrixelements=np.zeros(2*N, dtype = np.float32)
    #Getting the sigmas with the matrix elements
    slices, len_sigmas = J1J2Slices(J1,J2,Bz,samples, sigmas, H, sigmaH, matrixelements, Marshall_sign)

    #Process in steps to get log amplitudes
    steps = ceil(len_sigmas/numsamples)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)
        log_amps=wf.log_amplitude(sigmas[cut],inputdim=2)    
        log_amplitudes[cut] = log_amps

    #Generating the local energies
    for n in range(len(slices)):
        s=slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

    return local_energies

#----------------------------------------------------------------------------------------------------
def cost_fn(Eloc, log_amplitudes_):
        #cost = tf.reduce_mean(tf.multiply(log_probs_,Eloc)) - tf.reduce_mean(Eloc)*tf.reduce_mean(log_probs_)
        cost = 2*tf.math.real(tf.reduce_mean(tf.math.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) 
                - tf.math.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
        return cost
#----------------------------------------------------------------------------------------------------
def train_step(wf, numsamples, input_dim, J1, J2, Bz, Marshall_sign, optimizer):
        with tf.GradientTape() as tape:
            wfmodel = wf.model
            tsamp = wf.sample(numsamples, input_dim)
            log_amplitudes_ = wf.log_amplitude(tsamp,input_dim)
            total_vars = wfmodel.trainable_variables
            N = tsamp.shape[1]
            Eloc = J1J2_local_energies(wf,N,J1,J2,Bz,numsamples, tsamp, Marshall_sign)
            loss_value = cost_fn(Eloc, log_amplitudes_)

        grads = tape.gradient(loss_value, total_vars)
        optimizer.apply_gradients(zip(grads, total_vars))
        return loss_value, Eloc, wfmodel

#----------------------------------------------------------------------------------------------------
def train_step_with_hypvars(wf, numsamples, input_dim, J1, J2, Bz, Marshall_sign, opt_eucl, opt_hyp):
        with tf.GradientTape(persistent=True) as tape:
            wfmodel = wf.model
            tsamp = wf.sample(numsamples, input_dim)
            log_amplitudes_ = wf.log_amplitude(tsamp,input_dim)
            N = tsamp.shape[1]
            Eloc = J1J2_local_energies(wf,N,J1,J2,Bz,numsamples, tsamp, Marshall_sign)
            loss_value = cost_fn(Eloc, log_amplitudes_)

            wf_hyp_vars = wfmodel.rnn.hyp_vars
            wf_eucl_vars = []
            for var in wfmodel.rnn.eucl_vars:
                wf_eucl_vars.append(var)
            for var in wfmodel.dense_a.trainable_variables:
                wf_eucl_vars.append(var) 
            for var in wfmodel.dense_p.trainable_variables:
                wf_eucl_vars.append(var) 

        grads_eucl = tape.gradient(loss_value, wf_eucl_vars)
        grads_hyp = tape.gradient(loss_value, wf_hyp_vars)
        opt_eucl.apply_gradients(zip(grads_eucl, wf_eucl_vars))
        opt_hyp.apply_gradients(grads_hyp, wf_hyp_vars)
        return loss_value, Eloc, wfmodel


# ---------------- Running VMC with RNNs for J1J2 Model with only Euclidean parameters -------------------------------------
def run_J1J2(wf, numsteps, systemsize, var_tol, J1_  = 1.0, J2_ = 0.0, Marshall_sign = False, 
            numsamples = 500, learningrate = 1e-2, seed = 111, fname = 'results'):

    lr = np.float64(learningrate)   
    J1=+J1_*np.ones(systemsize) # nearest neighbours couplings
    J2=+J2_*np.ones(systemsize) # next-nearest neighbours couplings
    Bz=+0.0*np.ones(systemsize) # magnetic field along z
    
    #Seeding
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)  

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)


    #Running the training -------------------
    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learningrate,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) 

    meanEnergy=[]
    varEnergy=[]
    best_E_list = []
    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, input_dim, J1, J2, Bz, Marshall_sign, optimizer)
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns{numsamples}_Ms{Marshall_sign}_meanE.npy',meanEnergy)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_varE.npy',varEnergy)
        if step==0:
            best_E = meanE
            best_E_list.append(best_E)
        if step !=0:
            if np.real(meanE) < min(np.real(best_E_list)) and  np.abs(varE) < var_tol:
                best_E = meanE
                best_E_list.append(meanE)
                #only save best models
                wfmodel.save_weights(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_checkpoint.weights.h5') 
                print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
        if step %10 ==0:
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}')
                        
    return meanEnergy, varEnergy

# ---------------- Running VMC with RNNs for J1J2 Model with hyperbolic parameters -------------------------------------
def run_J1J2_hypvars(wf, numsteps, systemsize, var_tol, J1_  = 1.0, J2_ = 0.0, Marshall_sign = True, 
                   numsamples = 50,  lr1=1e-2, lr2=1e-2, seed = 111, fname = 'results'):
    
    J1=+J1_*np.ones(systemsize) # nearest neighbours couplings
    J2=+J2_*np.ones(systemsize) # next-nearest neighbours couplings
    Bz=+0.0*np.ones(systemsize) # magnetic field along z
    
    #Seeding
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)  

    input_dim=2

    #Running the training -------------------
    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr1,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    opt_eucl=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) 
    opt_hyp=RSGD('rsgd', learning_rate=lr2)

    meanEnergy=[]
    varEnergy=[]
    best_E_list = []
    for step in range(numsteps):
        cost, E, wfmodel = train_step_with_hypvars(wf, numsamples, input_dim, J1, J2, Bz, Marshall_sign, opt_eucl, opt_hyp)
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns{numsamples}_Ms{Marshall_sign}_meanE.npy',meanEnergy)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_varE.npy',varEnergy)
        if step==0:
            best_E = meanE
            best_E_list.append(best_E)
        if step !=0:
            if np.real(meanE) < min(np.real(best_E_list)) and  np.abs(varE)< var_tol:
                best_E = meanE
                best_E_list.append(meanE)
                #only save best models
                wfmodel.save_weights(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_checkpoint.weights.h5') 
                print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
        if step %10 ==0:
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}')
                        
    return meanEnergy, varEnergy
 
