#Code written by HLD - This is the TF2 rewritten and adapted version of the following TF1 code:
#https://github.com/mhibatallah/RNNWavefunctions/blob/master/2DTFIM_2DRNN/Training2DRNN_2DTFIM.py

import tensorflow as tf
import numpy as np
import os
import time
import random
from math import ceil

from tfim2d_2drnn import *
from tfim2d_MDRNNcell import*

# Loading Functions --------------------------
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, wf):
    """ To get the local energies of 2D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny) np array
    - Bx: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx,Ny) an empty allocated np array to store the non diagonal elements
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    """

    numsamples = samples.shape[0]

    N = Nx*Ny #Total number of spins
    queue_samples = np.zeros(((Nx*Ny+1), numsamples, Nx,Ny),dtype =np.int32)
    #Array to store all the diagonal and non diagonal matrix elements

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice :)))
        values = samples[:,:,i]+samples[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i]), axis = 1)


    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(Nx):  #Non-diagonal elements
            for j in range(Ny):
                valuesT = np.copy(samples)
                valuesT[:,i,j][samples[:,i,j]==1] = 0 #Flip
                valuesT[:,i,j][samples[:,i,j]==0] = 1 #Flip

                queue_samples[i*Ny+j+1] = valuesT

    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/numsamples)

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, Nx,Ny])

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

# ---------------- Running VMC with 2DRNNs -------------------------------------
def run_2DTFIM(numsteps = 200, Nx = 5, Ny = 5, Bx = +2,
                        num_units = 50, numsamples = 50, learningrate = 5e-3, var_tol = 1.0,
                        seed = 111, fname = 'results/'):

    #Seeding
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    Jz = +np.ones((Nx,Ny)) #Ferromagnetic couplings
    lr=np.float64(learningrate)
    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    wf=RNNwavefunction(Nx,Ny, num_units, 2, seed = seed)

    learning_rate_withexpdecay  = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learningrate,
                                                        decay_steps=100,
                                                        decay_rate=0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer

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
