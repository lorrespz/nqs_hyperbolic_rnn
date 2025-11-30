#Code written for arXiv:2505.22083 
#[Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz]

import tensorflow as tf
import keras
import numpy as np
import os
import time
import sys
import random
from math import ceil
from hyp_util import *

class RSGD(tf.Module):

  def __init__(self, hyp_opt, learning_rate, c_val = 1.0):
    # Initialize parameters
    self.learning_rate = learning_rate
    self.c_val = c_val
    self.title = f'RSGD with learning rate = {self.learning_rate}'
    self.hyp_opt = hyp_opt

  def apply_gradients(self, grads, vars):
    # Update variables
    for grad, var in zip(grads, vars):
        if self.hyp_opt == 'rsgd':
            var = tf_exp_map_x(var, - self.learning_rate * grad, self.c_val)
        else:
            # Use approximate RSGD based on a simple retraction.
            var=var- self.learning_rate * grad
            var = tf_project_hyp_vecs([[var]], self.c_val)
        