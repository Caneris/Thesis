#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:20:30 2018

@author: canerates
"""

from quantecon import MarkovChain
import numpy as np

def exp_val(z_vals, P_z):
    """
    This function gives you the weighted average of all states of a 
    markov chain in the long run.
    
    Parameters
    -----------
    
    z_vals : state values of the markov chain
    P_z : transition matrix
    """
    mc = MarkovChain(P_z, state_values = z_vals) # get the stationary probs
    return np.sum(mc.stationary_distributions * z_vals) # compute weighted average
