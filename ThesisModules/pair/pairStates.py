#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:38:19 2018

@author: canerates
"""

import numpy as np

class Pair:
    """
    This class, takes two markov processes, combines their state spaces
    and gives computes the corresponding transition matrix.
    """    
    
    def __init__(self,
                 a_vals = [0, 1], # state values of the first mc
                 b_vals = [1, 0], # state values of the second mc
                 P_a = [[0.9, 0.1], [0.1, 0.9]],
                 P_b = [[0.8, 0.2], [0.2, 0.8]]):
        
        self.a_vals = np.asarray(a_vals)
        self.b_vals = np.asarray(b_vals)
        self.P_a = np.asarray(P_a)
        self.P_b = np.asarray(P_b)
        self.n = len(a_vals) * len(b_vals)
        
        # Build the array with combined state values
        self.s_vals = make_statespace(self.a_vals, self.b_vals)
        
        # Combine both transition matrices
        self.Q = np.kron(self.P_b, self.P_a)
    
    
    


def make_statespace(a, b):
    """
    returns state space s_vals, when two statespaces of two independend markov 
    chains a, b are combined to s:=(a_state, b_state)
    
    Parameters
    ----------
    a,b : state spaces in form of numpy arrays containing the state values of
          the two markov chains a and b.
          
    Returns
    ---------
    state space vector : Returns an two dimentional array 
                         [[a_state1, b_state2], [a_state1, b_state2], ...etc]
    """
    
    s_vals = []

    for j in range(len(b)):
        for i in range(len(a)):
            s_vals.append([a[i], b[j]])
    return np.array(s_vals)





