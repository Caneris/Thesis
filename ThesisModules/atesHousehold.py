#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:38:19 2018

@author: canerates
"""

import numpy as np
import ThesisModules.pair.pairStates as pair
#from numba import jit


class Caner_Household:
    """
    This class is based on the Household class for an Aiyagari household
    written by John Stachurski and Thomas Sargent. You can find the original
    code by going to the following link:
        
        https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/aiyagari/aiyagari_household.py
    This code was modified to fit the slightly modified vesion of the Aiyagari model
    for my master's thesis.
    
    the constraint looks as follows:
        
        a_{t+1} + c_t <= w * z_w + (1 + r * z_r) * a_t
        
    z_r/ z_w are shocks to the interest rate respectively to the labor income
    """
    
    def __init__(self,
                 r = 0.03, # interest rate
                 w = 1, # wages
                 beta = 0.96, # discount factor
                 a_min=1e-10,
                 P_z_w = [[0.9, 0.1], [0.1, 0.9]],
                 P_z_r = [[0.8, 0.2], [0.2, 0.8]],
                 z_w_vals = [1, 0.1],
                 z_r_vals = [1, 0.2],
                 a_max = 18,
                 a_size = 200):
        
        # Store values, set up grids over a
        self.r, self.w, self.beta = r, w, beta
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size
        
        self.a_vals = np.linspace(a_min, a_max, a_size)
        
        # First, combine both exogenous shocks and extract the new
        # Markov Chain, z_t := (z_r_t, z_w_t)
        # Use the class Pair() in your 
        # module "pair_mc_states" from package "ThesisModules"
        
        self.z = pair.Pair(a_vals = z_r_vals, b_vals = z_w_vals,
                           P_a = P_z_r, P_b = P_z_w)
        
                # then extract the new transition matrix T[z, z']
        self.T = self.z.Q
        
        # extract the new state value pairs
        self.z_vals = self.z.s_vals
        
        # number of states in new mc
        self.z_size = len(self.z_vals)
        
        # number of states if combined with action grid for 
        # state defined as s_t := (z_t, a_t) = (z_r_t, z_w_t, a_t)
        self.n = a_size * self.z_size
        
        # -> n for the n*n transition matrix Q[s, a, s'] 
        #    with a = a_{t+1} as the action to be chosen by the agent
        #    s = s_t, s' = s_{t+1}
        #    and with s_t:=(z_w_t, z_w_t, a_{t+1})
        
        # Build the array Q[s, a, s']
        self.Q = np.zeros((self.n, self.a_size, self.n))
        self.build_Q()
        
        # Build the array R[s, a] where a = a_{t+1} 
        # (reward at state s under action a)
        self.R = np.empty((self.n, self.a_size))
        self.build_R()
        
    def set_prices(self, r, w):
        """
        Use this method to reset prices.  Calling the method will trigger a 
        re-build of R.
        """
        self.r, self.w = r, w 
        self.build_R()
        
    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.T)
        
    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R, self.a_size, self.z_size, self.a_vals, self.z_vals, self.r, self.w)
        
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size #a_i stays constant till s_i == z_size 
        z_i = s_i % z_size # z_i iterates from 0 to z_size while a_i constant, until a_i == s_size
        
        # next we have to extract the 3 following values, a, z_r, z_w
        # remember that z[i] = [z_r_i, z_w_i]
        # extract both values by taking z[i][0] and z[i][1]
        
        a = a_vals[a_i]
        z_r = z_vals[z_i][0]
        z_w = z_vals[z_i][1]
        
        # Now that you got the values, compute consumption for every possible
        # action a_{t+1} by using a for loop
        
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z_w + (1 + r * z_r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = np.log(c) # Utility
                
def populate_Q(Q, a_size, z_size, T):
    n = a_size * z_size
    for s_i in range(n): # index for state s_t := (z_t, a_t)
        z_i = s_i % z_size # index for z_t
        for a_i in range(a_size): # index for action a_{t+1}
            for next_z_i in range(z_size):  # index for z_{t+1}
                # iterate for index of next by using a_i * z_size + next_z_i
                # which iterates from 0 to z_size every tim after updating
                # a_i and so in a whole iterates from 0 to n-1
                # so index for s' is a_i * z_size + next_z_i
                Q[s_i, a_i, a_i * z_size + next_z_i] = T[z_i, next_z_i]
                
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i * z_size + z_i]
    return a_probs

