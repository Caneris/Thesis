#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:55:44 2018

@author: canerates
"""

import numpy as np
import ThesisModules.pair as pair
from scipy.optimize import brentq
import quantecon as qe
from quantecon import MarkovChain
import matplotlib.pyplot as plt

class canersPessimist:
    """
    This class is a version of the 'canersProblem' class with 
    pessimistic beliefs. It has a wage and interest rate shock.
    The agents have pessimistic expactations on the interest rates
    development. They think the interest rate is more likely to
    become negative.
    
    Use a transition matrix for z_r in the following form:
        
        Π_r = ((P1, P2, 0), (P2 - epsilon, P1, epsilon), (P4, P4, P3))
        
        with P2 = 1-P1, P4 = 0.5*(1-P3)
        
    This class will give a corresponding belief matrix and a real matrix. 
    Π_belief and Π_real.
    
    """
    
    def __init__(self, 
                 r=0.01,
                 w=1,
                 β=0.96, 
                 Π_w=((0.6, 0.4), (0.05, 0.95)),
                 Π_r=((0.6, 0.4, 0), (0.35, 0.6, 0.05), (0.35, 0.35, 0.3)),
                 z_w_vals=(0.5, 1.0),
                 z_r_vals=(1.5, 0.2, -3),
                 b=0, 
                 grid_max=16, 
                 grid_size=50,
                 u=np.log, 
                 du=lambda x: 1/x):
        
        # simplify names, set up array and change the input
        # in a way so that you can adjust the coleman operator
        # and the simulation of assets, 
        # belief trans. matrix -> in coleman operator
        
        self.u, self.du, self.w, self.r = u, du, w, r
        self.grid_max, self.grid_size = grid_max, grid_size
        self.Π_r, self.Π_w, self.z_w_vals, self.z_r_vals_raw = Π_r, Π_w, z_w_vals, z_r_vals
        self.β, self.b = β, b
        self.asset_grid = np.linspace(-b, grid_max, grid_size)
        
        # get probabilites
        
        P1, P2, P3, P4 = Π_r[0][0], Π_r[0][1], Π_r[2][2], Π_r[2][0]
        epsilon = Π_r[1][2]
        
        # make pessimistic beliefs for r
        self.Π_r_belief = (( P1, 0.5*P2, 0.5*P2, 0), 
                    (0.5*(P2-epsilon), P1, 0.5*(P2-epsilon), epsilon), 
                    ( 0, 0, P1, 1 - P1), (P4, P4, 0, P3))
        
        # rewrite the real transition matrix
        self.Π_r_real = ((P1, 0.5*P2, 0.5*P2, 0),
                  (P2-epsilon, 0.5*P1, 0.5*P1, epsilon), 
                  (P2-epsilon, 0.5*P1, 0.5*P1, epsilon), 
                  (P4, 0.5*P4, 0.5*P4, P3))
        
        self.z_r_vals = np.insert(z_r_vals, 1, z_r_vals[1]) 
        
        # take the new transition matrices and combine them with
        # the transition matrix for the wage shock in order
        # to get Π_belief, Π_real and z_vals
        self.z_belief = pair.Pair(a_vals=self.z_r_vals,
                                  b_vals=z_w_vals,
                                  P_a=self.Π_r_belief,
                                  P_b=Π_w)
        self.z_real = pair.Pair(a_vals=self.z_r_vals,
                                b_vals=z_w_vals,
                                P_a=self.Π_r_real,
                                P_b=Π_w)
        self.z_vals = self.z_real.s_vals
        self.Π_belief, self.Π_real = self.z_belief.Q, self.z_real.Q


def belief_operator(c, cp):
    
    # === simplify names, set up arrays === #
    r, Π_belief, β, du, b, w = cp.r, cp.Π_belief, cp.β, cp.du, cp.b, cp.w
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_size = len(z_vals)
    vals = np.empty(z_size)
    
    def cf(a):
        for i in range(z_size):
            vals[i] = np.interp(a, asset_grid, c[:,i])
        return vals
    
    Kc = np.empty(c.shape)
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            z_r, z_w = z[0], z[1]
            def h(t):
                expectation = np.dot(du(cf((1+r*z_r) * a + w * z_w - t)), Π_belief[i_z,:])
                return du(t) - max((1+r*z_r) * β * expectation, du((1+r*z_r) * a + w * z_w + b))
            Kc[i_a, i_z] = brentq(h, 1e-8, (1+r*z_r) * a + w * z_w + b)
    return Kc

def initialize(cp):
    # === Simplify names, set up arrays === #
    r, b, w = cp.r, cp.b, cp.w
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = len(asset_grid), len(z_vals)
    c = np.empty(shape)
    
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
                z_r, z_w = z[0], z[1]
                c_max = (1+r*z_r) * a + w * z_w + b
                c[i_a, i_z] = c_max
    return c

def pessimistic_asset_series(cp,
                         T=500000,
                         verbose=False):
    """
    Simulates a time series of length T for assets, given optimal savings
    behavior and pessimistic Beliefs. Parameter cp is an instance of canersPessimist.
    """
    
    Π_real, z_vals, r, w = cp.Π_real, cp.z_vals, cp.r, cp.w
    mc = MarkovChain(Π_real)
    c_init = initialize(cp)
    K = lambda c: belief_operator(c, cp)
    c = qe.compute_fixed_point(K, c_init, verbose=verbose, max_iter=100)
    cf = lambda a, i_z: np.interp(a, cp.asset_grid, c[:, i_z])
    a = np.zeros(T+1)
    z_seq = mc.simulate(T)
    for t in range(T):
        i_z = z_seq[t]
        a[t+1] = (1+r*z_vals[i_z][0]) * a[t] + w * z_vals[i_z][1] - cf(a[t], i_z)
    return a

def pessi_asset_mean(cp, T=250000):
    
    asset_mean = np.mean(pessimistic_asset_series(cp, T=T))
    return asset_mean

def pessi_capital_to_r(cp, r_min=1e-8, r_max=0.04, grid_points=5, T=250000, plot=False):
    
    β, u, du, b, w = cp.β, cp.u, cp.du, cp.b, cp.w 
    grid_max, grid_size = cp.grid_max, cp.grid_size
    Π_w, Π_r = cp.Π_w, cp.Π_r
    z_w_vals, z_r_vals = cp.z_w_vals, cp.z_r_vals_raw # use cp.z_r_vals_raw
    
    r_vals = np.linspace(r_min, r_max, grid_points)
    asset_means = []
    
    for i, r_val in enumerate(r_vals):
        hh = canersPessimist(r=r_val,
                             w=w,
                             β=β,
                             Π_w=Π_w,
                             Π_r=Π_r,
                             z_w_vals=z_w_vals,
                             z_r_vals=z_r_vals, # this can be a problem in new class
                             b=b,
                             grid_max=grid_max,
                             grid_size=grid_size,
                             u=u,
                             du=du)
        mean = pessi_asset_mean(hh, T=T)
        asset_means.append(mean)
        print('Finished iterating {:01.0f} %'.format(((i+1)/ grid_points)*100))
        
        if r_val == r_vals[-4]:
            print('\n We are almost there!!')
            print('\n Only THREE more r values left!! :D')
        elif r_val == r_vals[-3]:
            print('\n Wait for it!....')
        elif r_val == r_vals[-2]:
            print('\n WAIT FOR IT!........')
        elif r_val == r_vals[-1]:
            print('\n FINISHED!!!! YAY!! ;D')

    if plot:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(np.asarray(asset_means), r_vals)
        ax.set_xlabel('capital')
        ax.set_ylabel('interest rate')
        ax.grid(True)
        plt.show()


    return np.asarray(asset_means)      
    
    
    
    
        