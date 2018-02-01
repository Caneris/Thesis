#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:28:39 2018

@author: canerates
"""

import numpy as np
import ThesisModules.pair.pairStates as pair
from scipy.optimize import brentq
import quantecon as qe
from quantecon import MarkovChain
import matplotlib.pyplot as plt

class canersProblem():

    
    def __init__(self, 
                 r=0.01,
                 w=1,
                 β=0.96, 
                 Π_w=((0.6, 0.4), (0.05, 0.95)),
                 Π_r=((0.6, 0.4), (0.4, 0.6)),
                 z_w_vals=(0.5, 1.0),
                 z_r_vals=(2, 0.5),
                 b=0, 
                 grid_max=16, 
                 grid_size=50,
                 u=np.log, 
                 du=lambda x: 1/x,
                 interest_rate_shock = False):

        self.u, self.du, self.w, self.r = u, du, w, r
        self.grid_max, self.grid_size = grid_max, grid_size
        self.Π_w, self.Π_r = Π_w, Π_r
        self.z_w_vals, self.z_r_vals = z_w_vals, z_r_vals
        self.β, self.b, self.interest_rate_shock = β, b, interest_rate_shock
        self.asset_grid = np.linspace(-b, grid_max, grid_size)
        
        if interest_rate_shock:
            self.z = pair.Pair(a_vals = z_r_vals,
                               b_vals = z_w_vals,
                               P_a = Π_r,
                               P_b = Π_w)
            self.Π, self.z_vals = self.z.Q, self.z.s_vals
        else:
            self.Π, self.z_vals = np.array(Π_w), tuple(z_w_vals)
        
    
def coleman_operator(c, cp):

    
    # === simplify names, set up arrays === #
    r, Π, β, du, b, w = cp.r, cp.Π, cp.β, cp.du, cp.b, cp.w
    interest_rate_shock = cp.interest_rate_shock
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_size = len(z_vals)
    
    vals = np.empty(z_size)
    
    # === linear interpolation to get consumption function === #
    def cf(a):
        """
        The call cf(a) returns an array containing the values c(a, z)
        for each z in z_vals.  For each such z, the value c(a, z)
        is constructed by univariate linear approximation over asset
        space, based on the values in the array c
        """
        for i in range(z_size):
            vals[i] = np.interp(a, asset_grid, c[:, i])
        return vals    
    
    Kc = np.empty(c.shape)
    
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            if interest_rate_shock:
                z_r, z_w = z[0], z[1]
                def h(t):
                    expectation = np.dot(du(cf((1+r*z_r) * a + w * z_w - t)), Π[i_z, :])
                    return du(t) - max((1+r*z_r) * β * expectation, du((1+r*z_r) * a + w * z_w + b))
                Kc[i_a, i_z] = brentq(h, 1e-8, (1+r*z_r) * a + w * z_w + b)
            else:
                def h(t):
                    expectation = np.dot(du(cf((1+r) * a + w * z - t)), Π[i_z, :])
                    return du(t) - max((1+r) * β *  expectation, du((1+r) * a + w * z + b))
                Kc[i_a, i_z] = brentq(h, 1e-8, (1+r) * a + w * z + b)
    return Kc

def initialize(cp):

    # === Simplify names, set up arrays === #
    r, β, u, b, w = cp.r, cp.β, cp.u, cp.b, cp.w
    interest_rate_shock = cp.interest_rate_shock
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = len(asset_grid), len(z_vals)
    V, c = np.empty(shape), np.empty(shape)
    
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            if interest_rate_shock:
                z_r, z_w = z[0], z[1]
                c_max = (1+r*z_r) * a + w * z_w + b
            else:
                c_max = (1+r) * a + w * z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) / (1 - β)
    return V, c

def compute_asset_series(cp,
                         T=500000,
                         verbose=False):
    """
    Simulates a time series of length T for assets, given optimal savings
    behavior.  Parameter cp is an instance of canersProblem
    """
    
    Π, z_vals, r, w = cp.Π, cp.z_vals, cp.r, cp.w    
    interest_rate_shock = cp.interest_rate_shock
    mc = MarkovChain(Π)
    v_init, c_init = initialize(cp)
    K = lambda c: coleman_operator(c, cp)
    c = qe.compute_fixed_point(K, c_init, verbose=verbose)
    cf = lambda a, i_z: np.interp(a, cp.asset_grid, c[:, i_z])
    a = np.zeros(T+1)
    z_seq = mc.simulate(T)
    if interest_rate_shock:
        for t in range(T):
            i_z = z_seq[t]
            a[t+1] = (1+r*z_vals[i_z][0]) * a[t] + w * z_vals[i_z][1] - cf(a[t], i_z)
        return a
    else:
        for t in range(T):
            i_z = z_seq[t]
            a[t+1] = (1+r) * a[t] + w * z_vals[i_z] - cf(a[t], i_z)
        return a

def asset_mean(cp, T=250000):
    
    asset_mean = np.mean(compute_asset_series(cp, T=T))
    return asset_mean

def capital_supply_r(cp, r_min=1e-8, r_max=0.04, grid_points=5, T=250000, plot=False):
    
    β, u, du, b, w = cp.β, cp.u, cp.du, cp.b, cp.w 
    grid_max, grid_size = cp.grid_max, cp.grid_size
    Π_w, Π_r = cp.Π_w, cp.Π_r
    z_w_vals, z_r_vals = cp.z_w_vals, cp.z_r_vals
    interest_rate_shock = cp.interest_rate_shock
    
    r_vals = np.linspace(r_min, r_max, grid_points)
    asset_means = []
    
    for r_val in r_vals:
        hh = canersProblem(r=r_val,
                           w=w,
                           β=β,
                           Π_w=Π_w,
                           Π_r=Π_r,
                           z_w_vals=z_w_vals,
                           z_r_vals=z_r_vals,
                           b=b,
                           grid_max=grid_max,
                           grid_size=grid_size,
                           u=u,
                           du=du,
                           interest_rate_shock=interest_rate_shock)
        mean = asset_mean(hh, T=T)
        asset_means.append(mean)
        print('Finished iterating {:01.0f} %'.format(((i+1)/ grid_points)*100))
        
        if r_val == r_vals[-4]:
            print('We are almost there!!')
            print('Only three more r values left!! :D')
        elif r_val == r_vals[-3]:
            print('Wait for it!....')
        elif r_val == r_vals[-2]:
            print('WAIT FOR IT!........')
        elif r_val == r_vals[-1]:
            print('FINISHED!!!! YAY!! ;D')

    if plot:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(np.asarray(asset_means), r_vals)

    return np.asarray(asset_means)

def capital_supply_r2(cp, r_vals = np.linspace(1e-8, 0.04, 5), T=250000, plot=False):
    
    β, u, du, b, w = cp.β, cp.u, cp.du, cp.b, cp.w 
    grid_max, grid_size = cp.grid_max, cp.grid_size
    Π_w, Π_r = cp.Π_w, cp.Π_r
    z_w_vals, z_r_vals = cp.z_w_vals, cp.z_r_vals
    interest_rate_shock = cp.interest_rate_shock
    
    asset_means = []
    
    for i, r_val in enumerate(r_vals):
        hh = canersProblem(r=r_val,
                           w=w,
                           β=β,
                           Π_w=Π_w,
                           Π_r=Π_r,
                           z_w_vals=z_w_vals,
                           z_r_vals=z_r_vals,
                           b=b,
                           grid_max=grid_max,
                           grid_size=grid_size,
                           u=u,
                           du=du,
                           interest_rate_shock=interest_rate_shock)
        mean = asset_mean(hh, T=T)
        asset_means.append(mean)
        print('Finished iterating {:01.0f} %'.format(((i+1)/ len(r_vals))*100))
        
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
        

        







