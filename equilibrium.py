#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:01:15 2018

@author: canerates
"""

# Compute the equilibrium
import time
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from quantecon.markov import DiscreteDP
import ThesisModules as tm

start_time = time.time()

A = 1.0
N = 1.0
alpha = 0.33
beta = 0.96
delta = 0.05
epsilon = 0
r = 0.03

# Calibrate the interest rate shock
z_r_vals = [1, 0.01, -0.005]
P1 = 0.8
P2 = 1 - P1
P3 = 0.3
P_z_r = [[P1, P2, 0], [P2 - epsilon, P1, epsilon], [0.5 * (1-P3), 0.5 * (1-P3), P3]]

# Calibrate the wage shock
z_w_vals = [1, 0.1]
p1 = 0.4
p2 = 1 - p1
P_z_w = [[p1, p2], [p2, p1]]

# the following code is almost entirely taken from sargent and stachurski's
# quantecon lectures

def r_to_w(r):
    """
    Equilibrium wages associated with agiven interest rate r.
    """
    return A * (1 - alpha) * (A * alpha / (r + delta))**(alpha/(1-alpha))

def rd(K):
    """
    Inverse demand curve for capital.  The interest rate associated with a
    given demand for capital K.
    """
    return alpha * A * (N / K)**(1-alpha) - delta

def prices_to_capital_stock(hh, r):
    """
    Map prices to the induced level of capital stock.
    
    Parameters:
    ----------
    
    hh : Household
        An instance of an aiyagari_household.Household 
    r : float
        The interest rate
    """
    w = r_to_w(r)
    hh.set_prices(r, w)
    caner_ddp = DiscreteDP(hh.R, hh.Q, beta)
    # Compute optimal plicy
    results = caner_ddp.solve(method='policy_iteration')
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    # Extract the marginal distribution for assets
    asset_probs = tm.atesHousehold.asset_marginal(stationary_probs,
                                                  hh.a_size,
                                                  hh.z_size)
    # Return K
    return np.sum(asset_probs * hh.a_vals)

# Create an instance of the household
hh = tm.atesHousehold.Caner_Household(r = 0.03,
                                      z_r_vals = z_r_vals,
                                      z_w_vals = z_w_vals,
                                      P_z_r = P_z_r,
                                      P_z_w = P_z_w,
                                      a_max=20)

# Use the instance to build a discrete dynamic program
hh_ddp = DiscreteDP(hh.R, hh.Q, hh.beta)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 60
r_vals = np.linspace(0.005, 0.20, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(hh, r)
    print("--- %s seconds ---" % (time.time() - start_time))
    
# Plot against demand for capital by firm
fig, ax = plt.subplots(figsize=(11,8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
ax.plot(k_vals, rd(k_vals), lw=2, alpha=0.6, label='demand for capital')
ax.grid()
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

plt.show()




    
    
    
    
    

    