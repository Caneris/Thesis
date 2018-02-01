#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:17:10 2018

@author: canerates
"""

from ThesisModules.thesisHousehold import *
import ThesisModules.expectedSSdistValue as expect
import numpy as np
import matplotlib.pyplot as plt

# Here I want to compare how a shocked interest rate with a 
# possible negative shock affects the savings behavior.

r = 0.03 # the interest rate which is going to be shocked

P1, P3, epsilon = 0.6, 0.3, 0.005
P2, P4 = 1-P1, 0.5*(1-P3)

Π_r = ((P1, P2, 0), (P2 - epsilon, P1, epsilon), (P4, P4, P3))

z_r_vals = (2, 0.2, -1)


# Compute the expected shock to the interest rate E_z
E_z = expect.exp_val(z_vals=z_r_vals, P_z=Π_r)

# I am going to use E_z in an economy without any shocks to
# the interest rate as a multiplikator for the interest rates in 
# order to compare the differences in the savings behavior.

cp_r_shock = canersProblem(r=r,
                           Π_r=Π_r,
                           z_r_vals=z_r_vals,
                           interest_rate_shock=True)

capital_r_shock = capital_supply_r(cp_r_shock, 
                                   r_max=0.04,
                                   grid_points=10,
                                   T=100000)

# since I need to multiply every grid point by E_z
# I had to write another function
# where I can use the interest rate grid as an argument.
# In that way I can multiply every gridpoint by E_z

cp_no_r_shock = canersProblem()

r_vals=np.linspace(1e-8, 0.04, 10)*E_z

capital_no_r_shock = capital_supply_r2(cp_no_r_shock,
                                       r_vals=np.linspace(1e-8, 0.04, 10)*E_z,
                                       T=100000)

fig, ax = plt.subplots(figsize=(8,10))

ax.plot(capital_no_r_shock, r_vals, label='no r shock')
ax.plot(capital_r_shock, r_vals, label='r shock')

ax.set_yticks(np.arange(0, 0.038, .01))
ax.set_xticks(np.arange(0, 3, 1))
plt.ylim(0, 0.04)
plt.xlim(0, 1.8)
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.grid(True)
ax.legend(loc='upper left')

plt.show()
