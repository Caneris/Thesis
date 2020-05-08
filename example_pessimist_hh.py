#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:12:26 2018

@author: canerates
"""

from ThesisModules.thesisHousehold import *
import ThesisModules.pessimisticHousehold as pessimist
import ThesisModules.expectedSSdistValue as expect
import numpy as np
import matplotlib.pyplot as plt

# Here I want to how pessimistic expectations impact the savingsbehavior

r = 0.03 # the interest rate which is going to be shocked

P1, P3, epsilon = 0.6, 0.3, 0.05
P2, P4 = 1-P1, 0.5*(1-P3)

Π_r = ((P1, P2, 0), (P2 - epsilon, P1, epsilon), (P4, P4, P3))

z_r_vals = (1.5, 0.2, -3)


# simulate model without pessimistic beliefs

M=25
MM=250000

cp_r_shock = canersProblem(r=r,
                           Π_r=Π_r,
                           z_r_vals=z_r_vals,
                           interest_rate_shock=True)

capital_r_shock = capital_supply_r(cp_r_shock, 
                                   r_max=0.04,
                                   grid_points=M,
                                   T=MM)

# simulate model with pessimistic beliefs

pessimist_r_shock = pessimist.canersPessimist(r=r,
                                              Π_r=Π_r,
                                              z_r_vals=z_r_vals)


pessi_capital = pessimist.pessi_capital_to_r(pessimist_r_shock,
                                             r_max=0.04,
                                             grid_points=M,
                                             T=MM)

fig, ax = plt.subplots(figsize=(12,8))

ax.plot(pessi_capital, np.linspace(1e-8, 0.04, M),'o--', alpha=0.6, label='pessimistic beliefs')
ax.plot(capital_r_shock, np.linspace(1e-8, 0.04, M),'^--', alpha=0.6, label='no pessimistic beliefs')

#ax.set_yticks(np.arange(0, 0.04, .01))
#ax.set_xticks(np.arange(0, 0.5, 0.1))
#plt.ylim(0, 0.04)
#plt.xlim(0, 0.5)
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.grid(True)
ax.legend(loc='upper left')

plt.show()
