#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:10:23 2018

@author: canerates
"""


import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from quantecon.markov import DiscreteDP
import ThesisModules as tm



# Example prices
r = 0.03
w = 0.956
epsilon = 0.1

#------------First simulation---------------------

# Calibrate the interest rate shock
z_r_vals = [2, 0.5, -3]
P1 = 0.6
P2 = 1 - P1
P3 = 0.3
P_z_r = [[P1, P2, 0], [P2 - epsilon, P1, epsilon], [0.5 * (1-P3), 0.5 * (1-P3), P3]]

# Calibrate the wage shock
z_w_vals = [1, 0.6]
p1 = 0.5
p2 = 1 - p1
P_z_w = [[p1, p2], [p2, p1]]


hh = tm.atesHousehold.Caner_Household(a_max=20,
                                      r=r,
                                      w=w,
                                      z_r_vals=z_r_vals,
                                      z_w_vals=z_w_vals,
                                      P_z_r=P_z_r,
                                      P_z_w=P_z_w)

hh_ddp = DiscreteDP(hh.R, hh.Q, hh.beta)

results = hh_ddp.solve(method='policy_iteration')

z_size, a_size = hh.z_size, hh.a_size
z_vals, a_vals = hh.z_vals, hh.a_vals


n = a_size * z_size

# Get all the optimal allocation with z fixed in each row

a_star = np.empty((z_size, a_size))
for s_i in range(n):
    a_i = s_i // z_size
    z_i = s_i % z_size
    a_star[z_i, a_i] = a_vals[results.sigma[s_i]]







    
#------------Second simulation---------------------

epsilon = 0

# Calibrate the interest rate shock

P1 = 0.6
P2 = 1 - P1
P3 = 0.3
P_z_r = [[P1, P2, 0], [P2 - epsilon, P1, epsilon], [0.5 * (1-P3), 0.5 * (1-P3), P3]]

# Calibrate the wage shock
z_w_vals = [1, 0.6]
p1 = 0.5
p2 = 1 - p1
P_z_w = [[p1, p2], [p2, p1]]


hh2 = tm.atesHousehold.Caner_Household(a_max=20,
                                      r=r,
                                      w=w,
                                      z_r_vals=z_r_vals,
                                      z_w_vals=z_w_vals,
                                      P_z_r=P_z_r,
                                      P_z_w=P_z_w)

hh2_ddp = DiscreteDP(hh2.R, hh2.Q, hh2.beta)

results2 = hh2_ddp.solve(method='policy_iteration')

z_size2, a_size2 = hh2.z_size, hh2.a_size
z_vals2, a_vals2 = hh2.z_vals, hh2.a_vals


n2 = a_size2 * z_size2

# Get all the optimal allocation with z fixed in each row

a_star2 = np.empty((z_size2, a_size2))
for s_i in range(n2):
    a_i = s_i // z_size2
    z_i = s_i % z_size2
    a_star2[z_i, a_i] = a_vals2[results2.sigma[s_i]]








#f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')
#ax1.plot(a_vals, a_vals, 'k--')
#for i in range(z_size):
#    lb = r'$z = ({0},{1})$'.format(z_vals[i][0], z_vals[i][1], '.2f')
#    ax1.plot(a_vals, a_star[i,:], lw=2, alpha=0.6, label=lb)
#    ax1.set_xlabel('current assets')
#    ax1.set_ylabel('next period assets')
#ax1.legend(loc='upper left', ncol=1, prop={'size': 5})
#
#ax2.plot(a_vals2, a_vals2, 'k--')
#for i in range(z_size):
#    lb = r'$z = ({0},{1})$'.format(z_vals2[i][0], z_vals2[i][1], '.2f')
#    ax2.plot(a_vals2, a_star2[i,:], lw=2, alpha=0.6, label=lb)
#    ax2.set_xlabel('current assets')
#    ax2.set_ylabel('next period assets')
#ax2.legend(loc='upper left', ncol=1, prop={'size': 5})


fig2, ax = plt.subplots(figsize=(9, 9))
ax.plot(a_vals, a_vals, 'k--') # 45 degrees
ax.plot(a_vals, a_star[3,:], lw=2, alpha=0.6, label = '$epsilon = 0.3$')
ax.plot(a_vals, a_star2[3,:], lw=2, alpha=0.6, label = '$epsilon = 0$')
ax.legend(loc='upper left', ncol=1, prop={'size': 8})
 
plt.show()







