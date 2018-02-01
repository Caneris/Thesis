#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:17:27 2018

@author: canerates
"""

# Here is an example 

# We are going to simulate an aiyagari model with interest rate and income 
# being shocked

from ThesisModules.thesisHousehold import *

# create an instance, with borrowing constraint b = 2
cp = canersProblem(interest_rate_shock=True, b=2)

# use capital_supply_r to plot the capital suplly for varying r values
capital_supply_r(cp, r_max=0.03, grid_points=25, plot=True)
