#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:56:00 2023

@author: djordjemihajlovic
"""
import random
import numpy as np
from numba import njit
    

@njit  # Class decorator -  compiles in C resulting in (much) faster computation
def Order_Param(order_param, c_p, N, M, dx, dt):    # To speed up computation functions defined outside of update class - allows for implementation of njit decorator
    
    update_param = np.zeros((N, N))  # Set up new lattice for updated values after timestep, note () used in np.zeros rather than [], running in C requires () to denote that update_param is 2d array
    for i in range(0, N):
        for j in range(0, N):
            
            T = c_p[(i+1)%N][j] #neighbours to the right, left, top, bottom
            B = c_p[(i-1)%N][j]
            R = c_p[i][(j+1)%N]
            L = c_p[i][(j-1)%N] 
            
            c_p_point = c_p[i][j]  # Defined c_p_point so equation below was easier to follow

            update_param[i][j] = order_param[i][j] + M*dt/(dx**2) * (T+B+R+L-4*c_p_point)  # Coded update rule 
            
    order_param = update_param
    return order_param
    

@njit         
def Chemical_Potential(c_p, order_param, N, a, k, dx, dt):
    
    update_cp = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            
            T = order_param[(i+1)%N][j] #neighbours to the right, left, top, bottom
            B = order_param[(i-1)%N][j]
            R = order_param[i][(j+1)%N]
            L = order_param[i][(j-1)%N] 
            order_param_point = order_param[i][j]
            
            update_cp[i][j] = -(a * order_param_point) + a * (order_param_point**3) - (k/dx**2)*(T+B+R+L-(4*order_param_point))  # Update rule
    
    c_p = update_cp
    return c_p

@njit
def Free_Energy_Density(order_param, N, a, k, dx):
  
    update_f = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            
            T = order_param[(i+1)%N][j] #neighbours to the right, left, top, bottom
            B = order_param[(i-1)%N][j]
            R = order_param[i][(j+1)%N]
            L = order_param[i][(j-1)%N]

            dphi_dx = (T-B)/(2*dx)  # Computed single derivatives for grad used in update rule below
            dphi_dy = (R-L)/(2*dx)
            
            order_param_point = order_param[i][j]  # Introduced s.t. equation below is easier to understand
            
            update_f[i][j] = -(a/2) * (order_param_point**2) + (a/4) * (order_param_point**4) + (k/2)*((dphi_dx**2)+(dphi_dy**2))  # Update rule
            
    f = update_f
    density_f = np.sum(f) 
    return density_f


class Update(object):
    
    def __init__(self, phi, N, M, a, k, dx, dt): #Random initializer
        
        self.phi = phi
        self.M = M
        self.a = a
        self.k = k
        lower_bound = phi - 0.1
        upper_bound = phi + 0.1
        self.order_param = np.random.uniform(lower_bound, upper_bound, size=(N, N))
        self.c_p = np.zeros([N, N])
        self.dim = N
        self.dt = dt
        self.dx = dx
        self.f = np.zeros([N, N])
        
    def O_P(self):
        self.order_param = Order_Param(self.order_param, self.c_p, self.dim, self.M, self.dx, self.dt)
        return self.order_param
    
    def C_P(self):
        self.c_p = Chemical_Potential(self.c_p, self.order_param, self.dim, self.a, self.k, self.dx, self.dt)
        return self.c_p
    
    def F_E_D(self):
        self.f = Free_Energy_Density(self.order_param, self.dim, self.a, self.k, self.dx)
        return self.f
    
    
        
