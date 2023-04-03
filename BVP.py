#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:28:02 2023

@author: djordjemihajlovic
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import math

#Usually would've set up a class for dealing with all my functions however for this specific part of the checkpoint decided against it


def EField(phi, N, plane):  # Chooses plane to find Efield values

    Ex = []
    Ey = []
    Ez = []
    for i in range(1, N-1):
        for j in range(1, N-1):
                
            F = phi[(plane+1)][i][j] #neighbours to the right, left, top, bottom, forward & backward
            Ba = phi[(plane-1)][i][j]
            R = phi[plane][(i+1)][j]
            L = phi[plane][(i-1)][j]
            T = phi[plane][i][(j+1)]
            Bo = phi[plane][i][(j-1)]
            
            y = -(R-L)*(1/2)
            x = (T-Bo)*(1/2)
            z = (F-Ba)*(1/2)
            
            normx = x/math.sqrt((x**2) + (y**2) + (z**2))
            normy = y/math.sqrt((x**2) + (y**2) + (z**2))
            normz = z/math.sqrt((x**2) + (y**2) + (z**2))
            
            Ex.append(normx)
            Ey.append(normy)
            Ez.append(normz)
        
    return Ex, Ey, Ez               
    return EUpd

@njit
def JacobiPotential(phi, rho, N):
    update_phi = np.zeros((N, N, N))  # Set up new lattice for updated values after timestep, note () used in np.zeros rather than [], running in C requires () to denote that update_param is 2d array
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):  # range 1 to N-1 to keep boundaries 0.
                
                
                T = phi[(i+1)][j][k] #neighbours to the right, left, top, bottom, forward & backward
                Bo = phi[(i-1)][j][k]
                R = phi[i][(j+1)][k]
                L = phi[i][(j-1)][k]
                F = phi[i][j][(k+1)]
                Ba = phi[i][j][(k-1)]  # NOTE - CHECK BOUNDARY AFFECT -
            
                rho_point = rho[i][j][k]  # Defined rho_point so equation below was easier to follow
    
                update_phi[i][j][k] = (1/6) * (T+Bo+R+L+F+Ba+rho_point)  # Coded update rule 

    phi = update_phi    
    return phi

@njit
def GaussPotential(phi, rho, N):
    init = np.sum(np.abs(phi))
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):  # range 1 to N-1 to keep boundaries 0.
                
                
                T = phi[(i+1)][j][k] #neighbours to the right, left, top, bottom, forward & backward
                Bo = phi[(i-1)][j][k]
                R = phi[i][(j+1)][k]
                L = phi[i][(j-1)][k]
                F = phi[i][j][(k+1)]
                Ba = phi[i][j][(k-1)]  # NOTE - CHECK BOUNDARY AFFECT -
            
                rho_point = rho[i][j][k]  # Defined rho_point so equation below was easier to follow
    
                phi[i][j][k] = (1/6) * (T+Bo+R+L+F+Ba+rho_point)  # Coded update rule 
    fin = np.sum(np.abs(phi))
   
    return phi, init, fin


def Data(phi, N):
    data_point = []
    distance = []
    center = N/2
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                
                x_dist = i - center
                y_dist = j - center
                z_dist = k - center     
                centre_dist = math.sqrt((x_dist**2)+ (y_dist**2) + (z_dist**2))
                
                data_point.append(phi[i][j][k])
                distance.append(centre_dist)
                
    return distance, data_point            ## Too tired rn but need to set ALL data to arrays
                

def ChargeDist(N, system):
    if system == 0:
        rho = np.random.uniform(size=(N, N, N))
    elif system == 1:
        rho = np.zeros((N, N, N))
        mid = int(N/2)
        rho[mid][mid][mid] = 1
        
    return rho

##########################################################################################

@njit
def JacobiPotentialMag(Az, j, N):
    update_Az = np.zeros((N, N, N))  # Set up new lattice for updated values after timestep, note () used in np.zeros rather than [], running in C requires () to denote that update_param is 2d array
    for m in range(1, N-1):
        for l in range(1, N-1):
            for n in range(1, N-1):  # range 1 to N-1 to keep boundaries 0.
                
                
                T = Az[(m+1)][l][n] #neighbours to the right, left, top, bottom, forward & backward
                Bo = Az[(m-1)][l][n]
                R = Az[m][(l+1)][n]
                L = Az[m][(l-1)][n]
                F = Az[m][l][(n+1)]
                Ba = Az[m][l][(n-1)]  # NOTE - CHECK BOUNDARY AFFECT -
            
                j_point = j[m][l][n]  # Defined rho_point so equation below was easier to follow
    
                update_Az[m][l][n] = (1/6) * (T+Bo+R+L+F+Ba+j_point)  # Coded update rule 
                
    Az = update_Az    
    return Az

@njit
def GaussPotentialMag(Az, j, N):
    init = np.sum(np.abs(Az))
    for m in range(1, N-1):
        for l in range(1, N-1):
            for n in range(1, N-1):  # range 1 to N-1 to keep boundaries 0.
                
                
                T = Az[(m+1)][l][n] #neighbours to the right, left, top, bottom, forward & backward
                Bo = Az[(m-1)][l][n]
                R = Az[m][(l+1)][n]
                L = Az[m][(l-1)][n]
                F = Az[m][l][(n+1)]
                Ba = Az[m][l][(n-1)]  # NOTE - CHECK BOUNDARY AFFECT -
            
                j_point = j[m][l][n]  # Defined rho_point so equation below was easier to follow
    
                Az[m][l][n] = (1/6) * (T+Bo+R+L+F+Ba+j_point)  # Coded update rule 
    fin = np.sum(np.abs(Az))
   
    return Az, init, fin


def CurrentDist(N):
    j = np.random.uniform(size=(N, N, N))
    j = np.zeros((N, N, N))
    mid = int(N/2)
    for n in range(1, N-1):
        j[n][mid][mid] = 1
    
    return j

def MagField(Az, N, plane):  # Chooses plane to find Efield values

    Mx = []
    My = []
    Mz = []
    for i in range(1, N-1):
        for j in range(1, N-1):
                
            R = Az[plane][(i+1)][j] #neighbours to the right, left, top, bottom, forward & backward
            L = Az[plane][(i-1)][j]
            T = Az[plane][i][(j+1)]
            Bo = Az[plane][i][(j-1)]
            F = Az[plane+1][i][j]
            Ba = Az[plane-1][i][j]
            
            z = (F-Ba)*(1/2)
            x = (T-Bo)*(1/2)
            y = (R-L)*(1/2)
            
            normx = x/math.sqrt((x**2) + (y**2) + (z**2))
            normy = y/math.sqrt((x**2) + (y**2) + (z**2))
            normz = z/math.sqrt((x**2) + (y**2) + (z**2))
            
            Mx.append(normx)
            My.append(normy)
            Mz.append(normz)
        
    return Mx, My, Mz    


################################################################################
    
                
def main():
    
    print("Boundary value problem simulator - Poisson's equation. ")
    problem = int(input("Simulator (0 for Electric field), (1 for Magnetic field): "))
    algo = int(input("Algorithm (0 for Jacobi), (1 for Gauss): "))
    dim = int(input("Dimension of simulation: "))
    tolerance = float(input("Accuracy of simulation: "))
    
    if problem == 0:
        
        sim = int(input("Simulation type, 0 for arbitrary charge dist., 1 for single charge in box center: "))
    
        if algo == 1:
        
            charge_dist = ChargeDist(dim, sim)
            potential = np.zeros((dim, dim, dim))
            potential, init, fin = GaussPotential(potential, charge_dist, dim)
            error = fin - init
            
            while error > tolerance:
                potential, init, fin = GaussPotential(potential, charge_dist, dim)        
                error = fin - init
    
        if algo == 0:
        
            charge_dist = ChargeDist(dim, sim)
            init_potential = np.zeros((dim, dim, dim))
            potential = JacobiPotential(init_potential, charge_dist, dim)
            error = abs(potential - init_potential)
            error = np.sum(error)
            
            while error > tolerance:
                init_potential = potential
                potential = JacobiPotential(potential, charge_dist, dim)        
                error = abs(potential - init_potential)
                error = np.sum(error)
                
                
        X,Y = Data(potential, dim)
        efieldx, efieldy, efieldz = EField(potential, dim, int(dim/2))
                
            
        fig, ax = plt.subplots()
        im = ax.imshow(potential[int(dim/2), :, :], extent = (0, dim, 0, dim), cmap='magma')  #Zth slice of array "mirror plane on z axis"
        cbar = ax.figure.colorbar(im, ax=ax)
             
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(X, Y, marker="x")
        
    
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        plt.show()
        x,y = np.meshgrid(np.arange(dim-1,1,-1),np.arange(1,dim-1,1))
        u = efieldx
        v = efieldy
        
        ax2.quiver(x,y,u,v)
        plt.show()
            
        ## FOR DATA CHECK BOOK
        
    elif problem == 1:
    
        if algo == 1:
        
            j = CurrentDist(dim)
            gaugez = np.zeros((dim, dim, dim))
            gaugez, init, fin = GaussPotentialMag(gaugez, j, dim)
            error = fin - init
            
            while error > tolerance:
                gaugez, init, fin = GaussPotentialMag(gaugez, j, dim)        
                error = fin - init
                #print(i)
    
        if algo == 0:
        
            charge_dist = CurrentDist(dim)
            init_gaugez = np.zeros((dim, dim, dim))
            gaugez = JacobiPotentialMag(init_gaugez, charge_dist, dim)
            error = abs(gaugez - init_gaugez)
            error = np.sum(error)
            
            while error > tolerance:
                init_gaugez = gaugez
                gaugez = JacobiPotentialMag(gaugez, charge_dist, dim)        
                error = abs(gaugez - init_gaugez)
                error = np.sum(error)
                
        mfieldx, mfieldy, mfieldz = MagField(gaugez, dim, int(dim/2))
                
        
        fig, ax = plt.subplots()
        im = ax.imshow(gaugez[int(dim/2), :, :], extent = (0, dim, 0, dim), cmap='magma')
        cbar = ax.figure.colorbar(im, ax=ax)
        
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        plt.show()
        x,y = np.meshgrid(np.arange(1,dim-1,1),np.arange(dim-1,1,-1))
        u = mfieldy
        v = mfieldx
        ax2.quiver(x,y,u,v)
        plt.show()
            
                
        
main()
        
                    
        
    