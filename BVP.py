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

################################################################################

@njit
def EField(phi, N):  # Chooses plane to find Efield values

    Ex = np.zeros((N,N,N))
    Ey = np.zeros((N,N,N))
    Ez = np.zeros((N,N,N))
    E_total = np.zeros((N,N,N))
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                
                R = phi[i][j][k+1] # Neighbours to the right, left, top, bottom, forward & backward
                L = phi[i][j][k-1]
                T = phi[i][(j+1)][k]
                Bo = phi[i][(j-1)][k]
                F = phi[(i+1)][j][k]
                Ba = phi[(i-1)][j][k]
                
                x = -(R-L)*(1/2)  # Derivatives wrt x, y, z etc
                z = -(F-Ba)*(1/2)
                y = -(T-Bo)*(1/2)
                
                E_total[i][j][k] = math.sqrt((x**2)+(y**2)+(z**2))  # Vector magnitude
                
                normx = x/E_total[i][j][k]  # Normalization of field
                normy = y/E_total[i][j][k]
                normz = z/E_total[i][j][k]
                
                Ex[i][j][k] = normx
                Ey[i][j][k] = normy
                Ez[i][j][k] = normz
                
        
    return Ex, Ey, Ez, E_total              

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
                Ba = phi[i][j][(k-1)]  
            
                rho_point = rho[i][j][k]  # Defined rho_point so equation below was easier to follow
    
                update_phi[i][j][k] = (1/6) * (T+Bo+R+L+F+Ba+rho_point)  # Coded update rule 

    phi = update_phi    
    return phi

@njit
def GaussPotential(phi, rho, N):  # Note here just updating array as we go through lattice (Gauss Seidl)
    init = np.sum(phi)
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):  # range 1 to N-1 to keep boundaries 0.
                
                
                T = phi[(i+1)][j][k] #neighbours to the right, left, top, bottom, forward & backward
                Bo = phi[(i-1)][j][k]
                R = phi[i][(j+1)][k]
                L = phi[i][(j-1)][k]
                F = phi[i][j][(k+1)]
                Ba = phi[i][j][(k-1)]  
            
                rho_point = rho[i][j][k]  # Defined rho_point so equation below was easier to follow
    
                phi[i][j][k] = (1/6) * (T+Bo+R+L+F+Ba+rho_point)  # Coded update rule 
                
    fin = np.sum(phi)   
    return phi, init, fin


def Data(phi, e, N):  # Takes calculated results and compares with distance to center
    e_data_point = []
    potential_data_point = []
    distance = []
    center = N/2
    for i in range(1, N-1):
        for j in range(1, N-1):
                
            x_dist = i - center
            y_dist = j - center   
            centre_dist = math.sqrt((x_dist**2)+ (y_dist**2))
            
            potential_data_point.append(phi[int(center)][i][j])
            e_data_point.append(e[int(center)][i][j])
            distance.append(centre_dist)
                
    return distance, potential_data_point, e_data_point           
                

def ChargeDist(N, system):  # Set up charge distribution (choice of central charge or arbitrary charge dist.)
    if system == 0:
        rho = np.random.uniform(size=(N, N, N))
    elif system == 1:
        rho = np.zeros((N, N, N))
        mid = int(N/2)
        rho[mid][mid][mid] = 1
        
    return rho

##########################################################################################

## Below are methods used for the magnetic problem, most of which are recycled from the electric field problem.

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
                Ba = Az[m][l][(n-1)]  
            
                j_point = j[m][l][n]  # Defined j_point so equation below was easier to follow
    
                update_Az[m][l][n] = (1/6) * (T+Bo+R+L+F+Ba+j_point)  # Coded update rule 
                
    Az = update_Az    
    return Az

@njit
def GaussPotentialMag(Az, j, N):  #Similar methods to Electric problem
    init = np.sum(Az)
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
    fin = np.sum(Az)
   
    return Az, init, fin


def CurrentDist(N):  # Sets wire in center spanning the entire lattice cube 
    j = np.random.uniform(size=(N, N, N))
    j = np.zeros((N, N, N))
    mid = int(N/2)
    for n in range(1, N-1):
        j[n][mid][mid] = 1
    
    return j

@njit
def MagField(Az, N, plane):  # Chooses plane to find Efield values

    Mx = np.zeros((N,N,N))
    My = np.zeros((N,N,N))
    Mz = np.zeros((N,N,N))
    M_total = np.zeros((N,N,N))
    for j in range(1, N-1):
        for k in range(1, N-1):
            
            R = Az[plane][j][k+1] #neighbours to the right, left, top, bottom, forward & backward
            L = Az[plane][j][k-1]
            F = Az[(plane+1)][j][k]
            Ba = Az[(plane-1)][j][k]
            T = Az[plane][(j+1)][k]
            Bo = Az[plane][(j-1)][k]
            
            y = -(R-L)*(1/2)  # By = -dA/dx
            z = (F-Ba)*(1/2)
            x = (T-Bo)*(1/2)  # Bx = dA/dy
            
            M_total[plane][j][k] = math.sqrt((x**2) + (y**2) + (z**2))  #Magnetic vector magnitude
        
            normx = x/M_total[plane][j][k]
            normy = y/M_total[plane][j][k]
            normz = z/M_total[plane][j][k]
            
            Mx[plane][j][k] = normx
            My[plane][j][k] = normy
            Mz[plane][j][k] = normz
        
    return Mx, My, Mz, M_total

def DataMag(gaugez, m, N):  #Same method as electric problem to scale data according to distance from center
    m_data_point = []
    A_data_point = []
    distance = []
    center = N/2
    for i in range(1, N-1):
        for j in range(1, N-1):
                
            x_dist = i - center
            y_dist = j - center   
            centre_dist = math.sqrt((x_dist**2)+ (y_dist**2))
            
            A_data_point.append(gaugez[int(center)][i][j])
            m_data_point.append(m[int(center)][i][j])
            distance.append(centre_dist)
                
    return distance, A_data_point, m_data_point   


################################################################################

@njit
def SORPotential(phi, rho, N, w):  #SOR
    
    init = phi
    diff = np.zeros((N,N,N))
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
                phi_point = phi[i][j][k]
    
                phi[i][j][k] = ((1-w)*phi_point) + (w*((1/6) * (T+Bo+R+L+F+Ba+rho_point)))  # Coded update rule 
                diff[i][j][k] = phi[i][j][k]-phi_point
   
    return phi, diff


################################################################################
    
                
# The main function here essentially prompts the user for the required simulation and runs the code accordingly
    
def main():
    
    print("Boundary value problem simulator - Poisson's equation. ")
    problem = int(input("Simulator (0 for Electric field), (1 for Magnetic field): "))
    algo = int(input("Algorithm (0 for Jacobi), (1 for Gauss), (2 for SOR *only electric field*): "))
    dim = int(input("Dimension of simulation: "))
    tolerance = float(input("Accuracy of simulation: "))
    
    if problem == 0:
        
        sim = int(input("Simulation type, 0 for arbitrary charge dist., 1 for single charge in box center: "))
        
        if algo == 2:
            
            m = int(input("Find optimum w? Yes(1)/No(0): "))
            if m == 0:
                w = float(input("Relaxation parameter: "))
                time = 0
            
                charge_dist = ChargeDist(dim, sim)
                potential = np.zeros((dim, dim, dim))
                potential, diff = SORPotential(potential, charge_dist, dim, w)
                error = np.sum(np.abs(diff))
                
                while error > tolerance:
                    potential, diff = SORPotential(potential, charge_dist, dim, w)        
                    error = np.sum(np.abs(diff))
                    time += 1
            
            
            if m == 1:
                
                time_data = []
                
                w = np.arange(1, 2, 0.01)
                for i in range(0, len(w)):
                    omega = w[i]
                    time = 0
                    charge_dist = ChargeDist(dim, sim)
                    potential = np.zeros((dim, dim, dim))
                    potential, diff = SORPotential(potential, charge_dist, dim, omega)
                    error = np.sum(np.abs(diff))
                    
                    while error > tolerance:
                        time += 1
                        potential, diff = SORPotential(potential, charge_dist, dim, omega)        
                        error = np.sum(np.abs(diff))
                        
                    time_data.append(time)
                    print("Done " + str(time_data[i]))
                    #Below is commented out so as to not overwrite data already generated
                    
                    # file = open("SOR w relaxation value vs time.txt", "a")
                    # file.write(str(omega) + " " + str(time) + "\n")
                    # file.close()
                    
                f_sor = plt.figure()
                ax_sor = f_sor.add_subplot(111)
                ax_sor.plot(w, time_data, marker="x")
                f_sor.suptitle("Value of over relaxation vs iterations to convergence.")
                print(time_data)
                        
                    
                    
    
        if algo == 1:
            time = 0
        
            charge_dist = ChargeDist(dim, sim)
            potential = np.zeros((dim, dim, dim))
            potential, init, fin = GaussPotential(potential, charge_dist, dim)
            error = abs(fin - init)
            
            while error > tolerance:
                potential, init, fin = GaussPotential(potential, charge_dist, dim)        
                error = abs(fin - init)
                time +=1
            print(time)

    
        if algo == 0:
            time = 0
        
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
                time +=1
                
            print(time)
                
                

        efieldx, efieldy, efieldz, efield_total = EField(potential, dim)
        
        X,Y,Z = Data(potential, efield_total, dim)
        
        ## Below is kept commented - was only needed to generate data files and graphs; kept commented so data doesn't get written over
        
        # for i in range(1, dim-1):
        #     for j in range(1, dim-1):
        #         file = open("50:[100x100] Potential vs Efield(x,y,x).txt", "a") 
        #         file.write(str(i) + " " + str(j) + " 50 " + " " + str(potential[int(dim/2)][j][i]) + " " + str(efieldx[int(dim/2)][j][i]) + " " + str(efieldy[int(dim/2)][j][i]) + " " + str(efieldz[int(dim/2)][j][i]) + "\n")
        #         file.close()
        
        # for k in range(0, len(X)):
        #     file2 = open("50:[100x100] Distance vs Potential vs Efield(magnitude).txt", "a")
        #     file2.write(str(X[k]) + " " + str(Y[k]) + " " + str(Z[k]) + "\n")
        #     file2.close()
            
            
        fig, ax = plt.subplots()
        plt.title("Electric Potential 'Heatmap' at cut z=[50]")
        im = ax.imshow(potential[int(dim/2), :, :], extent = (0, dim, 0, dim), cmap='magma')  #Zth slice of array "mirror plane on z axis"
        cbar = ax.figure.colorbar(im, ax=ax)
             
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(X, Y, marker="x")
        f1.suptitle("Potential vs distance to center")
        
        f1_alt = plt.figure()
        ax1_alt = f1_alt.add_subplot(111)
        ax1_alt.loglog(X, Y, marker="x")
        f1_alt.suptitle("Log log plot - Potential vs distance to center")

        
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.plot(X, Z, marker="x")
        f2.suptitle("Efield vs distance to center")
        
        f2_alt = plt.figure()
        ax2_alt = f2_alt.add_subplot(111)
        ax2_alt.loglog(X, Z, marker="x")
        f2_alt.suptitle("Log log plot - Efield vs distance to center")


        f3 = plt.figure()
        ax3 = f3.add_subplot(111)
        x,y = np.meshgrid(np.arange(0,dim,1),np.arange(0,dim,1))
        u = efieldx[int(dim/2), :, :]
        v = efieldy[int(dim/2), :, :]
        ax3.quiver(x,y,u,v)
        f3.suptitle("Electric field at cut z = [50]")
        
        plt.show()
            
        ## FOR DATA CHECK BOOK
        
    elif problem == 1:
    
        if algo == 1:
        
            j = CurrentDist(dim)
            gaugez = np.zeros((dim, dim, dim))
            gaugez, init, fin = GaussPotentialMag(gaugez, j, dim)
            error = abs(fin - init)
            
            while error > tolerance:
                gaugez, init, fin = GaussPotentialMag(gaugez, j, dim)        
                error = abs(fin - init)
    
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
                
        mfieldx, mfieldy, mfieldz, mfield_total = MagField(gaugez, dim, int(dim/2))
        
        X,Y,Z = DataMag(gaugez, mfield_total, dim)
        
        ## Kept commented to not overrwrite data files already calculated - see folder 'Data'
        
        # for i in range(1, dim-1):
        #     for j in range(1, dim-1):
        #         file = open("50:[100x100] Potential(Az) vs Magfield(x,y,x).txt", "a") 
        #         file.write(str(i) + " " + str(j) + " 50 " + " " + str(gaugez[int(dim/2)][j][i]) + " " + str(mfieldx[int(dim/2)][j][i]) + " " + str(mfieldy[int(dim/2)][j][i]) + " " + str(mfieldz[int(dim/2)][j][i]) + "\n")
        #         file.close()
        
        # for k in range(0, len(X)):
        #     file2 = open("50:[100x100] Distance vs Potential(Az) vs Magfield(magnitude).txt", "a")
        #     file2.write(str(X[k]) + " " + str(Y[k]) + " " + str(Z[k]) + "\n")
        #     file2.close()
                
        
        fig, ax = plt.subplots()
        plt.title("Magnetic potential 'Heatmap' at cut z=[50]")
        im = ax.imshow(gaugez[int(dim/2), :, :], extent = (0, dim, 0, dim), cmap='magma')
        cbar = ax.figure.colorbar(im, ax=ax)
        
        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(X, Y, marker="x")
        f1.suptitle("MagPotential vs distance to center")
        
        f1_alt = plt.figure()
        ax1_alt = f1_alt.add_subplot(111)
        ax1_alt.plot(X, Y, marker="x")
        plt.semilogx()
        f1_alt.suptitle("Log log plot - MagPotential vs distance to center")

        
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.plot(X, Z, marker="x")
        f2.suptitle("Magfield vs distance to center")
        
        f2_alt = plt.figure()
        ax2_alt = f2_alt.add_subplot(111)
        ax2_alt.loglog(X, Z, marker="x")
        f2_alt.suptitle("Log log plot - Magfield vs distance to center")
        
        f3 = plt.figure()
        ax3 = f3.add_subplot(111)
        plt.show()
        x,y = np.meshgrid(np.arange(0,dim,1),np.arange(0,dim,1))
        v = mfieldy[int(dim/2), :, :]
        u = mfieldx[int(dim/2), :, :]
        ax3.quiver(x,y,u,v)
        f3.suptitle("Magfield vector plot at cut z = [50]")
        plt.show()
            
                
        
main()
        
                    
        
    