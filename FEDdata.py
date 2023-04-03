#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:42:55 2023

@author: djordjemihajlovic
"""

from CH import Update
import matplotlib.pyplot as plt

def main():
    
    phi = float(input("Initial phi = "))
    N = int(input("Dimension = "))
    M = float(input("Value parameter M = "))
    a = float(input("Value parameter a = "))
    k = float(input("Value parameter k = "))
    data = []
    iteration = []
    CH_sim = Update(phi, N, M, a, k, dx=1, dt=1) 
    for i in range (0, 100000):
        CH_sim.C_P()
        CH_sim.O_P()
        x = CH_sim.F_E_D()
        
        if phi == 0:

            file = open("Free Energy Density phi[50x50]0.txt", "a") 
            file.write("iteration: " + str(i) + ". Free energy density: " + str(x) +".\n")
            file.close()
            
        elif phi == 0.5:
            
            file = open("Free Energy Density phi[50x50]0.5.txt", "a") 
            file.write("iteration: " + str(i) + ". Free energy density: " + str(x) +".\n")
            file.close()
            
        data.append(x)
        iteration.append(i)
    
    plt.plot(iteration, data)
    plt.ylabel("Free Energy Density.")
    plt.xlabel("Iteration.")
    if phi == 0:
        plt.title("ϕ0 free energy vs time.")
    elif phi == 0.5:
        plt.title("ϕ0.5 free energy vs time.")
    
main()