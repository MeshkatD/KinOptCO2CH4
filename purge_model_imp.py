""" SIMULATION MODEL - PURGE """
# Implicit model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.optimize import fsolve


# Function to simulate the model with adsorption of CO2 and H2O
def simulate_purge_model(k2, k3, k4, k5, k6, K_CO2, m, 
                         E6, R, T, P_prg, Alfa,
                         D, L, Time_prg, u, epsilon, 
                         C_feed_purge_CO2, C_feed_purge_H2O, C_CO2_init_prg, C_H2O_init_prg,
                         C_CH4_init_prg, C_H2_init_prg,
                         theta_H2O_init_prg, theta_CO2_init_prg, theta_CO2_H2O_init_prg, 
                         rho, Omega, Nx, Nt):
    dx = L / (Nx - 1)
    dt = Time_prg / Nt
    x = np.linspace(0, L, Nx)
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    Theta_CO2_H2O = np.zeros((Nt + 1, Nx))
    
    R_ = 0.08206               # (L.atm / mol.K)
    C_CO2_init = (C_feed_purge_CO2/100) * P_prg / (R_ * T)
    C_H2O_init = (C_feed_purge_H2O/100) * P_prg / (R_ * T)
    
    # Initial condition (linked to the previous stage)
    C_CO2[0, :] = C_CO2_init_prg                # Initial concentration of CO2 in the column before purging
    C_H2O[0, :] = C_H2O_init_prg                # Initial concentration of CO2 in the column before purging
    C_CH4[0, :] = C_CH4_init_prg                # Initial concentration of CO2 in the column before purging
    C_H2[0, :] = C_H2_init_prg                # Initial concentration of CO2 in the column before purging
    theta_H2O[0, :] = theta_H2O_init_prg        # Initial coverage factor of H2O from adsorption stage
    theta_CO2[0, :] = theta_CO2_init_prg        # Initial coverage factor of CO2 from adsorption stage
    Theta_CO2_H2O[0, :] = theta_CO2_H2O_init_prg             # Initial coverage factor of CO2/H2O joint adsorption
    
    # Pre-calculate constants
    #k6_exp = k6 * np.exp(-E6 / (R * T))
    #k6_exp = 0.005
    def backward_euler_equations(y, i):
        #C_CO2_next, C_H2O_next, C_CH4_next, C_H2_next, theta_CO2_next, theta_H2O_next, Theta_CO2_H2O_next = y
        C_CO2_next, C_H2O_next, C_CH4_next, C_H2_next, theta_CO2_next, theta_H2O_next = y

        # Diffusion terms
        if i == Nx - 1:  # Outlet boundary condition
            diffusion_CO2_next = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (-C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (-C_H2_next + C_H2[t, i - 1]) / dx**2

        elif i == 1:  # Inlet boundary condition
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - C_CH4_next) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - C_H2_next) / dx**2

        else:  # Internal nodes
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - 2 * C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - 2 * C_H2_next + C_H2[t, i - 1]) / dx**2


        # Convection terms
        convection_CO2_next = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2O_next = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx
        convection_CH4_next = - (u * dt / epsilon) * (C_CH4_next - C_CH4[t, i - 1]) / dx
        convection_H2_next = - (u * dt / epsilon) * (C_H2_next - C_H2[t, i - 1]) / dx

        # Reaction terms  (The math contradiction (i.e., + instead of x) must be taken into account in future)
        CO2_formation_rate_next = k6 * np.exp((-E6 / (R * T)) * (1 - Alfa * theta_CO2_next)) * theta_CO2_next  # I've change "*" (as in the original paper) to "+" for multiplication of exponents
        H2O_formation_rate_next = k2 * C_CO2_next * theta_H2O_next - k3 * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) \
            - k4 * (C_H2O_next * theta_CO2_next / (1 + K_CO2 * C_CO2_next**m)) #+ k5 * C_CO2_next * Theta_CO2_H2O_next

        # Implicit equations
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2_next - convection_CO2_next - (rho * CO2_formation_rate_next * dt / epsilon)
        eq2 = C_H2O_next - C_H2O[t, i] - diffusion_H2O_next - convection_H2O_next - (rho * H2O_formation_rate_next * dt / epsilon)
        eq3 = theta_CO2_next - theta_CO2[t, i] + (CO2_formation_rate_next * dt / Omega)
        eq4 = theta_H2O_next - theta_H2O[t, i] + (H2O_formation_rate_next * dt / Omega)
        #eq5 = Theta_CO2_H2O_next - Theta_CO2_H2O[t, i] + ((CO2_formation_rate_next + H2O_formation_rate_next) * dt / Omega)
        eq6 = C_CH4_next - C_CH4[t, i] - diffusion_CH4_next - convection_CH4_next
        eq7 = C_H2_next - C_H2[t, i] - diffusion_H2_next - convection_H2_next

        #return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
        return [eq1, eq2, eq3, eq4, eq6, eq7]

 
    for t in range(0,Nt):
        C_CO2[t + 1, 0] = C_CO2_init
        C_H2O[t + 1, 0] = C_H2O_init
        C_CH4[t + 1, 0] = 0
        C_H2[t + 1, 0] = 0

        for i in range(1, Nx):
            #initial_guess = [C_CO2[t, i], C_H2O[t, i], C_CH4[t, i], C_H2[t, i], theta_CO2[t, i], theta_H2O[t, i], Theta_CO2_H2O[t, i]]
            initial_guess = [C_CO2[t, i], C_H2O[t, i], C_CH4[t, i], C_H2[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))
            #C_CO2[t + 1, i], C_H2O[t + 1, i], C_CH4[t + 1, i], C_H2[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i], Theta_CO2_H2O[t + 1, i] = solution
            C_CO2[t + 1, i], C_H2O[t + 1, i], C_CH4[t + 1, i], C_H2[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i] = solution

        # Apply boundary conditions for inlet
        theta_CO2[t + 1, i] = np.clip(theta_CO2[t + 1, i], 0, 1)
        theta_H2O[t + 1, i] = np.clip(theta_H2O[t + 1, i], 0, 1)
        #Theta_CO2_H2O[t + 1, i] = np.clip(Theta_CO2_H2O[t + 1, i], 0, 1)

    #return C_CO2, C_H2O, C_CH4, C_H2, theta_CO2, theta_H2O, Theta_CO2_H2O
    return C_CO2, C_H2O, C_CH4, C_H2, theta_CO2, theta_H2O
