""" SIMULATION MODEL - ADSORPTION """
# Implicit model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.optimize import fsolve


# Function to simulate the model with adsorption of CO2 and H2O
def simulate_adsorption_model(k1, k2, k3, k4, k5, K_CO2, m, 
                              T, P_ads, D, L, Time_ads, u, epsilon, 
                              C_feed_ads_CO2, C_feed_ads_H2O, C_CO2_init_ads, C_H2O_init_ads, theta_CO2_init_ads, theta_H2O_init_ads, 
                              rho, Omega, Nx, Nt):
    dx = L / (Nx - 1)
    dt = Time_ads / Nt

    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    Theta_CO2_H2O = np.zeros((Nt + 1, Nx))

    R_ = 0.08206
    C_CO2_init = (C_feed_ads_CO2 / 100) * P_ads / (R_ * T)
    C_H2O_init = (C_feed_ads_H2O / 100) * P_ads / (R_ * T)

    C_CO2[0, 0] = C_CO2_init
    C_H2O[0, 0] = C_H2O_init
    theta_H2O[0, :] = theta_H2O_init_ads            # Initial coverage factor of H2O
    theta_CO2[0, :] = theta_CO2_init_ads            # Initial coverage factor of CO2
    Theta_CO2_H2O[0, :] = 0                     # Initial coverage factor of CO2/H2O joint adsorption

    def backward_euler_equations(y, i):
        C_CO2_next, C_H2O_next, theta_CO2_next, theta_H2O_next, Theta_CO2_H2O_next = y

        # Diffusion terms
        if i == Nx - 1:  # Outlet boundary condition
            diffusion_CO2 = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
        elif i == 1:  # Inlet boundary condition
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
        else:  # Internal nodes
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2

        # Convection terms
        convection_CO2 = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2O = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx

        # Reaction terms
        CO2_formation_rate = - k1 * C_CO2_next * (1 - theta_CO2_next - theta_H2O_next) - k2 * C_CO2_next * theta_H2O_next
        H2O_formation_rate = (
            k2 * C_CO2_next * theta_H2O_next -
            k3 * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) -
            k4 * (C_H2O_next * theta_CO2_next / (1 + K_CO2 * C_CO2_next**m)) +
            k5 * C_CO2_next * Theta_CO2_H2O_next
        )

        # Implicit equations
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2 - convection_CO2 - (rho * CO2_formation_rate * dt / epsilon)
        eq2 = C_H2O_next - C_H2O[t, i] - diffusion_H2O - convection_H2O - (rho * H2O_formation_rate * dt / epsilon)
        eq3 = theta_CO2_next - theta_CO2[t, i] + (CO2_formation_rate * dt / Omega)
        eq4 = theta_H2O_next - theta_H2O[t, i] + (H2O_formation_rate * dt / Omega)
        eq5 = Theta_CO2_H2O_next - Theta_CO2_H2O[t, i] + ((CO2_formation_rate + H2O_formation_rate) * dt / Omega)

        return [eq1, eq2, eq3, eq4, eq5]

    for t in range(0,Nt):
        # Apply boundary conditions for inlet
        C_CO2[t + 1, 0] = C_CO2_init
        C_H2O[t + 1, 0] = C_H2O_init
        # Update coverage factors

        for i in range(1, Nx):
            initial_guess = [C_CO2[t, i], C_H2O[t, i], theta_CO2[t, i], theta_H2O[t, i], Theta_CO2_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))
            C_CO2[t + 1, i], C_H2O[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i], Theta_CO2_H2O[t + 1, i] = solution
        
        theta_CO2[t + 1, i] = np.clip(theta_CO2[t + 1, i], 0, 1)
        theta_H2O[t + 1, i] = np.clip(theta_H2O[t + 1, i], 0, 1)
        Theta_CO2_H2O[t + 1, i] = np.clip(Theta_CO2_H2O[t + 1, i], 0, 1)

    return C_CO2, C_H2O, theta_CO2, theta_H2O, Theta_CO2_H2O

