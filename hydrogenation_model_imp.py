""" SIMULATION MODEL - HYDROGENATION """
# Implicit model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ma
from time import perf_counter
from scipy.optimize import fsolve


def simulate_hydrogenation_model(k7, k8, k9, k10, 
                                 E7, E8, E10, R, T, P_hyd, n, 
                                 D, L, Time_hyd, u, epsilon, 
                                 C_feed_hyd_H2, C_feed_hyd_N2, C_CO2_init_hyd, C_H2O_init_hyd,
                                 theta_CO2_init_hyd, theta_H2O_init_hyd,
                                 rho, Omega, Nx, Nt):
    
    dx = L / (Nx - 1)
    dt = Time_hyd / Nt
    
    R_ = 0.08206               # (L.atm / mol.K)

    # Initialize concentration arrays
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_N2 = np.zeros((Nt + 1, Nx))
    C_total = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    r_CH4_hyd_array = np.zeros((Nt + 1, Nx))
    r_CO2_hyd_array = np.zeros((Nt + 1, Nx))
    r_H2O_hyd_array = np.zeros((Nt + 1, Nx))

    # Initialize partial pressure arrays
    P_CO2 = np.zeros((Nt + 1, Nx))
    P_H2 = np.zeros((Nt + 1, Nx))
    P_CH4 = np.zeros((Nt + 1, Nx))
    P_H2O = np.zeros((Nt + 1, Nx))
    P_N2 = np.zeros((Nt + 1, Nx))
    
    # Initial conditions
    P_CO2[0, 0] = 0                                 # No CO2 at the beginning
    P_H2[0, 0] = (C_feed_hyd_H2/100) * P_hyd        # H2 is being fed at the inlet at t = 0
    P_CH4[0, 0] = 0                                 # No CH4 at the beginning
    P_H2O[0, 0] = 0                                 # No H2O at the beginning
    P_N2[0, 0] = (C_feed_hyd_N2/100) * P_hyd
    
    C_CO2[0, :] = C_CO2_init_hyd
    C_H2[0, 0] = P_H2[0, 0] / (R_ * T)              # mmol/mL
    C_H2O[0, :] = C_H2O_init_hyd
    C_CH4[0, 0] = 0
    C_N2[0, 0] = P_N2[0, 0] / (R_ * T)              # mmol/mL
    
    theta_CO2[0, :] = theta_CO2_init_hyd            # Initial coverage factor from purge stage
    theta_H2O[0, :] = theta_H2O_init_hyd            # Initial coverage factor from purge stage (will be changed in the cycle model)

    # Equilibrium constant for the reaction at given temperature
    K_eq = np.exp(0.5032 * ((56000 / T**2) + (34633 / T) - (16.4 * np.log(T)) + (0.00557 * T)) + 33.165)

    # Pre-calculate constants
    k7_exp = k7 * np.exp(-E7 / (R * T))
    k8_exp = k8 * np.exp(-E8 / (R * T))
    k10_exp = k10 * np.exp(-E10 / (R * T))

    def backward_euler_equations(y, i):
        C_CO2_next, C_H2_next, C_CH4_next, C_H2O_next, C_N2_next, theta_CO2_next, theta_H2O_next = y

        # Calculate total concentration at t+1
        C_total_next = P_hyd * 101325 / (R * T) / 1000000

        # Calculate partial pressures at t+1
        P_CO2_next = C_CO2_next / C_total_next * P_hyd
        P_H2_next = C_H2_next / C_total_next * P_hyd
        P_CH4_next = C_CH4_next / C_total_next * P_hyd
        P_H2O_next = C_H2O_next / C_total_next * P_hyd

        # Reaction rate calculations at time t+1
        # Rate of CO2 desorption
        r_CO2_ads_next = k7_exp * theta_CO2_next * C_H2_next
        
        # Rate of formation of CH4
        Approach_to_Equilibrium_next = P_CO2_next * (P_H2_next**4) - (P_CH4_next * P_H2O_next**2) / K_eq
        absApproach_to_Equilibrium_next = abs(Approach_to_Equilibrium_next)

        if absApproach_to_Equilibrium_next <= 1e-6:
            r_CH4_hyd_next = k8_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium_next)) * absApproach_to_Equilibrium_next, Approach_to_Equilibrium_next)
        else:
            r_CH4_hyd_next = k8_exp * ma.copysign(absApproach_to_Equilibrium_next**n, Approach_to_Equilibrium_next)

        r_H2O_ads_next = k10_exp * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) - k9 * theta_H2O_next

        # Diffusion terms with Dankwert boundary conditions at time t+1
        if i == Nx - 1:  # Outlet boundary (last node)
            diffusion_CO2_next = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (-C_H2_next + C_H2[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (-C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (-C_N2_next + C_N2[t, i - 1]) / dx**2

        elif i == 1:  # Inlet boundary (first internal node)
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - C_H2_next) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - C_CH4_next) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (C_N2[t, i + 1] - C_N2_next) / dx**2

        else:  # Internal nodes
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - 2 * C_H2_next + C_H2[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - 2 * C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (C_N2[t, i + 1] - 2 * C_N2_next + C_N2[t, i - 1]) / dx**2

        # Convection terms (upwind scheme) at time t+1
        convection_CO2_next = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2_next = - (u * dt / epsilon) * (C_H2_next - C_H2[t, i - 1]) / dx
        convection_CH4_next = - (u * dt / epsilon) * (C_CH4_next - C_CH4[t, i - 1]) / dx
        convection_H2O_next = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx
        convection_N2_next = - (u * dt / epsilon) * (C_N2_next - C_N2[t, i - 1]) / dx

        # Implicit Euler update
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2_next - convection_CO2_next - (rho * (r_CO2_ads_next - r_CH4_hyd_next) * dt) / epsilon
        eq2 = C_CH4_next - C_CH4[t, i] - diffusion_CH4_next - convection_CH4_next - (rho * r_CH4_hyd_next * dt) / epsilon
        eq3 = C_H2_next - C_H2[t, i] - diffusion_H2_next - convection_H2_next - (rho * (-4 * r_CH4_hyd_next) * dt) / epsilon
        eq4 = C_H2O_next - C_H2O[t, i] - diffusion_H2O_next - convection_H2O_next - (rho * (-r_H2O_ads_next + 2 * r_CH4_hyd_next) * dt) / epsilon
        eq5 = C_N2_next - C_N2[t, i] - diffusion_N2_next - convection_N2_next

        # Update coverage factors to time t + 1
        # Desorption of CO2 from CO2-sites
        delta_theta_CO2_next = -r_CO2_ads_next * dt / Omega
        eq6 = theta_CO2_next - theta_CO2[t, i] - delta_theta_CO2_next

        delta_theta_H2O_next = (r_H2O_ads_next) * dt / Omega
        eq7 = theta_H2O_next - theta_H2O[t, i] - delta_theta_H2O_next

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]  # Corrected return statement

    # Time loop
    for t in range(0, Nt):

        # Boundary conditions at the inlet (i = 0)
        C_CO2[t+1, 0] = C_CO2[t, 0]                      # No CO2 is being fed
        C_H2[t+1, 0] = C_feed_hyd_H2 / 100 * P_hyd / (R_ * T)  # H2 is being fed at the inlet
        C_CH4[t+1, 0] = C_CH4[t, 0]                     # No CH4 is being fed
        C_H2O[t+1, 0] = C_H2O[t, 0]                      # No H2O is being fed
        C_N2[t+1, 0] = C_feed_hyd_N2 / 100 * P_hyd / (R_ * T)  # N2 is being fed at the inlet

        # Spatial loop (using backward Euler)
        for i in range(1, Nx):
            # Solve the implicit equations
            initial_guess = [C_CO2[t, i], C_H2[t, i], C_CH4[t, i], C_H2O[t, i], C_N2[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))  # Pass initial_guess as y_prev

            # Unpack the solution
            C_CO2[t+1, i], C_H2[t+1, i], C_CH4[t+1, i], C_H2O[t+1, i], C_N2[t+1, i], theta_CO2[t+1, i], theta_H2O[t+1, i] = solution

        # Calculate total concentration at t+1
        C_total[t+1, :] = P_hyd * 101325 / (R * T) / 1000000  # Assuming constant total pressure

        # Calculate partial pressures at t+1
        P_CO2[t+1, :] = C_CO2[t+1, :] / C_total[t+1, :] * P_hyd
        P_H2[t+1, :] = C_H2[t+1, :] / C_total[t+1, :] * P_hyd
        P_CH4[t+1, :] = C_CH4[t+1, :] / C_total[t+1, :] * P_hyd
        P_H2O[t+1, :] = C_H2O[t+1, :] / C_total[t+1, :] * P_hyd
        P_N2[t+1, :] = C_N2[t+1, :] / C_total[t+1, :] * P_hyd

        # Calculate reaction rates and store them in arrays
        for i in range(1, Nx):
            r_CO2_ads = k7_exp * theta_CO2[t+1, i] * C_H2[t+1, i]

            Approach_to_Equilibrium = P_CO2[t+1, i] * (P_H2[t+1, i]**4) - (P_CH4[t+1, i] * P_H2O[t+1, i]**2) / K_eq
            absApproach_to_Equilibrium = abs(Approach_to_Equilibrium)

            if absApproach_to_Equilibrium <= 1e-6:
                r_CH4_hyd = k8_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium)) * absApproach_to_Equilibrium, Approach_to_Equilibrium)
            else:
                r_CH4_hyd = k8_exp * ma.copysign(absApproach_to_Equilibrium**n, Approach_to_Equilibrium)

            r_H2O_ads = k10_exp * C_H2O[t+1, i] * (1 - theta_CO2[t+1, i] - theta_H2O[t+1, i]) - k9 * theta_H2O[t+1, i]

            r_CH4_hyd_array[t+1, i] = r_CH4_hyd
            r_CO2_hyd_array[t+1, i] = r_CO2_ads  # Corrected to r_CO2_ads
            r_H2O_hyd_array[t+1, i] = r_H2O_ads

    return (C_CO2, C_CH4, C_H2, C_H2O, 
            theta_CO2, theta_H2O, 
            P_H2, P_CO2, P_CH4, P_H2O,
            r_CH4_hyd_array, r_CO2_hyd_array, r_H2O_hyd_array)

