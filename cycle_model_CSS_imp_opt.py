""" DFM CYCLE OPTIMISATION MODEL """
# Cyclic Steady State Condition (CSSC)
# Using Optuna for Bayesian Optimisation
# Generating Pareto Front Curves


import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from adsorption_model_imp import simulate_adsorption_model
from purge_model_imp import simulate_purge_model
from hydrogenation_model_imp import simulate_hydrogenation_model
from analyser_model import apply_analyser_delay

# -------------------------------------  Parameters  ------------------------------------- #


F_gas = 1200/60             # Flowrate of gas (mL/s)                                ~~~ VAR ~~~
#SV = 45000                  # Space Velocity (h-1)
#V_react = F_gas * 3600 / SV # Reactor (Tube) Volue (mL)
Di = 1.03                   # Reactor (Tube) Inside Diameter (cm)                   ~~~ VAR ~~~
Ai = 3.14 * (Di**2) / 4     # Reactor (Tube) Internal Surface Area (cm2)
#L = 1.92                   # Length of the Reactor (tube) (cm)
u = F_gas / Ai              # Linear velocity (cm/s) 24
W_DFM = 3.00                # Weight of DFM in the reactor (gr) 2.24 >> rho = 1.4   ~~~ VAR ~~~
rho = 1.4                 # Adsorption bed Density (g/cm^3) 1.4                   ~~~ VAR ~~~
V_react = W_DFM / rho       # Reactor (Tube) Volue (mL)
L = V_react / Ai            # Length of the Reactor (tube) (cm)

Omega = 0.38                # Maximum adsorption capacity (mmol/g)
Alfa = 0.8                  # Adsorption strength correction factor
D = 0.16                    # Diffusion coefficient (cm^2/s)
epsilon = 0.35              # Porosity 0.35
R = 8.314e-3                # Ideal gas constant (J/K.mmol)
R_ = 0.08206                # (L.atm / mol.K)
T = 623                     # Temperature (K) (Assuming isothermal operation)
P_ads = 1                   # Total pressure during adsorption (atm)
P_prg = 1                   # Total pressure during purge (atm)
P_hyd = 1                   # Total pressure during hydrogenation (atm)             ~~~ VAR ~~~

#Time_ads = 150              # Total simulation time (s) for adsorption 150          ~~~ VAR ~~~
#Time_prg = 120              # Total simulation time (s) for purge 120               ~~~ VAR ~~~
#Time_hyd = 300              # Total simulation time (s) for hydrogenation 300       ~~~ VAR ~~~
#Time_prg2 = 120             # Total simulation time (s) for the 2nd purge 120       ~~~ VAR ~~~
Nx = 5                      # Number of spatial grid points
Nt_ads = 3000               # Number of time steps 3000
Nt_prg = 2000               # Number of time steps 2000
Nt_hyd = 5000               # Number of time steps 5000
Nt_prg2 = 2000              # Number of time steps 2000

C_feed_ads_CO2 = 5.7        # CO2 %vol in feed gas - Adsorption stage (mmol/cm^3)   ~~~ VAR ~~~
C_feed_ads_H2O = 0          # H2O %vol in feed gas - Adsorption stage (mmol/cm^3)
C_feed_purge_CO2 = 0        # CO2 %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_purge_H2O = 0        # H2O %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_hyd_H2 = 5.7         # H2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 5.7%   ~~~ VAR ~~~
C_feed_hyd_N2 = 100 - C_feed_hyd_H2        # N2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 94.3%

E6 = 25                   # Activation energy (J/mmol) Check the unit!! 27.7
E7 = 14                     # Activation energy for desorption of CO2 from the adsorption sites during Hydrogenation stage(J/mmol) 15 ; opt:12
E8 = 52                   # Activation energy for formation of CH4 during Hydrogenation stage (J/mmol) 68.1 >> 55
E10 = 10                    # Activation energy for adsorption of H2O on the adsorption sites during Hydrogenation stage (J/mmol) 10

# Optimized parameters from fitting process
k1 = 100                    # Optimized kinetic constant for ads. of CO2 on CO2 sites (cm3/s.g) 60
k2 = 100                     # Optimized kinetic constant for ads. of H2O on CO2 sites (cm3/s.g) 12
k3 = 300                     # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g) 19.6
k4 = 200                    # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g) 75.1
k5 = 100                     # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g) 19.8
k6 = 0.005                  # Optimized pre-exponential constant for purge of CO2 from CO2 sites (mmol/g.s) 0.005
k7 = 60                     # Optimized kinetic constant for desorption of CO2 from CO2 sites during hydrogenation (cm3/s.g) 24.9
k8 = 18.26e3                # Optimized kinetic constant for CO2 hydrogenation (mmol/g.s.atm^5n) 18.26
k9 = 0.005                  # Optimized kinetic constant for des. of H2O from H2O sites during hydrogenation(cm3/s.g) 0.003
k10 = 175                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g) 174.7

K_CO2 = 400                 # Optimized adsorption constant of CO2 (cm3/mmol) 375.3
m = 1                       # Adjust parameter in H2O adsorption 1
n = 0.14                    # Adjust parameter in CH4 production 0.14


def calculate_recovery(C_feed_ads_CO2, C_CH4_hyd, F_gas, Time_ads, Time_hyd):
    """Calculate recovery using steady-state data."""
    # Calculate uniform time step sizes
    dt_hyd = Time_hyd / (len(C_CH4_hyd) - 1)  # Hydrogenation time step

    # Integrate using trapezoidal rule
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)  # CH4 produced in hydrogenation
    total_CO2_in = F_gas * (P_ads*C_feed_ads_CO2/(R_ * T * 100)) * Time_ads  # CO2 fed in adsorption
    recovery = round(total_CH4_out / total_CO2_in , 2)

    return recovery


def calculate_productivity(C_CH4_hyd, F_gas, W_DFM, Time_cycle, Time_hyd):
    """Calculate productivity using steady-state data."""
    dt_hyd = Time_hyd / (len(C_CH4_hyd) - 1)  # Hydrogenation time step
    
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)  # Total CH4 produced
    productivity = round(total_CH4_out / (W_DFM * Time_cycle) , 5)  # Productivity
    return productivity*1000


def calculate_purity(C_CH4_hyd, C_H2_hyd, F_gas, Time_hyd):
    """Calculate purity using steady-state data."""
    dt_hyd = Time_hyd / (len(C_CH4_hyd) - 1)  # Hydrogenation time step
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)  # CH4 produced
    total_hydrogenation_out = np.trapz(F_gas * (C_CH4_hyd + C_H2_hyd), dx=dt_hyd)  # Total gas output
    purity = round(total_CH4_out / total_hydrogenation_out , 2)
    return purity


def objective(trial):
    """
    Objective function for Optuna optimization, running until CSSC.

    Args:
        trial: Optuna trial object.

    Returns:
        tuple: A tuple containing recovery, productivity, and purity.
    """
    Time_ads = trial.suggest_int('Time_ads', 20, 200)
    Time_prg = trial.suggest_int('Time_prg', 20, 180)
    Time_hyd = trial.suggest_int('Time_hyd', 20, 400)
    Time_prg2 = trial.suggest_int('Time_prg2', 20, 180)

    total_cycle_time = Time_ads + Time_prg + Time_hyd + Time_prg2

    # Initialize for CSSC loop
    C_CO2, C_H2O, theta_CO2, theta_H2O = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
    C_CO2_init_ads, C_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)             #initial concentration of CO2 and H2O inside the column (before running the cycle)
    theta_CO2_init_ads, theta_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)     #initial coverage area for CO2 and H2O inside the column (before running the cycle)
    
    # --- Begin CSSC loop --- #
    for cycle in range(10):
        print(f"Cycle {cycle+1}")

        # Adsorption
        C_CO2_ads, C_H2O_ads, theta_CO2_ads, theta_H2O_ads, Theta_CO2_H2O_ads = simulate_adsorption_model(
            k1, k2, k3, k4, k5, K_CO2, m, T, P_ads, D, L, Time_ads, u, epsilon,
            C_feed_ads_CO2, C_feed_ads_H2O, C_CO2_init_ads, C_H2O_init_ads, theta_CO2_init_ads, theta_H2O_init_ads,
            rho, Omega, Nx, Nt_ads
        )

        # Purge (Note: Theta_CO2_H2O_prg is deleted in the relevant code and hence, not received here)
        C_CO2_prg, C_H2O_prg, C_CH4_prg, C_H2_prg, theta_CO2_prg, theta_H2O_prg= simulate_purge_model(
            k2, k3, k4, k5, k6, K_CO2, m, E6, R, T, P_prg, Alfa, D, L, Time_prg, u, epsilon,
            C_feed_purge_CO2, C_feed_purge_H2O, 
            C_CO2_ads[-1, :], C_H2O_ads[-1, :], 0, 0, theta_H2O_ads[-1, :], theta_CO2_ads[-1, :], Theta_CO2_H2O_ads[-1, :], 
            rho, Omega, Nx, Nt_prg
        )

        # Hydrogenation
        C_CO2_hyd, C_CH4_hyd, C_H2_hyd, C_H2O_hyd, theta_CO2_hyd, theta_H2O_hyd, *_ = simulate_hydrogenation_model(
            k7, k8, k9, k10, E7, E8, E10, R, T, P_hyd, n, D, L, Time_hyd, u, epsilon,
            C_feed_hyd_H2, C_feed_hyd_N2,
            C_CO2_prg[-1, :], C_H2O_prg[-1, :], theta_CO2_prg[-1, :], theta_H2O_prg[-1, :], 
            rho, Omega, Nx, Nt_hyd
        )

        # Final Purge (CH4 and H2 needs to be added in future) , Note: Theta_CO2_H2O_prg2 is deleted in the relevant code and hence, not received here
        C_CO2_prg2, C_H2O_prg2, C_CH4_prg2, C_H2_prg2, theta_CO2_prg2, theta_H2O_prg2 = simulate_purge_model(
            k2, k3, k4, k5, k6, K_CO2, m, E6, R, T, P_prg, Alfa, D, L, Time_prg, u, epsilon,
            C_feed_purge_CO2, C_feed_purge_H2O, 
            C_CO2_hyd[-1, :], C_H2O_hyd[-1, :], C_CH4_hyd[-1, :], C_H2_hyd[-1, :], theta_H2O_hyd[-1, :], theta_CO2_hyd[-1, :], 0, 
            rho, Omega, Nx, Nt_prg2
        )

        # Convergence check
        if (
            np.all(np.abs(C_CO2_prg2[-1, :] - C_CO2) < 1e-4) and
            np.all(np.abs(C_H2O_prg2[-1, :] - C_H2O) < 1e-4) and
            np.all(np.abs(theta_CO2_prg2[-1, :] - theta_CO2) < 1e-4) and
            np.all(np.abs(theta_H2O_prg2[-1, :] - theta_H2O) < 1e-4)
        ):
            break       # Exit CSSC loop if converged

        # Update for next cycle
        C_CO2, C_H2O = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
        theta_CO2, theta_H2O = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
        C_CO2_init_ads, C_H2O_init_ads = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
        theta_CO2_init_ads, theta_H2O_init_ads = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
    # --- End CSSC loop --- #

    # Calculate metrics using the steady-state cycle data
    
    recovery = calculate_recovery(C_feed_ads_CO2, C_CH4_hyd[:, -1], F_gas, Time_ads, Time_hyd)
    productivity = calculate_productivity(C_CH4_hyd[:, -1], F_gas, W_DFM, total_cycle_time, Time_hyd)
    purity = calculate_purity(C_CH4_hyd[:, -1], C_H2_hyd[:, -1], F_gas, Time_hyd)
    
    #return recovery, productivity
    return recovery, productivity, purity
    #return productivity

# Defining a custom sampler in Optuna that prioritises the lower and upper bounds of search space
class BoundaryPrioritizedSampler(BaseSampler):
    def __init__(self, base_sampler=None):
        self.base_sampler = base_sampler or optuna.samplers.TPESampler()
        self.bound_priority_list = []  # Track which bounds to prioritize next

    def infer_relative_search_space(self, study, trial):
        return self.base_sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        # Default to base sampler for relative search space sampling
        return self.base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # If bounds are not yet exhausted, prioritize bounds
        if param_name not in self.bound_priority_list:
            self.bound_priority_list.append(param_name)
            return param_distribution.low  # Prioritize lower bound first
        elif self.bound_priority_list.count(param_name) == 1:
            self.bound_priority_list.append(param_name)
            return param_distribution.high  # Then prioritize upper bound
        else:
            # Default to base sampler once bounds are sampled
            return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

   
if __name__ == "__main__":
    start_time = perf_counter()

    # Create Optuna study with multi-objective optimization
    study = optuna.create_study(
        #directions=["maximize", "maximize"],
        directions=["maximize", "maximize", "maximize"],
        #directions=["maximize"],
        sampler=optuna.samplers.NSGAIISampler()         # for multi-objective optimisation
        #sampler=BoundaryPrioritizedSampler()           # for single objective and to priorotise the boundes
    )
    study.optimize(objective, n_trials=5)

    # Extract Pareto front solutions
    pareto_front = study.best_trials                    # for Multi-objectives
    #best_trial = study.best_trial                      # for Single objectives

    #print(f"Best parameters: {best_trial.params}")
    #print(f"Best productivity: {best_trial.value}")

    # Measure execution time
    duration = perf_counter() - start_time
    print(f"Execution Time: {duration / 60:.1f} minutes")

    # Plotting and Reporting the Pareto points
    # Store Pareto points with variable data
    pareto_points = []
    for trial in pareto_front:
        pareto_points.append({
            "recovery": trial.values[0],
            "productivity": trial.values[1],
            "purity": trial.values[2],
            "Time_ads": trial.params["Time_ads"],
            "Time_prg": trial.params["Time_prg"],
            "Time_hyd": trial.params["Time_hyd"],
            "Time_prg2": trial.params["Time_prg2"]
        })
    # Print or store the pareto_points list
    for point in pareto_points:
        print(point)
    # Convert pareto_points list to a DataFrame
    pareto_df = pd.DataFrame(pareto_points)

    # Save the DataFrame to an Excel file
    pareto_df.to_excel("pareto_points.xlsx", index=False)

    print("Pareto points exported to 'pareto_points.xlsx'")
    # Generate Pareto curves
    recovery_pareto = [trial.values[0] for trial in pareto_front]
    productivity_pareto = [trial.values[1] for trial in pareto_front]
    purity_pareto = [trial.values[2] for trial in pareto_front]

    # Recovery vs. Productivity
    plt.figure()
    plt.scatter(recovery_pareto, productivity_pareto)
    plt.xlabel("Recovery")
    plt.ylabel("Productivity")
    plt.title("Pareto Front: Recovery vs. Productivity")

    # Productivity vs. Purity
    plt.figure()
    plt.scatter(productivity_pareto, purity_pareto)
    plt.xlabel("Productivity")
    plt.ylabel("Purity")
    plt.title("Pareto Front: Productivity vs. Purity")

    # Recovery vs. Purity
    plt.figure()
    plt.scatter(recovery_pareto, purity_pareto)
    plt.xlabel("Recovery")
    plt.ylabel("Purity")
    plt.title("Pareto Front: Recovery vs. Purity")

    plt.show()

    # Measure execution time
    duration = perf_counter() - start_time
    print(f"Execution Time: {duration / 60:.2f} minutes")