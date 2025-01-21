""" ANALYZER MODEL """

import numpy as np

# Apply the analyzer delay

def apply_analyser_delay(C_model, dt, tau_increase, tau_decrease):
    Nt = len(C_model)
    C_measured = np.zeros(Nt)
    C_measured[0] = C_model[0]  # Initial condition

    for t in range(1, Nt):  # Iterate up to Nt only
        if C_model[t] > C_model[t - 1]:  # Concentration is increasing
            tau = tau_increase
        else:  # Concentration is decreasing
            tau = tau_decrease

        # Apply the delay based on the respective time constant
        dC_measured_dt = (C_model[t - 1] - C_measured[t - 1]) / tau
        C_measured[t] = C_measured[t - 1] + dC_measured_dt * dt

    # Ensure output length is exactly Nt
    C_measured = C_measured[:Nt]

    return C_measured
