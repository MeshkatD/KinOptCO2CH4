# DFM Kinetics Simulation and Optimisation

This repository contains Python models for simulating and optimising cyclic processes using Dual Function Materials (DFMs). The focus is on CO₂ adsorption, conversion to CH₄, and regeneration stages.

## Features
- **Adsorption Stage**: Simulates CO₂ and H₂O adsorption on DFM 
- **Purge Stage**: Models the desorption of CO₂ and removal of residual gases.
- **Hydrogenation Stage**: Simulates CO₂ hydrogenation to CH₄ with detailed reaction kinetics.
- **Cycle Model**: Combines stages to simulate the cyclic steady state (CSS) of the process.
- A semi-implicit finite difference method is used for modelling.

## Files
- `adsorption_model_imp.py`: Simulates the adsorption process with adsorption kinetics.
- `purge_model_imp.py`: Simulates the purge process with desorption kinetics.
- `hydrogenation_model_imp.py`: Models the hydrogenation stage using detailed kinetics.
- `analyser_model.py`: Adds delay effects to mimic analyzer readings.
- `cycle_model_CSS_imp.py`: Combines stages and analyzes cyclic steady state.

## Requirements
- Python 3.x
- Required Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

Install dependencies using:
```bash
pip install numpy pandas matplotlib scipy
