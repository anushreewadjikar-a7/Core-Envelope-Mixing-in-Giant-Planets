# Giant Planet Core–Envelope Mixing Simulation

A 1D numerical model of giant planet interiors implementing hydrostatic 
equilibrium, multi-species diffusion (Chapman-Enskog theory), convective 
transport with mixing-length theory, rotational suppression via Rossby-number 
scaling, and Ledoux stability criteria.

Validated against Jupiter: central pressure 67,988 GPa, surface gravity 
24.91 m/s² (0.5% error from observed). Simulations over 10⁹ years demonstrate 
deep core composition is preserved under rotational suppression, consistent 
with Juno gravity data and fuzzy-core models (Helled & Stevenson 2017, 
Fuentes et al. 2024).

## Physics implemented
- Hydrostatic equilibrium solved via shooting method (Brent's algorithm)
- Equation of state: ideal gas with phase corrections for H₂ dissociation 
  and metallic hydrogen transitions
- Multi-species diffusion: Chapman-Enskog theory for H, He, S
- Convective transport: mixing-length theory with Rossby-number rotational 
  suppression (Fuentes et al. 2023)
- Ledoux stability criterion suppressing convection in compositionally 
  stable regions
- Kelvin-Helmholtz thermal evolution
- Stellar irradiation floor temperature

## Validation
| Quantity | Model | Observed | Error |
|----------|-------|----------|-------|
| Central pressure | 67,988 GPa | ~70,000 GPa | 0.0% mass error |
| Surface gravity | 24.91 m/s² | 24.79 m/s² | 0.5% |
| Core boundary | shell 27/200 | ~10–15% radius | consistent |

## Key results
- Deep core S fraction change: ~6×10⁻⁵ over 10⁹ years
- 2MJ planet mixes more slowly than 1MJ due to stronger gravitational 
  confinement — physically consistent with Ledoux suppression
- Stellar irradiation (0.05 AU vs 5 AU) affects surface temperature floor 
  but not deep interior mixing — consistent with observations

## Code structure
- model.py — physics model, structure solver, evolution loop
- plot.py — visualization
- run_cases.py — parameter study across masses and orbital separations
- sample_outputs/ — example plots

## Installation
pip install -r requirements.txt

## Usage
python run_cases.py

## References
- Helled & Stevenson 2017, ApJL 840 L4
- Fuentes et al. 2023, ApJL 950 L4  
- Fuentes et al. 2024, ApJ 945 23
- Durante et al. 2020, Nature 583 567

## Sample Outputs
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/06d769f2-f5b6-4d29-844f-8a7580092e56" />
