# Giant Planet Core–Envelope Mixing Simulation

This project is a numerical simulation of how the interior of giant planets (like Jupiter) evolves over time, especially how material from the solid core mixes into the surrounding gaseous envelope.

It was developed as part of an MSc research project in planetary physics.

---

## What problem does this project solve?

Giant planets are believed to form with:
- A dense, heavy-element core
- A large envelope made mostly of hydrogen and helium

Over billions of years, the boundary between the core and the envelope may not stay sharp. Heavy elements from the core can gradually mix into the envelope due to:

- Convection (fluid motion)
- Diffusion
- Thermal evolution

Understanding this mixing helps scientists:
- Explain the internal structure of planets like Jupiter and Saturn
- Interpret spacecraft data (e.g., from the Juno mission)
- Model the formation and evolution of exoplanets

---

## What does this code actually do?

This code builds a simplified model of a giant planet and simulates its interior over time.

The model:

1. Creates a planet with:
   - A sulfur-rich core
   - A hydrogen–helium envelope

2. Solves the internal structure using:
   - Hydrostatic equilibrium (balance between gravity and pressure)

3. Simulates thermal evolution:
   - The planet cools over billions of years

4. Simulates compositional mixing:
   - Core material diffuses into the envelope
   - Convection affects how fast mixing occurs

5. Tracks how:
   - Temperature
   - Luminosity
   - Composition (H, He, S)
   change with time and radius.

---

## What skills this project demonstrates

This project involves:

- Physics-based modelling
- Numerical simulation
- Time-dependent differential equations
- Multi-species diffusion
- Data visualization
- Scientific programming in Python

It shows experience with:
- Building a simulation from physical equations
- Parameter studies across different planetary conditions
- Interpreting model outputs

---

## Example result

The simulation produces plots like:

- How the core–envelope boundary changes over time
- How temperature evolves over billions of years
- How hydrogen, helium, and heavy elements are distributed inside the planet

Example output:

![Core-envelope mixing](<sample outputs/Figure_ Elements mixing.png>)


## Code structure
model.py → Main physics model and evolution
plot.py → Plotting and visualization
run_cases.py → Runs different simulation scenarios
requirements.txt
sample_outputs/ → Example plots


## Installation

Install required libraries:
pip install -r requirements.txt

## How to run the simulation

Run:

python run_cases.py


This will:
Simulate several planet cases
Generate evolution and composition plots

## Project scope and limitations

This is a simplified research-style model designed to explore qualitative behaviour, not to exactly reproduce a specific planet.

Some simplifications:

Approximate equation of state

Simplified cooling model

1D radial structure

The goal is to study general trends in core–envelope mixing.