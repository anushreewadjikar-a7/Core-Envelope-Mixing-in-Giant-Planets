import numpy as np
import matplotlib.pyplot as plt
from model import CorrectedJupiterModel
M_JUPITER = 1.8982e27             # Juno mission [Durante et al. 2020]
R_JUPITER = 7.1492e7              # [Lindal 1992]



def plot_results(model, hist):

        """Plot results with improved visualization"""

        # Temperature evolution
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(hist["times"], [T[0] for T in hist["T"]], 'r-', linewidth=2)
        plt.xlabel("Time [yr]")
        plt.ylabel("Central Temperature [K]")
        plt.title("Central Temperature Evolution")
        plt.grid(True)

        # Luminosity evolution
        plt.subplot(1, 3, 2)
        plt.plot(hist["times"], hist["L"], 'orange', linewidth=2)
        plt.xlabel("Time [yr]")
        plt.ylabel("Luminosity [W]")
        plt.title("Intrinsic Luminosity Evolution")
        plt.yscale('log')
        plt.grid(True)

        # Final composition profile
        plt.subplot(1, 3, 3)
        r_norm = model.r / model.R_planet
        for sp in model.species:
            if np.any(model.X[sp] > 0.01):  # Only plot significant species
                plt.plot(r_norm, model.X[sp], label=sp, linewidth=2)

        plt.ylabel("Radius / R_planet")
        plt.xlabel("Mass Fraction")
        plt.title("Final Composition Profile")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Species mixing evolution (heatmaps)
        for sp in model.species:
            if np.any(model.X[sp] > 0.01):  # Only significant species
                data = np.array(hist["species"][sp])
                plt.figure(figsize=(10, 6))
                plt.imshow(
                    data.T, aspect="auto", origin="lower",
                    extent=[hist["times"][0], hist["times"][-1], 0, 1],  # x=time, y=radius
            cmap="plasma")
                plt.colorbar(label=f"{sp} mass fraction")
                plt.ylabel("Radius / R_planet")
                plt.xlabel("Time [yr]")
                plt.title(f"{sp} Mixing Evolution")
                plt.show()
def plot_core_envelope_contour(model, hist):
      """Combined contour map for S core + H and He envelopes"""
      r_norm = model.r / model.R_planet
      t = hist["times"]

    # Data arrays
      S_data = np.array(hist["species"]["S"])
      H_data = np.array(hist["species"]["H"])
      He_data = np.array(hist["species"]["He"])

      plt.figure(figsize=(10, 6))

    # Filled contour for S (core material)
      cs = plt.contourf( hist["times"], model.r/model.R_planet, np.array(hist["species"]["S"]).T,
    levels=np.linspace(0,1,21), cmap="twilight")
      plt.colorbar(cs, label="S mass fraction")
      vmax = np.max(S_data)
      cs = plt.contourf(hist["times"], model.r/model.R_planet,
                  np.array(hist["species"]["S"]).T,
                  levels=np.linspace(0, vmax, 21), cmap="twilight")

    # Overlay line contours for H and He
      plt.contour(hist["times"], model.r/model.R_planet, np.array(hist["species"]["H"]).T,
            levels=[0.1,0.3,0.5,0.7], colors="blue")
      plt.contour(hist["times"], model.r/model.R_planet, np.array(hist["species"]["He"]).T,
            levels=[0.1,0.3], colors="orange")


      plt.xlabel("Time [yr]")
      plt.ylabel("Radius / R_planet")
      plt.title("Core–Envelope Mixing ")
      plt.show()

def run_and_plot_case(MJ, core_frac, S_core, S_env, a_AU, total_time=1e9, dt=1e4):
      """
      Run a single planetary evolution case and plot results.
      """
      global PLANET_MASS_MJ, CORE_RADIUS_FRACTION, CORE_COMPOSITION, ENVELOPE_COMPOSITION

    # Override global composition settings
      PLANET_MASS_MJ = MJ
      CORE_DENSITY = 10000.0  # kg/m³, adjust as needed
      M_core = core_frac * (MJ * M_JUPITER)    # core mass in kg
      R_core = ((3 * M_core) / (4 * np.pi * CORE_DENSITY))**(1/3)  # radius in m
      CORE_DENSITY = 10000.0  # kg/m³, adjust as needed

# Normalize by planet radius
      PLANET_RADIUS_RJ=1
      CORE_RADIUS_FRACTION = min(R_core / (PLANET_RADIUS_RJ * R_JUPITER), 0.99)
      CORE_COMPOSITION = {'H': 0.7*(1-S_core), 'He': 0.30*(1-S_core), 'S': S_core}
      ENVELOPE_COMPOSITION = {'H': 0.70*(1- S_env) , 'He': 0.30*(1-S_env), 'S': S_env}

      print("\n" + "="*60)
      print(f"🚀 RUNNING CASE: M={MJ} M_J, CoreFrac={core_frac}, S_core={S_core}, S_env={S_env}, a={a_AU} AU")

      model = CorrectedJupiterModel(n_radial=200, a_AU=a_AU,core_frac=core_frac)
      hist = model.run_evolution_corrected(total_time=total_time, dt=dt)
      plot_core_envelope_contour(model,hist)
      plot_results(model,hist)
      return hist
def plot_final_sulfur_profiles(results, timescale=1e9):
      """
    results = dict of simulation histories, keyed by case tuple
              (M_J, core_frac, S_core, S_env, a_AU)
    timescale = which final time snapshot to take
      """
      plt.figure(figsize=(7,5))

      for case, hist in results.items():
        M_J, cf, S_core, S_env, a_AU = case

        initial_S_profile = hist["species"]["S"][0]
        final_S_profile   = hist["species"]["S"][-1]

        # Normalized radius
        r =np.linspace(0,1,len(final_S_profile))

        label = f"{M_J:.1f} M_J, S_env={S_env}"
        plt.plot(final_S_profile, r, lw=2, label=label,alpha=0.5)
        plt.plot(initial_S_profile, r, lw=2, linestyle="--",alpha=0.3)
        shell_mass = np.ones_like(final_S_profile)  # if ρ not stored, assume equal weight
        total_mass = np.sum(shell_mass)

        init_S_mass = np.sum(initial_S_profile * shell_mass)
        final_S_mass = np.sum(final_S_profile * shell_mass)

        init_S_frac  = init_S_mass / total_mass
        final_S_frac = final_S_mass / total_mass

      print(f"CASE {label}:")
      print(f"   Initial global S fraction = {init_S_frac:.4f}")
      print(f"   Final   global S fraction = {final_S_frac:.4f}")
      print(f"   Change  = {final_S_frac - init_S_frac:+.4f}\n")
      plt.xlim(0, 1.1*np.max(final_S_profile))
       # instead of full 0–1
      delta_S = final_S_profile - initial_S_profile
      plt.plot(delta_S, r, lw=2, color="red", label="ΔS (final - initial)")

      plt.gca().invert_yaxis()  # put core at bottom, surface at top
      plt.xlabel("Sulfur Mass Fraction")
      plt.ylabel("Radius / R_planet")
      plt.title(f"Final Sulfur Profiles after {timescale:.1e} yr")

      plt.gca().invert_yaxis()  # put core at bottom, surface at top
      plt.xlabel("Sulfur Mass Fraction")
      plt.ylabel("Radius / R_planet")
      plt.title(f"Final Sulfur Profiles after {timescale:.1e} yr")
      plt.legend()
      plt.grid(False)
      plt.tight_layout()
      plt.show()
      plt.figure(figsize=(8,6))

      colors = {1.0: "blue", 2.0: "red"}
      linestyles = {5.0: "-", 0.05: "--"}

      for case, hist in results.items():
        M_J, cf, S_core, S_env, a_AU = case

        initial_S_profile = hist["species"]["S"][0]
        final_S_profile   = hist["species"]["S"][-1]
        r = np.linspace(0, 1, len(final_S_profile))

        label = f"{M_J:.0f} M_J, a={a_AU} AU"
        plt.plot(final_S_profile, r,
                 color=colors.get(M_J,"black"),
                 ls=linestyles.get(a_AU,"-"),
                 lw=2, label=label)

        # (optional) also plot initial profiles as faint dashed gray
        plt.plot(initial_S_profile, r, lw=1, ls=":", color="gray", alpha=0.5)

      plt.gca().invert_yaxis()  # core at bottom
      plt.xlabel("Sulfur Mass Fraction")
      plt.ylabel("Radius / R_planet")
      plt.title(f"Sulfur Mixing after {timescale:.1e} yr")
      plt.legend()
      plt.grid(True, ls="--", alpha=0.5)
      plt.tight_layout()
      plt.show()