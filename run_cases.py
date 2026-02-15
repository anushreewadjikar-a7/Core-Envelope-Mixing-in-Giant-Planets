from model import CorrectedJupiterModel
from plot import run_and_plot_case, plot_final_sulfur_profiles
import numpy as np

if __name__ == "__main__":

      planet_masses = [1.0, 2.0]
      core_fracs = [0.1]
      S_core_vals = [0.01, 0.0001]      # 1%, 0.01%
      S_env_vals  = [1e-6]              # 0.0001%
      a_vals = [5.0,0.05]                    # 5 AU

      cases = []
      for MJ in planet_masses:
        for cf in core_fracs:
            for S_core in S_core_vals:
                for S_env in S_env_vals:
                    for a_AU in a_vals:
                        cases.append((MJ, cf, S_core, S_env, a_AU))

      results = {}
      for case in cases:
          MJ, cf, S_core, S_env, a_AU = case
          hist = run_and_plot_case(MJ, cf, S_core, S_env, a_AU, total_time=1e9, dt=1e4)
          results[case] = hist
      plot_final_sulfur_profiles(results, timescale=1e9)
      initial_S_profile=hist["species"]["S"][0]
      final_S_profile=hist["species"]["S"][-1]
      print(f"   Maximum S composition change: {np.max(np.abs(np.array(final_S_profile) - np.array(initial_S_profile))):.4f}")