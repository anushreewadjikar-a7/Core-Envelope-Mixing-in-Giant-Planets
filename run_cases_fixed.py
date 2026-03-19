from model import CorrectedJupiterModel, M_JUPITER, R_JUPITER
from plot import plot_results, plot_core_envelope_contour, plot_final_sulfur_profiles
import numpy as np


def run_and_plot_case(MJ, core_frac, S_core, S_env, a_AU,
                      total_time=1e8, dt=1e4):
    core_comp = {'H': 0.0, 'He': 0.0, 'S': 1.0}   # pure S core
    env_comp  = {'H': 0.70*(1-S_env), 'He': 0.30*(1-S_env), 'S': S_env}

    # normalize
    for comp in [core_comp, env_comp]:
        tot = sum(comp.values())
        if tot > 0:
            for k in comp: comp[k] /= tot

    print("\n" + "="*60)
    print(f"CASE: {MJ} MJ | core_frac={core_frac} | "
          f"S_core={S_core} | S_env={S_env} | a={a_AU} AU")

    model = CorrectedJupiterModel(
        n_radial=200,
        a_AU=a_AU,
        core_frac=core_frac,
        core_composition=core_comp,
        envelope_composition=env_comp,
        mass_MJ=MJ,          # ← FIX: planet mass now passed correctly
        radius_RJ=1.0,
    )

    hist = model.run_evolution_corrected(total_time=total_time, dt=dt)
    plot_core_envelope_contour(model, hist)
    plot_results(model, hist)
    return model, hist


if __name__ == "__main__":
    cases = [
        (MJ, cf, S_core, S_env, a_AU)
        for MJ      in [1.0, 2.0]
        for cf      in [0.1]
        for S_core  in [0.01]
        for S_env   in [1e-6]
        for a_AU    in [5.0, 0.05]
    ]

    results = {}
    for case in cases:
        MJ, cf, S_core, S_env, a_AU = case
        model, hist = run_and_plot_case(MJ, cf, S_core, S_env, a_AU,
                                total_time=1e9, dt=1e5)
        results[case] = hist

    plot_final_sulfur_profiles(results, timescale=1e8)

    for case, hist in results.items():
        i_S = np.array(hist["species"]["S"][0])
        f_S = np.array(hist["species"]["S"][-1])
        print(f"Case {case}: max|ΔS| = {np.max(np.abs(f_S-i_S)):.5f}  "
              f"ΔS_core = {f_S[0]-i_S[0]:+.5f}")
