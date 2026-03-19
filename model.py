import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ================================
# PHYSICAL CONSTANTS & REFERENCE VALUES
# ================================
G = 6.67430e-11
k_B = 1.380649e-23
m_u = 1.66053906660e-27
M_JUPITER = 1.8982e27             # [Durante et al. 2020]
R_JUPITER = 7.1492e7              # [Lindal 1992]
L_JUPITER = 8.7e17                # [Pearl & Conrath 1991]
OMEGA_JUPITER = 2*np.pi/(9.925*3600)  # [Seidelmann et al. 2007]
L_SUN   = 3.828e26
AU      = 1.496e11
ALBEDO  = 0.3
SIGMA   = 5.670374419e-8
P_H2_DISSOCIATION = 1.3e11        # [French et al. 2009]
P_H_METALLIZATION = 1.4e11        # [Fortney & Nettelmann 2010]

MOLECULAR_DATA = {
    'H':  {'mass': 2.016,  'cp': 14300, 'sigma': 2.89e-10, 'eps_k': 59.7},
    'He': {'mass': 4.003,  'cp': 5200,  'sigma': 2.64e-10, 'eps_k': 10.9},
    'S':  {'mass': 32.06,  'cp': 700,   'sigma': 2.89e-10, 'eps_k': 59.7}
}

# Default compositions (used if none passed in)
CORE_COMPOSITION     = {'H': 0.0,  'He': 0.0,  'S': 1.0}
ENVELOPE_COMPOSITION = {'H': 0.70, 'He': 0.30, 'S': 0.0}


# ================================
class CorrectedJupiterModel:

    def __init__(self, n_radial=200, tolerance=1e-4, a_AU=5.0, core_frac=0.1,
                 core_composition=None, envelope_composition=None,
                 mass_MJ=1.0, radius_RJ=1.0):
        """
        Parameters
        ----------
        mass_MJ    : planet mass in Jupiter masses  ← FIX: was hardcoded to 1.0
        radius_RJ  : planet radius in Jupiter radii
        """
        # FIX: planet mass now passed in correctly per case
        self.M_planet = mass_MJ * M_JUPITER
        self.R_planet = radius_RJ * R_JUPITER
        self.mass_MJ  = mass_MJ

        self.n_radial = n_radial
        self.dr = self.R_planet / n_radial
        self.r  = np.linspace(self.dr/2, self.R_planet - self.dr/2, n_radial)
        self.tolerance = tolerance
        self.core_frac = core_frac

        self.core_composition     = core_composition     or CORE_COMPOSITION
        self.envelope_composition = envelope_composition or ENVELOPE_COMPOSITION

        self.a_AU = a_AU
        self.a_m  = a_AU * AU
        self.T_eq = self._compute_T_eq()
        print(f"   T_eq = {self.T_eq:.1f} K at {a_AU} AU  |  M = {mass_MJ} MJ")

        # Radial arrays
        self.P     = np.zeros(n_radial)
        self.T     = np.zeros(n_radial)
        self.rho   = np.zeros(n_radial)
        self.M_enc = np.zeros(n_radial)
        self.g     = np.zeros(n_radial)
        self.cp    = np.zeros(n_radial)
        self.H_p   = np.zeros(n_radial)

        self.shell_volume = np.zeros(n_radial)
        for i in range(n_radial):
            r_inner = (self.r[i] - self.dr/2) if i > 0 else 0.0
            r_outer =  self.r[i] + self.dr/2
            self.shell_volume[i] = (4/3)*np.pi*(r_outer**3 - r_inner**3)
        self.shell_mass = np.zeros(n_radial)

        self.species = sorted(set(self.core_composition) | set(self.envelope_composition))
        self.X = {sp: np.zeros(n_radial) for sp in self.species}

        print(f"   Grid: {n_radial} pts, dr = {self.dr/1e3:.1f} km")

        self.initialize_composition()
        success = self.solve_structure()
        if success:
            print("✅ Model ready")
        else:
            print("⚠️  Structure solve had >5% mass error — results approximate")

    # ------------------------------------------------------------------
    def _compute_T_eq(self):
        flux = L_SUN / (4*np.pi*self.a_m**2)
        return (flux*(1-ALBEDO)/(4*SIGMA))**0.25

    # ------------------------------------------------------------------
    def initialize_composition(self):
        """
        Smooth tanh transition at core-envelope boundary.
        A hard step causes Ledoux suppression to freeze all mixing.
        Transition width ~5% of radius (Helled & Stevenson 2017).
        """
        core_idx = int(self.core_frac * self.n_radial)
        width    = max(5, int(0.05 * self.n_radial))

        for i in range(self.n_radial):
            f = 0.5*(1.0 + np.tanh((i - core_idx)/width))   # 0=core, 1=envelope
            for sp in self.species:
                xc = self.core_composition.get(sp, 0)
                xe = self.envelope_composition.get(sp, 0)
                self.X[sp][i] = xc*(1-f) + xe*f

        # Normalize
        for i in range(self.n_radial):
            tot = sum(self.X[sp][i] for sp in self.species)
            if tot > 0:
                for sp in self.species:
                    self.X[sp][i] /= tot

        print(f"   Core:    S={self.X['S'][0]:.3f}  H={self.X['H'][0]:.3f}  He={self.X['He'][0]:.3f}")
        print(f"   Surface: S={self.X['S'][-1]:.3f} H={self.X['H'][-1]:.3f} He={self.X['He'][-1]:.3f}")

    # ------------------------------------------------------------------
    def _mu(self, i):
        mu_inv = 0.0
        for sp in self.species:
            if self.X[sp][i] <= 0:
                continue
            if sp == 'H':
                if self.P[i] > P_H_METALLIZATION:  mu_H = 0.5
                elif self.P[i] > P_H2_DISSOCIATION: mu_H = 1.0
                else:                                mu_H = 2.0
                mu_inv += self.X[sp][i] / mu_H
            else:
                mu_inv += self.X[sp][i] / MOLECULAR_DATA[sp]['mass']
        return 1.0/mu_inv if mu_inv > 0 else 32.0

    # ------------------------------------------------------------------
    def _eos(self, P, T, i):
        P = max(P, 1e5);  T = max(T, 100)
        mu  = self._mu(i)
        rho_gas = P*mu*m_u/(k_B*T)
        if   P > P_H_METALLIZATION:   rho_gas *= 1.3
        elif P > P_H2_DISSOCIATION:   rho_gas *= 1.1
        rho_S = min(2000.0*(1 + 1e-11*P), 20000.0)
        XS    = self.X['S'][i] if 'S' in self.X else 0.0
        Xgas  = 1.0 - XS
        return float(np.clip(Xgas*rho_gas + XS*rho_S, 100, 25000))

    # ------------------------------------------------------------------
    def _integrate(self, Pc):
        self.P[0] = Pc;  self.T[0] = 20000
        self.rho[0] = self._eos(Pc, 20000, 0)
        self.M_enc[0] = (4/3)*np.pi*((self.r[0]+self.dr/2)**3)*self.rho[0]
        self.g[0] = 0.0
        for i in range(1, self.n_radial):
            ri   = self.r[i-1] - self.dr/2
            ro   = self.r[i]   + self.dr/2
            Vsh  = (4/3)*np.pi*(ro**3 - ri**3)
            self.M_enc[i] = self.M_enc[i-1] + Vsh*self.rho[i-1]
            self.g[i]     = G*self.M_enc[i]/(self.r[i]**2 + 1e-20)
            dP            = -self.g[i]*self.rho[i-1]*self.dr
            self.P[i]     = max(self.P[i-1]+dP, 1e3)
            gam = 5/3 if self.P[i] > P_H2_DISSOCIATION else 7/5
            self.T[i]     = max(20000*(self.P[i]/Pc)**((gam-1)/gam), 100)
            self.rho[i]   = self._eos(self.P[i], self.T[i], i)
        return self.M_enc[-1] - self.M_planet

    def solve_structure(self):
        print("🔄 Solving hydrostatic structure...")
        try:
            Pc = brentq(self._integrate, 1e10, 1e16, xtol=1e6)
        except:
            P_scan = np.logspace(11, 14, 40)
            errs   = [self._integrate(p) for p in P_scan]
            Pc     = P_scan[np.argmin(np.abs(errs))]
        self._integrate(Pc)

        g_expected  = G*self.M_planet/self.R_planet**2
        mass_err    = abs(self.M_enc[-1]-self.M_planet)/self.M_planet*100
        grav_err    = abs(self.g[-1]-g_expected)/g_expected*100
        print(f"   Pc = {Pc/1e9:.0f} GPa | mass err = {mass_err:.2f}% | "
              f"g_surf = {self.g[-1]:.2f} m/s² (err {grav_err:.1f}%)")

        M_core_target   = self.core_frac*self.M_planet
        self.core_idx   = max(1, int(np.argmax(self.M_enc >= M_core_target)))
        self.shell_mass = self.shell_volume*self.rho
        print(f"   Core boundary: shell {self.core_idx}/{self.n_radial}")
        return mass_err < 5

    # ------------------------------------------------------------------
    def _calc_cp(self):
        for i in range(self.n_radial):
            cp, tot = 0.0, 0.0
            for sp in self.species:
                if self.X[sp][i] > 0:
                    base = MOLECULAR_DATA[sp]['cp']
                    if sp=='H' and self.T[i]>2000:
                        base *= (1 + np.exp(-4000/self.T[i]))
                    cp  += self.X[sp][i]*base
                    tot += self.X[sp][i]
            self.cp[i] = max(cp/max(tot,1e-12), 1000)

    def _calc_Hp(self):
        for i in range(self.n_radial):
            mu = self._mu(i)
            self.H_p[i] = max(k_B*self.T[i]/(mu*m_u*max(self.g[i],1e-10)), 1e3)

    # ------------------------------------------------------------------
    def _diffusion(self, verbose=False):
        """
        Returns D_conv, D_mol per species.
        Ledoux suppression applied to convection only; molecular diffusion
        is allowed through stable regions (it drives slow Gyr-scale erosion).
        """
        # --- Convective diffusion (MLT + Rossby suppression) ---
        F_conv = L_JUPITER/(4*np.pi*self.r**2 + 1e-20)
        lmix   = self.H_p
        v_nr   = (F_conv*self.g*lmix / (self.rho*self.cp*self.T + 1e-20))**(1/3)

        # FIX 1: correct Rossby number (operator precedence was wrong before)
        Ro     = v_nr / (2*OMEGA_JUPITER*self.H_p + 1e-20)
        v_conv = v_nr * (Ro**0.2)
        D_conv = (1/3)*v_conv*lmix

        # --- Molecular diffusion (Chapman–Enskog) ---
        n_tot  = self.rho/(m_u*np.mean([MOLECULAR_DATA[s]['mass'] for s in self.species]))
        D_mol  = {}
        for sp_i in self.species:
            mi   = MOLECULAR_DATA[sp_i]['mass']*m_u
            denom = np.zeros(self.n_radial)
            for sp_j in self.species:
                if sp_j == sp_i: continue
                mj     = MOLECULAR_DATA[sp_j]['mass']*m_u
                mu_ij  = mi*mj/(mi+mj)
                sig_ij = 0.5*(MOLECULAR_DATA[sp_i]['sigma']+MOLECULAR_DATA[sp_j]['sigma'])
                Dij    = (3/16)*np.sqrt(2*np.pi*k_B*self.T/mu_ij)/(n_tot*sig_ij**2)
                denom += self.X[sp_j]/(Dij + 1e-30)
            D_eff        = 1.0/(denom + 1e-30)
            pres_factor  = 1.0/(1.0 + 0.1*(self.P/1e11)**0.3)
            mass_scaling = 1.0/MOLECULAR_DATA[sp_i]['mass']
            D_mol[sp_i]  = np.clip(D_eff*mass_scaling*pres_factor, 1e-15, 1e6)

        # --- Ledoux suppression on convection only ---
        mu_arr  = np.array([self._mu(i) for i in range(self.n_radial)])
        grad_mu = np.gradient(mu_arr, np.log(self.P + 1e-30))
        D_conv_s = D_conv.copy()
        for i in range(self.n_radial):
            if grad_mu[i] > 0:
                D_conv_s[i] *= np.exp(-min(grad_mu[i], 2.0))

        if verbose:
            print(f"   D_conv  range: {D_conv_s.min():.2e} – {D_conv_s.max():.2e} m²/s")
            for sp in self.species:
                print(f"   D_mol[{sp}] range: {D_mol[sp].min():.2e} – {D_mol[sp].max():.2e} m²/s")

        return D_conv_s, D_mol

    # ------------------------------------------------------------------
    def run_evolution_corrected(self, total_time, dt):
        SEC = 3.154e7
        total_s = total_time*SEC;  dt_s = dt*SEC
        print(f"🚀 Evolving {total_time:.1e} yr  (dt={dt:.0f} yr)")

        self._calc_cp();  self._calc_Hp()

        # Print diffusion diagnostic once so we know mixing is non-zero
        print("--- Diffusion diagnostic (t=0) ---")
        self._diffusion(verbose=True)
        print("----------------------------------")

        history = {"T":[], "L":[], "times":[],
                   "species":{sp:[] for sp in self.species},
                   "mass":   {sp:[] for sp in self.species}}

        n_steps    = int(total_s/dt_s)
        save_every = max(1, n_steps//100)
        t_KH = G*self.M_planet**2/(self.R_planet*L_JUPITER)

        t, step = 0.0, 0
        while t < total_s:
            t_yr = t/SEC;  t_KH_yr = t_KH/SEC

            if   t_yr < 1e-6:       L = L_JUPITER
            elif t_yr < t_KH_yr:    L = L_JUPITER*np.exp(-t_yr/t_KH_yr)
            else:                   L = L_JUPITER*(t_KH_yr/t_yr)

            Teff = (L/(4*np.pi*self.R_planet**2*SIGMA) + self.T_eq**4)**0.25
            dT   = -L/(self.M_planet*np.mean(self.cp))*dt_s
            self.T = np.maximum(self.T+dT, Teff)

            D_conv, D_mol = self._diffusion()

            for sp in self.species:
                for D in [D_conv, D_mol[sp]]:
                    lap          = np.zeros(self.n_radial)
                    lap[1:-1]    = (self.X[sp][2:] - 2*self.X[sp][1:-1] + self.X[sp][:-2])/self.dr**2
                    lap[0]       = 2*(self.X[sp][1] - self.X[sp][0])/self.dr**2
                    D_safe       = np.minimum(D, self.dr**2/(2*dt_s))
                    self.X[sp]  += dt_s*D_safe*lap

            # Clip and renormalize (FIX 2 & 3)
            for sp in self.species:
                self.X[sp] = np.clip(np.nan_to_num(self.X[sp]), 0, 1)
            tot_X = sum(self.X[sp] for sp in self.species)
            tot_X = np.where(tot_X > 0, tot_X, 1.0)
            for sp in self.species:
                self.X[sp] /= tot_X

            if step % save_every == 0:
                history["T"].append(self.T.copy())
                history["L"].append(L)
                history["times"].append(t_yr)
                for sp in self.species:
                    history["species"][sp].append(self.X[sp].copy())
                    history["mass"][sp].append((self.X[sp]*self.shell_mass).copy())

            if step % max(1, n_steps//10) == 0:
                print(f"   t={t_yr:.2e} yr | Tc={self.T[0]:.0f} K | "
                      f"S_core={self.X['S'][0]:.5f} | S_surf={self.X['S'][-1]:.5f}")
            t    += dt_s
            step += 1

        print("✅ Evolution complete")
        return history
