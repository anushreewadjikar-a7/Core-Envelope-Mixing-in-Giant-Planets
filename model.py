import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ================================
# PLANET PARAMETERS (in SI Units)
# ================================
PLANET_MASS_MJ = 1.0
PLANET_RADIUS_RJ = 1.0

CORE_RADIUS_FRACTION = 0.30


# Your composition (
CORE_COMPOSITION = {'H': 0, 'He': 0, 'S': 1.0}
ENVELOPE_COMPOSITION = {'H': 0.70, 'He': 0.30, 'S': 0}

# Constants [literature values]
G = 6.67430e-11                    #
k_B = 1.380649e-23                 #
m_u = 1.66053906660e-27           #
M_JUPITER = 1.8982e27             # Juno mission [Durante et al. 2020]
R_JUPITER = 7.1492e7              # [Lindal 1992]
L_JUPITER = 8.7e17                # [Pearl & Conrath 1991]
g_JUPITER_SURFACE = 24.79         # Expected surface gravity
OMEGA_JUPITER = 2*np.pi/(9.925*3600)  # [Seidelmann et al. 2007]
# Stellar irradiation constants
L_SUN   = 3.828e26          # Solar luminosity [W]
AU      = 1.496e11          # 1 AU in m
ALBEDO  = 0.3               # Bond albedo (Jupiter-like)
SIGMA   = 5.670374419e-8    # Stefan–Boltzmann constant

# Phase transitions [experimental literature]
P_H2_DISSOCIATION = 1.3e11        # [French et al. 2009]
P_H_METALLIZATION = 1.4e11        # [Fortney & Nettelmann 2010]

# Molecular data [NIST + literature]
MOLECULAR_DATA = {
    'H':    {'mass': 2.016, 'cp': 14300, 'sigma': 2.89e-10, 'eps_k': 59.7},
    'He':   {'mass': 4.003, 'cp': 5200,  'sigma': 2.64e-10, 'eps_k': 10.9},
    'S':    {'mass': 32.06, 'cp': 700,   'sigma': 2.89e-10, 'eps_k': 59.7}
}


# ================================
class CorrectedJupiterModel:
    def __init__(self, n_radial=200, tolerance=1e-4,a_AU=5.0,core_frac=0.1):

        # Planet parameters
        self.M_planet = PLANET_MASS_MJ * M_JUPITER
        self.R_planet = PLANET_RADIUS_RJ * R_JUPITER
        self.n_radial = n_radial
        self.dr = self.R_planet / n_radial
        self.r = np.linspace(self.dr/2, self.R_planet - self.dr/2, n_radial)
        self.tolerance = tolerance
        self.core_frac = core_frac
        # Irradiation setup
        self.a_AU = a_AU
        self.a_m  = a_AU * AU
        self.T_eq = self.compute_equilibrium_temperature()
        print(f"   Irradiation floor temperature: T_eq = {self.T_eq:.1f} K at {self.a_AU} AU")


        # Arrays
        self.P = np.zeros(n_radial)
        self.T = np.zeros(n_radial)
        self.rho = np.zeros(n_radial)
        self.M_enc = np.zeros(n_radial)
        self.g = np.zeros(n_radial)
        self.cp = np.zeros(n_radial)
        self.H_p = np.zeros(n_radial)
        # Precompute shell volumes and masses (assuming density ~ initial rho)
        self.shell_volume = np.zeros(self.n_radial)
        self.shell_mass   = np.zeros(self.n_radial)

        for i in range(self.n_radial):
          r_inner = (self.r[i] - self.dr/2) if i > 0 else 0.0
          r_outer = self.r[i] + self.dr/2
          self.shell_volume[i] = (4/3) * np.pi * (r_outer**3 - r_inner**3)
# Later, after structure solved, update shell_mass with self.rho

        # Species and composition
        self.species = sorted(set(CORE_COMPOSITION) | set(ENVELOPE_COMPOSITION))
        self.X = {sp: np.zeros(n_radial) for sp in self.species}

        print(f"   Species: {self.species}")
        print(f"   Grid: {n_radial} points, dr = {self.dr/1000:.1f} km")

        # Initialize and solve
        self.initialize_composition()
        success = self.solve_structure_corrected()

        if success:
            print("✅ Model initialized successfully")
        else:
            print("❌ Model initialization failed")

    def initialize_composition(self):
        """Your composition initialization (kept exactly)"""

        core_total = sum(CORE_COMPOSITION.values())
        env_total = sum(ENVELOPE_COMPOSITION.values())
    
        core_norm = CORE_COMPOSITION
        env_norm = ENVELOPE_COMPOSITION
        core_idx = getattr(self, "core_idx", int(self.core_frac*self.n_radial))  # fallback if not set



        for i in range(self.n_radial):
            if i < core_idx:
                # Pure sulfur core
                for sp in self.species:
                    self.X[sp][i] = core_norm.get(sp, 0)

            else:
                # H-He envelope
                for sp in self.species:
                    self.X[sp][i] = env_norm.get(sp, 0)

            

        print("✅ Composition initialized:")
        print(f"   Core: S={self.X['S'][0]:.3f}, H={self.X['H'][0]:.3f}, He={self.X['He'][0]:.3f}")
        print(f"   Surface: S={self.X['S'][-1]:.3f}, H={self.X['H'][-1]:.3f}, He={self.X['He'][-1]:.3f}")

    def calculate_mu(self, i):
        """Mean molecular weight with phase transitions"""

        mu_inv = 0.0
        for sp in self.species:
            if self.X[sp][i] > 0:
                # Pressure-dependent hydrogen phases
                if sp == 'H':
                    if hasattr(self, 'P') and self.P[i] > P_H_METALLIZATION:
                        mu_H = 0.5  # Metallic H: H⁺ + e⁻
                    elif hasattr(self, 'P') and self.P[i] > P_H2_DISSOCIATION:
                        mu_H = 1.0  # Atomic H
                    else:
                        mu_H = 2.0  # Molecular H₂
                    mu_inv += self.X[sp][i] / mu_H
                else:
                    mu_inv += self.X[sp][i] / MOLECULAR_DATA[sp]['mass']

        return 1.0 / mu_inv if mu_inv > 0 else 32.0  # Default to sulfur
    def compute_equilibrium_temperature(self):
        """Compute irradiation floor temperature from star at distance a_AU"""
        flux = L_SUN / (4 * np.pi * self.a_m**2)
        return (flux * (1 - ALBEDO) / (4 * SIGMA))**0.25

    def modern_equation_of_state(self, P, T, i):
      """
    Equation of State (EOS) with species-dependent treatment:
    - H, He: ideal gas law + corrections (SCVH95)
    - S (heavy element): condensed density with weak pressure scaling (Militzer+2008, Nettelmann+2012)

    Parameters:
        P : float
            Pressure [Pa]
        T : float
            Temperature [K]
        i : int
            Radial index for composition lookup
      """

    # Safety floors
      P = max(P, 1e5)   # avoid unphysical low pressure
      T = max(T, 100)   # avoid division by zero

    # Composition fractions
      XH  = self.X['H'][i]  if 'H' in self.X  else 0.0
      XHe = self.X['He'][i] if 'He' in self.X else 0.0
      XS  = self.X['S'][i]  if 'S' in self.X  else 0.0

    # -------------------
    # 1. GAS COMPONENTS (H/He)
    # -------------------
      mu = self.calculate_mu(i)
      rho_gas = P * mu * m_u / (k_B * T)

    # Apply SCVH-type corrections (approximate)
      if P > P_H_METALLIZATION:
        correction = 1.3   # metallic H
      elif P > P_H2_DISSOCIATION:
        correction = 1.1   # atomic H regime
      else:
        correction = 1.0   # molecular regime
      rho_gas *= correction

    # -------------------
    # 2. HEAVY ELEMENTS (S as proxy)
    # -------------------
    # Reference density of condensed sulfur ~2 g/cm³ at STP
      rho_S0 = 2000.0  # [kg/m³]
      K = 1e-11        # compressibility factor (Pa^-1), tuned so density rises with P
      rho_sulfur = rho_S0 * (1 + K * P)

    # Cap sulfur density at realistic planetary values
      rho_sulfur = min(rho_sulfur, 20000.0)  # [kg/m³]

    # -------------------
    # 3. MIXING BY MASS FRACTION
    # -------------------
      rho = XH * rho_gas + XHe * rho_gas + XS * rho_sulfur

    # Safety clipping
      return np.clip(rho, 100.0, 25000.0)
    def _mass_difference(self, Pc):
      """Return difference between enclosed and target mass for root-finding"""
      self.P[0] = Pc
      self.T[0] = 20000
      self.rho[0] = self.modern_equation_of_state(Pc, self.T[0], 0)
      self.M_enc[0] = (4/3) * np.pi * ((self.r[0] + self.dr/2)**3) * self.rho[0]

      for i in range(1, self.n_radial):
        r_inner = self.r[i-1] - self.dr/2
        r_outer = self.r[i] + self.dr/2
        V_shell = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        M_shell = V_shell * self.rho[i-1]
        self.M_enc[i] = self.M_enc[i-1] + M_shell

        self.g[i] = G * self.M_enc[i] / (self.r[i]**2 + 1e-20)
        dP = -self.g[i] * self.rho[i-1] * self.dr
        self.P[i] = max(self.P[i-1] + dP, 1e3)

        gamma = 5/3 if self.P[i] > P_H2_DISSOCIATION else 7/5
        self.T[i] = max(self.T[0] * (self.P[i] / self.P[0])**((gamma-1)/gamma), 100)
        self.rho[i] = self.modern_equation_of_state(self.P[i], self.T[i], i)

      return self.M_enc[-1] - self.M_planet



    def solve_structure_corrected(self):


      print("🔄 Solving structure (corrected method)...")

      def structure_error(Pc):


        # Initial conditions at center
          self.P[0] = Pc
          self.T[0] = 20000  # central T
          self.rho[0] = self.modern_equation_of_state(Pc, self.T[0], 0)
          self.M_enc[0] = (4/3) * np.pi * ((self.r[0] + self.dr/2)**3) * self.rho[0]
          self.g[0] = 0.0  # must be zero at center

        # Radial integration
          for i in range(1, self.n_radial):
            # Enclosed mass first (use rho[i-1])

            r_inner = self.r[i-1] - self.dr/2 if i > 0 else 0.0
            r_outer = self.r[i] + self.dr/2
            V_shell = (4/3) * np.pi * (r_outer**3 - r_inner**3)
            M_shell = V_shell * self.rho[i-1]
            self.M_enc[i] = self.M_enc[i-1] + M_shell

            # Gravity at current radius
            self.g[i] = G * self.M_enc[i] / (self.r[i]**2 + 1e-20)

            # Hydrostatic equilibrium
            dP = -self.g[i] * self.rho[i-1] * self.dr
            self.P[i] = max(self.P[i-1] + dP, 1e3)

            # Adiabatic T profile
            gamma = 5/3 if self.P[i] > P_H2_DISSOCIATION else 7/5
            self.T[i] = max(self.T[0] * (self.P[i] / self.P[0])**((gamma-1)/gamma), 100)

            # Update density with EOS
            self.rho[i] = self.modern_equation_of_state(self.P[i], self.T[i], i)

        # Final error in planet mass
          mass_error = abs(self.M_enc[-1] - self.M_planet) / self.M_planet
          return mass_error

    # Find central pressure using shooting method
      try:
              P_center = brentq(
               self._mass_difference,
              1e10, 1e16, xtol=1e6
            )
      except:
        # fallback log scan
            P_test = np.logspace(11, 14, 30)
            diffs = [self._mass_difference(p) for p in P_test]
            P_center = P_test[np.argmin(np.abs(diffs))]
    # Final solution
      #final_error = structure_error(P_center)
      self._mass_difference(P_center)
    # Validation
      M_expected = self.M_planet
      R_expected = self.R_planet
      g_expected = G * M_expected / (R_expected**2)

      mass_error = abs(self.M_enc[-1] - M_expected) / M_expected * 100
      gravity_error = abs(self.g[-1] - g_expected) / g_expected * 100

      print(f"✅ Structure solved (corrected method):")
      print(f"   Central pressure: {P_center/1e9:.0f} GPa")
      print(f"   Mass error: {mass_error:.2f}% (target: {M_expected:.2e} kg)")
      print(f"   Surface gravity: {self.g[-1]:.2f} m/s² (target: {g_expected:.2f} m/s²)")
      print(f"   Gravity error: {gravity_error:.2f}% ")
        # Core boundary defined by mass fraction
      M_core_target = self.core_frac * self.M_planet
      self.core_idx = np.argmax(self.M_enc >= M_core_target)
      if self.core_idx == 0:
            self.core_idx = 1
      self.shell_mass = self.shell_volume * self.rho

      print(f"   Core boundary: {self.core_idx}/{self.n_radial} points")
      print(f"   Core mass: {self.M_enc[self.core_idx]:.2e} kg")
      print(f"final enclosed mass={self.M_enc[-1]:.2e}, target={self.M_planet:.2e}")
      return mass_error < 5


    def calculate_cp(self):
        """Calculate specific heat capacity"""

        for i in range(self.n_radial):
            cp_mix = 0.0
            mass_total = 0.0

            for sp in self.species:
                if self.X[sp][i] > 0:
                    # Temperature-dependent cp for H₂ dissociation
                    if sp == 'H' and self.T[i] > 2000:
                        cp_base = MOLECULAR_DATA[sp]['cp'] * (1 + np.exp(-4000/self.T[i]))
                    else:
                        cp_base = MOLECULAR_DATA[sp]['cp']

                    cp_mix += self.X[sp][i] * cp_base
                    mass_total += self.X[sp][i]

            self.cp[i] = cp_mix / max(mass_total, 1e-12)
            self.cp[i] = max(self.cp[i], 1000)  # Minimum cp

    def calculate_scale_height(self):
        """Calculate pressure scale height H_p = kT/(μmg)"""

        self.H_p = np.zeros(self.n_radial)

        for i in range(self.n_radial):
            mu = self.calculate_mu(i)
            g_safe = max(self.g[i], 1e-10)
            self.H_p[i] = k_B * self.T[i] / (mu * m_u * g_safe)
            self.H_p[i] = max(self.H_p[i], 1e3)  # Minimum scale height

    def calculate_diffusion_corrected(self):


        # Convective diffusion from mixing length theory
        F_conv = L_JUPITER / (4 * np.pi * self.r**2 + 1e-20)

        # Convective velocity [standard mixing length theory]
        mixing_length = 1.0 * self.H_p  # α = 1.0

        numerator = F_conv * self.g * mixing_length
        denominator = self.rho * self.cp * self.T + 1e-20
        v_nr = (numerator / denominator)**(1/3)

        # Rotation effects [Rossby number calculation]
        Ro = v_nr / 2*(OMEGA_JUPITER * self.H_p + 1e-20)

        # Rotation suppression [literature-based]
        # Based on numerical simulations [Jones & Kuzanyan 2009]
        #rotation_factor = 1.0 / (1.0 + (1.0 / (Ro + 1e-10))**0.5)
        v_conv = v_nr * (Ro**0.2)

        D_conv = (1/3) * v_conv * mixing_length

        # Molecular diffusion [Chapman-Enskog theory ]
        D_mol = {sp: np.zeros_like(self.T) for sp in self.species}
        n_total = self.rho / (m_u * np.mean([MOLECULAR_DATA[sp]['mass'] for sp in self.species]))

        for i, sp_i in enumerate(self.species):
          m_i = MOLECULAR_DATA[sp_i]['mass'] * m_u
          D_eff = np.zeros_like(self.T)

        # Effective D_i = harmonic mean over collisions (abundance-weighted)
          denom = 0.0
          for j, sp_j in enumerate(self.species):
            if sp_i == sp_j:
              continue
            m_j = MOLECULAR_DATA[sp_j]['mass'] * m_u
            mu_ij = (m_i * m_j) / (m_i + m_j)
            sigma_ij = 0.5 * (MOLECULAR_DATA[sp_i]['sigma'] + MOLECULAR_DATA[sp_j]['sigma'])

            D_ij = (3/16) * np.sqrt(2 * np.pi * k_B * self.T / mu_ij) / (n_total * sigma_ij**2)

    # Weight collisions by partner abundance
            denom += self.X[sp_j] / (D_ij + 1e-30)

# Effective harmonic mean with abundance weighting
          D_eff = 1.0 / (denom + 1e-30)

# ---------------------------
#  Extra explicit scaling to enforce physical order H > He > S
# ---------------------------
          mass_scaling = 1.0 / MOLECULAR_DATA[sp_i]['mass']  # lighter species diffuse faster
          D_mol[sp_i] = D_eff*mass_scaling

# Apply pressure suppression

          pressure_factor = 1.0 / (1.0 + 0.1 * (self.P / 1e11)**0.3)
          D_mol[sp_i] *= pressure_factor

# Clip to physical bounds
          D_mol[sp_i] = np.clip(D_mol[sp_i], 1e-15, 1e6)


    # Ledoux suppression
        mu_prof = [self.calculate_mu(i) for i in range(self.n_radial)]
        grad_mu = np.gradient(mu_prof, np.log(self.P + 1e-30))


        D_mol_suppressed = {sp: np.copy(D_mol[sp]) for sp in self.species}
        D_conv_s = np.copy(D_conv)

        for i in range(self.n_radial):
          if grad_mu[i] > 0:
            suppression = np.exp(-min(10.0 * grad_mu[i], 5.0))
            for sp in self.species:   # inner loop safe
              D_mol_suppressed[sp][i] *= suppression
            D_conv_s[i] *= suppression
          else:
                # Unstable: allow full diffusion
                pass

        return D_conv_s, D_mol_suppressed

    def run_evolution_corrected(self, total_time, dt):
        """CORRECTED evolution with proper boundary conditions"""

        SEC_PER_YEAR = 3.154e7
        total_time = total_time * SEC_PER_YEAR
        dt = dt * SEC_PER_YEAR

        print(f"🚀 Running corrected evolution for {total_time/SEC_PER_YEAR:.1f} years")
        print(f"   Timestep: {dt/SEC_PER_YEAR:.1f} years")

        t = 0

        # Calculate properties
        self.calculate_cp()
        self.calculate_scale_height()


        # History storage
        history = {
            "T": [], "L": [], "times": [],
            "species": {sp: [] for sp in self.species}
        }

        while t < total_time:
            # Realistic Kelvin-Helmholtz cooling
            # Based on cooling timescale t_KH ~ GM²/RL
            t_KH = G * self.M_planet**2 / (self.R_planet * L_JUPITER)
            t_years = t / SEC_PER_YEAR
            t_KH_years = t_KH / SEC_PER_YEAR

            # Realistic cooling law
            if t_years < t_KH_years:
                L_current = L_JUPITER * np.exp(-t_years / t_KH_years)
            else:
                L_current = L_JUPITER * (t_KH_years / t_years)
            if t_years < 1e-6:  # avoid division by zero
               L_current = L_JUPITER
            elif t_years < t_KH_years:
              L_current = L_JUPITER * np.exp(-t_years / t_KH_years)
            else:
              L_current = L_JUPITER * (t_KH_years / t_years)
            # Effective temperature including irradiation
            Teff = (L_current / (4 * np.pi * self.R_planet**2 * SIGMA) + self.T_eq**4)**0.25

            # Temperature evolution
            # Temperature evolution (cooling limited by Teff, not hard Teq floor)
            dT = -L_current / (self.M_planet * np.mean(self.cp)) * dt
            self.T = np.maximum(self.T + dT, Teff)


            # Diffusion with CORRECTED boundary conditions
            D_conv, D_mol_suppressed = self.calculate_diffusion_corrected()
            #D_conv = np.ones_like(self.r) * D_conv

            # --- Step 1: convection (bulk, same for all species) ---
            for sp in self.species:
              lap = np.zeros_like(self.X[sp])
              lap[1:-1] = (self.X[sp][2:] - 2*self.X[sp][1:-1] + self.X[sp][:-2]) / self.dr**2
              lap[0] = 2 * (self.X[sp][1] - self.X[sp][0]) / self.dr**2
              lap[-1] = 0

             
              self.X[sp] += dt * np.minimum(D_conv, self.dr**2/(2*dt)) * lap

# --- Step 2: molecular diffusion (species-dependent) ---
            for sp in self.species:
              lap = np.zeros_like(self.X[sp])
              lap[1:-1] = (self.X[sp][2:] - 2*self.X[sp][1:-1] + self.X[sp][:-2]) / self.dr**2
              lap[0] = 2 * (self.X[sp][1] - self.X[sp][0]) / self.dr**2
              lap[-1] = 0
              CFL_limit = self.dr**2 / (2*dt)  # stability limit
              D_safe = np.minimum(D_mol_suppressed[sp], self.dr**2/(2*dt))
              self.X[sp] += dt * D_safe * lap

              #D_safe = D_mol_suppressed[sp]
              #self.X[sp] += dt * D_safe * lap

    # Clip to keep physical bounds
            self.X[sp] = np.nan_to_num(self.X[sp], nan=0.0, posinf=1.0, neginf=0.0)
            self.X[sp] = np.clip(self.X[sp], 0, 1)

# --- Step 3: renormalize ---
           

            # Store data
            history["T"].append(self.T.copy())
            history["L"].append(L_current)
            history["times"].append(t / SEC_PER_YEAR)
            for sp in self.species:
                history["species"][sp].append(self.X[sp].copy())
            if "mass" not in history:
              history["mass"] = {sp: [] for sp in self.species}
            for sp in self.species:
              history["mass"][sp].append((self.X[sp] * self.shell_mass).copy())


            t += dt

            # Progress output
            if len(history["times"]) % max(1, int(total_time/dt/10)) == 0:
                print(f"   t = {t/SEC_PER_YEAR:.0f} yr: T_center = {self.T[0]:.0f} K, "
                      f"L = {L_current:.2e} W")

        print("✅ Evolution finished ")
        return history