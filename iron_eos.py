import numpy as np
from scipy.optimize import newton, brentq
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu) # erg/K/g to kb/baryon
dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion
dyn_to_GPa = (u.dyn/u.cm**2).to('GPa') # dyn/cm² to GPa conversion
U_conv_cgs = (1.0 * u.J/u.kg).to(u.erg/u.g).value          # 1 J/kg -> erg/g
S_conv_cgs = (1.0 * u.J/u.kg/u.K).to(u.erg/u.g/u.K).value  # 1 J/kg/K -> erg/g/K

class Fe_EOS_analytic:
    """
    Dorogokupets et al. (2017) thermodynamic model for Fe phases.

    Explicit phases: 'bcc', 'fcc', 'hcp', 'liquid'
    Auto phase: 'auto'  (switches between solid_phase (default 'hcp') and 'liquid'
                         using a chosen melting curve)

    Inputs:
      rho : kg/m^3
      T   : K

    Core outputs:
      P : Pa
      s : J/kg/K
      u : J/kg
      f : J/kg  (Helmholtz free energy)

    Added thermo:
      KT    : Pa         isothermal bulk modulus
      alpha : 1/K        thermal expansivity
      cv    : J/kg/K
      cp    : J/kg/K

    Auto melt curves (P in Pa internally):
      - 'gonzalez2023' : 6469 * (1 + (P_GPa-300)/434.822)**1.839
      - 'zhang2015'    : 1900 * (P_GPa/31.3 + 1)**(1/1.99)
    """

    # Physical constants (SI)
    R = 8.31446261815324  # J/mol/K
    M_Fe = 55.845e-3      # kg/mol
    n = 1.0               # atoms per formula unit

    def __init__(
        self,
        phase="auto",
        *,
        solid_phase="hcp",
        melt_curve="gonzalez2023",
        phase_hysteresis_K=0.0,
        melt_smooth_width_K=200.0,
        auto_max_iter=3,
        solid_lowP="bcc",
        solid_switch_P_GPa=13.0,
        solid_switch_width_GPa=2.0,
        # NEW: tensile-failure pressure cap
        enforce_P_nonneg=False,
        P_nonneg_smooth_Pa=1e5,   # ~0.01 GPa smoothing band near P=0
    ):


        phase = phase.lower()
        solid_phase = solid_phase.lower()
        melt_curve = melt_curve.lower()

        self.melt_curve = melt_curve
        self.enforce_P_nonneg = bool(enforce_P_nonneg)
        self.P_nonneg_smooth_Pa = float(P_nonneg_smooth_Pa)


        if phase == "auto":
            if solid_phase not in ("bcc", "fcc", "hcp"):
                raise ValueError("solid_phase must be one of: 'bcc','fcc','hcp'")
            if melt_curve not in ("gonzalez2023", "zhang2015"):
                raise ValueError("melt_curve must be 'gonzalez2023' or 'zhang2015'")

            self.phase = "auto"
            self.solid_phase = solid_phase
            self.melt_curve = melt_curve
            self.melt_smooth_width_K = float(melt_smooth_width_K)
            self.phase_hysteresis_K = float(phase_hysteresis_K)
            self.auto_max_iter = int(auto_max_iter)

            # NEW: low-P solid for tension-avoidance + physical α→ε behavior
            self.solid_lowP = solid_lowP.lower()
            if self.solid_lowP not in ("bcc", "fcc", "hcp"):
                raise ValueError("solid_lowP must be one of: 'bcc','fcc','hcp'")

            self.solid_switch_P = float(solid_switch_P_GPa) * 1e9
            self.solid_switch_width_P = float(solid_switch_width_GPa) * 1e9

            # Internal explicit-phase EOS objects
            self._solid_lo = Fe_EOS_analytic(phase=self.solid_lowP)     # typically bcc
            self._solid_hi = Fe_EOS_analytic(phase=self.solid_phase)    # typically hcp
            self._liquid   = Fe_EOS_analytic(phase="liquid")
            return


        if phase not in ("bcc", "fcc", "hcp", "liquid"):
            raise ValueError("phase must be one of: 'bcc', 'fcc', 'hcp', 'liquid', 'auto'")
        self.phase = phase

        params = {
            "bcc": dict(U0_kJmol=0.0,     V0_cm3mol=7.092,  K0_GPa=164.0, K0p=5.50,
                        Theta0_K=303.0,   gamma0=1.736,    beta=1.125,  gamma_inf=0.0,
                        e0_1e6Kinv=198.0, g=1.0,
                        T_star_K=1043.0,  B0=2.22, a_s=0.0,
                        Tref_K=298.15),
            "fcc": dict(U0_kJmol=4.470,   V0_cm3mol=6.9285, K0_GPa=146.2, K0p=4.67,
                        Theta0_K=222.5,   gamma0=2.203,    beta=0.01,   gamma_inf=0.0,
                        e0_1e6Kinv=198.0, g=0.5,
                        T_star_K=None,    B0=0.0,  a_s=0.0,
                        Tref_K=298.15),
            "hcp": dict(U0_kJmol=4.500,   V0_cm3mol=6.8175, K0_GPa=148.0, K0p=5.86,
                        Theta0_K=227.0,   gamma0=2.20,     beta=0.01,   gamma_inf=0.0,
                        e0_1e6Kinv=126.0, g=-0.83,
                        T_star_K=None,    B0=0.0,  a_s=0.0,
                        Tref_K=298.15),
            "liquid": dict(U0_kJmol=-100.204, V0_cm3mol=7.957,  K0_GPa=83.7,  K0p=5.97,
                           Theta0_K=263.0,   gamma0=2.033,     beta=1.168, gamma_inf=0.0,
                           e0_1e6Kinv=198.0, g=0.884,
                           T_star_K=None,    B0=0.0,  a_s=2.12,
                           Tref_K=1811.0),
        }[phase]

        # Convert to SI
        self.U0 = params["U0_kJmol"] * 1e3
        self.V0 = params["V0_cm3mol"] * 1e-6
        self.K0 = params["K0_GPa"] * 1e9
        self.K0p = params["K0p"]
        self.Theta0 = params["Theta0_K"]
        self.gamma0 = params["gamma0"]
        self.beta = params["beta"]
        self.gamma_inf = params["gamma_inf"]
        self.e0 = params["e0_1e6Kinv"] * 1e-6
        self.g = params["g"]
        self.a_s = params["a_s"]
        self.Tref = params["Tref_K"]

        self.T_star = params["T_star_K"]
        self.B0 = params["B0"]

        self.eta = 1.5 * (self.K0p - 1.0)
        self.p_mag = 0.4 if phase == "bcc" else 0.28

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _as_arrays(a, b):
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A, B
    @staticmethod
    def _smoothstep5(u):
        """
        Quintic smoothstep: 0->1 with zero 1st/2nd derivatives at endpoints.
        u should be in [0,1].
        """
        return u*u*u*(u*(u*6.0 - 15.0) + 10.0)

    def _enforce_nonnegative_pressure(self, P):
        """
        Enforce P >= 0 with a smooth transition near P=0.

        Behavior:
        - P_raw <= 0        -> 0 exactly
        - 0 < P_raw < dP    -> smoothly ramps up (C2 smooth)
        - P_raw >= dP       -> P_raw (unchanged)

        dP = self.P_nonneg_smooth_Pa
        """
        P = np.asarray(P, dtype=float)
        if not self.enforce_P_nonneg:
            return P

        dP = float(self.P_nonneg_smooth_Pa)
        if dP <= 0.0:
            return np.maximum(P, 0.0)

        Ppos = np.maximum(P, 0.0)
        u = np.clip(Ppos / dP, 0.0, 1.0)
        s = self._smoothstep5(u)

        # For tiny positive pressures, damp toward 0 smoothly; above dP leave unchanged.
        return np.where(Ppos < dP, Ppos * s, Ppos)


    @staticmethod
    def _log1mexp_neg(y):
        return np.log(-np.expm1(-y))

    # -------------------------
    # Molar volume from density
    # -------------------------
    def V_molar(self, rho):
        rho = np.asarray(rho, dtype=float)
        return self.M_Fe / rho

    def x(self, V):
        return V / self.V0

    def gamma_V(self, V):
        x = self.x(V)
        return self.gamma_inf + (self.gamma0 - self.gamma_inf) * np.power(x, self.beta)

    def q_V(self, V):
        # Eq. (14): q = β x^β (γ0-γ∞) / γ
        x = self.x(V)
        gam = self.gamma_V(V)
        return self.beta * np.power(x, self.beta) * (self.gamma0 - self.gamma_inf) / gam

    def Theta_V(self, V):
        x = self.x(V)
        ln_x = np.log(x)
        one_minus_xb = -np.expm1(self.beta * ln_x)  # 1 - x^beta

        factor = (self.gamma0 - self.gamma_inf)
        if abs(self.beta) < 1e-6:
            expo = -factor * ln_x
        else:
            expo = (factor / self.beta) * one_minus_xb

        return self.Theta0 * np.exp(-self.gamma_inf * ln_x + expo)

    def e_V(self, V):
        x = self.x(V)
        return self.e0 * np.power(x, self.g)

    # -------------------------
    # Vinet cold: P0, E0, K0(V)
    # -------------------------
    def P_cold(self, V):
        X = np.power(V / self.V0, 1.0 / 3.0)
        return 3.0 * self.K0 * np.power(X, -2.0) * (1.0 - X) * np.exp(self.eta * (1.0 - X))

    def E_cold(self, V):
        X = np.power(V / self.V0, 1.0 / 3.0)
        eta = self.eta
        term = (1.0 - eta * (1.0 - X)) * np.exp(eta * (1.0 - X))
        return 9.0 * self.K0 * self.V0 * (1.0 / (eta * eta)) * (1.0 - term)

    def K_cold(self, V):
        # Eq. (3)
        X = np.power(V / self.V0, 1.0 / 3.0)
        eta = self.eta
        return self.K0 * np.power(X, -2.0) * np.exp(eta * (1.0 - X)) * (1.0 + (1.0 - X) * (eta * X + 1.0))

    # -------------------------
    # Einstein lattice: F_th, S_th, E_th, Cv_th, P_th, K_Tth
    # -------------------------
    def F_th(self, V, T):
        T = np.asarray(T, dtype=float)
        Theta = self.Theta_V(V)
        y = Theta / T
        return 3.0 * self.n * self.R * T * self._log1mexp_neg(y)

    def S_th(self, V, T):
        T = np.asarray(T, dtype=float)
        Theta = self.Theta_V(V)
        y = Theta / T
        yclip = np.clip(y, None, 700.0)
        denom = np.expm1(yclip)
        frac = np.where(y > 700.0, 0.0, y / denom)
        return 3.0 * self.n * self.R * (-self._log1mexp_neg(y) + frac)

    def E_th(self, V, T):
        T = np.asarray(T, dtype=float)
        Theta = self.Theta_V(V)
        y = Theta / T
        yclip = np.clip(y, None, 700.0)
        denom = np.expm1(yclip)
        Eth = 3.0 * self.n * self.R * Theta / denom
        return np.where(y > 700.0, 0.0, Eth)

    def Cv_th(self, V, T):
        # Eq. (9)
        T = np.asarray(T, dtype=float)
        Theta = self.Theta_V(V)
        y = Theta / T
        yclip = np.clip(y, None, 700.0)
        expy = np.exp(yclip)
        denom = np.expm1(yclip)
        val = 3.0 * self.n * self.R * (y**2) * expy / (denom**2)
        return np.where(y > 700.0, 0.0, val)

    def P_th(self, V, T):
        gam = self.gamma_V(V)
        Eth = self.E_th(V, T)
        return gam * Eth / V

    def K_Tth(self, V, T):
        gam = self.gamma_V(V)
        q = self.q_V(V)
        Pth = self.P_th(V, T)
        Cv = self.Cv_th(V, T)  # J/mol/K
        return Pth * (1.0 + gam - q) - (gam * gam) * T * Cv / V

    # -------------------------
    # Electronic: F_e, S_e, E_e, Cv_e, P_e, K_Te
    # -------------------------
    def F_e(self, V, T):
        T = np.asarray(T, dtype=float)
        e = self.e_V(V)
        return -1.5 * self.n * self.R * e * T * T

    def S_e(self, V, T):
        T = np.asarray(T, dtype=float)
        e = self.e_V(V)
        return 3.0 * self.n * self.R * e * T

    def E_e(self, V, T):
        T = np.asarray(T, dtype=float)
        e = self.e_V(V)
        return 1.5 * self.n * self.R * e * T * T

    def Cv_e(self, V, T):
        T = np.asarray(T, dtype=float)
        e = self.e_V(V)
        return 3.0 * self.n * self.R * e * T

    def P_e(self, V, T):
        Ee = self.E_e(V, T)
        return self.g * Ee / V

    def K_Te(self, V, T):
        Pe = self.P_e(V, T)
        return Pe * (1.0 - self.g)

    # -------------------------
    # Magnetic (bcc only): F_mag, S_mag, E_mag, Cv_mag
    # -------------------------
    def _mag_D(self):
        p = self.p_mag
        return 518.0/1125.0 + 11692.0/15975.0 * (1.0/p - 1.0)

    def _f_tau(self, tau):
        p = self.p_mag
        D = self._mag_D()
        tau = np.asarray(tau, dtype=float)
        f = np.empty_like(tau)

        m = tau <= 1.0
        if np.any(m):
            t = tau[m]
            A1 = 79.0 * t**(-1.0) / (140.0 * p)
            A2 = (474.0/497.0) * (1.0/p - 1.0) * (t**3/6.0 + t**9/135.0 + t**15/600.0)
            f[m] = 1.0 - (A1 + A2)/D

        mp = ~m
        if np.any(mp):
            t = tau[mp]
            f[mp] = -(t**(-5.0)/10.0 + t**(-15.0)/315.0 + t**(-25.0)/1500.0)/D

        return f

    def _df_dtau(self, tau):
        p = self.p_mag
        D = self._mag_D()
        tau = np.asarray(tau, dtype=float)
        df = np.empty_like(tau)

        m = tau <= 1.0
        if np.any(m):
            t = tau[m]
            factor = (474.0/497.0) * (1.0/p - 1.0)
            dA1 = -79.0 / (140.0 * p) * t**(-2.0)
            dA2 = factor * (0.5 * t**2 + (1.0/15.0) * t**8 + (1.0/40.0) * t**14)
            df[m] = -(dA1 + dA2)/D

        mp = ~m
        if np.any(mp):
            t = tau[mp]
            df[mp] = (0.5 * t**(-6.0) + (1.0/21.0) * t**(-16.0) + (1.0/60.0) * t**(-26.0))/D

        return df

    def _d2f_dtau2(self, tau):
        p = self.p_mag
        D = self._mag_D()
        tau = np.asarray(tau, dtype=float)
        d2 = np.empty_like(tau)

        m = tau <= 1.0
        if np.any(m):
            t = tau[m]
            factor = (474.0/497.0) * (1.0/p - 1.0)
            d2A1 = 2.0 * 79.0 / (140.0 * p) * t**(-3.0)
            d2A2 = factor * (t + (8.0/15.0)*t**7 + (7.0/20.0)*t**13)
            d2[m] = -(d2A1 + d2A2)/D

        mp = ~m
        if np.any(mp):
            t = tau[mp]
            d2[mp] = (-3.0*t**(-7.0) - (16.0/21.0)*t**(-17.0) - (13.0/30.0)*t**(-27.0))/D

        return d2

    def F_mag(self, T):
        if (self.B0 is None) or (self.B0 <= 0.0) or (self.T_star is None):
            return 0.0 * np.asarray(T, dtype=float)
        T = np.asarray(T, dtype=float)
        A = self.R * np.log(self.B0 + 1.0)
        tau = T / self.T_star
        f = self._f_tau(tau)
        return A * T * (f - 1.0)

    def S_mag(self, T):
        if (self.B0 is None) or (self.B0 <= 0.0) or (self.T_star is None):
            return 0.0 * np.asarray(T, dtype=float)
        T = np.asarray(T, dtype=float)
        A = self.R * np.log(self.B0 + 1.0)
        tau = T / self.T_star
        f = self._f_tau(tau)
        df = self._df_dtau(tau)
        return -A * ((f - 1.0) + tau * df)

    def E_mag(self, T):
        if (self.B0 is None) or (self.B0 <= 0.0) or (self.T_star is None):
            return 0.0 * np.asarray(T, dtype=float)
        T = np.asarray(T, dtype=float)
        A = self.R * np.log(self.B0 + 1.0)
        tau = T / self.T_star
        df = self._df_dtau(tau)
        return -A * T * tau * df

    def Cv_mag(self, T):
        # C_Vmag = dE_mag/dT (mag depends only on T)
        if (self.B0 is None) or (self.B0 <= 0.0) or (self.T_star is None):
            return 0.0 * np.asarray(T, dtype=float)
        T = np.asarray(T, dtype=float)
        A = self.R * np.log(self.B0 + 1.0)
        tau = T / self.T_star
        df = self._df_dtau(tau)
        d2f = self._d2f_dtau2(tau)
        Tstar = self.T_star
        return -(A/Tstar) * (2.0*T*df + (T*T/Tstar)*d2f)

    # -------------------------
    # Full molar F, P, S, U
    # -------------------------
    def F_molar(self, V, T):
        Tref = self.Tref
        F = self.U0 + self.E_cold(V)
        F += (self.F_th(V, T) - self.F_th(V, Tref))
        F += (self.F_e(V, T)  - self.F_e(V, Tref))
        if self.phase == "liquid":
            F += -self.a_s * self.R * (T - Tref)
        else:
            F += (self.F_mag(T) - self.F_mag(Tref))
        return F

    def P(self, rho, T):
        if self.phase == "auto":
            return self.get_p_rhot(rho, T)

        V = self.V_molar(rho)
        Tref = self.Tref
        P_raw = self.P_cold(V) + (self.P_th(V, T) - self.P_th(V, Tref)) + (self.P_e(V, T) - self.P_e(V, Tref))
        return self._enforce_nonnegative_pressure(P_raw)


    def S_molar(self, V, T):
        S = self.S_th(V, T) + self.S_e(V, T)
        if self.phase == "liquid":
            S += self.a_s * self.R
        else:
            S += self.S_mag(T)
        return S

    def U_molar(self, V, T):
        Tref = self.Tref
        U = self.U0 + self.E_cold(V)
        U += (self.E_th(V, T) - self.E_th(V, Tref))
        U += (self.E_e(V, T)  - self.E_e(V, Tref))
        if self.phase == "liquid":
            U += self.a_s * self.R * Tref
        else:
            U += (self.E_mag(T) - self.E_mag(Tref))
        return U

    # -------------------------
    # KT, alpha, Cv, Cp (molar)
    # -------------------------
    def KT_molar(self, V, T):
        Tref = self.Tref
        KT = self.K_cold(V)
        KT += (self.K_Tth(V, T) - self.K_Tth(V, Tref))
        KT += (self.K_Te(V, T)  - self.K_Te(V, Tref))
        return KT

    def alpha(self, rho, T):
        if self.phase == "auto":
            return self.get_alpha_rhot(rho, T)

        V = self.V_molar(rho)
        KT = self.KT_molar(V, T)  # Pa
        # (∂P/∂T)_V = gamma*Cv_th/V + g*Cv_e/V ; magnetic has no P(V,T)
        dPdT_V = (self.gamma_V(V) * self.Cv_th(V, T) + self.g * self.Cv_e(V, T)) / V
        return dPdT_V / KT

    def Cv_molar(self, V, T):
        Cv = self.Cv_th(V, T) + self.Cv_e(V, T)
        if self.phase != "liquid":
            Cv += self.Cv_mag(T)
        return Cv

    def Cp_molar(self, V, T):
        KT = self.KT_molar(V, T)
        a = self.alpha(self.M_Fe / V, T)  # alpha expects rho; rho = M/V
        Cv = self.Cv_molar(V, T)
        return Cv + (a*a) * T * V * KT

    # -------------------------
    # Auto phase logic + melting curves
    # -------------------------
    def Tmelt(self, P_pa, curve=None):
        curve = (self.melt_curve if curve is None else curve).lower()
        P_pa = np.asarray(P_pa, dtype=float)
        P_GPa = P_pa / 1e9

        if curve == "gonzalez2023":
            return 6469.0 * (1.0 + (P_GPa - 300.0) / 434.822) ** 1.839
        if curve == "zhang2015":
            return 1900.0 * (P_GPa / 31.3 + 1.0) ** (1.0 / 1.99)

        raise ValueError("curve must be 'gonzalez2023' or 'zhang2015'")

    def _auto_phase_mask(self, rho, T):
        """
        Return boolean mask: True->liquid, False->solid.
        Uses a short self-consistency iteration to reduce flip-flop.
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        h = self.phase_hysteresis_K

        # start assuming solid everywhere
        is_liq = np.zeros_like(rho_arr, dtype=bool)

        for _ in range(max(1, self.auto_max_iter)):
            # pressure using current phase assignment
            P_now = np.where(is_liq,
                             self._liquid.P(rho_arr, T_arr),
                             self._solid.P(rho_arr, T_arr))
            Tm = self.Tmelt(np.maximum(P_now, 0.0), curve=self.melt_curve)

            # hysteresis band to avoid chatter near Tm
            to_liq = T_arr > (Tm + h)
            to_sol = T_arr < (Tm - h)
            new_is_liq = np.where(to_liq, True, np.where(to_sol, False, is_liq))

            if np.all(new_is_liq == is_liq):
                break
            is_liq = new_is_liq

        return is_liq

    def _smooth_weight(self, x):
        """Return w in [0,1] using tanh smoothstep."""
        return 0.5 * (1.0 + np.tanh(x))

    def _auto_blend_weights_and_pressure(self, rho, T):
        """
        Auto mode with TWO smooth transitions:
        (1) solid-solid: lowP solid (usually bcc) -> highP solid (usually hcp), in pressure-space
        (2) solid-liquid: via melt curve, in temperature-space

        Returns:
        w_melt : liquid fraction in [0,1]
        w_hi   : highP-solid fraction in [0,1]  (e.g. hcp fraction within the solid)
        P_blend: blended pressure [Pa]
        P_sol  : blended solid pressure [Pa]
        P_liq  : liquid pressure [Pa]
        Tm     : melt temperature evaluated at final P_blend [K]
        """
        rho_arr, T_arr = self._as_arrays(rho, T)

        # Endmember pressures at same (rho, T)
        P_lo  = self._solid_lo.P(rho_arr, T_arr)
        P_hi  = self._solid_hi.P(rho_arr, T_arr)
        P_liq = self._liquid.P(rho_arr, T_arr)

        P_lo  = self._enforce_nonnegative_pressure(P_lo)
        P_hi  = self._enforce_nonnegative_pressure(P_hi)
        P_liq = self._enforce_nonnegative_pressure(P_liq)


        dT_melt = self.melt_smooth_width_K
        Ptr = self.solid_switch_P
        dP  = self.solid_switch_width_P

        # initial P guess
        P = 0.5 * (P_lo + P_hi)

        for _ in range(max(1, self.auto_max_iter)):
            # --- solid-solid weight (pressure-space) ---
            if dP <= 0.0:
                w_hi = (P >= Ptr).astype(float)
            else:
                w_hi = self._smooth_weight((P - Ptr) / dP)

            P_sol = (1.0 - w_hi) * P_lo + w_hi * P_hi

            # --- melt weight (temperature-space, using melt curve evaluated at current P) ---
            Tm = self.Tmelt(np.maximum(P, 0.0), curve=self.melt_curve)

            if dT_melt <= 0.0:
                w_melt = (T_arr >= Tm).astype(float)
            else:
                w_melt = self._smooth_weight((T_arr - Tm) / dT_melt)

            P_new = (1.0 - w_melt) * P_sol + w_melt * P_liq

            if np.allclose(P_new, P, rtol=1e-10, atol=0.0):
                P = P_new
                break
            P = P_new

        # final update with converged P
        if dP <= 0.0:
            w_hi = (P >= Ptr).astype(float)
        else:
            w_hi = self._smooth_weight((P - Ptr) / dP)
        P_sol = (1.0 - w_hi) * P_lo + w_hi * P_hi

        Tm = self.Tmelt(np.maximum(P, 0.0), curve=self.melt_curve)
        if dT_melt <= 0.0:
            w_melt = (T_arr >= Tm).astype(float)
        else:
            w_melt = self._smooth_weight((T_arr - Tm) / dT_melt)

        P_blend = (1.0 - w_melt) * P_sol + w_melt * P_liq
        return w_melt, w_hi, P_blend, P_sol, P_liq, Tm


    # -------------------------
    # Public rho,T API (dispatching if auto)
    # -------------------------
    def eos_rhoT(self, rho, T):
        rho_arr, T_arr = self._as_arrays(rho, T)

        if self.phase == "auto":
            is_liq = self._auto_phase_mask(rho_arr, T_arr)

            P = np.where(is_liq,
                         self._liquid.get_p_rhot(rho_arr, T_arr),
                         self._solid.get_p_rhot(rho_arr, T_arr))

            u = np.where(is_liq,
                         self._liquid.get_u_rhot(rho_arr, T_arr),
                         self._solid.get_u_rhot(rho_arr, T_arr))

            s = np.where(is_liq,
                         self._liquid.get_s_rhot(rho_arr, T_arr),
                         self._solid.get_s_rhot(rho_arr, T_arr))

            f = np.where(is_liq,
                         self._liquid.get_f_rhot(rho_arr, T_arr),
                         self._solid.get_f_rhot(rho_arr, T_arr))

            return P.reshape(rho_arr.shape), u.reshape(rho_arr.shape), s.reshape(rho_arr.shape), f.reshape(rho_arr.shape)

        V = self.V_molar(rho_arr)
        P = self.P(rho_arr, T_arr)
        Fm = self.F_molar(V, T_arr)
        Sm = self.S_molar(V, T_arr)
        Um = self.U_molar(V, T_arr)

        f = Fm / self.M_Fe
        s = Sm / self.M_Fe
        u = Um / self.M_Fe

        return P.reshape(rho_arr.shape), u.reshape(rho_arr.shape), s.reshape(rho_arr.shape), f.reshape(rho_arr.shape)

    def get_p_rhot(self, rho, T):
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, P_blend, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)
            return self._enforce_nonnegative_pressure(P_blend).reshape(rho_arr.shape)
        return self._enforce_nonnegative_pressure(self.P(rho_arr, T_arr)).reshape(rho_arr.shape)

    def get_u_rhot(self, rho, T):
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            u_lo  = self._solid_lo.get_u_rhot(rho_arr, T_arr)
            u_hi  = self._solid_hi.get_u_rhot(rho_arr, T_arr)
            u_liq = self._liquid.get_u_rhot(rho_arr, T_arr)

            u_sol = (1.0 - w_hi) * u_lo + w_hi * u_hi
            return ((1.0 - w_melt) * u_sol + w_melt * u_liq).reshape(rho_arr.shape)

        V = self.V_molar(rho_arr)
        return (self.U_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * U_conv_cgs
    
    def get_s_rhot(self, rho, T):
        """
        Get the entropy at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          s: erg/g/K
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            s_lo  = self._solid_lo.get_s_rhot(rho_arr, T_arr)
            s_hi  = self._solid_hi.get_s_rhot(rho_arr, T_arr)
            s_liq = self._liquid.get_s_rhot(rho_arr, T_arr)

            s_sol = (1.0 - w_hi) * s_lo + w_hi * s_hi
            return ((1.0 - w_melt) * s_sol + w_melt * s_liq).reshape(rho_arr.shape)
    
        V = self.V_molar(rho_arr)
        return (self.S_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * S_conv_cgs
    
    def get_f_rhot(self, rho, T):
        """
        Get the Helmholtz free energy at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          f: erg/g
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            f_lo  = self._solid_lo.get_f_rhot(rho_arr, T_arr)
            f_hi  = self._solid_hi.get_f_rhot(rho_arr, T_arr)
            f_liq = self._liquid.get_f_rhot(rho_arr, T_arr)

            f_sol = (1.0 - w_hi) * f_lo + w_hi * f_hi
            return ((1.0 - w_melt) * f_sol + w_melt * f_liq).reshape(rho_arr.shape)
    
        V = self.V_molar(rho_arr)
        return (self.F_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * U_conv_cgs


    def get_KT_rhot(self, rho, T):
        """
        Get the isothermal bulk modulus at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          KT: Pa
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            KT_lo  = self._solid_lo.get_KT_rhot(rho_arr, T_arr)
            KT_hi  = self._solid_hi.get_KT_rhot(rho_arr, T_arr)
            KT_liq = self._liquid.get_KT_rhot(rho_arr, T_arr)

            KT_sol = (1.0 - w_hi) * KT_lo + w_hi * KT_hi
            return ((1.0 - w_melt) * KT_sol + w_melt * KT_liq).reshape(rho_arr.shape)
    
        V = self.V_molar(rho_arr)
        return self.KT_molar(V, T_arr).reshape(rho_arr.shape)
    
    def get_alpha_rhot(self, rho, T):
        """
        Get the thermal expansivity at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          alpha: 1/K
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            alpha_lo  = self._solid_lo.get_alpha_rhot(rho_arr, T_arr)
            alpha_hi  = self._solid_hi.get_alpha_rhot(rho_arr, T_arr)
            alpha_liq = self._liquid.get_alpha_rhot(rho_arr, T_arr)

            alpha_sol = (1.0 - w_hi) * alpha_lo + w_hi * alpha_hi
            return ((1.0 - w_melt) * alpha_sol + w_melt * alpha_liq).reshape(rho_arr.shape)
        return self.alpha(rho_arr, T_arr).reshape(rho_arr.shape)
    
    def get_cv_rhot(self, rho, T):
        """
        Get the specific heat at constant volume at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          cv: erg/g/K
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            cv_lo  = self._solid_lo.get_cv_rhot(rho_arr, T_arr)
            cv_hi  = self._solid_hi.get_cv_rhot(rho_arr, T_arr)
            cv_liq = self._liquid.get_cv_rhot(rho_arr, T_arr)

            cv_sol = (1.0 - w_hi) * cv_lo + w_hi * cv_hi
            return ((1.0 - w_melt) * cv_sol + w_melt * cv_liq).reshape(rho_arr.shape)
    
        V = self.V_molar(rho_arr)
        return (self.Cv_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * S_conv_cgs
    
    def get_cp_rhot(self, rho, T):
        """
        Get the specific heat at constant pressure at a given density and temperature.
        rho: kg/m^3
        T: K
        Returns:
          cp: erg/g/K
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)

            cp_lo  = self._solid_lo.get_cp_rhot(rho_arr, T_arr)
            cp_hi  = self._solid_hi.get_cp_rhot(rho_arr, T_arr)
            cp_liq = self._liquid.get_cp_rhot(rho_arr, T_arr)

            cp_sol = (1.0 - w_hi) * cp_lo + w_hi * cp_hi
            return ((1.0 - w_melt) * cp_sol + w_melt * cp_liq).reshape(rho_arr.shape)
    
        cv = self.get_cv_rhot(rho_arr, T_arr)
        a  = self.get_alpha_rhot(rho_arr, T_arr)
        KT = self.get_KT_rhot(rho_arr, T_arr)
        cp = cv + (a*a) * T_arr * KT / rho_arr
        return cp.reshape(rho_arr.shape) * S_conv_cgs


    # -------------------------
    # P(rho,T) -> rho(P,T) inversion in SI (P in Pa, rho in kg/m^3)
    # -------------------------
    def get_rho_pt_inv(
        self,
        P,
        T,
        rho0=None,
        *,
        rho_bracket=(1000, 100000.0),
        tol=1e-8,
        maxiter=50,
        newton_first=True,
        dPdrho_eps_rel=1e-6,
    ):
        """
        Invert P(rho,T) -> rho(P,T) by root finding.

        Units:
          P: Pa
          T: K
          rho: kg/m^3

        Strategy:
          1) Newton (optional) with numerical dP/drho
          2) Brent bracketing fallback on rho_bracket
        """

        P_arr, T_arr = self._as_arrays(P, T)
        out = np.empty_like(P_arr, dtype=float)

        if rho0 is not None:
            rho0_arr, _ = self._as_arrays(rho0, T_arr)
        else:
            rho0_arr = None

        a_br, b_br = float(rho_bracket[0]), float(rho_bracket[1])

        def dP_drho_num(r, t):
            r = float(r)
            h = dPdrho_eps_rel * max(abs(r), 1.0)
            p_hi = float(self.get_p_rhot(r + h, t))
            p_lo = float(self.get_p_rhot(r - h, t))
            return (p_hi - p_lo) / (2.0 * h)

        for idx in np.ndindex(P_arr.shape):
            p_tgt = float(P_arr[idx])
            t = float(T_arr[idx])

            def f(r):
                return float(self.get_p_rhot(r, t)/p_tgt - 1)

            def fp(r):
                return float(dP_drho_num(r, t))

            r_guess = float(rho0_arr[idx]) if rho0_arr is not None else 0.5 * (a_br + b_br)

            solved = False
            if newton_first:
                try:
                    r_new = newton(f, r_guess, fprime=fp, tol=tol, maxiter=maxiter)
                    out[idx] = float(r_new)
                    solved = True
                except Exception:
                    solved = False

            if not solved:
                fa = f(a_br)
                fb = f(b_br)
                if np.isnan(fa) or np.isnan(fb) or fa * fb > 0.0:
                    raise ValueError(
                        f"brentq: target P={p_tgt:g} Pa at T={t:g} K not bracketed on "
                        f"[{a_br:g}, {b_br:g}] kg/m^3. f(a)={fa:g}, f(b)={fb:g}."
                    )
                out[idx] = float(brentq(f, a_br, b_br, xtol=tol, maxiter=maxiter))

        return out.reshape(P_arr.shape)

    def get_s_pt_inv(self,
        P,
        T,
        rho0=None,
        *,
        rho_bracket=(1000, 100000.0),
        tol=1e-8,
        maxiter=50,
        newton_first=True,
        dPdrho_eps_rel=1e-6,
    ):
        """
        Obtains S(P,T) via rho(P,T) inversion.
        P: Pa
        T: K
        rho0: kg/m^3
        rho_bracket: tuple of floats (min, max)
        tol: float
        maxiter: int
        newton_first: bool
        dPdrho_eps_rel: float
        Returns:
          S: erg/g/K
        """
        
        rho = self.get_rho_pt_inv(P, T)

        return self.get_s_rhot(rho, T)

    def get_T_sp_inv(self, _s, _P, bracket = (0, 200000), xtol=1e-8, maxiter=500):
        """
        Invert s(P, T) → T, i.e. find T such that get_s_pt(P, T) == s.

        Parameters
        ----------
        _s : float or array_like
            Entropy value(s) in kB/baryon.
        _P : float or array_like
            Pressure value(s) in Pa.
        xtol : float, optional
            Tolerance on the temperature root (passed to brentq).
        maxiter : int, optional
            Maximum number of iterations for brentq.

        Returns
        -------
        T_sol : float or ndarray
            Temperature(s) in K.  If any root‐finding fails, the corresponding
            entry is set to np.nan.
        """
        s_arr = np.atleast_1d(_s)
        P_arr = np.atleast_1d(_P)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        def _find_T(s_val, P_val):
            # The function whose root in T we seek:
            def err(T_val):
                return self.get_s_pt_inv(P_val, T_val) * erg_to_kbbar / s_val - 1

            try:
                return brentq(err, bracket[0], bracket[1], xtol=xtol, maxiter=maxiter)
            except ValueError:
                # e.g. f(T_min)*f(T_max) ≥ 0 or NaN → no bracket
                return np.nan

        # vectorize over the pair (s_val, P_val)
        T_roots = np.vectorize(_find_T)(s_arr, P_arr)

        # return scalar if inputs were scalars
        if T_roots.size == 1:
            return float(T_roots)
        return T_roots

    def get_T_srho_inv(self, _s, _rho, bracket = (0, 200000), xtol=1e-8, maxiter=500):
        """
        Invert s(P, T) → T, i.e. find T such that get_s_pt(P, T) == s.

        Parameters
        ----------
        _s : float or array_like
            Entropy value(s) in kB/baryon.
        _P : float or array_like
            Pressure value(s) in Pa.
        xtol : float, optional
            Tolerance on the temperature root (passed to brentq).
        maxiter : int, optional
            Maximum number of iterations for brentq.

        Returns
        -------
        T_sol : float or ndarray
            Temperature(s) in K.  If any root‐finding fails, the corresponding
            entry is set to np.nan.
        """
        s_arr = np.atleast_1d(_s)
        rho_arr = np.atleast_1d(_rho)
        s_arr, rho_arr = np.broadcast_arrays(s_arr, rho_arr)

        def _find_T(s_val, rho_val):
            # The function whose root in T we seek:
            def err(T_val):
                return self.get_s_rhot(rho_val, T_val) * erg_to_kbbar / s_val - 1

            try:
                return brentq(err, bracket[0], bracket[1], xtol=xtol, maxiter=maxiter)
            except ValueError:
                # e.g. f(T_min)*f(T_max) ≥ 0 or NaN → no bracket
                return np.nan

        # vectorize over the pair (s_val, P_val)
        T_roots = np.vectorize(_find_T)(s_arr, rho_arr)

        # return scalar if inputs were scalars
        if T_roots.size == 1:
            return float(T_roots)
        return T_roots

class Fe_EOS(Fe_EOS_analytic):
    """
    Fe EOS with optional PT tables (rho, s, u) loaded from an NPZ.

    Behavior:
      - tab=True  -> PT-table interpolation
      - tab=False -> analytic inversion via Fe_EOS.get_rho_pt_inv + analytic getters

    Table expectations (default keys):
      pvals_pt     [Pa]
      tvals_pt     [K]
      rho_grid_pt  [g/cm^3 by default]  (converted to kg/m^3 internally)
      s_grid_pt    [erg/g/K]
      u_grid_pt    [erg/g]
    """

    def __init__(
        self,
        *args,
        rgi_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if rgi_kwargs is None:
            rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)

        data = np.load('eos/dorogokupets_iron_eos/iron_eos_PT_liquid.npz')

        self.pvals_pt = np.asarray(data['pvals_pt'], dtype=float)  # Pa
        self.tvals_pt = np.asarray(data['tvals_pt'], dtype=float)  # K

        self.rho_grid_pt = np.asarray(data['rho_grid_pt'], dtype=float) # kg/m^3
        self.s_grid_pt = np.asarray(data['s_grid_pt'], dtype=float)  # erg/g/K
        self.u_grid_pt = np.asarray(data['u_grid_pt'], dtype=float)  # erg/g

        # Interpolators: input points are (P, T)
        self.rho_rgi_pt = RGI((self.pvals_pt, self.tvals_pt), self.rho_grid_pt, **rgi_kwargs)
        self.s_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.s_grid_pt, **rgi_kwargs)
        self.u_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.u_grid_pt, **rgi_kwargs)

    @staticmethod
    def _broadcast_PT(P, T):
        scalar = np.isscalar(P) and np.isscalar(T)
        P_arr = np.array(P, ndmin=1, dtype=float)
        T_arr = np.array(T, ndmin=1, dtype=float)
        if P_arr.shape != T_arr.shape:
            P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
        return scalar, P_arr, T_arr

    def _interp_PT(self, rgi, P_arr, T_arr):
        pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
        return rgi(pts).reshape(P_arr.shape)

    # -------------------------
    # PT table API (override)
    # -------------------------
    def get_rho_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        rho(P,T) in kg/m^3.
        If tab=False, uses Fe_EOS.get_rho_pt_inv.
        """
        scalar, P_arr, T_arr = self._broadcast_PT(P, T)

        if tab:
            vals = self._interp_PT(self.rho_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        # If you have tables, using them as an initial guess can speed up inversion
        if rho0 is None:
            try:
                rho0 = self._interp_PT(self.rho_rgi_pt, P_arr, T_arr)
            except Exception:
                rho0 = None

        vals = self.get_rho_pt_inv(P_arr, T_arr, rho0=rho0, **inv_kwargs)
        return float(vals) if scalar else vals

    def get_s_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        s(P,T) in erg/g/K.
        If tab=False, inverts for rho then evaluates analytic s(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast_PT(P, T)

        if tab:
            vals = self._interp_PT(self.s_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_s_rhot(rho, T_arr)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_u_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        u(P,T) in erg/g.
        If tab=False, inverts for rho then evaluates analytic u(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast_PT(P, T)

        if tab:
            vals = self._interp_PT(self.u_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_u_rhot(rho, T_arr)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_T_sp_inv(self, _s, _P, *, bracket = (0, 200000), xtol=1e-8, maxiter=500):
        """
        Invert s(P, T) → T, i.e. find T such that get_s_pt(P, T) == s.

        Parameters
        ----------
        _s : float or array_like
            Entropy value(s) in kB/baryon.
        _P : float or array_like
            Pressure value(s) in Pa.
        xtol : float, optional
            Tolerance on the temperature root (passed to brentq).
        maxiter : int, optional
            Maximum number of iterations for brentq.

        Returns
        -------
        T_sol : float or ndarray
            Temperature(s) in K.  If any root‐finding fails, the corresponding
            entry is set to np.nan.
        """
        s_arr = np.atleast_1d(_s)
        P_arr = np.atleast_1d(_P)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        def _find_T(s_val, P_val):
            # The function whose root in T we seek:
            def err(T_val):
                return self.get_s_pt(P_val, T_val) * erg_to_kbbar / s_val - 1

            try:
                return brentq(err, bracket[0], bracket[1], xtol=xtol, maxiter=maxiter)
            except ValueError:
                # e.g. f(T_min)*f(T_max) ≥ 0 or NaN → no bracket
                return np.nan

        # vectorize over the pair (s_val, P_val)
        T_roots = np.vectorize(_find_T)(s_arr, P_arr)

        # return scalar if inputs were scalars
        if T_roots.size == 1:
            return float(T_roots)
        return T_roots
