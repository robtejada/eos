from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Literal, Tuple, Union, Optional

ArrayLike = Union[float, np.ndarray]

from scipy.optimize import newton, brentq, brenth, least_squares
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

# --- scalar conversion factors (plain floats) ---
ERG_GK_TO_KBBAR = float((u.erg/u.Kelvin/u.gram).to(k_B/amu))  # (erg/g/K) -> (kB/baryon)
DYNCM2_TO_PA    = float((u.dyn/u.cm**2).to('Pa'))
DYNCM2_TO_GPA   = float((u.dyn/u.cm**2).to('GPa'))
U_CONV_CGS      = float((u.J/u.kg).to('erg/g'))       # J/kg -> erg/g
S_CONV_CGS      = float((u.J/u.kg/u.K).to('erg/(g * K)'))  # J/kg/K -> erg/g/K
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

    erg_to_kbbar: float = ERG_GK_TO_KBBAR
    dyn_to_Pa: float = DYNCM2_TO_PA
    dyn_to_GPa: float = DYNCM2_TO_GPA
    L: float = 1.2e6 * (u.J/u.kg).to('erg/g')  # latent heat of fusion of Fe-Si alloy in erg/g (Anderson & Duba 1997)
    kb: float = k_B.to('erg/K') # ergs/K

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

        self.p = params

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
        return (self.U_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * U_CONV_CGS
    
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
        return (self.S_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * S_CONV_CGS
    
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
        return (self.F_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * U_CONV_CGS


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
        return (self.Cv_molar(V, T_arr) / self.M_Fe).reshape(rho_arr.shape) * S_CONV_CGS
    
    def get_cp_rhot(self, rho, T):
        rho_arr, T_arr = self._as_arrays(rho, T)
        if self.phase == "auto":
            w_melt, w_hi, *_ = self._auto_blend_weights_and_pressure(rho_arr, T_arr)
            cp_lo  = self._solid_lo.get_cp_rhot(rho_arr, T_arr)
            cp_hi  = self._solid_hi.get_cp_rhot(rho_arr, T_arr)
            cp_liq = self._liquid.get_cp_rhot(rho_arr, T_arr)
            cp_sol = (1.0 - w_hi) * cp_lo + w_hi * cp_hi
            return ((1.0 - w_melt) * cp_sol + w_melt * cp_liq).reshape(rho_arr.shape)

        cv_cgs = self.get_cv_rhot(rho_arr, T_arr)         # erg/g/K
        a      = self.get_alpha_rhot(rho_arr, T_arr)      # 1/K
        KT     = self.get_KT_rhot(rho_arr, T_arr)         # Pa
        # convert the pressure term (SI) -> cgs and add to cv (already cgs)
        cp_term_cgs = (a*a) * T_arr * KT / rho_arr * S_CONV_CGS  # erg/g/K
        cp_cgs = cv_cgs + cp_term_cgs
        return cp_cgs.reshape(rho_arr.shape)



    # -------------------------
    # P(rho,T) -> rho(P,T) inversion in SI (P in Pa, rho in kg/m^3)
    # -------------------------
    def get_rho_pt_inv(
        self,
        P: ArrayLike,
        T_K: ArrayLike,
        rho_bracket_kgm3: Optional[Tuple[float, float]] = None,
        max_iter: int = 200,
        rtol: float = 1e-10,
        rho_guess0: Optional[float] = None,
        use_lsq_first: bool = True,
        lsq_max_nfev: int = 60,
        bracket_expand_steps: int = 30,
        bracket_expand_factor: float = 1.6,
        on_fail: str = "nan",  # "nan" or "raise"
    ) -> np.ndarray:
        """
        Invert P(rho,T)=P_target for rho at given (P,T).

        Units
        -----
        P: Pa
        T: K
        rho: kg/m^3
        """
        P_in = np.asarray(P, dtype=float)
        T_in = np.asarray(T_K, dtype=float)

        if np.any(T_in <= 0):
            raise ValueError("Temperature must be > 0 K")

        shape = np.broadcast(P_in, T_in).shape
        P_b = np.broadcast_to(P_in, shape)
        T_b = np.broadcast_to(T_in, shape)

        out = np.full(shape, np.nan, dtype=float)

        # Default bracket (tune if you want, but keep positive + wide)
        if rho_bracket_kgm3 is None:
            rho_min, rho_max = 1.0e3, 4.0e4
        else:
            rho_min, rho_max = map(float, rho_bracket_kgm3)

        if rho_min <= 0 or rho_max <= 0 or rho_min >= rho_max:
            raise ValueError("rho_bracket_kgm3 must be positive and increasing (rho_min < rho_max).")

        log_rho_lo, log_rho_hi = np.log(rho_min), np.log(rho_max)

        # Seed density: reference V0 if available (explicit phases), else midpoint
        if hasattr(self, "V0"):
            rho_V0 = float(self.M_Fe / self.V0)  # kg/m^3 (since V0 is m^3/mol)
        else:
            rho_V0 = np.sqrt(rho_min * rho_max)

        rho_seed_first = float(rho_guess0) if rho_guess0 is not None else rho_V0
        rho_seed_first = min(max(rho_seed_first, rho_min), rho_max)

        def P_of_rho_scalar(rho: float, Ti: float) -> float:
            val = self.get_p_rhot(rho, Ti)
            return float(np.asarray(val))

        rho_prev = None

        for idx in np.ndindex(shape):
            Pt = float(P_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(Pt)) or (not np.isfinite(Ti)) or Ti <= 0:
                continue

            rho_guess = rho_seed_first if (rho_prev is None or not np.isfinite(rho_prev)) else float(rho_prev)
            rho_guess = min(max(rho_guess, rho_min), rho_max)

            P_scale = max(abs(Pt), 1.0)

            def resid(logrho_vec):
                rho = float(np.exp(logrho_vec[0]))
                Pm = P_of_rho_scalar(rho, Ti)
                if not np.isfinite(Pm):
                    return np.array([1e30], dtype=float)
                return np.array([(Pm - Pt) / P_scale], dtype=float)

            rho_sol = np.nan

            # (1) bounded least-squares in log(rho)
            if use_lsq_first:
                x0 = np.array([np.log(rho_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([log_rho_lo], [log_rho_hi]),
                        xtol=rtol, ftol=rtol, gtol=rtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        rho_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(rho_try)]))[0]
                        if np.isfinite(r) and abs(r) <= 1e-10:
                            rho_sol = rho_try
                except Exception:
                    pass

            # (2) fallback to brentq with local bracket expansion
            if not np.isfinite(rho_sol):
                def f(rho):
                    Pm = P_of_rho_scalar(rho, Ti)
                    if not np.isfinite(Pm):
                        return np.nan
                    return Pm - Pt

                left = right = rho_guess
                f_left = f_right = f(rho_guess)

                if not np.isfinite(f_left):
                    rho_guess = np.sqrt(rho_min * rho_max)
                    left = right = rho_guess
                    f_left = f_right = f(rho_guess)

                if np.isfinite(f_left) and f_left == 0.0:
                    rho_sol = rho_guess
                else:
                    bracketed = False
                    for _ in range(bracket_expand_steps):
                        left = max(rho_min, left / bracket_expand_factor)
                        right = min(rho_max, right * bracket_expand_factor)
                        f_left = f(left)
                        f_right = f(right)
                        if np.isfinite(f_left) and np.isfinite(f_right) and (f_left == 0.0 or f_right == 0.0 or f_left * f_right < 0.0):
                            bracketed = True
                            if f_left == 0.0:
                                rho_sol = left
                            elif f_right == 0.0:
                                rho_sol = right
                            else:
                                try:
                                    rho_sol = brentq(f, left, right, xtol=rtol, maxiter=max_iter)
                                except Exception:
                                    rho_sol = np.nan
                            break

                    if not bracketed:
                        fA, fB = f(rho_min), f(rho_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0.0):
                            try:
                                rho_sol = brentq(f, rho_min, rho_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                rho_sol = np.nan

            if np.isfinite(rho_sol):
                out[idx] = rho_sol
                rho_prev = rho_sol
            else:
                if on_fail == "raise":
                    raise RuntimeError(
                        f"Failed rho(P,T) inversion: P={Pt:.3e} Pa, T={Ti:.3f} K within "
                        f"[{rho_min}, {rho_max}] kg/m^3"
                    )

        return float(out) if out.size == 1 else out



    def get_s_pt_inv(self, P, T):
        rho = self.get_rho_pt_inv(P, T)
        return self.get_s_rhot(rho, T)  # cgs: erg/g/K


    def _as_float(self, x):
        # Handles python float, numpy scalar, 0-d array cleanly
        return float(np.asarray(x))

    def get_T_srho_inv(
        self,
        _s,
        _rho,
        bracket=(1.0, 200000.0),
        xtol=1e-10,
        maxiter=200,
        s_units="kbbar",
        T_guess0=None,
        use_lsq_first=True,
        lsq_max_nfev=80,
        bracket_expand_steps=30,
        bracket_expand_factor=1.6,
    ):
        """
        Invert s(rho,T) -> T

        _s:
        - if s_units="cgs":   erg/g/K
        - if s_units="kbbar": kB/baryon
        _rho: kg/m^3
        """
        s_arr = np.asarray(_s, dtype=float)
        rho_arr = np.asarray(_rho, dtype=float)
        s_arr, rho_arr = np.broadcast_arrays(s_arr, rho_arr)
        shape = s_arr.shape

        Tmin, Tmax = map(float, bracket)
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")

        # Convert target entropy to cgs (erg/g/K) because get_s_rhot returns cgs
        if str(s_units).lower() == "kbbar":
            s_target_cgs = s_arr / float(self.erg_to_kbbar)
        else:
            s_target_cgs = s_arr

        T_out = np.full(shape, np.nan, dtype=float)

        def S_cgs(rho_val, T_val) -> float:
            return float(np.asarray(self.get_s_rhot(rho_val, T_val)))

        logT_lo, logT_hi = np.log(Tmin), np.log(Tmax)

        # seed temperature: prefer explicit-phase Tref if present
        if T_guess0 is None:
            T_seed_first = float(getattr(self, "Tref", 7000.0))
        else:
            T_seed_first = float(T_guess0)
        T_seed_first = min(max(T_seed_first, Tmin), Tmax)

        T_prev = None

        for idx in np.ndindex(shape):
            rho = float(rho_arr[idx])
            s_t = float(s_target_cgs[idx])

            if (not np.isfinite(rho)) or rho <= 0 or (not np.isfinite(s_t)):
                continue

            T_guess = T_seed_first if (T_prev is None or not np.isfinite(T_prev)) else float(T_prev)
            T_guess = min(max(T_guess, Tmin), Tmax)

            S_scale = max(abs(s_t), 1.0)

            def resid(logT_vec):
                T = float(np.exp(logT_vec[0]))
                Sm = S_cgs(rho, T)
                if not np.isfinite(Sm):
                    return np.array([1e30], dtype=float)
                return np.array([(Sm - s_t) / S_scale], dtype=float)

            T_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(T_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([logT_lo], [logT_hi]),
                        xtol=xtol, ftol=xtol, gtol=xtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        T_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(T_try)]))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            T_sol = T_try
                except Exception:
                    pass

            if not np.isfinite(T_sol):
                def f(T):
                    Sm = S_cgs(rho, T)
                    if not np.isfinite(Sm):
                        return np.nan
                    return Sm - s_t

                left = right = T_guess
                f_left = f_right = f(T_guess)

                if not np.isfinite(f_left):
                    T_guess = np.sqrt(Tmin * Tmax)
                    left = right = T_guess
                    f_left = f_right = f(T_guess)

                if np.isfinite(f_left) and f_left == 0.0:
                    T_sol = T_guess
                else:
                    for _ in range(bracket_expand_steps):
                        left = max(Tmin, left / bracket_expand_factor)
                        right = min(Tmax, right * bracket_expand_factor)
                        f_left = f(left)
                        f_right = f(right)
                        if np.isfinite(f_left) and np.isfinite(f_right) and (f_left == 0.0 or f_right == 0.0 or f_left * f_right < 0.0):
                            if f_left == 0.0:
                                T_sol = left
                            elif f_right == 0.0:
                                T_sol = right
                            else:
                                try:
                                    T_sol = brentq(f, left, right, xtol=xtol, maxiter=maxiter)
                                except Exception:
                                    T_sol = np.nan
                            break

                    if not np.isfinite(T_sol):
                        fA, fB = f(Tmin), f(Tmax)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0.0):
                            try:
                                T_sol = brentq(f, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
                            except Exception:
                                T_sol = np.nan

            if np.isfinite(T_sol):
                T_out[idx] = T_sol
                T_prev = T_sol

        return float(T_out) if T_out.size == 1 else T_out

    def get_T_sp_inv(self, _s, _P, bracket=(1.0, 20000.0), xtol=1e-8, maxiter=500, s_units="kbbar"):
        """
        Invert s(P,T) -> T, where s(P,T) is computed via rho(P,T) inversion.

        _P in Pa
        _s in erg/g/K (default) or kB/baryon if s_units="kbbar"
        """
        s_arr = np.atleast_1d(_s).astype(float)
        P_arr = np.atleast_1d(_P).astype(float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        if s_units.lower() == "kbbar":
            s_target_cgs = s_arr / self.erg_to_kbbar
        else:
            s_target_cgs = s_arr

        Tmin, Tmax = float(bracket[0]), float(bracket[1])
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")

        def _find_T(s_cgs, P_val):
            if P_val <= 0 or not np.isfinite(P_val) or not np.isfinite(s_cgs):
                return np.nan

            def err(T):
                # get_s_pt_inv returns cgs entropy already (erg/g/K)
                return self._as_float(self.get_s_pt_inv(P_val, T)) - s_cgs

            try:
                f_lo = err(Tmin)
                f_hi = err(Tmax)
                if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
                    return np.nan
                if f_lo == 0.0:
                    return Tmin
                if f_hi == 0.0:
                    return Tmax
                if f_lo * f_hi > 0:
                    return np.nan

                return brenth(err, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
            except (ValueError, FloatingPointError, RuntimeError):
                # RuntimeError can come from get_rho_pt_inv failing to bracket rho
                return np.nan

        T_roots = np.vectorize(_find_T)(s_target_cgs, P_arr)
        return float(T_roots) if T_roots.size == 1 else T_roots

    def get_rhot_sp_2d_inv(
        self,
        s_target,
        P_target,
        *,
        s_units="kbbar",                 # "cgs" (erg/g/K) or "kbbar" (kB/baryon)
        guess="auto",                  # "auto" or (rho_guess, T_guess)
        T_guess0=None,                 # used only if guess="auto"
        bounds_rho=(1e3, 1e6),         # kg/m^3
        bounds_T=(1.0, 2e5),           # K (must be > 0)
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
        fail_value=np.nan,
        return_diagnostics=False,
    ):
        """
        Solve the coupled system:
            P(rho, T) = P_target   (Pa)
            S(rho, T) = s_target   (cgs by default: erg/g/K)

        Warm-start behavior for array inputs:
        - The first element uses `guess` (or "auto")
        - Each subsequent element uses the previous converged (rho, T) as the initial guess

        Returns
        -------
        rho_sol, T_sol  (arrays with broadcasted shape)
        optionally diagnostics dict if return_diagnostics=True
        """

        # Broadcast inputs
        s_arr = np.asarray(s_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)
        shape = s_arr.shape

        # Convert target entropy to cgs (erg/g/K) because get_s_rhot returns cgs
        if str(s_units).lower() == "kbbar":
            s_cgs = s_arr / self.erg_to_kbbar
        else:
            s_cgs = s_arr

        rho_out = np.full(shape, fail_value, dtype=float)
        T_out   = np.full(shape, fail_value, dtype=float)

        # Optional diagnostics
        info = {
            "success": np.zeros(shape, dtype=bool),
            "cost": np.full(shape, np.nan),
            "nfev": np.full(shape, np.nan),
            "resid_P_frac": np.full(shape, np.nan),
            "resid_S_frac": np.full(shape, np.nan),
            "message": np.empty(shape, dtype=object),
        } if return_diagnostics else None

        # Bounds in log-space
        rho_lo, rho_hi = map(float, bounds_rho)
        T_lo,   T_hi   = map(float, bounds_T)
        if rho_lo <= 0 or T_lo <= 0:
            raise ValueError("bounds_rho and bounds_T must be > 0 on the low end.")

        lb = np.array([np.log(rho_lo), np.log(T_lo)], dtype=float)
        ub = np.array([np.log(rho_hi), np.log(T_hi)], dtype=float)

        # Initial guess seed for the first element
        if guess == "auto":
            # Choose a temperature to seed rho(P,T) inversion
            Tseed = float(getattr(self, "Tref", 7000.0) if T_guess0 is None else T_guess0)

            # If Tseed violates bounds, clamp it
            Tseed = min(max(Tseed, T_lo), T_hi)
            # We’ll compute rho seed per-element (because P differs)
            rho_seed = None
            T_seed   = Tseed
        else:
            rho_seed, T_seed = map(float, guess)
            rho_seed = min(max(rho_seed, rho_lo), rho_hi)
            T_seed   = min(max(T_seed,   T_lo),   T_hi)

        # Continuation guess that updates as we solve
        rho_guess_cur = rho_seed
        T_guess_cur   = T_seed

        # Iterate in given order (warm-start for “subsequent attempts”)
        for idx in np.ndindex(shape):
            Pt = float(P_arr[idx])
            St = float(s_cgs[idx])

            if not (np.isfinite(Pt) and np.isfinite(St)) or Pt <= 0:
                if return_diagnostics:
                    info["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            # Auto-seed rho using your existing rho(P,T) inversion at the current T guess
            if guess == "auto" and rho_guess_cur is None:
                try:
                    rho_guess_cur = float(self.get_rho_pt_inv(Pt, T_guess_cur))
                except Exception:
                    # Fall back to mid-range density if rho(P,Tguess) fails
                    rho_guess_cur = 0.5 * (rho_lo + rho_hi)

            # Clamp guesses into bounds
            rho_guess_cur = min(max(float(rho_guess_cur), rho_lo), rho_hi)
            T_guess_cur   = min(max(float(T_guess_cur),   T_lo),   T_hi)

            x0 = np.array([np.log(rho_guess_cur), np.log(T_guess_cur)], dtype=float)

            # Scaling to make the two equations comparable
            P_scale = max(abs(Pt), 1.0e9)      # Pa
            S_scale = max(abs(St), 1.0)        # erg/g/K

            def residuals(x):
                # log-variables => always positive
                rho = float(np.exp(x[0]))
                T   = float(np.exp(x[1]))

                # Evaluate model
                Pm = float(self.get_p_rhot(rho, T))      # Pa
                Sm = float(self.get_s_rhot(rho, T))      # erg/g/K (cgs)

                # Guard against NaNs/Infs from the EOS
                if not (np.isfinite(Pm) and np.isfinite(Sm)):
                    return np.array([1e30, 1e30], dtype=float)

                return np.array([(Pm - Pt) / P_scale, (Sm - St) / S_scale], dtype=float)

            try:
                sol = least_squares(
                    residuals,
                    x0,
                    bounds=(lb, ub),
                    xtol=xtol,
                    ftol=ftol,
                    gtol=gtol,
                    max_nfev=max_nfev,
                    method="trf",
                )

                if sol.success and np.all(np.isfinite(sol.x)):
                    rho_sol = float(np.exp(sol.x[0]))
                    T_sol   = float(np.exp(sol.x[1]))

                    rho_out[idx] = rho_sol
                    T_out[idx]   = T_sol

                    # Update continuation guess (key feature you asked for)
                    rho_guess_cur = rho_sol
                    T_guess_cur   = T_sol

                    if return_diagnostics:
                        info["success"][idx] = True
                        info["cost"][idx] = sol.cost
                        info["nfev"][idx] = sol.nfev

                        # Report fractional residuals at the solution
                        r = residuals(sol.x)
                        info["resid_P_frac"][idx] = abs(r[0])
                        info["resid_S_frac"][idx] = abs(r[1])
                        info["message"][idx] = sol.message
                else:
                    if return_diagnostics:
                        info["message"][idx] = getattr(sol, "message", "least_squares failed")
                    # Do NOT update guess on failure; keep last good guess
            except Exception as e:
                if return_diagnostics:
                    info["message"][idx] = f"Exception: {e}"
                # Do NOT update guess on exception

        if return_diagnostics:
            return rho_out, T_out, info
        return rho_out, T_out

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
