"""
dorogokupets2017_hcp_fe_eos.py

hcp iron EOS following Dorogokupets et al. (2017) thermodynamic formulation.

Model (paper equations, solid phases):
Helmholtz free energy (relative to reference isotherm Tref):
  F(V,T) = U0 + E0(V)
           + [F_th(V,T) - F_th(V,Tref)]
           + [F_e (V,T) - F_e (V,Tref)]
           + [F_mag(T)  - F_mag(Tref)]   (excluded for hcp; moment ~ 0)

Cold curve at reference Tref=298.15 K:
  Vinet–Rydberg pressure P0(V) (eq. 2)
  Potential energy E0(V) (eq. 5)
  Bulk modulus K0(V) (eq. 3)

Thermal (vibrational) contribution uses Einstein model:
  F_th(V,T) = 3 n R T ln(1 - exp(-Theta/T))  (eq. 6)
  S_th, E_th, C_Vth (eqs. 7–9)
  P_th = 3 n R (gamma/V) [Theta/(exp(Theta/T)-1)] (eq. 10)
  K_Tth via eq. (11)

Volume dependence:
  gamma(V) = gamma_inf + (gamma0 - gamma_inf) x^beta,  x=V/V0  (eq. 13)
  q(V)     = beta x^beta (gamma0 - gamma_inf) / gamma   (eq. 14)
  Theta(V) = Theta0 x^{-gamma_inf} exp[(gamma0-gamma_inf)/beta * (1 - x^beta)] (eq. 15)

Electronic contribution (eqs. 16–17):
  e(V) = e0 x^g,  where e0 has units 1/K (Table: e0 in 1e-6 K^-1)
  F_e = -3/2 n R e(V) T^2
  S_e = 3 n R e T
  E_e = 3/2 n R e T^2
  C_Ve = 3 n R e T
  P_e = (g/V) E_e
  K_Te = P_e (1 - g)

Total pressure:
  P(V,T) = P0(V) + [P_th(V,T)-P_th(V,Tref)] + [P_e(V,T)-P_e(V,Tref)]

Total internal energy:
  U(V,T) = U0 + E0(V) + [E_th(V,T)-E_th(V,Tref)] + [E_e(V,T)-E_e(V,Tref)]

Total entropy:
  S(V,T) = [S_th(V,T)-S_th(V,Tref)] + [S_e(V,T)-S_e(V,Tref)]

Public API matches Ichikawa-style names and units.

Units:
- Inputs: rho in kg/m^3, T in K
- Outputs:
    unit_system="si":  P in Pa,        u in J/kg,   s in J/kg/K
    unit_system="cgs": P in dyn/cm^2,  u in erg/g,  s in erg/g/K
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union, Optional

import numpy as np
from scipy.optimize import brenth, least_squares

import astropy.units as u
from astropy.constants import k_B
from astropy.constants import u as amu


ArrayLike = Union[float, np.ndarray]
UnitSystem = Literal["cgs", "si"]


@dataclass(frozen=True)
class Dorogokupets2017HCPFeParams:
    """
    hcp-Fe fitting parameters (Table 1 in Dorogokupets et al. 2017).
    """
    U0_kJmol: float = 4.500
    V0_cm3mol: float = 6.8175
    K0_GPa: float = 148.0
    K0p: float = 5.86

    Theta0_K: float = 227.0
    gamma0: float = 2.20
    beta: float = 0.01
    gamma_inf: float = 0.0

    e0_1e6Kinv: float = 126.0   # e0 in units of 1e-6 K^-1
    g_elec: float = -0.83

    Tref_K: float = 298.15      # reference isotherm temperature for solids


class Fe_EOS_Dorogokupets2017_HCP:
    """
    hcp Fe EOS using Dorogokupets et al. (2017) formulation.

    Public API (rho,T):
      - get_p_rhot(rho, T)
      - get_u_rhot(rho, T)
      - get_s_rhot(rho, T)
      - get_cv_rhot(rho, T)
      - get_cp_rhot(rho, T)
      - get_alpha_rhot(rho, T)
      - get_kt_rhot(rho, T)
      - get_ks_rhot(rho, T)
      - get_rho_pt_inv(P, T)
      - get_s_pt_inv(P, T)
    """

    # constants
    R_J_per_molK: float = 8.31446261815324
    M_molar_kg_per_mol: float = 55.845e-3  # Fe
    n_atoms_formula_unit: float = 1.0

    # scalar conversions
    erg_to_kbbar: float = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
    dyn_to_Pa: float = float((u.dyn / u.cm**2).to("Pa"))

    def __init__(
        self,
        params: Dorogokupets2017HCPFeParams = Dorogokupets2017HCPFeParams(),
        unit_system: UnitSystem = "si",
        kt_deriv_eps: float = 1e-5,
        apply_s_u_offsets: bool = True,
        ref_state_rhoT: Optional[Tuple[float, float]] = (20000.0, 2000.0),  # (kg/m^3, K)
    ) -> None:
        self.p = params
        self.unit_system = unit_system
        self._eps = float(kt_deriv_eps)

        # parameters in SI
        self.U0_Jmol = self.p.U0_kJmol * 1e3
        self.V0_m3mol = self.p.V0_cm3mol * 1e-6
        self.K0_Pa = self.p.K0_GPa * 1e9
        self.K0p = float(self.p.K0p)

        self.Theta0 = float(self.p.Theta0_K)
        self.gamma0 = float(self.p.gamma0)
        self.beta = float(self.p.beta)
        self.gamma_inf = float(self.p.gamma_inf)

        self.e0_perK = float(self.p.e0_1e6Kinv) * 1e-6  # 1/K
        self.g_elec = float(self.p.g_elec)

        self.Tref = float(self.p.Tref_K)

        # Vinet helper: eta = 3/2 (K0' - 1)
        self._eta_vinet = 1.5 * (self.K0p - 1.0)

        # optional reference offsets to make u,s zero at chosen state (API compatibility)
        self._S_offset_molar = 0.0  # J/mol/K
        self._U_offset_molar = 0.0  # J/mol
        if apply_s_u_offsets and (ref_state_rhoT is not None):
            rho_ref, T_ref = ref_state_rhoT
            V_ref = self._V_molar_from_rho(rho_ref)
            self._S_offset_molar = -float(self._S_molar(V_ref, T_ref))
            self._U_offset_molar = -float(self._U_molar(V_ref, T_ref))

    # -------------------------
    # Unit conversions
    # -------------------------
    @staticmethod
    def _pa_to_barye(P_pa: ArrayLike) -> np.ndarray:
        return np.asarray(P_pa, dtype=float) * 10.0  # 1 Pa = 10 dyn/cm^2

    @staticmethod
    def _barye_to_pa(P_barye: ArrayLike) -> np.ndarray:
        return np.asarray(P_barye, dtype=float) * 0.1

    @staticmethod
    def _jkg_to_ergg(x_jkg: ArrayLike) -> np.ndarray:
        return np.asarray(x_jkg, dtype=float) * 1e4  # 1 J/kg = 1e4 erg/g

    # -------------------------
    # V, rho utilities
    # -------------------------
    def _V_molar_from_rho(self, rho_kgm3: ArrayLike) -> np.ndarray:
        rho = np.asarray(rho_kgm3, dtype=float)
        if np.any(rho <= 0):
            raise ValueError("Density rho must be > 0")
        return self.M_molar_kg_per_mol / rho  # m^3/mol

    def _rho_from_V_molar(self, V_m3mol: ArrayLike) -> np.ndarray:
        V = np.asarray(V_m3mol, dtype=float)
        if np.any(V <= 0):
            raise ValueError("Molar volume must be > 0")
        return self.M_molar_kg_per_mol / V

    def _x(self, V_m3mol: ArrayLike) -> np.ndarray:
        return np.asarray(V_m3mol, dtype=float) / self.V0_m3mol  # x = V/V0

    def _X(self, V_m3mol: ArrayLike) -> np.ndarray:
        # X = (V/V0)^(1/3)
        return np.power(self._x(V_m3mol), 1.0 / 3.0)

    # -------------------------
    # Numerically stable helpers
    # -------------------------
    @staticmethod
    def _log1mexp_pos(z: np.ndarray) -> np.ndarray:
        """
        Compute log(1 - exp(-z)) for z>0 stably.
        """
        z = np.asarray(z, dtype=float)
        if np.any(z <= 0):
            raise ValueError("log1mexp requires z > 0")
        # for z < ln2 use log(-expm1(-z)), else log1p(-exp(-z))
        ln2 = np.log(2.0)
        out = np.empty_like(z)
        m = z < ln2
        out[m] = np.log(-np.expm1(-z[m]))
        out[~m] = np.log1p(-np.exp(-z[~m]))
        return out

    @staticmethod
    def _safe_expm1(z: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(z, dtype=float))

    # -------------------------
    # Dorogokupets model pieces (SI)
    # -------------------------
    def _P0_vinet_pa(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        Vinet–Rydberg cold curve pressure at Tref (eq. 2):
          P0(V) = 3 K0 X^{-2} (1 - X) exp[eta (1 - X)]
        """
        X = self._X(V_m3mol)
        if np.any(X <= 0):
            raise ValueError("X must be > 0")
        return (
            3.0
            * self.K0_Pa
            * np.power(X, -2.0)
            * (1.0 - X)
            * np.exp(self._eta_vinet * (1.0 - X))
        )

    def _E0_potential_jmol(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        Potential (cold) energy at Tref (eq. 5):
          E0(V) = 9 K0 V0 eta^{-2} { 1 - [1 - eta(1-X)] exp[(1-X)eta] }
        """
        X = self._X(V_m3mol)
        eta = self._eta_vinet
        if abs(eta) < 1e-14:
            # pathological; not expected for real K0p
            raise ValueError("eta is too small; check K0p.")
        bracket = 1.0 - (1.0 - eta * (1.0 - X)) * np.exp((1.0 - X) * eta)
        return 9.0 * self.K0_Pa * self.V0_m3mol * (eta**-2) * bracket

    def _K0_vinet_pa(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        Isothermal bulk modulus of cold curve at Tref (eq. 3):
          K0(V) = K0 X^{-2} exp[eta(1-X)] [1 + (1-X)(eta X + 1)]
        """
        X = self._X(V_m3mol)
        eta = self._eta_vinet
        pref = self.K0_Pa * np.power(X, -2.0) * np.exp(eta * (1.0 - X))
        return pref * (1.0 + (1.0 - X) * (eta * X + 1.0))

    def _gamma(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        gamma(V) (eq. 13):
          gamma = gamma_inf + (gamma0 - gamma_inf) x^beta
        """
        x = self._x(V_m3mol)
        return self.gamma_inf + (self.gamma0 - self.gamma_inf) * np.power(x, self.beta)

    def _q(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        q(V) (eq. 14):
          q = beta x^beta (gamma0 - gamma_inf) / gamma
        """
        x = self._x(V_m3mol)
        gamma = self._gamma(V_m3mol)
        num = self.beta * np.power(x, self.beta) * (self.gamma0 - self.gamma_inf)
        return num / gamma

    def _Theta(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        Einstein temperature Theta(V) (eq. 15):
          Theta = Theta0 x^{-gamma_inf} exp[(gamma0-gamma_inf)/beta * (1 - x^beta)]
        """
        x = self._x(V_m3mol)
        if abs(self.beta) < 1e-12:
            # beta -> 0 limit: gamma ~ constant; Theta ~ Theta0 * x^{-gamma0}
            return self.Theta0 * np.power(x, -self.gamma0)
        expo = (self.gamma0 - self.gamma_inf) / self.beta * (1.0 - np.power(x, self.beta))
        return self.Theta0 * np.power(x, -self.gamma_inf) * np.exp(expo)

    # ---- Einstein vib thermodynamics ----
    def _F_th_jmol(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        F_th(V,T) = 3 n R T ln(1 - exp(-Theta/T)) (eq. 6)
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        Theta = self._Theta(V)
        z = Theta / T
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * T * self._log1mexp_pos(z)

    def _S_th_jmolK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        S_th (eq. 7):
          S = 3 n R [ -ln(1 - exp(-Theta/T)) + (Theta/T)/(exp(Theta/T)-1) ]
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        Theta = self._Theta(V)
        z = Theta / T
        term1 = -self._log1mexp_pos(z)
        denom = np.expm1(z)  # exp(z)-1
        term2 = z / denom
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * (term1 + term2)

    def _E_th_jmol(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        E_th (eq. 8):
          E_th = 3 n R [ Theta / (exp(Theta/T) - 1) ]
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        Theta = self._Theta(V)
        z = Theta / T
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * (Theta / np.expm1(z))

    def _Cv_th_jmolK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        C_Vth (eq. 9):
          C_V = 3 n R (Theta/T)^2 * exp(Theta/T) / (exp(Theta/T)-1)^2
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        Theta = self._Theta(V)
        z = Theta / T
        # exp(z)/(exp(z)-1)^2 = 1/(expm1(z))^2 * exp(z)
        ez = np.exp(z)
        denom = np.expm1(z)
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * (z**2) * (ez / (denom**2))

    def _P_th_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        P_th (eq. 10):
          P_th = 3 n R (gamma/V) [Theta/(exp(Theta/T)-1)]
              = (gamma/V) * E_th
        """
        V = np.asarray(V_m3mol, dtype=float)
        gamma = self._gamma(V)
        Eth = self._E_th_jmol(V, T_K)
        return gamma / V * Eth  # Pa

    def _K_Tth_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        K_Tth (eq. 11):
          K_Tth = P_th(1 + gamma - q) - gamma^2 T C_V / V
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        gamma = self._gamma(V)
        q = self._q(V)
        Pth = self._P_th_pa(V, T)
        Cv = self._Cv_th_jmolK(V, T)
        return Pth * (1.0 + gamma - q) - (gamma**2) * T * Cv / V

    # ---- Electronic contribution ----
    def _e_elec(self, V_m3mol: ArrayLike) -> np.ndarray:
        """
        e(V) = e0 x^g  (eq. 16)
        """
        x = self._x(V_m3mol)
        return self.e0_perK * np.power(x, self.g_elec)

    def _F_e_jmol(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        eV = self._e_elec(V_m3mol)
        T = np.asarray(T_K, dtype=float)
        return -1.5 * self.n_atoms_formula_unit * self.R_J_per_molK * eV * T**2

    def _S_e_jmolK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        eV = self._e_elec(V_m3mol)
        T = np.asarray(T_K, dtype=float)
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * eV * T

    def _E_e_jmol(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        eV = self._e_elec(V_m3mol)
        T = np.asarray(T_K, dtype=float)
        return 1.5 * self.n_atoms_formula_unit * self.R_J_per_molK * eV * T**2

    def _Cv_e_jmolK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        eV = self._e_elec(V_m3mol)
        T = np.asarray(T_K, dtype=float)
        return 3.0 * self.n_atoms_formula_unit * self.R_J_per_molK * eV * T

    def _P_e_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        P_e = (g/V) E_e  (eq. 17)
        """
        V = np.asarray(V_m3mol, dtype=float)
        Ee = self._E_e_jmol(V, T_K)
        return self.g_elec * Ee / V

    def _K_Te_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        K_Te = P_e (1 - g) (eq. 17)
        """
        Pe = self._P_e_pa(V_m3mol, T_K)
        return Pe * (1.0 - self.g_elec)

    # -------------------------
    # Total thermodynamics in molar SI units
    # -------------------------
    def _P_total_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        P0 = self._P0_vinet_pa(V)
        Pth = self._P_th_pa(V, T) - self._P_th_pa(V, self.Tref)
        Pe = self._P_e_pa(V, T) - self._P_e_pa(V, self.Tref)
        return P0 + Pth + Pe

    def _K_T_total_pa(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        K0V = self._K0_vinet_pa(V)
        Kth = self._K_Tth_pa(V, T) - self._K_Tth_pa(V, self.Tref)
        Ke = self._K_Te_pa(V, T) - self._K_Te_pa(V, self.Tref)
        return K0V + Kth + Ke

    def _Cv_total_jmolK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        return self._Cv_th_jmolK(V_m3mol, T_K) + self._Cv_e_jmolK(V_m3mol, T_K)

    def _dP_dT_V_pa_perK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        (∂P/∂T)_V = gamma/V * C_Vth + g/V * C_Ve  (eqs. 12 and 17)
        """
        V = np.asarray(V_m3mol, dtype=float)
        gamma = self._gamma(V)
        Cv_th = self._Cv_th_jmolK(V, T_K)
        Cv_e = self._Cv_e_jmolK(V, T_K)
        return (gamma * Cv_th + self.g_elec * Cv_e) / V

    def _alpha_perK(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        KT = self._K_T_total_pa(V_m3mol, T_K)
        dP_dT = self._dP_dT_V_pa_perK(V_m3mol, T_K)
        return dP_dT / KT

    def _U_molar(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        U(V,T) = U0 + E0(V) + [E_th(T)-E_th(Tref)] + [E_e(T)-E_e(Tref)]
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        U = (
            self.U0_Jmol
            + self._E0_potential_jmol(V)
            + (self._E_th_jmol(V, T) - self._E_th_jmol(V, self.Tref))
            + (self._E_e_jmol(V, T) - self._E_e_jmol(V, self.Tref))
        )
        return U

    def _S_molar(self, V_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        S(V,T) = [S_th(T)-S_th(Tref)] + [S_e(T)-S_e(Tref)]
        """
        V = np.asarray(V_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        S = (
            (self._S_th_jmolK(V, T) - self._S_th_jmolK(V, self.Tref))
            + (self._S_e_jmolK(V, T) - self._S_e_jmolK(V, self.Tref))
        )
        return S

    # -------------------------
    # Public API (rho,T) with unit conversion
    # -------------------------
    def get_p_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        P_pa = self._P_total_pa(V, T_K)
        return P_pa

    def get_u_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        U_molar = self._U_molar(V, T_K) + self._U_offset_molar
        u_jkg = U_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(u_jkg)

    def get_s_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        S_molar = self._S_molar(V, T_K) + self._S_offset_molar
        s_jkgK = S_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(s_jkgK)

    def get_cv_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        Cv_molar = self._Cv_total_jmolK(V, T_K)
        cv_jkgK = Cv_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(cv_jkgK)

    def get_kt_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        KT_pa = self._K_T_total_pa(V, T_K)
        return KT_pa

    def get_alpha_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        V = self._V_molar_from_rho(rho_kgm3)
        return self._alpha_perK(V, T_K)

    def get_cp_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Use exact relation: Cp - Cv = alpha^2 K_T T / rho
        """
        rho = np.asarray(rho_kgm3, dtype=float)
        T = np.asarray(T_K, dtype=float)
        Cv = self.get_cv_rhot(rho, T)  # already in chosen unit system
        alpha = self.get_alpha_rhot(rho, T)

        # work in SI internally for the add-on term
        V = self._V_molar_from_rho(rho)
        KT_pa = self._K_T_total_pa(V, T)

        # Cp - Cv in J/kg/K
        dcp_jkgK = (alpha**2) * KT_pa * T / rho
        # convert addon to cgs (erg/g/K) then add to Cv(cgs)
        dcp_cgs = self._jkg_to_ergg(dcp_jkgK)
        return np.asarray(Cv, dtype=float) + dcp_cgs

    def get_ks_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        K_S/K_T = C_P/C_V  (general thermodynamic identity)
        """
        Cv = self.get_cv_rhot(rho_kgm3, T_K)
        Cp = self.get_cp_rhot(rho_kgm3, T_K)
        KT = self.get_kt_rhot(rho_kgm3, T_K)
        ratio = np.asarray(Cp, dtype=float) / np.asarray(Cv, dtype=float)
        return np.asarray(KT, dtype=float) * ratio

    # -------------------------
    # Inversion rho(P,T)
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
        P_in = np.asarray(P, dtype=float)
        T_in = np.asarray(T_K, dtype=float)
        if np.any(T_in <= 0):
            raise ValueError("Temperature must be > 0 K")

        # Convert target pressure to Pa
        P_target = P_in if self.unit_system == "si" else self._barye_to_pa(P_in)

        shape = np.broadcast(P_target, T_in).shape
        P_b = np.broadcast_to(P_target, shape)
        T_b = np.broadcast_to(T_in, shape)

        out = np.full(shape, np.nan, dtype=float)

        if rho_bracket_kgm3 is None:
            rho_min, rho_max = 1000.0, 40000.0
        else:
            rho_min, rho_max = float(rho_bracket_kgm3[0]), float(rho_bracket_kgm3[1])

        if rho_min <= 0 or rho_max <= 0 or rho_min >= rho_max:
            raise ValueError("rho_bracket_kgm3 must be positive and increasing (rho_min < rho_max).")

        log_rho_lo, log_rho_hi = np.log(rho_min), np.log(rho_max)

        rho_V0 = float(self.M_molar_kg_per_mol / self.V0_m3mol)
        rho_seed_first = float(rho_guess0) if rho_guess0 is not None else rho_V0
        rho_seed_first = min(max(rho_seed_first, rho_min), rho_max)

        def P_pa_of_rho_scalar(rho, Ti):
            try:
                V = self._V_molar_from_rho(rho)
                val = self._P_total_pa(V, Ti)
                return float(np.asarray(val))
            except Exception:
                return np.nan

        rho_prev = None

        for idx in np.ndindex(shape):
            Pt = float(P_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(Pt)) or (not np.isfinite(Ti)) or Ti <= 0:
                continue

            rho_guess = rho_seed_first if (rho_prev is None or not np.isfinite(rho_prev)) else float(rho_prev)
            rho_guess = min(max(rho_guess, rho_min), rho_max)

            P_scale = max(abs(Pt), 1.0)

            def resid(logrho):
                rho = float(np.exp(logrho[0]))
                Pm = P_pa_of_rho_scalar(rho, Ti)
                if not np.isfinite(Pm):
                    return np.array([1e30], dtype=float)
                return np.array([(Pm - Pt) / P_scale], dtype=float)

            rho_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(rho_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([log_rho_lo], [log_rho_hi]),
                        xtol=rtol,
                        ftol=rtol,
                        gtol=rtol,
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

            if not np.isfinite(rho_sol):
                def f(rho):
                    Pm = P_pa_of_rho_scalar(rho, Ti)
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
                    for _ in range(bracket_expand_steps):
                        left = max(rho_min, left / bracket_expand_factor)
                        right = min(rho_max, right * bracket_expand_factor)
                        f_left = f(left)
                        f_right = f(right)

                        if np.isfinite(f_left) and np.isfinite(f_right):
                            if f_left == 0.0:
                                rho_sol = left
                                break
                            if f_right == 0.0:
                                rho_sol = right
                                break
                            if f_left * f_right < 0:
                                try:
                                    rho_sol = brenth(f, left, right, xtol=rtol, maxiter=max_iter)
                                except Exception:
                                    rho_sol = np.nan
                                break

                    if not np.isfinite(rho_sol):
                        fA, fB = f(rho_min), f(rho_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                rho_sol = brenth(f, rho_min, rho_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                rho_sol = np.nan

            if np.isfinite(rho_sol):
                out[idx] = rho_sol
                rho_prev = rho_sol
            else:
                if on_fail == "raise":
                    raise RuntimeError(
                        f"Failed rho(P,T) inversion: P={Pt:.3e} Pa, T={Ti:.3f} K "
                        f"within rho bracket [{rho_min}, {rho_max}] kg/m^3"
                    )

        return float(out) if out.size == 1 else out

    def get_s_pt_inv(self, P: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        rho = self.get_rho_pt_inv(P, T_K)
        return self.get_s_rhot(rho, T_K)
    
    def _as_float(self, x):
        # Handles python float, numpy scalar, 0-d array cleanly
        return float(np.asarray(x))
    # -------------------------
    # Inversion: T(S, rho)
    # -------------------------
    def get_T_srho_inv(
        self,
        _s,
        _rho,
        bracket=(1.0, 200000.0),
        xtol=1e-10,
        maxiter=200,
        s_units="kbbar",
        # warm-start / robustness knobs
        T_guess0=None,              # if None: use previous solution; else scalar
        use_lsq_first=True,
        lsq_max_nfev=80,
        bracket_expand_steps=30,
        bracket_expand_factor=1.6,
    ):
        """
        Invert s(rho,T) -> T with warm-starts across array inputs.

        Parameters
        ----------
        _s : target entropy.
            If s_units="kbbar": kB/baryon
            If s_units="cgs":   erg/g/K
            If s_units="si":    J/kg/K
        _rho : density [kg/m^3]
        bracket : (Tmin, Tmax) in K, Tmin MUST be > 0
        Returns
        -------
        T in K (np.nan where no solution is found)
        """
        s_arr = np.asarray(_s, dtype=float)
        rho_arr = np.asarray(_rho, dtype=float)
        s_arr, rho_arr = np.broadcast_arrays(s_arr, rho_arr)
        shape = s_arr.shape

        Tmin, Tmax = map(float, bracket)
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")

        # Convert target entropy to cgs (erg/g/K), since get_s_rhot returns cgs
        if str(s_units).lower() == "kbbar":
            s_target_cgs = s_arr / self.erg_to_kbbar
        else:
            s_target_cgs = s_arr

        T_out = np.full(shape, np.nan, dtype=float)

        def S_model(rho_val, T_val):
            return self._as_float(self.get_s_rhot(rho_val, T_val))

        logT_lo, logT_hi = np.log(Tmin), np.log(Tmax)

        T_prev = None
        if T_guess0 is None:
            # sane seed (no p.T0_K in this EOS); pick something moderate
            T_seed_first = float(getattr(self, "Tref", 3000.0))
            T_seed_first = max(T_seed_first, 300.0)
        else:
            T_seed_first = float(T_guess0)

        T_seed_first = min(max(T_seed_first, Tmin), Tmax)

        for idx in np.ndindex(shape):
            rho = float(rho_arr[idx])
            s_t = float(s_target_cgs[idx])

            if (not np.isfinite(rho)) or rho <= 0 or (not np.isfinite(s_t)):
                continue

            T_guess = T_seed_first if (T_prev is None or not np.isfinite(T_prev)) else float(T_prev)
            T_guess = min(max(T_guess, Tmin), Tmax)

            S_scale = max(abs(s_t), 1.0)

            def resid(logT):
                T = float(np.exp(logT[0]))
                Sm = S_model(rho, T)
                if not np.isfinite(Sm):
                    return np.array([1e30], dtype=float)
                return np.array([(Sm - s_t) / S_scale], dtype=float)

            T_sol = np.nan

            # 1) bounded least_squares in logT
            if use_lsq_first:
                x0 = np.array([np.log(T_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([logT_lo], [logT_hi]),
                        xtol=xtol,
                        ftol=xtol,
                        gtol=xtol,
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

            # 2) fallback brenth with expanding local bracket
            if not np.isfinite(T_sol):
                def f(T):
                    Sm = S_model(rho, T)
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
                    bracket_found = False
                    for _ in range(bracket_expand_steps):
                        left = max(Tmin, left / bracket_expand_factor)
                        right = min(Tmax, right * bracket_expand_factor)

                        f_left = f(left)
                        f_right = f(right)

                        if np.isfinite(f_left) and np.isfinite(f_right):
                            if f_left == 0.0:
                                T_sol = left
                                bracket_found = True
                                break
                            if f_right == 0.0:
                                T_sol = right
                                bracket_found = True
                                break
                            if f_left * f_right < 0:
                                try:
                                    T_sol = brenth(f, left, right, xtol=xtol, maxiter=maxiter)
                                except Exception:
                                    T_sol = np.nan
                                bracket_found = True
                                break

                    if not bracket_found:
                        fA, fB = f(Tmin), f(Tmax)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                T_sol = brenth(f, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
                            except Exception:
                                T_sol = np.nan

            if np.isfinite(T_sol):
                T_out[idx] = T_sol
                T_prev = T_sol

        return float(T_out) if T_out.size == 1 else T_out
    
    # -------------------------
    # Inversion: T(S, P)
    # -------------------------
    def get_T_sp_inv(
        self,
        _s,
        _P,
        bracket=(1.0, 20000.0),
        xtol=1e-8,
        maxiter=500,
        s_units="kbbar",
    ):
        """
        Invert s(P,T) -> T, where s(P,T) is computed via rho(P,T) inversion.

        _P must be in the EOS pressure units:
          - unit_system="si":  Pa
          - unit_system="cgs": dyn/cm^2

        _s can be:
          - "kbbar": kB/baryon
          - "cgs":   erg/g/K
          - "si":    J/kg/K
        """
        s_arr = np.asarray(_s, dtype=float)
        P_arr = np.asarray(_P, dtype=float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        if s_units.lower() == "kbbar":
            s_target_cgs = s_arr / self.erg_to_kbbar
        else:
            s_target_cgs = s_arr

        Tmin, Tmax = float(bracket[0]), float(bracket[1])
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")

        def _find_T(s_eos, P_val):
            if (not np.isfinite(P_val)) or P_val <= 0 or (not np.isfinite(s_eos)):
                return np.nan

            def err(T):
                return self._as_float(self.get_s_pt_inv(P_val, T)) - s_eos

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
            except Exception:
                return np.nan

        T_roots = np.vectorize(_find_T)(s_target_cgs, P_arr)
        return float(T_roots) if T_roots.size == 1 else T_roots

    # ----------------------------
    # 2-D Inversion: (rho, T) from (S, P)
    # ----------------------------
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
            Tseed = float(T_guess0) if T_guess0 is not None else max(float(getattr(self, "Tref", 3000.0)), 300.0)
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