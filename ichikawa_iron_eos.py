"""
ichikawa2014_liquid_fe_eos.py

Liquid iron P–V–T equation of state (EOS) following:
Ichikawa, Tsuchiya & Tange (2014), JGR Solid Earth:
"The P-V-T equation of state and thermodynamic properties of liquid iron"

Model (paper equations):
- Mie–Grüneisen form:
    P(V,T) = P_T0(V) + ΔP_th(V,T)
- Reference isotherm at T0 = 7000 K:
    P_T0(V) is Vinet EOS with parameters (V0, K_T0, K'_T0)
- Thermal energy (per mol):
    E_th(V,T) = 3 n R [ T + e0 (V/V0)^(-g) T^2 ]
- Thermal pressure:
    ΔP_th(V,T) = γ(V)/V * [E_th(V,T) - E_th(V,T0)]
- Grüneisen parameter:
    γ(V) = γ0 [ 1 + a ( (V/V0)^b - 1 ) ]
  (Table 2 sets a=1, so γ(V)=γ0 (V/V0)^b, but this implementation supports general a.)
- Isochoric heat capacity:
    C_V = ∂E_th/∂T|_V = 3 n R [ 1 + 2 e0 (V/V0)^(-g) T ]
- Conversion (paper text):
    (K_S/K_T) = (C_P/C_V) = 1 + α γ T

This class provides:
- P(ρ,T)
- specific internal energy u(ρ,T) and entropy s(ρ,T), referenced such that u(V0,T0)=0 and s(V0,T0)=0
- C_V, C_P, α, K_T, K_S
- inversion ρ(P,T) via bisection

Units:
- Inputs: ρ in g/cm^3, P in GPa, T in K
- Outputs:
    P in GPa, rho in g/cm^3
    s in erg/g/K, u in erg/g, Cp/Cv in erg/g/K, alpha in 1/K
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union, Optional

import numpy as np
from scipy.optimize import brenth, least_squares
from scipy.interpolate import RegularGridInterpolator as RGI
import astropy.units as u
from astropy.constants import k_B
from astropy.constants import u as amu


ArrayLike = Union[float, np.ndarray]
UnitSystem = Literal["cgs", "si"]


@dataclass(frozen=True)
class Ichikawa2014LiquidFeParams:
    """
    Parameter set from Table 2 in Ichikawa et al. (2014).
    Subscript 0 refers to 0 GPa and T0 = 7000 K.

    V0 is given per atom in Å^3/atom in the paper.
    KT0 is in GPa.
    e0 is listed in units of 10^-4 / K.
    """
    V0_A3_per_atom: float = 17.87       # Å^3/atom
    KT0_GPa: float = 24.6               # GPa
    Kprime_T0: float = 6.65             # dimensionless
    gamma0: float = 1.85                # dimensionless
    a_gamma: float = 1.0                # dimensionless (Table 2 fixes a=1)
    b_gamma: float = 0.35               # dimensionless
    e0_times1e4_perK: float = 0.314     # (10^-4)/K
    g_e: float = -0.4                   # dimensionless
    T0_K: float = 7000.0                # K


class Fe_EOS_Ichikawa2014_Liquid:
    """
    Liquid Fe EOS class following Ichikawa et al. (2014).

    Public API (ρ,T):
      - get_p_rhot(rho, T)
      - get_u_rhot(rho, T)      specific internal energy (relative; u(V0,T0)=0)
      - get_s_rhot(rho, T)      specific entropy (relative; s(V0,T0)=0)
      - get_CV_rhot(rho, T)
      - get_CP_rhot(rho, T)
      - get_alpha_rhot(rho, T)
      - get_kt_rhot(rho, T)
      - get_ks_rhot(rho, T)
      - get_rho_pt(P, T)        invert EOS
    """

    # Constants
    R_J_per_molK: float = 8.31446261815324
    N_A: float = 6.02214076e23
    M_molar_kg_per_mol: float = 55.845e-3  # Fe
    erg_to_kbbar: float = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
    dyn_to_Pa: float = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion
    L: float = 1.2e6 * (u.J/u.kg).to('erg/g')  # latent heat of fusion of Fe-Si alloy in erg/g (Anderson & Duba 1997)
    kb: float = k_B.to('erg/K') # ergs/K

    def __init__(
        self,
        params: Ichikawa2014LiquidFeParams = Ichikawa2014LiquidFeParams(),
        unit_system: UnitSystem = "si",
        n_atoms_formula_unit: int = 1,
        kt_deriv_eps: float = 1e-5,
        apply_s_u_offsets: bool = True,
        ref_state_rhoT: Optional[Tuple[float, float]] = (20000.0, 2000.0),  # (kg/m^3, K)
    ) -> None:
        self.p = params
        self.unit_system = unit_system
        self.n = float(n_atoms_formula_unit)
        self._eps = float(kt_deriv_eps)

        # Convert V0 to molar volume [m^3/mol]
        V0_m3_per_atom = self.p.V0_A3_per_atom * 1e-30
        self.V0_molar_m3_per_mol = V0_m3_per_atom * self.N_A

        # Convert parameters
        self.KT0_Pa = self.p.KT0_GPa * 1e9
        self.e0_perK = self.p.e0_times1e4_perK * 1e-4  # -> 1/K

        # Vinet helper coefficient a = 3/2 (K' - 1)
        self._a_vinet = 1.5 * (self.p.Kprime_T0 - 1.0)

        if apply_s_u_offsets and (ref_state_rhoT is not None):
            rho_ref, T_ref = ref_state_rhoT
            V_ref = self._V_molar_from_rho(rho_ref)
            # shift so that S(ref)=0 and U(ref)=0
            self._S_offset_molar = -float(self._s_molar_jmolK(V_ref, T_ref))
            self._U_offset_molar = -float(self._u_molar_jmol(V_ref, T_ref))

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
    
    @staticmethod
    def _as_array_single(x):
        return np.array(x, ndmin=1, dtype=float)

    # -------------------------
    # V, rho utilities
    # -------------------------
    def _V_molar_from_rho(self, rho_kgm3: ArrayLike) -> np.ndarray:
        rho = np.asarray(rho_kgm3, dtype=float)
        if np.any(rho <= 0):
            raise ValueError("Density rho must be > 0")
        return self.M_molar_kg_per_mol / rho  # m^3/mol

    def _rho_from_V_molar(self, V_molar_m3mol: ArrayLike) -> np.ndarray:
        V = np.asarray(V_molar_m3mol, dtype=float)
        if np.any(V <= 0):
            raise ValueError("Molar volume must be > 0")
        return self.M_molar_kg_per_mol / V

    def _y(self, V_molar_m3mol: ArrayLike) -> np.ndarray:
        return np.asarray(V_molar_m3mol, dtype=float) / self.V0_molar_m3_per_mol

    # -------------------------
    # EOS pieces (SI)
    # -------------------------
    def _p_T0_vinet_pa(self, V_molar_m3mol: ArrayLike) -> np.ndarray:
        """
        Reference isotherm P_T0(V) in Pa via Vinet EOS.
        """
        y = self._y(V_molar_m3mol)
        if np.any(y <= 0):
            raise ValueError("V/V0 must be > 0")

        x = np.power(y, 1.0 / 3.0)
        P = (
            3.0
            * self.KT0_Pa
            * np.power(y, -2.0 / 3.0)
            * (1.0 - x)
            * np.exp(self._a_vinet * (1.0 - x))
        )
        return P

    def _eth_molar_jmol(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Thermal internal energy E_th(V,T) in J/mol.
        """
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        y = self._y(V_molar_m3mol)
        return 3.0 * self.n * self.R_J_per_molK * (T + self.e0_perK * np.power(y, self.p.g_e) * T**2)

    def _cv_molar_jmolK(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Isochoric heat capacity C_V in J/mol/K.
        """
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")
        y = self._y(V_molar_m3mol)
        return 3.0 * self.n * self.R_J_per_molK * (1.0 + 2.0 * self.e0_perK * np.power(y, self.p.g_e) * T)

    def _gamma(self, V_molar_m3mol: ArrayLike) -> np.ndarray:
        """
        Grüneisen parameter γ(V).
        γ(V) = γ0 [ 1 + a ( (V/V0)^b - 1 ) ]
        """
        y = self._y(V_molar_m3mol)
        return self.p.gamma0 * (1.0 + self.p.a_gamma * (np.power(y, self.p.b_gamma) - 1.0))

    def _delta_p_th_pa(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Thermal pressure increment ΔP_th in Pa:
        ΔP_th = γ(V)/V * [E_th(V,T) - E_th(V,T0)].
        """
        V = np.asarray(V_molar_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        dEth = self._eth_molar_jmol(V, T) - self._eth_molar_jmol(V, self.p.T0_K)
        return self._gamma(V) / V * dEth  # Pa

    def _p_total_pa(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Total pressure P(V,T) in Pa.
        """
        return self._p_T0_vinet_pa(V_molar_m3mol) + self._delta_p_th_pa(V_molar_m3mol, T_K)

    # -------------------------
    # Thermodynamics (SI)
    # -------------------------
    def _dP_dT_v_pa_perK(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        (∂P/∂T)_V in Pa/K.

        From ΔP_th = γ/V * ΔE_th and ΔE_th depends only on E_th:
          ∂P/∂T|_V = γ/V * ∂E_th/∂T|_V = γ/V * C_V
        """
        V = np.asarray(V_molar_m3mol, dtype=float)
        return self._gamma(V) / V * self._cv_molar_jmolK(V, T_K)

    @staticmethod
    def _safe_log(x: np.ndarray) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError("log argument must be > 0")
        return np.log(x)

    def _s_molar_jmolK(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Entropy S(V,T) in J/mol/K, referenced such that S(V0,T0)=0.

        Constructed from exact differential:
            dS = (C_V/T)dT + (∂P/∂T)_V dV
        using path (V0,T0)->(V,T0)->(V,T).

        This is thermodynamically consistent with P(V,T), up to an arbitrary constant
        fixed by S(V0,T0)=0.

        Supports general a_gamma, though Table 2 uses a_gamma=1.
        """
        V = np.asarray(V_molar_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        if np.any(T <= 0):
            raise ValueError("Temperature must be > 0 K")

        y = self._y(V)
        T0 = self.p.T0_K
        b = self.p.b_gamma
        g = self.p.g_e
        a = self.p.a_gamma
        e0 = self.e0_perK

        ln_y = self._safe_log(y)

        # Volume integral at T0:
        # S(V,T0) = ∫_{V0}^{V} (∂P/∂T)_V'|_{T0} dV'
        # with (∂P/∂T)_V = (γ/V) C_V
        #
        # Using γ = γ0[(1-a)+a y^b] and C_V(T0)=3nR[1+2 e0 y^{-g} T0],
        # the integral reduces to:
        # 3 n R γ0 [ (1-a) ln y + a (y^b-1)/b
        #         + 2 e0 T0 * ( (1-a)*(y^{-g}-1)/(-g) + a*(y^{b-g}-1)/(b-g) ) ]
        def powm1_over(p: float, yy: np.ndarray) -> np.ndarray:
            # (y^p - 1)/p with stable p->0 limit => ln y
            if abs(p) < 1e-12:
                return ln_y
            return (np.power(yy, p) - 1.0) / p

        term_b = powm1_over(b, y)
        term_mg = powm1_over(g, y)  # (y^{-g}-1)/(-g)
        term_bg = powm1_over(b + g, y)

        S_V_T0 = 3.0 * self.n * self.R_J_per_molK * self.p.gamma0 * (
            (1.0 - a) * ln_y
            + a * term_b
            + 2.0 * e0 * T0 * ((1.0 - a) * term_mg + a * term_bg)
        )

        # Temperature integral at fixed V:
        # ∫_{T0}^{T} C_V(V,T')/T' dT' = 3 n R [ ln(T/T0) + 2 e0 y^{-g} (T - T0) ]
        dS_T = 3.0 * self.n * self.R_J_per_molK * (
            self._safe_log(T / T0) + 2.0 * e0 * np.power(y, g) * (T - T0)
        )

        return S_V_T0 + dS_T

    def _vinet_work_jmol(self, V_molar_m3mol: ArrayLike) -> np.ndarray:
        """
        Work integral along reference isotherm:
            W(V) = ∫_{V0}^{V} P_T0(V') dV'  [J/mol]

        Analytic integral of Vinet EOS gives:
            ∫_{1}^{y} P(y') dy' = 9 K0 / a^2 [ (a x - a + 1) exp(a(1-x)) - 1 ]
        where y=V/V0, x=y^(1/3), a=3/2(K'-1).
        Then W = V0_molar * ∫ P dy.
        """
        y = self._y(V_molar_m3mol)
        x = np.power(y, 1.0 / 3.0)
        a = self._a_vinet
        K0 = self.KT0_Pa
        integral_P_dy = 9.0 * K0 / (a**2) * ((a * x - a + 1.0) * np.exp(a * (1.0 - x)) - 1.0)  # Pa
        return self.V0_molar_m3_per_mol * integral_P_dy  # J/mol

    def _u_molar_jmol(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Internal energy U(V,T) in J/mol, referenced so U(V0,T0)=0.

        We use:
          U(V,T) = U(V,T0) + [E_th(V,T) - E_th(V,T0)]
        and define U(V,T0) from dU = T dS - P dV along T=T0:
          U(V,T0) = T0*S(V,T0) - ∫_{V0}^{V} P_T0(V') dV' = T0*S(V,T0) - W(V)
        """
        V = np.asarray(V_molar_m3mol, dtype=float)
        T = np.asarray(T_K, dtype=float)
        T0 = self.p.T0_K

        S_V_T0 = self._s_molar_jmolK(V, T0)
        U_V_T0 = T0 * S_V_T0 - self._vinet_work_jmol(V)

        dEth = self._eth_molar_jmol(V, T) - self._eth_molar_jmol(V, T0)
        return U_V_T0 + dEth

    # -------------------------
    # K_T and alpha (SI)
    # -------------------------
    def _kt_pa(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Isothermal bulk modulus K_T = -V (∂P/∂V)_T in Pa.

        Computed via centered finite difference in ln V:
            K_T = - dP / d(ln V)
        """
        V = np.asarray(V_molar_m3mol, dtype=float)
        eps = self._eps

        Vp = V * np.exp(eps)
        Vm = V * np.exp(-eps)
        Pp = self._p_total_pa(Vp, T_K)
        Pm = self._p_total_pa(Vm, T_K)
        dP_dlnV = (Pp - Pm) / (2.0 * eps)
        return -dP_dlnV

    def _alpha_perK(self, V_molar_m3mol: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """
        Thermal expansion coefficient:
            α = (1/V)(∂V/∂T)_P = (1/K_T) (∂P/∂T)_V
        """
        dP_dT = self._dP_dT_v_pa_perK(V_molar_m3mol, T_K)
        KT = self._kt_pa(V_molar_m3mol, T_K)
        return dP_dT / KT

    # -------------------------
    # Public API (rho,T)
    # -------------------------
    def get_p_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """P(rho, T). rho in g/cm^3, T in K. Returns P in GPa."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        P_pa = self._p_total_pa(V, T_K)
        return P_pa * 1e-9

    def get_u_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """u(rho, T). rho in g/cm^3, T in K. Returns u in erg/g."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        U_molar = self._u_molar_jmol(V, T_K) + self._U_offset_molar
        u_jkg = U_molar / self.M_molar_kg_per_mol + 1.5e7
        return self._jkg_to_ergg(u_jkg)

    def get_s_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """s(rho, T). rho in g/cm^3, T in K. Returns s in erg/g/K."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        S_molar = self._s_molar_jmolK(V, T_K) + self._S_offset_molar
        s_jkgK = S_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(s_jkgK)

    def get_CV_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """Cv(rho, T). rho in g/cm^3, T in K. Returns Cv in erg/g/K."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        Cv_molar = self._cv_molar_jmolK(V, T_K)
        cv_jkgK = Cv_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(cv_jkgK)

    def get_alpha_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """alpha(rho, T). rho in g/cm^3, T in K. Returns alpha in 1/K."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        return self._alpha_perK(V, T_K)

    def get_kt_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """KT(rho, T). rho in g/cm^3, T in K. Returns KT in GPa."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        KT_pa = self._kt_pa(V, T_K)
        return KT_pa * 1e-9

    def get_gamma_rhot(self, rho: ArrayLike) -> np.ndarray:
        """Gruneisen parameter. rho in g/cm^3."""
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        return self._gamma(V)

    def get_CP_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """Cp(rho, T). rho in g/cm^3, T in K. Returns Cp in erg/g/K.
        Uses C_P/C_V = 1 + alpha * gamma * T (paper text).
        """
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        T = np.asarray(T_K, dtype=float)

        Cv_molar = self._cv_molar_jmolK(V, T)
        alpha = self._alpha_perK(V, T)
        gamma = self._gamma(V)
        Cp_molar = Cv_molar * (1.0 + alpha * gamma * T)

        cp_jkgK = Cp_molar / self.M_molar_kg_per_mol
        return self._jkg_to_ergg(cp_jkgK)

    def get_ks_rhot(self, rho: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        """KS(rho, T). rho in g/cm^3, T in K. Returns KS in GPa.
        Uses K_S/K_T = 1 + alpha * gamma * T (paper text).
        """
        V = self._V_molar_from_rho(np.asarray(rho, dtype=float) * 1e3)
        T = np.asarray(T_K, dtype=float)

        KT = self._kt_pa(V, T)
        alpha = self._alpha_perK(V, T)
        gamma = self._gamma(V)

        KS_pa = KT * (1.0 + alpha * gamma * T)
        return KS_pa * 1e-9
    
    # -------------------------
    # Inversion rho(P,T)
    # -------------------------

    def get_rho_pt_inv(
        self,
        P: ArrayLike,
        T_K: ArrayLike,
        rho_bracket_gcm3: Optional[Tuple[float, float]] = None,
        max_iter: int = 200,
        rtol: float = 1e-10,
        # robustness / warm-start controls
        rho_guess0: Optional[float] = None,   # first guess in g/cm^3; if None use rho(V0)
        use_lsq_first: bool = True,
        lsq_max_nfev: int = 60,
        bracket_expand_steps: int = 30,
        bracket_expand_factor: float = 1.6,
        on_fail: str = "nan",  # "nan" or "raise"
    ) -> np.ndarray:
        """
        Invert the EOS for density at given (P,T).

        Parameters
        ----------
        P : pressure in GPa
        T_K : temperature [K]
        rho_bracket_gcm3 : optional (rho_lo, rho_hi) in g/cm^3.
        on_fail : "nan" or "raise"

        Returns rho in g/cm^3.
        """
        P_in = np.asarray(P, dtype=float)
        T_in = np.asarray(T_K, dtype=float)

        if np.any(T_in <= 0):
            raise ValueError("Temperature must be > 0 K")

        # Convert GPa target to Pa for internal inversion
        P_target_Pa = P_in * 1e9

        # Broadcast
        shape = np.broadcast(P_target_Pa, T_in).shape
        P_b = np.broadcast_to(P_target_Pa, shape)
        T_b = np.broadcast_to(T_in, shape)

        out = np.full(shape, np.nan, dtype=float)

        # Default bracket in kg/m^3 (internal SI)
        if rho_bracket_gcm3 is None:
            rho_min, rho_max = 1000.0, 40000.0  # kg/m^3
        else:
            rho_min, rho_max = float(rho_bracket_gcm3[0]) * 1e3, float(rho_bracket_gcm3[1]) * 1e3

        if rho_min <= 0 or rho_max <= 0 or rho_min >= rho_max:
            raise ValueError("rho_bracket_gcm3 must be positive and increasing (rho_min < rho_max).")

        log_rho_lo, log_rho_hi = np.log(rho_min), np.log(rho_max)

        # First-guess: density at V0 (reference), unless user supplies rho_guess0
        rho_V0 = float(self.M_molar_kg_per_mol / self.V0_molar_m3_per_mol)  # kg/m^3
        rho_seed_first = float(rho_guess0) * 1e3 if rho_guess0 is not None else rho_V0  # -> kg/m^3
        rho_seed_first = min(max(rho_seed_first, rho_min), rho_max)

        def P_pa_of_rho_scalar(rho_kgm3, Ti):
            # Safe scalar evaluation (rho in kg/m^3, returns Pa)
            try:
                V = self._V_molar_from_rho(rho_kgm3)
                val = self._p_total_pa(V, Ti)
                return float(np.asarray(val))
            except Exception:
                return np.nan

        # warm-start
        rho_prev = None

        for idx in np.ndindex(shape):
            Pt = float(P_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(Pt)) or (not np.isfinite(Ti)) or Ti <= 0:
                continue

            # Initial guess: previous solution if available, else seed
            if rho_prev is None or (not np.isfinite(rho_prev)):
                rho_guess = rho_seed_first
            else:
                rho_guess = float(rho_prev)
            rho_guess = min(max(rho_guess, rho_min), rho_max)

            # Residual (scaled) in log(rho)
            P_scale = max(abs(Pt), 1.0)

            def resid(logrho):
                rho = float(np.exp(logrho[0]))
                Pm = P_pa_of_rho_scalar(rho, Ti)
                if not np.isfinite(Pm):
                    return np.array([1e30], dtype=float)
                return np.array([(Pm - Pt) / P_scale], dtype=float)

            rho_sol = np.nan

            # 1) Bounded least_squares in log(rho): no bracketing required
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
                        # accept only if it's actually accurate
                        r = resid(np.array([np.log(rho_try)]))[0]
                        if np.isfinite(r) and abs(r) <= 1e-10:
                            rho_sol = rho_try
                except Exception:
                    pass

            # 2) Fallback: brenth with *local* bracket search around rho_guess
            if not np.isfinite(rho_sol):
                def f(rho):
                    Pm = P_pa_of_rho_scalar(rho, Ti)
                    if not np.isfinite(Pm):
                        return np.nan
                    return Pm - Pt

                # Try local bracket around rho_guess by expanding geometrically
                left = right = rho_guess
                f_left = f_right = f(rho_guess)

                # If f(rho_guess) is NaN, try a safer mid-point
                if not np.isfinite(f_left):
                    rho_guess = np.sqrt(rho_min * rho_max)
                    left = right = rho_guess
                    f_left = f_right = f(rho_guess)

                bracket_found = False
                if np.isfinite(f_left) and f_left == 0.0:
                    rho_sol = rho_guess
                    bracket_found = True

                if not bracket_found:
                    for _ in range(bracket_expand_steps):
                        left = max(rho_min, left / bracket_expand_factor)
                        right = min(rho_max, right * bracket_expand_factor)

                        f_left = f(left)
                        f_right = f(right)

                        if np.isfinite(f_left) and np.isfinite(f_right):
                            if f_left == 0.0:
                                rho_sol = left
                                bracket_found = True
                                break
                            if f_right == 0.0:
                                rho_sol = right
                                bracket_found = True
                                break
                            if f_left * f_right < 0:
                                try:
                                    rho_sol = brenth(f, left, right, xtol=rtol, maxiter=max_iter)
                                except Exception:
                                    rho_sol = np.nan
                                bracket_found = True
                                break

                    # Last resort: try full bracket once
                    if (not bracket_found):
                        fA, fB = f(rho_min), f(rho_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                rho_sol = brenth(f, rho_min, rho_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                rho_sol = np.nan

            if np.isfinite(rho_sol):
                out[idx] = rho_sol * 1e-3  # kg/m^3 -> g/cm^3
                rho_prev = rho_sol
            else:
                # keep rho_prev as warm-start if it exists; often best for smooth sequences
                if on_fail == "raise":
                    raise RuntimeError(
                        f"Failed to invert rho(P,T): P={Pt*1e-9:.3e} GPa, T={Ti:.3f} K "
                        f"within rho bracket [{rho_min*1e-3}, {rho_max*1e-3}] g/cm^3."
                    )

        return float(out) if out.size == 1 else out

    # -------------------------
    # Inversion S(P,T)
    # -------------------------

    def get_s_pt_inv(self,
        P,
        T,
    ):
        """
        Obtains S(P,T) via rho(P,T) inversion.
        P: GPa
        T: K
        Returns:
          S: erg/g/K
        """
        rho = self.get_rho_pt_inv(P, T)  # g/cm^3

        return self.get_s_rhot(rho, T) # output in cgs units

    def _as_float(self, x):
        # Handles python float, numpy scalar, 0-d array cleanly
        return float(np.asarray(x))

    # -------------------------
    # Inversion T(S,rho)
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
        T_guess0=None,              # if None: use previous solution; if first element, use T0_K
        use_lsq_first=True,
        lsq_max_nfev=80,
        bracket_expand_steps=30,    # for local bracket finding around guess
        bracket_expand_factor=1.6,  # geometric expansion
    ):
        """
        Invert s(rho,T) -> T with warm-starts across array inputs.

        Parameters
        ----------
        _s : target entropy.
            If s_units="cgs":   erg/g/K
            If s_units="kbbar": kB/baryon
        _rho : density [g/cm^3]
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
            raise ValueError("bracket[0] must be > 0 K (entropy uses log(T/T0)).")

        # Convert target entropy to cgs (erg/g/K), since get_s_rhot returns cgs
        if str(s_units).lower() == "kbbar":
            s_target_cgs = s_arr / self.erg_to_kbbar
        else:
            s_target_cgs = s_arr

        # Output
        T_out = np.full(shape, np.nan, dtype=float)

        # Helper: safe scalar S(rho,T)
        def S_cgs(rho_val, T_val):
            val = self.get_s_rhot(rho_val, T_val)
            # get_s_rhot might return array; force scalar float
            return float(np.asarray(val))

        # Helper: bounded least-squares solve in log(T)
        logT_lo, logT_hi = np.log(Tmin), np.log(Tmax)

        # Warm-start seed
        T_prev = None
        if T_guess0 is None:
            T_seed_first = float(getattr(self.p, "T0_K", 7000.0))
        else:
            T_seed_first = float(T_guess0)

        T_seed_first = min(max(T_seed_first, Tmin), Tmax)

        # Iterate in a deterministic order (warm-start works!)
        it = np.ndindex(shape)
        for idx in it:
            rho = float(rho_arr[idx])
            s_t = float(s_target_cgs[idx])

            if (not np.isfinite(rho)) or rho <= 0 or (not np.isfinite(s_t)):
                continue

            # Choose initial guess
            if T_prev is None or (not np.isfinite(T_prev)):
                T_guess = T_seed_first
            else:
                T_guess = float(T_prev)
            T_guess = min(max(T_guess, Tmin), Tmax)

            # Define residual in a scaled way (like your 2-D inversion)
            S_scale = max(abs(s_t), 1.0)

            def resid(logT):
                T = float(np.exp(logT[0]))
                Sm = S_cgs(rho, T)
                if not np.isfinite(Sm):
                    return np.array([1e30], dtype=float)
                return np.array([(Sm - s_t) / S_scale], dtype=float)

            # --------
            # 1) Try bounded least_squares in logT (fast, doesn’t need brackets)
            # --------
            T_sol = np.nan
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
                        # Accept if it actually hits the target reasonably well
                        r = resid(np.array([np.log(T_try)]))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            T_sol = T_try
                except Exception:
                    pass

            # --------
            # 2) Fallback: brenth with a *local bracket search* around the guess
            # --------
            if not np.isfinite(T_sol):
                def f(T):
                    Sm = S_cgs(rho, T)
                    if not np.isfinite(Sm):
                        return np.nan
                    return Sm - s_t

                # Try to find a bracket near T_guess by expanding outward geometrically
                T_lo = T_hi = T_guess
                f_lo = f_hi = f(T_guess)

                # If f(T_guess) is NaN, nudge to something safer (midpoint)
                if not np.isfinite(f_lo):
                    T_guess = np.sqrt(Tmin * Tmax)
                    T_lo = T_hi = T_guess
                    f_lo = f_hi = f(T_guess)

                bracket_found = False
                if np.isfinite(f_lo) and f_lo == 0.0:
                    T_sol = T_guess
                    bracket_found = True

                if not bracket_found:
                    left = T_guess
                    right = T_guess
                    f_left = f_lo
                    f_right = f_hi

                    for _ in range(bracket_expand_steps):
                        # Expand
                        left = max(Tmin, left / bracket_expand_factor)
                        right = min(Tmax, right * bracket_expand_factor)

                        f_left = f(left)
                        f_right = f(right)

                        if np.isfinite(f_left) and np.isfinite(f_right) and (f_left == 0.0 or f_right == 0.0 or f_left * f_right < 0):
                            # We have a bracket!
                            a, b = left, right
                            fa, fb = f_left, f_right
                            if fa == 0.0:
                                T_sol = a
                            elif fb == 0.0:
                                T_sol = b
                            else:
                                try:
                                    T_sol = brenth(f, a, b, xtol=xtol, maxiter=maxiter)
                                except Exception:
                                    T_sol = np.nan
                            bracket_found = True
                            break

                    # If still not bracketed, try the full bracket as a last resort
                    if (not bracket_found):
                        fA, fB = f(Tmin), f(Tmax)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                T_sol = brenth(f, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
                            except Exception:
                                T_sol = np.nan

            # Store, update warm-start
            if np.isfinite(T_sol):
                T_out[idx] = T_sol
                T_prev = T_sol
            # else: keep previous solution as the guess for the next point (usually best for smooth arrays)

        return float(T_out) if T_out.size == 1 else T_out

    # -------------------------
    # Inversion T(S,P)
    # -------------------------

    def get_T_sp_inv(self, _s, _P, bracket=(1.0, 20000.0), xtol=1e-8, maxiter=500, s_units="kbbar"):
        """
        Invert s(P,T) -> T, where s(P,T) is computed via rho(P,T) inversion.

        _P in GPa
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

    # ----------------------------
    # 2-D Inversion Rho, T -> S, P
    # ----------------------------
    def get_rhot_sp_2d_inv(
        self,
        s_target,
        P_target,
        *,
        s_units="kbbar",                 # "cgs" (erg/g/K) or "kbbar" (kB/baryon)
        guess="auto",                  # "auto" or (rho_guess, T_guess)
        T_guess0=None,                 # used only if guess="auto"
        bounds_rho=(1.0, 1e3),         # g/cm^3
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
            P(rho, T) = P_target   (GPa)
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
            Tseed = float(self.p.T0_K if T_guess0 is None else T_guess0)
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
            P_scale = max(abs(Pt), 1.0)        # GPa
            S_scale = max(abs(St), 1.0)        # erg/g/K

            def residuals(x):
                # log-variables => always positive
                rho = float(np.exp(x[0]))
                T   = float(np.exp(x[1]))

                # Evaluate model
                Pm = float(self.get_p_rhot(rho, T))      # GPa
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

class Fe_EOS(Fe_EOS_Ichikawa2014_Liquid):
    """
    Fe EOS with optional PT tables (rho, s, u) loaded from an NPZ.

    Behavior:
      - tab=True  -> PT-table interpolation
      - tab=False -> analytic inversion via Fe_EOS.get_rho_pt_inv + analytic getters

    Table expectations (default keys in NPZ files):
      pvals_pt     [Pa in file -> converted to GPa]
      tvals_pt     [K]
      rho_grid_pt  [kg/m^3 in file -> converted to g/cm^3]
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

        # P, T tables

        data_pt = np.load('eos/ichikawa_iron_eos/iron_eos_PT_liquid.npz')

        self.pvals_pt = np.asarray(data_pt['pvals_pt'], dtype=float) * 1e-9  # Pa -> GPa
        self.tvals_pt = np.asarray(data_pt['tvals_pt'], dtype=float)  # K

        self.rho_grid_pt = np.asarray(data_pt['rho_grid_pt'], dtype=float) * 1e-3  # kg/m^3 -> g/cm^3
        self.s_grid_pt = np.asarray(data_pt['s_grid_pt'], dtype=float)  # erg/g/K
        self.u_grid_pt = np.asarray(data_pt['u_grid_pt'], dtype=float)  # erg/g
        self.cp_grid_pt = np.asarray(data_pt['cp_grid_pt'], dtype=float)  # erg/g/K
        self.cv_grid_pt = np.asarray(data_pt['cv_grid_pt'], dtype=float)  # erg/g/K
        self.alpha_grid_pt = np.asarray(data_pt['alpha_grid_pt'], dtype=float)  # 1/K

        # S, P tables

        data_sp = np.load('eos/ichikawa_iron_eos/iron_eos_SP_liquid.npz')

        self.svals_sp = np.asarray(data_sp['svals_sp'], dtype=float)  # kb/baryon
        self.pvals_sp = np.asarray(data_sp['pvals_sp'], dtype=float) * 1e-9  # Pa -> GPa

        self.rho_grid_sp = np.asarray(data_sp['rho_grid_sp'], dtype=float) * 1e-3  # kg/m^3 -> g/cm^3
        self.t_grid_sp = np.asarray(data_sp['t_grid_sp'], dtype=float)  # K
        self.u_grid_sp = np.asarray(data_sp['u_grid_sp'], dtype=float)  # erg/g
        self.cp_grid_sp = np.asarray(data_sp['cp_grid_sp'], dtype=float)  # erg/g/K
        self.cv_grid_sp = np.asarray(data_sp['cv_grid_sp'], dtype=float)  # erg/g/K
        self.alpha_grid_sp = np.asarray(data_sp['alpha_grid_sp'], dtype=float)  # 1/K

        # S, Rho tables

        data_srho = np.load('eos/ichikawa_iron_eos/iron_eos_SRho_liquid.npz')

        self.svals_srho = np.asarray(data_srho['svals_srho'], dtype=float)  # kb/baryon
        self.rho_vals_srho = np.asarray(data_srho['rhovals_srho'], dtype=float) * 1e-3  # kg/m^3 -> g/cm^3

        self.t_grid_srho = np.asarray(data_srho['t_grid_srho'], dtype=float)  # K
        self.p_grid_srho = np.asarray(data_srho['p_grid_srho'], dtype=float) * 1e-9  # Pa -> GPa
        self.u_grid_srho = np.asarray(data_srho['u_grid_srho'], dtype=float)  # erg/g
        self.cp_grid_srho = np.asarray(data_srho['cp_grid_srho'], dtype=float)  # erg/g/K
        self.cv_grid_srho = np.asarray(data_srho['cv_grid_srho'], dtype=float)  # erg/g/K
        self.alpha_grid_srho = np.asarray(data_srho['alpha_grid_srho'], dtype=float)  # 1/K


        # Interpolators: input points are (P, T)
        self.rho_rgi_pt = RGI((self.pvals_pt, self.tvals_pt), self.rho_grid_pt, **rgi_kwargs)
        self.s_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.s_grid_pt, **rgi_kwargs)
        self.u_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.u_grid_pt, **rgi_kwargs)
        self.cp_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.cp_grid_pt, **rgi_kwargs)
        self.cv_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.cv_grid_pt, **rgi_kwargs)
        self.alpha_rgi_pt   = RGI((self.pvals_pt, self.tvals_pt), self.alpha_grid_pt, **rgi_kwargs)

        # Interpolators: input points are (S, P)
        self.rho_rgi_sp = RGI((self.svals_sp, self.pvals_sp), self.rho_grid_sp, **rgi_kwargs)
        self.t_rgi_sp = RGI((self.svals_sp, self.pvals_sp), self.t_grid_sp, **rgi_kwargs)
        self.u_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.u_grid_sp, **rgi_kwargs)
        self.cp_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.cp_grid_sp, **rgi_kwargs)
        self.cv_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.cv_grid_sp, **rgi_kwargs)
        self.alpha_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.alpha_grid_sp, **rgi_kwargs)

        # Interpolators: input points are (S, Rho)
        self.t_rgi_srho = RGI((self.svals_srho, self.rho_vals_srho), self.t_grid_srho, **rgi_kwargs)
        self.p_rgi_srho = RGI((self.svals_srho, self.rho_vals_srho), self.p_grid_srho, **rgi_kwargs)
        self.u_rgi_srho   = RGI((self.svals_srho, self.rho_vals_srho), self.u_grid_srho, **rgi_kwargs)
        self.cp_rgi_srho   = RGI((self.svals_srho, self.rho_vals_srho), self.cp_grid_srho, **rgi_kwargs)
        self.cv_rgi_srho   = RGI((self.svals_srho, self.rho_vals_srho), self.cv_grid_srho, **rgi_kwargs)
        self.alpha_rgi_srho   = RGI((self.svals_srho, self.rho_vals_srho), self.alpha_grid_srho, **rgi_kwargs)

    @staticmethod
    def _broadcast(P, Q):
        "Generalized for coordinates P and Q. Could be P, T; S, P; etc."
        scalar = np.isscalar(P) and np.isscalar(Q)
        P_arr = np.array(P, ndmin=1, dtype=float)
        Q_arr = np.array(Q, ndmin=1, dtype=float)
        if P_arr.shape != Q_arr.shape:
            P_arr, Q_arr = np.broadcast_arrays(P_arr, Q_arr)
        return scalar, P_arr, Q_arr

    def _interp(self, rgi, P_arr, Q_arr):
        "Generalized for coordinates P and Q. Could be P, T; S, P; etc."
        pts = np.stack((P_arr.ravel(), Q_arr.ravel()), axis=-1)
        return rgi(pts).reshape(P_arr.shape)

    # -------------------------
    # PT table API (override)
    # -------------------------
    def get_rho_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        rho(P,T) in g/cm^3.  P in GPa, T in K.
        If tab=False, uses Fe_EOS.get_rho_pt_inv.
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.rho_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        # If you have tables, using them as an initial guess can speed up inversion
        if rho0 is None:
            try:
                rho0 = self._interp(self.rho_rgi_pt, P_arr, T_arr)
            except Exception:
                rho0 = None

        vals = self.get_rho_pt_inv(P_arr, T_arr, rho0=rho0, **inv_kwargs)
        return float(vals) if scalar else vals

    def get_s_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        s(P,T) in erg/g/K.
        If tab=False, inverts for rho then evaluates analytic s(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.s_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_s_rhot(rho, T_arr)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_u_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        u(P,T) in erg/g.
        If tab=False, inverts for rho then evaluates analytic u(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.u_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_u_rhot(rho, T_arr)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CP_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        u(P,T) in erg/g.
        If tab=False, inverts for rho then evaluates analytic u(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.cp_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_CP_rhot(rho, T_arr)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CV_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        u(P,T) in erg/g.
        If tab=False, inverts for rho then evaluates analytic u(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.cv_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_CV_rhot(rho, T_arr)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_alpha_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        """
        u(P,T) in erg/g.
        If tab=False, inverts for rho then evaluates analytic u(rho,T).
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        if tab:
            vals = self._interp(self.alpha_rgi_pt, P_arr, T_arr)
            return float(vals) if scalar else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False, rho0=rho0, **inv_kwargs)
        vals = self.get_alpha_rhot(rho, T_arr)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    # -------------------------
    # SP table API (override)
    # -------------------------

    def get_rho_sp(self, S, P, tab=True, **inv_kwargs):
        """
        rho(S,P) in g/cm^3.  P in GPa.
        If tab=False, uses Fe_EOS.get_rho_sp_inv.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.rho_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_T_sp(self, S, P, tab=True, **inv_kwargs):
        """
        T(S,P) in K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.t_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(T) if scalar else T
    
    def get_u_sp(self, S, P, tab=True, **inv_kwargs):
        """
        u(S,P) in erg/g.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.u_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_rhot(rho, T)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CP_sp(self, S, P, tab=True, **inv_kwargs):
        """
        cp(S,P) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.cp_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CP_rhot(rho, T)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CV_sp(self, S, P, tab=True, **inv_kwargs):
        """
        cv(S,P) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.cv_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CV_rhot(rho, T)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals
    
    def get_alpha_sp(self, S, P, tab=True, **inv_kwargs):
        """
        alpha(S,P) in 1/K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.alpha_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_rhot(rho, T)  # 1/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    # -------------------------
    # SRho table API (override)
    # -------------------------

    def get_T_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        T(S,rho) in K.
        If tab=False, uses Fe_EOS.get_rho_srho_inv.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.t_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        return float(T) if scalar else T
    
    def get_p_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        p(S,rho) in GPa.  rho in g/cm^3.
        If tab=False, uses Fe_EOS.get_rho_srho_inv.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.p_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        p = self.get_p_rhot(rho_arr, T)  # Pa
        return float(p) if scalar else p

    def get_u_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        u(S,rho) in erg/g.
        If tab=False, uses Fe_EOS.get_rho_srho_inv then analytic getter.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.u_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_u_rhot(rho_arr, T)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CP_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        cp(S,rho) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_srho_inv then analytic getter.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.cp_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_CP_rhot(rho_arr, T)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_CV_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        cv(S,rho) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_srho_inv then analytic getter.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.cv_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_CV_rhot(rho_arr, T)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_alpha_srho(self, S, rho, tab=True, **inv_kwargs):
        """
        alpha(S,rho) in 1/K.
        If tab=False, uses Fe_EOS.get_rho_srho_inv then analytic getter.
        """
        scalar, S_arr, rho_arr = self._broadcast(S, rho)

        if tab:
            vals = self._interp(self.alpha_rgi_srho, S_arr, rho_arr)
            return float(vals) if scalar else vals

        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_alpha_rhot(rho_arr, T)  # 1/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_T_melt(self, P):
        """
        Zhang et al. (2015) melt curve used in ichikawa_iron_eos.
        P in GPa. Return T_melt(P) in K.
        """
        P_arr = self._as_array_single(P)
        Tm = 1900 * (P_arr / 31.3 + 1) ** (1/1.99) # Zhang et al. 2015
        return Tm.reshape(P_arr.shape)