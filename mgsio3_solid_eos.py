# Mie–Grüneisen–Debye P–V–T EOS with thermoelastic derivatives, entropy, and rho(P,T) inversion
# External I/O units:
#   Inputs:  P in GPa,  rho in g/cm^3,  T in K
#   Outputs: P and K_T in GPa; rho in g/cm^3; U in erg/g; S, C_V, C_P in erg/(g·K); alpha in 1/K; gamma unitless
# Internal engine uses SI (Pa, kg/m^3, J/mol, etc.). Facade handles all unit conversions without mutating caller arrays.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Dict
from scipy.optimize import newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import numpy as np

ArrayLike = Union[float, int, np.ndarray]

# constants
R = 8.31446261815324          # J/(mol K)
N_A = 6.02214076e23
A3_TO_M3 = 1e-30
GPa = 1e9

# -------- Debye helpers --------
try:
    from scipy.special import debye as _scipy_debye  # D_n(x)
except Exception:
    _scipy_debye = None


def _gauss8_integral(f, xmax: np.ndarray) -> np.ndarray:
    """8-pt Gauss–Legendre quadrature on [0, xmax], split into ~5-sized segments."""
    xmax = np.asarray(xmax, dtype=float)
    out = np.zeros_like(xmax)

    nodes = np.array([0.1834346424956498, 0.5255324099163290,
                      0.7966664774136267, 0.9602898564975363])
    weights = np.array([0.3626837833783620, 0.3137066458778873,
                        0.2223810344533745, 0.1012285362903763])

    it = np.nditer(xmax, flags=['multi_index'])
    for x in it:
        x = float(x)
        if x <= 0.0:
            val = 0.0
        else:
            nseg = max(1, int(np.ceil(x/5.0)))
            s = 0.0
            for k in range(nseg):
                a = (k    )*x/nseg
                b = (k + 1)*x/nseg
                c = 0.5*(a+b)
                h = 0.5*(b-a)
                t1 = c + h*nodes
                t2 = c - h*nodes
                s += h * np.sum(weights*(f(t1) + f(t2)))
            val = s
        out[it.multi_index] = val
    return out

def _debye_integral(n: int, xmax: np.ndarray) -> np.ndarray:
    """I_n(xmax) = ∫_0^{xmax} t^n/(e^t-1) dt."""
    return _gauss8_integral(lambda t: (t**n)/(np.exp(t) - 1.0 + 1e-300), xmax)


def _debye_integral_cv(xmax: np.ndarray) -> np.ndarray:
    """I_cv(x) = ∫_0^{x} t^4 e^t/(e^t-1)^2 dt."""
    return _gauss8_integral(lambda t: (t**4)*np.exp(t)/(np.expm1(t)**2 + 1e-300), xmax)


def debye_D2(x):
    x = np.asarray(x, float)
    if _scipy_debye is not None:
        return _scipy_debye(2, x)
    # series + integral fallback
    out = np.empty_like(x)
    small = x < 0.1
    large = x > 50.0
    mid = ~(small | large)
    xs = x[small]
    out[small] = 1.0 - xs/4.0 + xs**2/36.0 - xs**4/3600.0
    xl = x[large]
    out[large] = 4.0*1.202056903159594/(xl**2)  # 4*ζ(3)/x^2
    if np.any(mid):
        xm = x[mid]
        I = _debye_integral(2, xm)
        out[mid] = 2.0*I/(xm**2)
    return out


def debye_D3(x):
    x = np.asarray(x, float)
    if _scipy_debye is not None:
        return _scipy_debye(3, x)
    out = np.empty_like(x)
    small = x < 0.1
    large = x > 50.0
    mid = ~(small | large)
    xs = x[small]
    out[small] = 1.0 - 3.0*xs/8.0 + xs**2/20.0 - xs**4/1680.0
    xl = x[large]
    out[large] = (np.pi**4)/(5.0*xl**3)
    if np.any(mid):
        xm = x[mid]
        I = _debye_integral(3, xm)
        out[mid] = 3.0*I/(xm**3)
    return out


def dD3_dx(x):
    """D3'(x) = 3/x * (D3 - D2). Limit at x→0 handled."""
    x = np.asarray(x, float)
    D3 = debye_D3(x)
    D2 = debye_D2(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = 3.0/x * (D3 - D2)
        d = np.where(x == 0.0, -3.0/8.0, d)
    return d


def debye_integral_3(x):
    """∫_0^x t^3/(e^t-1) dt."""
    return _debye_integral(3, x)


def debye_integral_1(x):
    """∫_0^x t^3/(e^t-1) dt. Used in S(V,T)."""
    return _debye_integral(3, x)


# -------- Core EOS param container --------
@dataclass(frozen=True)
class MGDParams:
    V0: float           # m^3/mol
    K_T0: float         # Pa
    K_T0p: float        # dimensionless
    theta0: float       # K
    gamma0: float       # –
    a: float            # –
    b: float            # –
    n: float            # atoms per formula unit
    M_molar: float      # kg/mol
    T0: float = 300.0   # K
    cold_curve: str = "bm3"  # 'bm3' or 'vinet'


# -------- Engine EOS (single parameter set, SI; no phase logic) --------
class MGD_EOS:
    def __init__(self, p: MGDParams):
        self.p = p
        # Unit conversions for mass-specific CGS outputs
        self.U_conv_cgs = (1.0 * u.J/u.kg).to(u.erg/u.g).value          # 1 J/kg -> erg/g
        self.S_conv_cgs = (1.0 * u.J/u.kg/u.K).to(u.erg/u.g/u.K).value  # 1 J/kg/K -> erg/g/K
        self.erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
        self.dyn_to_Pa = (u.dyn/u.cm**2).to('Pa')
        self.dyn_to_GPa = (u.dyn/u.cm**2).to('GPa')

    @staticmethod
    def V0_molar_from_A3_per_cell(V0_A3_per_cell: float, z_fu_per_cell: int) -> float:
        return (V0_A3_per_cell / z_fu_per_cell) * A3_TO_M3 * N_A

    # volume dependences
    def gamma_of_V(self, V):
        p = self.p
        x = np.asarray(V, float)/p.V0
        return p.gamma0 * (1.0 + p.a*(x**p.b - 1.0))

    def dgamma_dV(self, V):
        p = self.p
        x = np.asarray(V, float)/p.V0
        return p.gamma0 * p.a * p.b * x**(p.b - 1.0) / p.V0

    def thetaD_of_V(self, V):
        p = self.p
        x = np.asarray(V, float)/p.V0
        g = self.gamma_of_V(V)
        return p.theta0 * x**(-p.gamma0*(1.0 - p.a)) * np.exp(-(g - p.gamma0)/p.b)

    def dtheta_dV(self, V):
        V = np.asarray(V, float)
        return - self.gamma_of_V(V) * self.thetaD_of_V(V) / V  # d ln θ / d ln V = -γ

    # cold-curve P(V, T0)
    def P_T0_of_V(self, V):
        p = self.p
        x = np.asarray(V, float)/p.V0
        if p.cold_curve.lower() == "bm3":
            f = x**(-7/3) - x**(-5/3)
            g = 1.0 + 0.75*(p.K_T0p - 4.0)*(x**(-2/3) - 1.0)
            return 1.5 * p.K_T0 * f * g
        elif p.cold_curve.lower() == "vinet":
            eta = x**(1/3)
            return 3.0*p.K_T0 * x**(-2/3) * (1.0 - eta) * np.exp(1.5*(p.K_T0p - 1.0)*(1.0 - eta))
        else:
            raise ValueError("cold_curve must be 'bm3' or 'vinet'.")

    def dP_T0_dV(self, V):
        p = self.p
        x = np.asarray(V, float)/p.V0
        if p.cold_curve.lower() == "bm3":
            f  = x**(-7/3) - x**(-5/3)
            g  = 1.0 + 0.75*(p.K_T0p - 4.0)*(x**(-2/3) - 1.0)
            df = (-7/3)*x**(-10/3) + (5/3)*x**(-8/3)
            dg = -0.5*(p.K_T0p - 4.0)*x**(-5/3)
            dPdx = 1.5 * p.K_T0 * (df*g + f*dg)
            return dPdx / p.V0
        else:  # vinet
            eta = x**(1/3)
            h = x**(-2/3)
            A = 1.5*(p.K_T0p - 1.0)
            E = np.exp(A*(1.0 - eta))
            dhdx = (-2/3)*x**(-5/3)
            detadx = (1/3)*x**(-2/3)
            dEdx = -A*E*detadx
            dPdx = 3.0*p.K_T0*( dhdx*(1-eta)*E + h*(-detadx)*E + h*(1-eta)*dEdx )
            return dPdx / p.V0

    # thermal energy U(V,T) (per mole)
    def U_of_V_T(self, V, T):
        """
        U = 9 n R T (T/θ)^3 ∫_0^{θ/T} x^3/(e^x-1) dx.
        """
        V = np.asarray(V, float)
        T = np.asarray(T, float)
        theta = self.thetaD_of_V(V)
        y = theta / np.maximum(T, 1e-12)
        I3 = debye_integral_3(y)
        return 9.0 * self.p.n * R * T * (T/np.maximum(theta, 1e-12))**3 * I3

    def dU_dV_T(self, V, T):
        # chain rule with y = θ/T, I3'(y) = y^3/(e^y-1)
        V = np.asarray(V, float)
        T = np.asarray(T, float)
        theta = self.thetaD_of_V(V)
        y = theta / np.maximum(T, 1e-12)
        I3 = debye_integral_3(y)
        I3prime = y**3/(np.exp(y) - 1.0 + 1e-300)
        dtheta = self.dtheta_dV(V)
        term1 = -3.0 * (T/theta)**3 * (dtheta/theta) * I3
        term2 = (T/theta)**3 * I3prime * (dtheta/np.maximum(T, 1e-12))
        return 9.0 * self.p.n * R * T * (term1 + term2)

    # thermal pressure and derivatives
    def dPth_of_V_T(self, V, T):
        V = np.asarray(V, float)
        g = self.gamma_of_V(V)
        U_T  = self.U_of_V_T(V, T)
        U_T0 = self.U_of_V_T(V, self.p.T0)
        return (g/V) * (U_T - U_T0)

    def dPth_dV_T(self, V, T):
        V = np.asarray(V, float)
        g = self.gamma_of_V(V)
        dg_dV = self.dgamma_dV(V)
        U_T   = self.U_of_V_T(V, T)
        U_T0  = self.U_of_V_T(V, self.p.T0)
        dU_dV_T   = self.dU_dV_T(V, T)
        dU_dV_T0  = self.dU_dV_T(V, self.p.T0)
        return (dg_dV/V - g/V**2)*(U_T - U_T0) + (g/V)*(dU_dV_T - dU_dV_T0)

    # P(V,T)
    def P_of_V_T(self, V, T):
        return self.P_T0_of_V(V) + self.dPth_of_V_T(V, T)

    # thermoelastic: K_T, alpha, C_V, C_P
    def K_T_of_V_T(self, V, T):
        dP_dV_T = self.dP_T0_dV(V) + self.dPth_dV_T(V, T)
        return - np.asarray(V, float) * dP_dV_T  # Pa

    def C_V_of_V_T(self, V, T):
        V = np.asarray(V, float)
        T = np.asarray(T, float)
        theta = self.thetaD_of_V(V)
        y = theta/np.maximum(T, 1e-12)
        Icv = _debye_integral_cv(y)
        return 9.0 * self.p.n * R * (np.maximum(theta, 1e-12)/np.maximum(T, 1e-12))**(-3) * Icv  # J/mol/K

    def dP_dT_V(self, V, T):
        V = np.asarray(V, float)
        return (self.gamma_of_V(V)/V) * self.C_V_of_V_T(V, T)

    def alpha_of_V_T(self, V, T):
        dP_dT = self.dP_dT_V(V, T)
        K_T = self.K_T_of_V_T(V, T)
        return dP_dT / K_T

    def C_P_of_V_T(self, V, T):
        alpha = self.alpha_of_V_T(V, T)
        gamma = self.gamma_of_V(V)
        C_V = self.C_V_of_V_T(V, T)
        return C_V * (1.0 + alpha*gamma*T)  # J/mol/K

    # entropy S(V,T) per mole
    def S_of_V_T(self, V, T):
        """
        S = 9 n R (T/θ)^3 I1(θ/T) - 3 n R ln(1 - exp(-θ/T)).
        """
        V = np.asarray(V, float)
        T = np.asarray(T, float)
        theta = self.thetaD_of_V(V)
        y = theta/np.maximum(T, 1e-12)
        I1 = debye_integral_1(y)
        term1 = 9.0 * self.p.n * R * (T/np.maximum(theta, 1e-12))**3 * I1
        term2 = -3.0 * self.p.n * R * np.log1p(-np.exp(-y))
        return term1 + term2  # J/mol/K

    # ---------- helpers ----------
    def _as_arrays(self, *vals):
        arrs = [np.asarray(v, dtype=float) for v in vals]
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
        return [np.broadcast_to(a, shape) for a in arrs]

    def _V_from_rho(self, rho):
        return self.p.M_molar / np.asarray(rho, float)

    # ---------- "get_" API in SI (used by facade) ----------
    def get_p_rhot(self, rho, T):
        return self.P_of_V_T(self._V_from_rho(rho), T)  # Pa

    def get_u_rhot(self, rho, T):
        # U(V,T): J/mol -> J/kg -> erg/g
        return self.U_of_V_T(self._V_from_rho(rho), T) / self.p.M_molar * self.U_conv_cgs

    def get_KT_rhot(self, rho, T):
        return self.K_T_of_V_T(self._V_from_rho(rho), T)  # Pa

    def get_alpha_rhot(self, rho, T):
        return self.alpha_of_V_T(self._V_from_rho(rho), T)  # 1/K

    def get_CV_rhot(self, rho, T):
        # J/mol/K -> J/kg/K -> erg/g/K
        return self.C_V_of_V_T(self._V_from_rho(rho), T) / self.p.M_molar * self.S_conv_cgs

    def get_CP_rhot(self, rho, T):
        # J/mol/K -> J/kg/K -> erg/g/K
        return self.C_P_of_V_T(self._V_from_rho(rho), T) / self.p.M_molar * self.S_conv_cgs

    def get_gamma_rhot(self, rho):
        return self.gamma_of_V(self._V_from_rho(rho))  # unitless

    def get_s_rhot(self, rho, T):
        # J/mol/K -> J/kg/K -> erg/g/K
        return self.S_of_V_T(self._V_from_rho(rho), T) / self.p.M_molar * self.S_conv_cgs

    # ---------- inversion (SI) ----------
    def _dP_drho_num(self, rho: float, T: float, eps_rel: float = 1e-6) -> float:
        rho = float(rho); T = float(T)
        dr = max(eps_rel*abs(rho), 1e-6 * self.p.M_molar / self.p.V0)
        r1 = max(rho - dr, 1e-6); r2 = rho + dr
        P1 = float(self.get_p_rhot(r1, T))
        P2 = float(self.get_p_rhot(r2, T))
        return (P2 - P1) / (r2 - r1)

    def _initial_rho_guess(self, P: float, T: float) -> float:
        rho0 = self.p.M_molar / self.p.V0
        Pth_V0 = float(self.dPth_of_V_T(self.p.V0, T))
        Peff = float(P) - Pth_V0
        rho_guess = rho0 * (1.0 + Peff / max(self.p.K_T0, 1.0))
        return max(rho_guess, 0.2*rho0)

    def get_rho_pt_inv(
        self,
        P: ArrayLike,   # Pa
        T: ArrayLike,
        rho0: Optional[ArrayLike] = None,
        *,
        tol: float = 1e-8,
        maxiter: int = 50,
        newton_first: bool = True,
        dPdrho_eps_rel: float = 1e-6,
    ) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        out = np.empty_like(P_arr, dtype=float)

        if rho0 is not None:
            rho0_arr, = self._as_arrays(rho0)
            rho0_arr = np.broadcast_to(rho0_arr, P_arr.shape)
        else:
            rho0_arr = None

        rho_ref = self.p.M_molar / self.p.V0

        for idx in np.ndindex(P_arr.shape):
            p = float(P_arr[idx]); t = float(T_arr[idx])

            def f(r):
                return float(self.get_p_rhot(r, t) - p)

            def fprime(r):
                r = float(r)
                dr = max(dPdrho_eps_rel*abs(r), 1e-6 * self.p.M_molar / self.p.V0)
                r1 = max(r - dr, 1e-6); r2 = r + dr
                return (f(r2) - f(r1)) / (r2 - r1)

            r_guess = (float(rho0_arr[idx]) if rho0_arr is not None
                       else float(self._initial_rho_guess(p, t)))
            r_guess = max(r_guess, 1e-6)

            solved = False
            if newton_first:
                try:
                    r_newton = newton(f, r_guess, fprime=fprime, tol=tol, maxiter=maxiter)
                    if np.isfinite(r_newton) and r_newton > 0:
                        out[idx] = float(r_newton); solved = True
                except Exception:
                    solved = False

            if not solved:
                a = max(0.1*r_guess, 0.05*rho_ref)
                b = max(2.0*r_guess, 0.5*rho_ref)
                fa, fb = f(a), f(b)
                expand = 1.6; tries = 0
                while (np.isnan(fa) or np.isnan(fb) or fa*fb > 0.0) and tries < 60:
                    a = max(a/expand, 1e-8); b = b*expand
                    fa, fb = f(a), f(b); tries += 1
                if np.isnan(fa) or np.isnan(fb) or fa*fb > 0.0:
                    raise ValueError(
                        f"Failed to bracket for P={p:g} Pa, T={t:g} K. "
                        f"Last bracket [{a:g},{b:g}] with f(a)={fa:g}, f(b)={fb:g}."
                    )
                out[idx] = float(brentq(f, a, b, xtol=tol, maxiter=maxiter))

        return out.reshape(P_arr.shape)

    # ---------- P,T -> X by inversion shortcuts (SI inputs P in Pa) ----------
    def get_s_pt_inv(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_s_rhot(rho_arr, T_arr)

    def get_u_pt_inv(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_u_rhot(rho_arr, T_arr)

    def get_cp_pt(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_CP_rhot(rho_arr, T_arr)

    def get_cv_pt(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_CV_rhot(rho_arr, T_arr)

    def get_alpha_pt(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_alpha_rhot(rho_arr, T_arr)

    def get_gamma_pt(self, P: ArrayLike, T: ArrayLike, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_arrays(P, T)
        rho_arr = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)
        return self.get_gamma_rhot(rho_arr)


# -------- MgSiO3 multi-phase façade with internal parameter dictionaries --------
class MGSIO3_SOLID_EOS:
    """
    User-facing class. Instantiate once: eos = MGSIO3_SOLID_EOS()
    Call get_* with phase in {'pv','ppv','en'} (synonyms allowed).
    This facade:
      - accepts P in GPa and rho in g/cm^3
      - returns P, K_T in GPa; rho in g/cm^3; heat/entropy in erg/g or erg/(g·K)
    """

    # Internal parameter dictionaries (per phase)
    _PARAMS_RAW: Dict[str, Dict[str, float]] = {
        # Perovskite (Pv): Tange et al. 2012
        "pv": dict(
            V0_A3_per_cell=162.373, z=4,  # convert to molar
            K_T0=256.7*GPa, K_T0p=4.09,
            theta0=950.0, gamma0=1.54,
            a=1.0, b=1.5,
            n=5.0, M_molar=100.387e-3, T0=300.0, cold_curve="bm3",
        ),
        # Post-perovskite (PPv): Sakai et al. 2016 fit 7
        "ppv": dict(
            V0_A3_per_cell=164.26, z=4,
            K_T0=203.0*GPa, K_T0p=5.35,
            theta0=848.0, gamma0=1.47,
            # map Al’tsuler γ(V)=γ∞+(γ0−γ∞)(V/V0)^β → γ=γ0[1+a((V/V0)^b−1)]
            a=(1.47 - 0.93)/1.47,  # ≈ 0.3673
            b=2.7,
            n=5.0, M_molar=100.387e-3, T0=300.0, cold_curve="bm3",
        ),
        # NOTE: CONLY A PLACE HOLDER... COMB IS A SAVED TABLE IN THE EOS CLASS. NO RHO,T EOS IS AVAILABLE.
        "comb": dict(
            V0_A3_per_cell=164.26, z=4,
            K_T0=203.0*GPa, K_T0p=5.35,
            theta0=848.0, gamma0=1.47,
            # map Al’tsuler γ(V)=γ∞+(γ0−γ∞)(V/V0)^β → γ=γ0[1+a((V/V0)^b−1)]
            a=(1.47 - 0.93)/1.47,  # ≈ 0.3673
            b=2.7,
            n=5.0, M_molar=100.387e-3, T0=300.0, cold_curve="bm3",
        ),

        # Enstatite (En, orthopyroxene): Perple_X / SLB-style set
        "en": dict(
            V0_molar=31.35e-6,  # m^3/mol
            K_T0=106.0*GPa, K_T0p=8.6,
            theta0=818.0, gamma0=0.92,
            a=1.0, b=2.0,    # γ(V)=γ0*(V/V0)^b
            n=5.0, M_molar=100.387e-3, T0=300.0, cold_curve="bm3",
        ),
    }

    _SYNONYMS = {
        "pv": ["pv", "perovskite"],
        "ppv": ["ppv", "post-perovskite", "postperovskite"],
        "en": ["en", "enstatite", "opx", "orthopyroxene"],
        "comb": ["comb", "combined", "whole"]
    }

    def __init__(self, default_phase: str = "comb"):
        self._eos_by_phase: Dict[str, MGD_EOS] = {}
        # Build canonical keys first
        for canon in ("pv", "ppv", "en", "comb"):
            par = self._build_params(canon)
            self._eos_by_phase[canon] = MGD_EOS(par)
        # Add synonyms
        for canon, keys in self._SYNONYMS.items():
            for k in keys:
                self._eos_by_phase[k] = self._eos_by_phase[canon]
        self._default_phase = 'comb' #self._normalize_phase(default_phase)

                # Unit conversions for mass-specific CGS outputs
        self.U_conv_cgs = (1.0 * u.J/u.kg).to(u.erg/u.g).value          # 1 J/kg -> erg/g
        self.S_conv_cgs = (1.0 * u.J/u.kg/u.K).to(u.erg/u.g/u.K).value  # 1 J/kg/K -> erg/g/K
        self.erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
        self.dyn_to_Pa = (u.dyn/u.cm**2).to('Pa')
        self.dyn_to_GPa = (u.dyn/u.cm**2).to('GPa')

        # ----------------------------- reading P,T basis -----------------------------

        """
        Load P,T grid data for interpolation (solid phases only).
        The pressure range for Enstatite is up to 25 GPa, for Pv up to 150 GPa,
        and for PPv up to 1500 GPa.
        The P-T functions below should interpolate between these three grids to produce a more physical transition.
        """

        self.pt_data_en = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_en_solid_rob_PT.npz')

        self.P_vals_en_pt = self.pt_data_en['P_grid']  # GPa range 1e-4 -- 25
        self.T_vals_en_pt = self.pt_data_en['T_grid']  # K range 300 -- 2000
        self.rho_vals_en_pt = self.pt_data_en['rho_grid']  # g/cm^3
        self.U_vals_en_pt = self.pt_data_en['u_grid']
        self.S_vals_en_pt = self.pt_data_en['s_grid']
        self.alpha_vals_en_pt = self.pt_data_en['alpha_grid']
        self.cp_vals_en_pt = self.pt_data_en['cp_grid']
        self.cv_vals_en_pt = self.pt_data_en['cv_grid']

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)

        self._rho_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.rho_vals_en_pt.T, **rgi_kwargs)
        self._u_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.U_vals_en_pt.T, **rgi_kwargs)
        self._s_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.S_vals_en_pt.T, **rgi_kwargs)
        self._alpha_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.alpha_vals_en_pt.T, **rgi_kwargs)
        self._cp_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.cp_vals_en_pt.T, **rgi_kwargs)
        self._cv_rgi_en_pt = RegularGridInterpolator((self.P_vals_en_pt, self.T_vals_en_pt), self.cv_vals_en_pt.T, **rgi_kwargs)

        self.pt_data_pv = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_pv_solid_rob_PT.npz')

        self.P_vals_pv_pt = self.pt_data_pv['P_grid']  # GPa range 0.1 -- 150
        self.T_vals_pv_pt = self.pt_data_pv['T_grid']  # K range 300 - 10000 
        self.rho_vals_pv_pt = self.pt_data_pv['rho_grid']  # g/cm^3
        self.U_vals_pv_pt = self.pt_data_pv['u_grid']
        self.S_vals_pv_pt = self.pt_data_pv['s_grid']
        self.alpha_vals_pv_pt = self.pt_data_pv['alpha_grid']
        self.cp_vals_pv_pt = self.pt_data_pv['cp_grid']
        self.cv_vals_pv_pt = self.pt_data_pv['cv_grid']

        self._rho_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.rho_vals_pv_pt.T, **rgi_kwargs)
        self._u_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.U_vals_pv_pt.T, **rgi_kwargs)
        self._s_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.S_vals_pv_pt.T, **rgi_kwargs)
        self._alpha_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.alpha_vals_pv_pt.T, **rgi_kwargs)
        self._cp_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.cp_vals_pv_pt.T, **rgi_kwargs)
        self._cv_rgi_pv_pt = RegularGridInterpolator((self.P_vals_pv_pt, self.T_vals_pv_pt), self.cv_vals_pv_pt.T, **rgi_kwargs)

        self.pt_data_ppv = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_ppv_solid_rob_PT.npz')

        self.P_vals_ppv_pt = self.pt_data_ppv['P_grid']  # GPa range 100 -- 1500
        self.T_vals_ppv_pt = self.pt_data_ppv['T_grid']  # K range 1000 -- 15000
        self.rho_vals_ppv_pt = self.pt_data_ppv['rho_grid']  # g/cm^3
        self.U_vals_ppv_pt = self.pt_data_ppv['u_grid']
        self.S_vals_ppv_pt = self.pt_data_ppv['s_grid']
        self.alpha_vals_ppv_pt = self.pt_data_ppv['alpha_grid']
        self.cp_vals_ppv_pt = self.pt_data_ppv['cp_grid']
        self.cv_vals_ppv_pt = self.pt_data_ppv['cv_grid']

        self._rho_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.rho_vals_ppv_pt.T, **rgi_kwargs)
        self._u_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.U_vals_ppv_pt.T, **rgi_kwargs)
        self._s_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.S_vals_ppv_pt.T, **rgi_kwargs)
        self._alpha_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.alpha_vals_ppv_pt.T, **rgi_kwargs)
        self._cp_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.cp_vals_ppv_pt.T, **rgi_kwargs)
        self._cv_rgi_ppv_pt = RegularGridInterpolator((self.P_vals_ppv_pt, self.T_vals_ppv_pt), self.cv_vals_ppv_pt.T, **rgi_kwargs)

        # ----------------------------- Combined P,T smoothed table -----------------------------

        ### The following was generated by combining the three phases along the rho, T basis. But it should be done with the PT basis using the tables above. 
        #self.pt_data = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_solid_rob_PT.npz')
        self.pt_data = np.load('eos/rock_eos/MgSiO3_solid_PT_auto_new.npz')

        self.P_vals_pt = self.pt_data['pvals_pt']  # GPa
        self.T_vals_pt = self.pt_data['tvals_pt']  # K
        self.rho_vals_pt = self.pt_data['rho_grid_pt']  # g/cm^3
        self.U_vals_pt = self.pt_data['u_grid_pt']
        self.S_vals_pt = self.pt_data['s_grid_pt']
        self.alpha_vals_pt = self.pt_data['alpha_grid_pt']
        self.cp_vals_pt = self.pt_data['cp_grid_pt']
        self.cv_vals_pt = self.pt_data['cv_grid_pt']

        self._rho_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.rho_vals_pt.T, **rgi_kwargs)
        self._u_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.U_vals_pt.T, **rgi_kwargs)
        self._s_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.S_vals_pt.T, **rgi_kwargs)
        self._alpha_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.alpha_vals_pt.T, **rgi_kwargs)
        self._cp_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.cp_vals_pt.T, **rgi_kwargs)
        self._cv_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.cv_vals_pt.T, **rgi_kwargs)

        # ----------------------------- phase switching controls -----------------------------
        self.P_en_pv = 25.0     # GPa
        self.P_pv_ppv = 125.0   # GPa

        # smooth transition half-widths (GPa). Tune these.
        self.phase_frac_width = 0.05
        self.dP_en_pv  = max(self.phase_frac_width * self.P_en_pv, 0.5)   # >= 0.5 GPa
        self.dP_pv_ppv = max(self.phase_frac_width * self.P_pv_ppv, 1.0)  # >= 1.0 GPa

        # If True: clamp (P,T) into each phase grid before RGI evaluation (prevents wild extrapolation)
        self.clip_phase_tables = True

        # Registry for easy access
        self._PT = {
            "en":  dict(
                P=self.P_vals_en_pt,  T=self.T_vals_en_pt,
                rho=self._rho_rgi_en_pt, u=self._u_rgi_en_pt, s=self._s_rgi_en_pt,
                alpha=self._alpha_rgi_en_pt, cp=self._cp_rgi_en_pt, cv=self._cv_rgi_en_pt,
            ),
            "pv":  dict(
                P=self.P_vals_pv_pt,  T=self.T_vals_pv_pt,
                rho=self._rho_rgi_pv_pt, u=self._u_rgi_pv_pt, s=self._s_rgi_pv_pt,
                alpha=self._alpha_rgi_pv_pt, cp=self._cp_rgi_pv_pt, cv=self._cv_rgi_pv_pt,
            ),
            "ppv": dict(
                P=self.P_vals_ppv_pt, T=self.T_vals_ppv_pt,
                rho=self._rho_rgi_ppv_pt, u=self._u_rgi_ppv_pt, s=self._s_rgi_ppv_pt,
                alpha=self._alpha_rgi_ppv_pt, cp=self._cp_rgi_ppv_pt, cv=self._cv_rgi_ppv_pt,
            ),
            # keep your combined table available if you still want it explicitly
            "comb": dict(
                P=self.P_vals_pt, T=self.T_vals_pt,
                rho=self._rho_rgi_pt, u=self._u_rgi_pt, s=self._s_rgi_pt,
                alpha=self._alpha_rgi_pt, cp=self._cp_rgi_pt, cv=self._cv_rgi_pt,
            ),
        }

        self._build_entropy_phase_offsets(nT=128, use_max=True)  # use_max=True enforces "always decreases"

        # ----------------------------- reading S,P basis -----------------------------

        self.sp_data_en = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_en_solid_rob_SP.npz')

        self.S_vals_en_sp = self.sp_data_en['s_grid']  # kb/baryon
        self.P_vals_en_sp = self.sp_data_en['P_grid']  # GPa
        self.T_vals_en_sp = self.sp_data_en['T_grid']  # K
        self.rho_vals_en_sp = self.sp_data_en['rho_grid']  # g/cm^3
        self.U_vals_en_sp = self.sp_data_en['u_grid'] # erg/g

        self._t_rgi_en_sp = RegularGridInterpolator((self.S_vals_en_sp, self.P_vals_en_sp), self.T_vals_en_sp, **rgi_kwargs)
        self._rho_rgi_en_sp = RegularGridInterpolator((self.S_vals_en_sp, self.P_vals_en_sp), self.rho_vals_en_sp, **rgi_kwargs)
        self._u_rgi_en_sp = RegularGridInterpolator((self.S_vals_en_sp, self.P_vals_en_sp), self.U_vals_en_sp, **rgi_kwargs)

        self.sp_data_pv = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_pv_solid_rob_SP.npz')

        self.S_vals_pv_sp = self.sp_data_pv['s_grid']  # kb/baryon
        self.P_vals_pv_sp = self.sp_data_pv['P_grid']  # GPa
        self.T_vals_pv_sp = self.sp_data_pv['T_grid']  # K
        self.rho_vals_pv_sp = self.sp_data_pv['rho_grid']  # g/cm^3
        self.U_vals_pv_sp = self.sp_data_pv['u_grid'] # erg/g

        self._t_rgi_pv_sp = RegularGridInterpolator((self.S_vals_pv_sp, self.P_vals_pv_sp), self.T_vals_pv_sp, **rgi_kwargs)
        self._rho_rgi_pv_sp = RegularGridInterpolator((self.S_vals_pv_sp, self.P_vals_pv_sp), self.rho_vals_pv_sp, **rgi_kwargs)
        self._u_rgi_pv_sp = RegularGridInterpolator((self.S_vals_pv_sp, self.P_vals_pv_sp), self.U_vals_pv_sp, **rgi_kwargs)

        self.sp_data_ppv = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_ppv_solid_rob_SP.npz')

        self.S_vals_ppv_sp = self.sp_data_ppv['s_grid']  # kb/baryon
        self.P_vals_ppv_sp = self.sp_data_ppv['P_grid']  # GPa
        self.T_vals_ppv_sp = self.sp_data_ppv['T_grid']  # K
        self.rho_vals_ppv_sp = self.sp_data_ppv['rho_grid']  # g/cm^3
        self.U_vals_ppv_sp = self.sp_data_ppv['u_grid'] # erg/g

        self._t_rgi_ppv_sp = RegularGridInterpolator((self.S_vals_ppv_sp, self.P_vals_ppv_sp), self.T_vals_ppv_sp, **rgi_kwargs)
        self._rho_rgi_ppv_sp = RegularGridInterpolator((self.S_vals_ppv_sp, self.P_vals_ppv_sp), self.rho_vals_ppv_sp, **rgi_kwargs)
        self._u_rgi_ppv_sp = RegularGridInterpolator((self.S_vals_ppv_sp, self.P_vals_ppv_sp), self.U_vals_ppv_sp, **rgi_kwargs)


         # ----------------------------- Combined S,P smoothed table -----------------------------

        self.sp_data = np.load('eos/rock_eos/solid_liquid_Bm3/MgSiO3_solid_rob_SP.npz')

        self.S_vals_sp = self.sp_data['s_grid']  # kb/baryon
        self.P_vals_sp = self.sp_data['P_grid']  # GPa
        self.T_vals_sp = self.sp_data['T_grid']  # K
        self.rho_vals_sp = self.sp_data['rho_grid']  # g/cm^3
        self.U_vals_sp = self.sp_data['u_grid'] # erg/g

        self._t_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.T_vals_sp, **rgi_kwargs)
        self._rho_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.rho_vals_sp, **rgi_kwargs)
        self._u_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.U_vals_sp, **rgi_kwargs)

    @staticmethod
    def _normalize_phase(phase: Optional[str]) -> str:
        return "comb" if phase is None else str(phase).lower()

    @staticmethod
    def _V0_molar_from_A3_per_cell(V0_A3_per_cell: float, z_fu_per_cell: int) -> float:
        return (V0_A3_per_cell / z_fu_per_cell) * A3_TO_M3 * N_A

    def _build_params(self, key: str) -> MGDParams:
        d = dict(self._PARAMS_RAW[key])  # copy
        # Volume conversion
        if "V0_molar" in d:
            V0_molar = float(d.pop("V0_molar"))
        else:
            V0_molar = self._V0_molar_from_A3_per_cell(d.pop("V0_A3_per_cell"), d.pop("z"))
        return MGDParams(
            V0=V0_molar,
            K_T0=float(d["K_T0"]),
            K_T0p=float(d["K_T0p"]),
            theta0=float(d["theta0"]),
            gamma0=float(d["gamma0"]),
            a=float(d["a"]),
            b=float(d["b"]),
            n=float(d["n"]),
            M_molar=float(d["M_molar"]),
            T0=float(d.get("T0", 300.0)),
            cold_curve=str(d.get("cold_curve", "bm3")),
        )

    def _eos(self, phase: Optional[str]) -> MGD_EOS:
        key = self._normalize_phase(phase)
        if key not in self._eos_by_phase:
            valid = sorted({"pv","ppv","en","comb"})
            raise ValueError(f"unknown phase '{phase}'. valid: {valid}")
        return self._eos_by_phase[key]

    # ---- rho,T -> property (I/O: rho in g/cm^3; returns with requested external units) ----
    def get_p_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3             # g/cm^3 -> kg/m^3
        P_pa   = self._eos(phase or self._default_phase).get_p_rhot(rho_si, T)
        return P_pa * 1e-9                                       # GPa

    def get_u_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_u_rhot(rho_si, T)      # erg/g

    def get_KT_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        KT_pa  = self._eos(phase or self._default_phase).get_KT_rhot(rho_si, T)
        return KT_pa * 1e-9                                      # GPa

    def get_alpha_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_alpha_rhot(rho_si, T)  # 1/K

    def get_CV_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_CV_rhot(rho_si, T)     # erg/g/K

    def get_CP_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_CP_rhot(rho_si, T)     # erg/g/K

    def get_gamma_rhot(self, rho, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_gamma_rhot(rho_si)     # –

    def get_s_rhot(self, rho, T, *, phase: Optional[str] = None):
        rho_si = np.asarray(rho, dtype=float) * 1e3
        return self._eos(phase or self._default_phase).get_s_rhot(rho_si, T)      # erg/g/K

    # ---- P,T -> rho inversion and derived props (I/O: P in GPa; rho returned in g/cm^3) ----
    def get_rho_pt_inv(
        self,
        P: ArrayLike,  # GPa
        T: ArrayLike,
        rho0: Optional[ArrayLike] = None,  # g/cm^3
        *,
        phase: Optional[str] = None,
        tol: float = 1e-6,
        maxiter: int = 150,
        newton_first: bool = True,
        dPdrho_eps_rel: float = 1e-4,
    ) -> np.ndarray:
        eos    = self._eos(phase or self._default_phase)
        P_si   = np.asarray(P, dtype=float) * 1e9                 # Pa (copy)
        rho0_si = None if rho0 is None else np.asarray(rho0, dtype=float) * 1e3
        rho_si = eos.get_rho_pt_inv(P_si, T, rho0=rho0_si, tol=tol, maxiter=maxiter,
                                    newton_first=newton_first, dPdrho_eps_rel=dPdrho_eps_rel)  # kg/m^3
        return rho_si * 1e-3                                      # g/cm^3

    # Helpers for *_pt wrappers
    @staticmethod
    def _P_to_Pa(P: ArrayLike) -> np.ndarray:
        return np.asarray(P, dtype=float) * 1e9

    def _as_PT(self, P, T):
        P_arr = np.asarray(P, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
        return P_arr, T_arr

    def _rgi_eval(self, rgi: RegularGridInterpolator, P_arr, T_arr):
        pts = np.stack([P_arr.ravel(), T_arr.ravel()], axis=-1)
        out = rgi(pts).reshape(P_arr.shape)
        return out

    def _clip_to_grid(self, P_arr, T_arr, Pgrid, Tgrid, *, clip_T=True):
        Pmin, Pmax = float(np.min(Pgrid)), float(np.max(Pgrid))
        Tmin, Tmax = float(np.min(Tgrid)), float(np.max(Tgrid))
        Pc = np.clip(P_arr, Pmin, Pmax)
        Tc = np.clip(T_arr, Tmin, Tmax) if clip_T else T_arr
        return Pc, Tc


    def _phase_weights_P(self, P_arr):
        """
        Smooth weights that sum to 1:
        en dominant below ~25 GPa
        pv dominant between ~25 and ~125 GPa
        ppv dominant above ~125 GPa
        """
        P = np.asarray(P_arr, dtype=float)

        s1 = 0.5 * (1.0 + np.tanh((P - self.P_en_pv)  / 0.5))   # en -> pv switch
        s2 = 0.5 * (1.0 + np.tanh((P - self.P_pv_ppv) / 0.5))  # pv -> ppv switch

        w_en  = 1.0 - s1
        w_pv  = s1 * (1.0 - s2)
        w_ppv = s2

        # numerical safety
        w_en  = np.clip(w_en,  0.0, 1.0)
        w_pv  = np.clip(w_pv,  0.0, 1.0)
        w_ppv = np.clip(w_ppv, 0.0, 1.0)
        w_sum = w_en + w_pv + w_ppv
        w_en, w_pv, w_ppv = w_en/w_sum, w_pv/w_sum, w_ppv/w_sum
        return w_en, w_pv, w_ppv

    def _pt_prop_phase(self, prop: str, phase: str, P_arr, T_arr, *, clip: bool | None = None):
        tab = self._PT[phase]
        Pq, Tq = P_arr, T_arr

        do_clip = self.clip_phase_tables if clip is None else bool(clip)
        if do_clip:
            # Clip P always, but DON'T clip T (especially for ppv)
            Pq, Tq = self._clip_to_grid(Pq, Tq, tab["P"], tab["T"], clip_T=False)
            out = self._rgi_eval(tab[prop], Pq, Tq)

            # --- entropy reference correction (constant offsets, cumulative with phase) ---
            if prop == "s":
                off = 0.0
                if hasattr(self, "_S_phase_offset"):
                    off = float(self._S_phase_offset.get(phase, 0.0))
                out = out - off

            return out

    def _pt_prop_phase_raw(self, prop, phase, P_arr, T_arr, *, clip=True):
        tab = self._PT[phase]
        Pq, Tq = P_arr, T_arr
        if clip:
            Pq, Tq = self._clip_to_grid(Pq, Tq, tab["P"], tab["T"], clip_T=True)
        return self._rgi_eval(tab[prop], Pq, Tq)


    def get_rho_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            # WARNING: auto + tab=False would require 3 inversions; use explicit phase here for speed.
            P_si = self._P_to_Pa(P_arr)
            rho_si = self._eos(ph if ph != "auto" else self._default_phase).get_rho_pt_inv(P_si, T_arr, **kwargs)
            return rho_si * 1e-3  # kg/m^3 -> g/cm^3  (BUGFIX vs your current code)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("rho", ph, P_arr, T_arr, clip=clip)

        # --- auto: smooth phase selection in P ---
        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)

        rho_en  = self._pt_prop_phase("rho", "en",  P_arr, T_arr, clip=clip)
        rho_pv  = self._pt_prop_phase("rho", "pv",  P_arr, T_arr, clip=clip)
        rho_ppv = self._pt_prop_phase("rho", "ppv", P_arr, T_arr, clip=clip)

        # blend in specific volume to avoid overshoots
        v = (w_en/np.maximum(rho_en, 1e-300) +
            w_pv/np.maximum(rho_pv, 1e-300) +
            w_ppv/np.maximum(rho_ppv, 1e-300))
        rho_mix = 1.0/np.maximum(v, 1e-300)
        return rho_mix

    def _build_entropy_phase_offsets(self, *, nT=128, use_max=True, percentile=95.0):
        """
        Build additive entropy offsets so S does NOT step upward with pressure across phase boundaries.

        Offsets are constants (independent of T) so they don't break dS/dT-derived quantities.
        We enforce:
            S_pv_corr(P=25,T) <= S_en(P=25,T)
            S_ppv_corr(P=125,T) <= S_pv_corr(P=125,T)

        If use_max=True, we use max_T(S_hi - S_lo) to guarantee "always decreases" at the boundary.
        Otherwise we use a robust percentile (e.g. 95th) to avoid a single noisy outlier dominating.
        """
        self._S_phase_offset = {"en": 0.0, "pv": 0.0, "ppv": 0.0, "comb": 0.0}

        def overlap_T(phase_lo, phase_hi):
            Tlo = self._PT[phase_lo]["T"]
            Thi = self._PT[phase_hi]["T"]
            Tmin = max(float(np.min(Tlo)), float(np.min(Thi)))
            Tmax = min(float(np.max(Tlo)), float(np.max(Thi)))
            if not np.isfinite(Tmin) or not np.isfinite(Tmax) or Tmax <= Tmin:
                return None
            return np.linspace(Tmin, Tmax, int(nT))

        def boundary_offset(Pb, phase_lo, phase_hi):
            Tvec = overlap_T(phase_lo, phase_hi)
            if Tvec is None:
                return 0.0

            Pvec = np.full_like(Tvec, float(Pb), dtype=float)

            # raw table entropies (no offsets applied here)
            s_lo = self._pt_prop_phase("s", phase_lo, Pvec, Tvec, clip=True)
            s_hi = self._pt_prop_phase("s", phase_hi, Pvec, Tvec, clip=True)

            good = np.isfinite(s_lo) & np.isfinite(s_hi)
            if not np.any(good):
                return 0.0

            delta = (s_hi - s_lo)[good]  # >0 means high-P phase is too entropic and must be shifted down
            if use_max:
                d = float(np.nanmax(delta))
            else:
                d = float(np.nanpercentile(delta, percentile))

            # tiny epsilon to enforce strict non-increase
            scale = float(np.nanmax(np.abs(s_lo[good])))
            eps = 1e-12 * max(1.0, scale)

            return max(0.0, d + eps)

        # 25 GPa: en -> pv
        off_pv = boundary_offset(self.P_en_pv, "en", "pv")
        self._S_phase_offset["pv"] = off_pv

        # 125 GPa: pv -> ppv  (ppv should include pv's baseline shift too)
        off_ppv_add = boundary_offset(self.P_pv_ppv, "pv", "ppv")
        self._S_phase_offset["ppv"] = off_pv + off_ppv_add


    def get_s_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_s_pt_inv(P_si, T_arr, **kwargs)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("s", ph, P_arr, T_arr, clip=clip)

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)
        s_en  = self._pt_prop_phase("s", "en",  P_arr, T_arr, clip=clip)
        s_pv  = self._pt_prop_phase("s", "pv",  P_arr, T_arr, clip=clip)
        s_ppv = self._pt_prop_phase("s", "ppv", P_arr, T_arr, clip=clip)
        return w_en*s_en + w_pv*s_pv + w_ppv*s_ppv


    def get_t_sp(self, S: ArrayLike, P: ArrayLike, *, tab: bool = True, phase: Optional[str] = 'ppv', **kwargs) -> np.ndarray:
        eos  = self._eos(phase or self._default_phase)

        if tab:
            if phase == 'comb':
                return self._t_rgi_sp((S, P))
            elif phase != 'comb':
                return self._t_rgi_en_sp((S, P)) if phase == 'en' else (self._t_rgi_pv_sp((S, P)) if phase == 'pv' else self._t_rgi_ppv_sp((S, P)))
        else:
            P_si = self._P_to_Pa(P)
            return self.get_t_sp_inv(S, P_si, **kwargs)

    def get_rho_sp(self, S: ArrayLike, P: ArrayLike, *, tab: bool = True, phase: Optional[str] = 'ppv', **kwargs) -> np.ndarray:
        eos  = self._eos(phase or self._default_phase)

        if tab:
            if phase == 'comb':
                return self._rho_rgi_sp((S, P))
            elif phase != 'comb':
                return self._rho_rgi_en_sp((S, P)) if phase == 'en' else (self._rho_rgi_pv_sp((S, P)) if phase == 'pv' else self._rho_rgi_ppv_sp((S, P)))
        else:
            P_si = self._P_to_Pa(P)
            T = eos.get_t_sp_inv(S, P_si, ** kwargs)
            return eos.get_rho_pt_inv(P_si, T, **kwargs)

    def get_u_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_u_pt_inv(P_si, T_arr, **kwargs)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("u", ph, P_arr, T_arr, clip=clip)

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)
        u_en  = self._pt_prop_phase("u", "en",  P_arr, T_arr, clip=clip)
        u_pv  = self._pt_prop_phase("u", "pv",  P_arr, T_arr, clip=clip)
        u_ppv = self._pt_prop_phase("u", "ppv", P_arr, T_arr, clip=clip)
        return w_en*u_en + w_pv*u_pv + w_ppv*u_ppv


    def get_alpha_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                    phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_alpha_pt(P_si, T_arr, **kwargs)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("alpha", ph, P_arr, T_arr, clip=clip)

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)
        a_en  = self._pt_prop_phase("alpha", "en",  P_arr, T_arr, clip=clip)
        a_pv  = self._pt_prop_phase("alpha", "pv",  P_arr, T_arr, clip=clip)
        a_ppv = self._pt_prop_phase("alpha", "ppv", P_arr, T_arr, clip=clip)
        return w_en*a_en + w_pv*a_pv + w_ppv*a_ppv


    def get_cp_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_cp_pt(P_si, T_arr, **kwargs)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("cp", ph, P_arr, T_arr, clip=clip)

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)
        cp_en  = self._pt_prop_phase("cp", "en",  P_arr, T_arr, clip=clip)
        cp_pv  = self._pt_prop_phase("cp", "pv",  P_arr, T_arr, clip=clip)
        cp_ppv = self._pt_prop_phase("cp", "ppv", P_arr, T_arr, clip=clip)
        return w_en*cp_en + w_pv*cp_pv + w_ppv*cp_ppv


    def get_cv_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_cv_pt(P_si, T_arr, **kwargs)

        if ph in ("en", "pv", "ppv", "comb"):
            return self._pt_prop_phase("cv", ph, P_arr, T_arr, clip=clip)

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)
        cv_en  = self._pt_prop_phase("cv", "en",  P_arr, T_arr, clip=clip)
        cv_pv  = self._pt_prop_phase("cv", "pv",  P_arr, T_arr, clip=clip)
        cv_ppv = self._pt_prop_phase("cv", "ppv", P_arr, T_arr, clip=clip)
        return w_en*cv_en + w_pv*cv_pv + w_ppv*cv_ppv

    def get_gamma_pt(self, P: ArrayLike, T: ArrayLike, *, tab: bool = True,
                    phase: Optional[str] = "auto", clip: bool | None = None, **kwargs) -> np.ndarray:
        P_arr, T_arr = self._as_PT(P, T)
        ph = self._normalize_phase(phase)

        # For tab=False: use the underlying EOS directly
        if not tab:
            P_si = self._P_to_Pa(P_arr)
            return self._eos(ph if ph != "auto" else self._default_phase).get_gamma_pt(P_si, T_arr, **kwargs)

        # For tab=True: compute gamma(V) from tabulated rho for each phase
        if ph in ("en", "pv", "ppv"):
            rho = self._pt_prop_phase("rho", ph, P_arr, T_arr, clip=clip)  # g/cm^3
            rho_si = rho * 1e3  # kg/m^3
            return self._eos(ph).get_gamma_rhot(rho_si)

        if ph == "comb":
            # "comb" has no single underlying EOS; use auto blend instead
            ph = "auto"

        if ph != "auto":
            raise ValueError(f"Unknown phase '{phase}'")

        w_en, w_pv, w_ppv = self._phase_weights_P(P_arr)

        rho_en  = self._pt_prop_phase("rho", "en",  P_arr, T_arr, clip=clip) * 1e3
        rho_pv  = self._pt_prop_phase("rho", "pv",  P_arr, T_arr, clip=clip) * 1e3
        rho_ppv = self._pt_prop_phase("rho", "ppv", P_arr, T_arr, clip=clip) * 1e3

        g_en  = self._eos("en").get_gamma_rhot(rho_en)
        g_pv  = self._eos("pv").get_gamma_rhot(rho_pv)
        g_ppv = self._eos("ppv").get_gamma_rhot(rho_ppv)

        return w_en*g_en + w_pv*g_pv + w_ppv*g_ppv


    # -------------------------- NEW: T(S,P) inversion (table-based) --------------------------
    @staticmethod
    def _as_arrays(*vals):
        arrs = [np.asarray(v, float) for v in vals]
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
        return [np.broadcast_to(a, shape) for a in arrs]

    def _select_s_table(self, phase):
        ph = self._normalize_phase(phase)
        if ph == 'en':
            return self._s_rgi_en_pt, self.T_vals_en_pt
        if ph == 'pv':
            return self._s_rgi_pv_pt, self.T_vals_pv_pt
        if ph == 'ppv':
            return self._s_rgi_ppv_pt, self.T_vals_ppv_pt
        if ph == 'comb':
            return self._s_rgi_pt, self.T_vals_pt
        raise ValueError(f"No S(P,T) table for phase '{phase}'")

    def _dS_dT_num_tab(self, P, T, *, phase, eps_rel=1e-6):
        s_tab, _ = self._select_s_table(phase)
        T = float(T)
        dT = max(eps_rel*max(abs(T), 1.0), 1e-6)
        return (s_tab((P, T + dT)) - s_tab((P, T - dT))) / (2.0*dT)

    def _initial_T_guess_from_tab(self, S_target, P, *, phase):
        s_tab, Tgrid = self._select_s_table(phase)
        if Tgrid is None or np.size(Tgrid) == 0:
            return 2000.0
        vals = np.array([s_tab((P, float(t))) for t in Tgrid], dtype=float)
        j = int(np.nanargmin(np.abs(vals - S_target)))
        return float(Tgrid[j])

    def get_t_sp_inv(
        self,
        S: ArrayLike,     # kb/baryon
        P: ArrayLike,     # GPa
        T0: Optional[ArrayLike] = None,
        *,
        phase: Optional[str] = None,
        tol: float = 1e-8,
        maxiter: int = 150,
        newton_first: bool = True,
        dSdT_eps_rel: float = 1e-6,
    ) -> np.ndarray:
        """
        Invert S(P,T) -> T(S,P) **using the S(P,T) tables** for the specified phase.
        """
        ph = self._normalize_phase(phase or self._default_phase)
        s_tab, Tgrid = self._select_s_table(ph)
        T_min = float(np.min(Tgrid)) if Tgrid is not None and Tgrid.size else 10.0
        T_max = float(np.max(Tgrid)) if Tgrid is not None and Tgrid.size else 6000.0

        S_arr, P_arr = self._as_arrays(S, P)
        S_arr = S_arr / self.erg_to_kbbar   # erg/g/K -> kb/baryon
        out = np.empty_like(S_arr, dtype=float)

        if T0 is not None:
            T0_arr, _ = self._as_arrays(T0, P_arr)
        else:
            T0_arr = None

        it = np.ndindex(S_arr.shape)
        for idx in it:
            s_tgt = float(S_arr[idx])
            p     = float(P_arr[idx])

            def g(T):
                return float(s_tab((p, T)) - s_tgt)

            def gprime(T):
                return float(self._dS_dT_num_tab(p, T, phase=ph, eps_rel=dSdT_eps_rel))

            T_guess = float(T0_arr[idx]) if T0_arr is not None else self._initial_T_guess_from_tab(s_tgt, p, phase=ph)

            solved = False
            if newton_first:
                try:
                    T_new = newton(g, T_guess, fprime=gprime, tol=tol, maxiter=maxiter)
                    if np.isfinite(T_new):
                        out[idx] = float(T_new); solved = True
                except Exception:
                    solved = False

            if not solved:
                a, b = T_min, T_max
                ga, gb = g(a), g(b)
                tries, expand = 0, 1.5
                while (np.isnan(ga) or np.isnan(gb) or ga*gb > 0.0) and tries < 40:
                    a = max(a/expand, 1.0); b = b*expand
                    ga, gb = g(a), g(b); tries += 1
                if np.isnan(ga) or np.isnan(gb) or ga*gb > 0.0:
                    raise ValueError(
                        f"T(S,P) bracketing failed at P={p:g} GPa, target S={s_tgt:g}. "
                        f"Last bracket [{a:g},{b:g}] with g(a)={ga:g}, g(b)={gb:g}"
                    )
                out[idx] = float(brentq(g, a, b, xtol=tol, maxiter=maxiter))

        return out.reshape(S_arr.shape)

    def get_T_melt(self, P):
        """
        Returns the melting temperature as a function of pressure (GPa) from Fei et al. (2021)
        """
        return 6295 * (P / 140 ) ** 0.317


# ---------- minimal usage example ----------
# if __name__ == "__main__":
#     eos = MGSIO3_EOS()  # defaults to perovskite
#     rho = 4.500  # g/cm^3
#     T = 2000.0   # K

#     print("P_pv [GPa]:", eos.get_p_rhot(rho, T, phase='pv'))
#     print("P_ppv [GPa]:", eos.get_p_rhot(rho, T, phase='ppv'))
#     print("P_en  [GPa]:", eos.get_p_rhot(rho, T, phase='en'))

#     # Invert back: provide P in GPa, get rho in g/cm^3
#     p_test = eos.get_p_rhot(rho, T, phase='pv')        # GPa
#     rho_back = eos.get_rho_pt_inv(p_test, T, phase='pv')
#     print("rho(p_test, 2000 K) [g/cm^3]:", rho_back)

#     rho_back = eos.get_rho_pt(p_test, T, phase='pv', tab=True)
#     print("rho(p_test, 2000 K) [g/cm^3]:", rho_back)

#     # Example property at fixed P,T
#     S = eos.get_s_pt(100.0, 2000.0, phase='pv', tab=False)        # P=100 GPa
#     print("S(100 GPa, 2000 K) [erg/g/K]:", S)
#     S = eos.get_s_pt(100.0, 2000.0, phase='pv', tab=True)        # P=100 GPa
#     print("S(100 GPa, 2000 K) [erg/g/K]:", S)

#     # New: invert T from S and P using S-table
#     T_sol = eos.get_t_sp_inv(S * eos.erg_to_kbbar, 100.0, phase='pv')
#     print("T(S,100 GPa) [K]:", T_sol)
#     T_sol_tab = eos.get_t_sp(S * eos.erg_to_kbbar, 100.0, phase='pv')
#     print("Ttable(S,100 GPa) [K]:", T_sol_tab)
