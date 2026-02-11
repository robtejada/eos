from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import newton, brentq

from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

kb = k_B.to('erg/K') # ergs/K

ArrayLike = Union[float, int, np.ndarray]

@dataclass
class Domain:
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float

class MGSIO3_LIQUID_EOS:
    """
    Build rectangular (T, rho) grids for P, U, S by:
      1) Reading the CSV table.
      2) For each unique T, fit 1-D P(ρ|T), U(ρ|T), S(ρ|T) with interp1d (linear, with extrapolation).
      3) Define a global uniform rho-grid [rho_min, rho_max] with N points.
      4) Evaluate those 1-D interpolators on the global rho-grid to form rectangular arrays
         P[T_i, rho_j], U[T_i, rho_j], S[T_i, rho_j].
      5) Build RegularGridInterpolator for each field over axes (T, rho).
      6) Provide getters get_p_rhot, get_u_rhot, get_s_rhot (support scalars/arrays),
         inversion get_rho_pt(P, T), and inversion get_t_sp(S, P) using Newton with fallback to Brent.

    Units are preserved from the input file.
      P:  GPa
      T:  K
      rho: g/cm^3
      U:  kJ/mol
      S:  J/mol/K
    """
    def __init__(
        self,
        csv_path: str = "eos/rock_eos/MgSiO3_liquid_EoS.csv",
        *,
        colmap: Optional[Dict[str, str]] = None,
        n_rho: int = 120,
        verbose: bool = True,
    ):
        self._df = pd.read_csv(csv_path)

        # Molar mass of MgSiO3 (using standard atomic weights)
        self.mu_MgSiO3 = (24.305*u.g/u.mol) + (28.0855*u.g/u.mol) + (3*15.999*u.g/u.mol)

        # 1) Internal energy: kJ/mol  ->  erg/g
        self.U_conv_cgs = (1.0 * u.kJ/u.mol / self.mu_MgSiO3).to(u.erg/u.g).value      # multiply your values [kJ/mol] by U_conv.value

        # 2) Entropy: J/mol/K  ->  erg/g/K
        self.S_conv_cgs = (1.0 * u.J/u.mol/u.K / self.mu_MgSiO3).to(u.erg/u.g/u.K).value  # multiply your values [J/mol/K] by S_conv.value

        self.erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

        self.dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion

        self.dyn_to_GPa = (u.dyn/u.cm**2).to('GPa') # dyn/cm² to GPa conversion

        # Column detection (fuzzy default)
        if colmap is None:
            colmap = self._auto_colmap(self._df.columns)
        for k in ("P","T","rho","U","S"):
            if k not in colmap or colmap[k] not in self._df.columns:
                raise ValueError(f"Column for '{k}' not found. Got colmap={colmap} and columns={list(self._df.columns)}")
        df = self._df[[colmap["P"], colmap["T"], colmap["rho"], colmap["U"], colmap["S"]]].copy()
        df.columns = ["P","T","rho","U","S"]

        # Unique temperatures, sorted
        self.T_vals = np.sort(df["T"].unique())

        # Build per-T 1-D interpolators (ρ -> field) with linear extrapolation
        self._interp_P: Dict[float, interp1d] = {}
        self._interp_U: Dict[float, interp1d] = {}
        self._interp_S: Dict[float, interp1d] = {}
        rho_all_min = np.inf
        rho_all_max = -np.inf

        for T in self.T_vals:
            sub = df[df["T"] == T].copy()
            # Enforce monotonic ρ and drop duplicates
            sub = sub.sort_values("rho").drop_duplicates(subset="rho", keep="last")
            rho_arr = sub["rho"].to_numpy()
            P_arr   = sub["P"  ].to_numpy()
            U_arr   = sub["U"  ].to_numpy()
            S_arr   = sub["S"  ].to_numpy()

            rho_all_min = min(rho_all_min, rho_arr.min())
            rho_all_max = max(rho_all_max, rho_arr.max())

            # Linear with extrapolation
            self._interp_P[T] = interp1d(rho_arr, P_arr, kind="linear", fill_value="extrapolate", bounds_error=False, assume_sorted=True)
            self._interp_U[T] = interp1d(rho_arr, U_arr, kind="linear", fill_value="extrapolate", bounds_error=False, assume_sorted=True)
            self._interp_S[T] = interp1d(rho_arr, S_arr, kind="linear", fill_value="extrapolate", bounds_error=False, assume_sorted=True)

        # Global uniform rho-grid (N points)
        self.rho_vals = np.linspace(rho_all_min, rho_all_max, int(n_rho))
        self.domain = Domain(rho_min=float(self.rho_vals.min()),
                             rho_max=float(self.rho_vals.max()),
                             T_min=float(self.T_vals.min()),
                             T_max=float(self.T_vals.max()))

        # Rectangular (T, rho) arrays
        nT, nR = len(self.T_vals), len(self.rho_vals)
        self.P_grid = np.empty((nT, nR), dtype=float)
        self.U_grid = np.empty((nT, nR), dtype=float)
        self.S_grid = np.empty((nT, nR), dtype=float)

        for i, T in enumerate(self.T_vals):
            self.P_grid[i, :] = self._interp_P[T](self.rho_vals)
            self.U_grid[i, :] = self._interp_U[T](self.rho_vals)
            self.S_grid[i, :] = self._interp_S[T](self.rho_vals)

        # Build RegularGridInterpolators over axes (T, rho)
        # Inside the box they interpolate; outside they return NaN. We handle extrap below.
        self._p_rgi = RegularGridInterpolator((self.T_vals, self.rho_vals), self.P_grid,
                                              method="linear", bounds_error=False, fill_value=None)
        self._u_rgi = RegularGridInterpolator((self.T_vals, self.rho_vals), self.U_grid,
                                              method="linear", bounds_error=False, fill_value=None)
        self._s_rgi = RegularGridInterpolator((self.T_vals, self.rho_vals), self.S_grid,
                                              method="linear", bounds_error=False, fill_value=None)

        # if verbose:
        #     print("Building mantle EOS grids")
        #     print(f"  T in [{self.domain.T_min:g}, {self.domain.T_max:g}] K  (N_T={nT})")
        #     print(f"  rho in [{self.domain.rho_min:g}, {self.domain.rho_max:g}] g/cm^3  (N_rho={nR})")

        # ----------------------------- reading P,T basis -----------------------------

        self.pt_data = np.load('eos/rock_eos/MgSiO3_liquid_PT.npz')

        self.P_vals_pt = self.pt_data['P_grid']  # GPa
        self.T_vals_pt = self.pt_data['T_grid']  # K
        self.rho_vals_pt = self.pt_data['rho_grid']  # g/cm^3
        self.U_vals_pt = self.pt_data['u_grid'] # kJ/mol
        self.S_vals_pt = self.pt_data['s_grid'] # J/mol/K
        self.alpha_vals_pt = self.pt_data['alpha_grid'] # 1/K
        self.cp_vals_pt = self.pt_data['cp_grid'] # J/mol/K

        self._rho_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.rho_vals_pt.T,
                                                    method="linear", bounds_error=False, fill_value=None)
        self._u_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.U_vals_pt.T,
                                                  method="linear", bounds_error=False, fill_value=None)
        self._s_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.S_vals_pt.T,
                                                  method="linear", bounds_error=False, fill_value=None)
        self._alpha_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.alpha_vals_pt.T,
                                                  method="linear", bounds_error=False, fill_value=None)
        self._cp_rgi_pt = RegularGridInterpolator((self.P_vals_pt, self.T_vals_pt), self.cp_vals_pt.T,
                                                  method="linear", bounds_error=False, fill_value=None)

        # ----------------------------- reading S,P basis -----------------------------

        self.sp_data = np.load('eos/rock_eos/MgSiO3_liquid_SP.npz')

        self.S_vals_sp = self.sp_data['S_grid']  # J/mol/K
        self.P_vals_sp = self.sp_data['P_grid']  # GPa
        self.T_vals_sp = self.sp_data['T_grid']  # K
        self.rho_vals_sp = self.sp_data['rho_grid']  # g/cm^3
        self.U_vals_sp = self.sp_data['u_grid'] # kJ/mol

        self._t_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.T_vals_sp,
                                                    method="linear", bounds_error=False, fill_value=None)
        self._rho_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.rho_vals_sp,
                                                  method="linear", bounds_error=False, fill_value=None)
        self._u_rgi_sp = RegularGridInterpolator((self.S_vals_sp, self.P_vals_sp), self.U_vals_sp,
                                                  method="linear", bounds_error=False, fill_value=None)

    # ----------------------------- Public API -----------------------------

    def get_p_rhot(self, _lgrho: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        """Return P(GPa) for given rho[g/cm^3], T[K]. Supports scalars or arrays (broadcasted)."""
        rho, T = 10**_lgrho, 10**_lgt
        return self._eval_with_extrap(self._p_rgi, self.P_grid, rho, T)
    def get_u_rhot(self, _lgrho: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        """Return U[kJ/mol] for given rho[g/cm^3], T[K]."""
        rho, T = 10**_lgrho, 10**_lgt
        return self._eval_with_extrap(self._u_rgi, self.U_grid, rho, T)
    def get_s_rhot(self, _lgrho: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        """Return S[J/mol/K] for given rho[g/cm^3], T[K]."""
        rho, T = 10**_lgrho, 10**_lgt
        return self._eval_with_extrap(self._s_rgi, self.S_grid, rho, T)

    def get_rho_pt_tab(self, _lgp: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        P, T = 10**(_lgp - 10), 10**_lgt
        return self._rho_rgi_pt(np.column_stack(self._as_arrays(P, T)))
    def get_u_pt_tab(self, _lgp: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        P, T = 10**(_lgp - 10), 10**_lgt
        return self._u_rgi_pt(np.column_stack(self._as_arrays(P, T)))
    def get_s_pt_tab(self, _lgp: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        P, T = 10**(_lgp - 10), 10**_lgt
        return self._s_rgi_pt(np.column_stack(self._as_arrays(P, T)))
    def get_alpha_pt_tab(self, _lgp: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        P, T = 10**(_lgp - 10), 10**_lgt
        return self._alpha_rgi_pt(np.column_stack(self._as_arrays(P, T)))
    def get_cp_pt_tab(self, _lgp: ArrayLike, _lgt: ArrayLike) -> np.ndarray:
        P, T = 10**(_lgp - 10), 10**_lgt
        return self._cp_rgi_pt(np.column_stack(self._as_arrays(P, T)))

    def get_t_sp_tab(self, S: ArrayLike, _lgp: ArrayLike) -> np.ndarray:
        P = 10**(_lgp - 10)
        return self._t_rgi_sp(np.column_stack(self._as_arrays(S, P)))
    def get_rho_sp_tab(self, S: ArrayLike, _lgp: ArrayLike) -> np.ndarray:
        P = 10**(_lgp - 10)
        return self._rho_rgi_sp(np.column_stack(self._as_arrays(S, P)))
    def get_u_sp_tab(self, S: ArrayLike, _lgp: ArrayLike) -> np.ndarray:
        P = 10**(_lgp - 10)
        return self._u_rgi_sp(np.column_stack(self._as_arrays(S, P)))

    def get_rho_pt_inv(
        self,
        P: ArrayLike,
        T: ArrayLike,
        rho0: Optional[ArrayLike] = None,
        *,
        tol: float = 1e-8,
        maxiter: int = 50,
        newton_first: bool = True,
        dPdrho_eps_rel: float = 1e-6,
    ) -> np.ndarray:
        """
        Invert P(rho, T) -> rho(P, T) by root-finding f(rho)=P(rho,T)-P_target.
        Strategy:
          1) Try Newton's method with an optional starting guess rho0 (broadcasted to shape of P,T).
             Derivative dP/drho is computed numerically via central difference.
          2) If Newton fails, fall back to Brent's method (brentq) **bounded** to [rho_min, rho_max].
        """
        P_arr, T_arr = self._as_arrays(P, T)
        out = np.empty_like(P_arr, dtype=float)

        # Prepare rho0 array (initial guesses)
        if rho0 is not None:
            rho0_arr, _ = self._as_arrays(rho0, T_arr)
        else:
            rho0_arr = None

        it = np.ndindex(P_arr.shape)
        for idx in it:
            p = float(P_arr[idx])
            t = float(T_arr[idx])

            def f(r):
                return float(self.get_p_rhot(r, t) - p)

            def fprime(r):
                return float(self._dP_drho_num(r, t, eps_rel=dPdrho_eps_rel))

            r_guess = (float(rho0_arr[idx]) if rho0_arr is not None
                       else float(self._initial_rho_guess(p, t)))

            solved = False
            if newton_first:
                try:
                    r_newton = newton(f, r_guess, fprime=fprime, tol=tol, maxiter=maxiter)
                    out[idx] = float(r_newton)
                    solved = True
                except Exception:
                    solved = False

            if not solved:
                a = -100
                b = 1000
                fa = f(a)
                fb = f(b)
                if np.isnan(fa) or np.isnan(fb) or fa * fb > 0.0:
                    raise ValueError(
                        f"brentq: target P={p:g} at T={t:g} K not bracketed on [rho_min={a:g}, rho_max={b:g}]. "
                        f"f(rho_min)={fa:g}, f(rho_max)={fb:g}."
                    )
                r_brent = brentq(f, a, b, xtol=tol, maxiter=maxiter)
                out[idx] = float(r_brent)

        return out.reshape(P_arr.shape)

    def get_s_pt(self, P: ArrayLike, T: ArrayLike, tab=True) -> np.ndarray:
        """Gets S(P,T) from rho(P,T) and S(rho,T)."""
        if tab:
            return self.get_s_pt_tab(P, T) * self.S_conv_cgs
        else:
            rho = self.get_rho_pt_inv(P, T)
            return self.get_s_rhot(rho, T) * self.S_conv_cgs

    def get_u_pt(self, P: ArrayLike, T: ArrayLike, tab=True) -> np.ndarray:
        """Gets U(P,T) from rho(P,T) and U(rho,T)."""
        if tab:
            return self.get_u_pt_tab(P, T) * self.U_conv_cgs
        else:
            rho = self.get_rho_pt_inv(P, T)
            return self.get_u_rhot(rho, T) * self.U_conv_cgs

    def get_rho_pt(self, P: ArrayLike, T: ArrayLike, tab=True, **kwargs) -> np.ndarray:
        """Return rho[g/cm^3] for given P[GPa], T[K]."""
        if tab:
            return self.get_rho_pt_tab(P, T) # returning log for mixing
        else:
            return self.get_rho_pt_inv(P, T, **kwargs)

    def get_t_sp_inv(
        self,
        S: ArrayLike,
        P: ArrayLike,
        T0: Optional[ArrayLike] = None,
        *,
        tol: float = 1e-8,
        maxiter: int = 50,
        newton_first: bool = True,
        dSdT_eps_rel: float = 1e-6,
    ) -> np.ndarray:
        """
        Invert S(P, T) -> T(S, P) by root-finding g(T) = S(P,T) - S_target.
        Strategy:
          1) Try Newton's method with optional starting guess T0 (broadcasted to S,P).
             Derivative dS/dT computed numerically via central difference.
          2) If Newton fails, fall back to Brent's bracketing (brentq) on [T_min, T_max].
        """
        S_arr, P_arr = self._as_arrays(S, P)
        out = np.empty_like(S_arr, dtype=float)

        # Initial guess array
        if T0 is not None:
            T0_arr, _ = self._as_arrays(T0, P_arr)
        else:
            T0_arr = None

        it = np.ndindex(S_arr.shape)
        for idx in it:
            s_tgt = float(S_arr[idx])
            p = float(P_arr[idx])

            def g(T):
                return float(self.get_s_pt_tab(p, T) - s_tgt)

            def gprime(T):
                return float(self._dS_dT_num(p, T, eps_rel=dSdT_eps_rel))

            # Choose starting guess: provided T0 or nearest T-grid to target S at fixed P
            T_guess = (float(T0_arr[idx]) if T0_arr is not None
                       else float(self._initial_T_guess(s_tgt, p)))

            solved = False
            if newton_first:
                try:
                    T_newton = newton(g, T_guess, fprime=gprime, tol=tol, maxiter=maxiter)
                    out[idx] = float(T_newton)
                    solved = True
                except Exception:
                    solved = False

            if not solved:
                a = -100
                b = 100000
                ga = g(a)
                gb = g(b)
                if np.isnan(ga) or np.isnan(gb) or ga * gb > 0.0:
                    raise ValueError(
                        f"brentq: target S={s_tgt:g} at P={p:g} GPa not bracketed on [T_min={a:g}, T_max={b:g}] K. "
                        f"g(T_min)={ga:g}, g(T_max)={gb:g}."
                    )
                T_brent = brentq(g, a, b, xtol=tol, maxiter=maxiter)
                out[idx] = float(T_brent)

        return out.reshape(S_arr.shape)

    def get_t_sp(self, S: ArrayLike, P: ArrayLike, tab=True) -> np.ndarray:
        #S_cgs = S / self.erg_to_kbbar  # convert to cgs
        #S_SI = S_cgs / self.S_conv_cgs  # convert to J/mol/K
        S_SI = S / (self.S_conv_cgs * self.erg_to_kbbar)  # convert to J/mol/K
        if tab:
            return self.get_t_sp_tab(S_SI, P)
        else:
            return self.get_t_sp_inv(S_SI, P)

    def get_rho_sp(self, S: ArrayLike, P: ArrayLike, tab=True) -> np.ndarray:
        """Gets rho(S,P) from T(S,P) and rho(T,P)."""
        #S_cgs = S / self.erg_to_kbbar  # convert to cgs
        #S_SI = S_cgs / self.S_conv_cgs  # convert to J/mol/K
        S_SI = S / (self.S_conv_cgs * self.erg_to_kbbar)  # convert to J/mol/K
        if tab:
            return self.get_rho_sp_tab(S_SI, P)
        else:
            T = self.get_t_sp_inv(S_SI, P)
            return self.get_rho_pt(P, T)

    def get_u_sp(self, S: ArrayLike, P: ArrayLike, tab=True) -> np.ndarray:
        """Gets U(S,P) from T(S,P) and U(T,P)."""
        #S_cgs = S / self.erg_to_kbbar  # convert to cgs
        #S_SI = S_cgs / self.S_conv_cgs  # convert to J/mol/K
        S_SI = S / (self.S_conv_cgs * self.erg_to_kbbar)  # convert to J/mol/K
        if tab:
            return self.get_u_sp_tab(S_SI, P) #* self.U_conv_cgs
        else:
            T = self.get_t_sp_inv(S_SI, P)
            return self.get_u_pt(P, T)#* self.U_conv_cgs


    # ---------------------------- Derivatives ------------------------------

    def get_alpha_pt(self, P: ArrayLike, T: ArrayLike, tab: bool = True, eps_rel: float = 0.05) -> np.ndarray:
        """
        Thermal expansivity:
            α = -(1/ρ) * (∂ρ/∂T)_P      [1/K]

        Uses a symmetric relative step in T: T*(1±eps_rel). The minus sign is IMPORTANT:
        since ρ = m/V, α = (1/V)(∂V/∂T)_P = -(1/ρ)(∂ρ/∂T)_P.
        """
        if tab:
            return self.get_alpha_pt_tab(P, T)  # 1/K
        else:
            P_arr, T_arr = self._as_arrays(P, T)
            T_hi = T_arr * (1.0 + eps_rel)
            T_lo = T_arr * (1.0 - eps_rel)

            rho     = self.get_rho_pt(P_arr, T_arr, tab=tab)
            rho_hi  = self.get_rho_pt(P_arr, T_hi,  tab=tab)
            rho_lo  = self.get_rho_pt(P_arr, T_lo,  tab=tab)

            drho_dT = (rho_hi - rho_lo) / (T_hi - T_lo)
            alpha   = -(drho_dT / rho)
            return alpha.reshape(P_arr.shape)

    def get_cv_rhot(self, rho: ArrayLike, T: ArrayLike, eps_rel: float = 0.1) -> np.ndarray:
        """
        Specific heat at constant volume (density) per mass:
            C_V = (∂U/∂T)_ρ               [erg g^-1 K^-1]

        U is tabulated in kJ/mol. We compute a symmetric derivative and convert to cgs with U_conv_cgs.
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = T_arr * (1.0 - eps_rel)

        U_hi = self.get_u_rhot(rho_arr, T_hi)   # kJ/mol
        U_lo = self.get_u_rhot(rho_arr, T_lo)   # kJ/mol

        dU_dT_kJmolK = (U_hi - U_lo) / (T_hi - T_lo)
        Cv_cgs = dU_dT_kJmolK * self.S_conv_cgs      # erg/g/K
        return Cv_cgs.reshape(np.shape(rho))

    def get_cp_pt(self, P: ArrayLike, T: ArrayLike, tab: bool = True, eps_rel: float = 0.1) -> np.ndarray:
        """
        Specific heat at constant pressure per mass:
            C_P = T * (∂S/∂T)_P           [erg g^-1 K^-1]

        S is tabulated in J/mol/K. We compute (∂S/∂T)_P and convert with S_conv_cgs, then multiply by T.
        """
        if tab:
            return self.get_cp_pt_tab(P, T) * self.S_conv_cgs  # erg/g/K
        else:
            P_arr, T_arr = self._as_arrays(P, T)
            T_hi = T_arr * (1.0 + eps_rel)
            T_lo = T_arr * (1.0 - eps_rel)

            # Evaluate S(P,T) along the isobar (prefer the PT table to match your precomputed basis)
            S_hi = self.get_s_pt(P_arr, T_hi, tab=tab)  # J/mol/K
            S_lo = self.get_s_pt(P_arr, T_lo, tab=tab)  # J/mol/K

            dS_dT_JmolK2 = (S_hi - S_lo) / (T_hi - T_lo)
            Cp_cgs = T_arr * dS_dT_JmolK2 * self.S_conv_cgs   # erg/g/K
            return Cp_cgs.reshape(P_arr.shape)

    # def get_gamma_sp(self, S: ArrayLike, P: ArrayLike, tab: bool = True, eps_rel: float = 0.1) -> np.ndarray:

    #     rho_ = self.get_rho_sp(S, P, tab=tab)  # g/cm^3
    #     rho_plus = self.get_rho_sp(S, P * (1.0 + eps_rel), tab=tab)
    #     rho_minus = self.get_rho_sp(S, P * (1.0 - eps_rel), tab=tab)

    #     lnrho_minus = np.log(rho_minus)
    #     lnrho_plus = np.log(rho_plus)

    #     dlnrho_dlnP = (np.log(P * (1.0 + eps_rel) * u.GPa.to('dyn/cm^2')) - np.log(P * (1.0 - eps_rel) * u.GPa.to('dyn/cm^2'))) / (lnrho_plus - lnrho_minus)

    #     return dlnrho_dlnP.reshape(np.shape(S))

    def get_gruneisen(self, rho, dlnrho = 0.05):
        """
        Smooth, vectorized Grüneisen parameter γ(P,y):
        • Olivine   for P < 23 GPa
        • Perovskite for 23 GPa ≤ P < 125 GPa
        • Post‑pv   for P ≥ 125 GPa

        Smooth blending (tanh) over ±5% of each boundary.
        """

        _PPV_PARAMS = {
        'rho0':      1374.7831663193047,  # kg/m³
        'gamma0':    1.495,
        'gamma_inf': 0.818,
        'lambda':    2.627
}
        # 1) broadcast inputs
        rho *= 1e3
        rho_arr = np.array(rho, ndmin=1)
        lnrho = np.log(rho_arr)
        lnrho_plus = lnrho*(1.0 + dlnrho)
        lnrho_minus = lnrho*(1.0 - dlnrho)

        p = _PPV_PARAMS
        x_pp    = rho / p['rho0']
        x_pp_plus = np.exp(lnrho_plus) / p['rho0']
        x_pp_minus = np.exp(lnrho_minus) / p['rho0']

        gamma_pp = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp**(-p['lambda'])
        gamma_pp_plus = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp_plus**(-p['lambda'])
        gamma_pp_minus = p['gamma_inf'] + (p['gamma0'] - p['gamma_inf']) * x_pp_minus**(-p['lambda'])

        dlngamma_dlnrho_pp = -(np.log(gamma_pp_plus) - np.log(gamma_pp_minus)) / (lnrho_plus - lnrho_minus)

        gamma    = gamma_pp

        return gamma, dlngamma_dlnrho_pp # derivative used in conductivity model

    def get_cv_pt(self, P: ArrayLike, T: ArrayLike, tab: bool = True, eps_rel: float = 0.1) -> np.ndarray:

        CP = self.get_cp_pt(P, T, tab=tab, eps_rel=eps_rel)  # erg/g/K
        alpha = self.get_alpha_pt(P, T, tab=tab, eps_rel=eps_rel)  # 1/K
        rho = self.get_rho_pt(P, T, tab=tab)  # g/cm^3
        gamma = self.get_gruneisen(rho)[0]  # dimensionless

        CV = CP / (1.0 + gamma * alpha * T)

        return CV

    # ---------------------------- Miscellaneous ------------------------------

    _SEC_IN_GYR = u.Gyr.to('s')
    SI_to_cgs = (u.W/u.kg).to('erg/(s * g)')

    # McDonough & Sun (1995) present-day specific heat-production rates (W kg⁻¹)
    _ISOTOPES = ('K40', 'Th232', 'U235', 'U238')
    _Q0       = np.array([8.69e-13, 2.24e-12, 8.48e-14, 1.97e-12]) * SI_to_cgs # erg s⁻¹ kg⁻¹
    _TAU_GYR  = np.array([1.25,     14.0,      0.704,    4.47   ])

    def radiogenic_heat(self, t_sec, *, weights=None, t_now_gyr=4.567):
        """
        Radiogenic mantle heat production per unit mass from ⁴⁰K, ²³²Th, ²³⁵U, ²³⁸U.

        Parameters
        ----------
        t_sec : float or array-like
            Planetary age since formation, **in seconds**.
        weights : dict | sequence | None, optional
            Relative inventory factors for each isotope.
            – None  → Earth-like (all 1.0)
            – dict  → keys 'K40','Th232','U235','U238' with scale factors
            – list/tuple/ndarray → length-4 array in the isotope order above
        t_now_gyr : float, default 4.567
            Present-day age used for q₀ (Earth ≈ 4.567 Gyr).

        Returns
        -------
        H_total : ndarray
            Total heat production (erg s⁻¹ g⁻¹) at the supplied age(s).
        H_isotopes : dict
            Individual isotope contributions (erg s⁻¹ g⁻¹).
        """
        # Build weight vector ---------------------------------------------------
        if weights is None:
            w = np.ones_like(_Q0)
        elif isinstance(weights, dict):
            w = np.array([weights.get(k, 1.0) for k in _ISOTOPES], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != _Q0.shape:
                raise ValueError("weights must have length 4")

        # Convert constants to seconds -----------------------------------------
        tau_sec = _TAU_GYR * _SEC_IN_GYR
        t_now_sec = t_now_gyr * _SEC_IN_GYR

        # Vectorised evaluation of Eq. (43) ------------------------------------
        t = np.asarray(t_sec, dtype=float)
        H_each = w * _Q0 * np.exp(np.log(2.0) * (t_now_sec - t) / tau_sec)
        H_tot  = H_each.sum(axis=0)

        return H_tot, dict(zip(_ISOTOPES, H_each))

    # ---------------------------- Internals ------------------------------

    def _eval_with_extrap(
        self,
        rgi: RegularGridInterpolator,
        field_grid: np.ndarray,  # shape (nT, nR)
        rho: ArrayLike,
        T: ArrayLike,
    ) -> np.ndarray:
        """
        Evaluate RGI inside the (T, rho) box. For any point outside, do
        **linear extrapolation** in T and/or rho using boundary slopes.
        """
        rho_arr, T_arr = self._as_arrays(rho, T)
        out = np.empty_like(rho_arr, dtype=float)

        # Inside mask
        in_rho = (rho_arr >= self.rho_vals[0]) & (rho_arr <= self.rho_vals[-1])
        in_T   = (T_arr   >= self.T_vals[0])   & (T_arr   <= self.T_vals[-1])
        inside = in_rho & in_T

        # Evaluate inside points via RGI
        if np.any(inside):
            pts = np.column_stack([T_arr[inside], rho_arr[inside]])  # axes order (T, rho)
            out[inside] = rgi(pts)

        # Handle outside points via piecewise-linear extrapolation
        if np.any(~inside):
            out[~inside] = self._bilinear_with_linear_extrap(field_grid, rho_arr[~inside], T_arr[~inside])

        return out.reshape(np.shape(rho))

    def _eval_with_extrap_pt(
        self,
        rgi: RegularGridInterpolator,
        field_grid: np.ndarray,  # shape (nT, nR)
        P: ArrayLike,
        T: ArrayLike,
    ) -> np.ndarray:
        """
        Evaluate RGI inside the (T, P) box. For any point outside, do
        **linear extrapolation** in T and/or P using boundary slopes.
        """
        P_arr, T_arr = self._as_arrays(P, T)
        out = np.empty_like(P_arr, dtype=float)

        # Inside mask
        in_P = (P_arr >= self.P_vals_pt[0]) & (P_arr <= self.P_vals_pt[-1])
        in_T = (T_arr >= self.T_vals_pt[0]) & (T_arr <= self.T_vals_pt[-1])
        inside = in_P & in_T

        # Evaluate inside points via RGI
        if np.any(inside):
            pts = np.column_stack([T_arr[inside], P_arr[inside]])  # axes order (T, P)
            out[inside] = rgi(pts)

        # Handle outside points via piecewise-linear extrapolation
        if np.any(~inside):
            out[~inside] = self._bilinear_with_linear_extrap(field_grid, P_arr[~inside], T_arr[~inside])

        return out.reshape(np.shape(P))

    def _bilinear_with_linear_extrap(
        self,
        field_grid: np.ndarray,  # (nT, nR)
        rho_q: np.ndarray,
        T_q: np.ndarray,
    ) -> np.ndarray:
        """
        Bilinear interpolation inside; linear extrapolation in rho and/or T outside.
        Implemented using the nearest two grid lines on each relevant axis.
        """
        nT, nR = field_grid.shape
        Tv = self.T_vals
        Rv = self.rho_vals

        def val_along_rho_at_row(row_idx: int, rho_val: float) -> float:
            row = field_grid[row_idx, :]
            if rho_val <= Rv[0]:
                dy = row[1] - row[0]
                dx = Rv[1] - Rv[0]
                return row[0] + dy/dx * (rho_val - Rv[0])
            if rho_val >= Rv[-1]:
                dy = row[-1] - row[-2]
                dx = Rv[-1] - Rv[-2]
                return row[-1] + dy/dx * (rho_val - Rv[-1])
            j = np.searchsorted(Rv, rho_val) - 1
            j = np.clip(j, 0, nR-2)
            t = (rho_val - Rv[j]) / (Rv[j+1] - Rv[j])
            return (1.0 - t) * row[j] + t * row[j+1]

        result = np.empty_like(rho_q, dtype=float)
        for k in range(rho_q.size):
            r = rho_q[k]
            tT = T_q[k]

            if tT <= Tv[0]:
                j0, j1 = 0, 1
                w = (tT - Tv[0]) / (Tv[1] - Tv[0])
            elif tT >= Tv[-1]:
                j0, j1 = nT-2, nT-1
                w = (tT - Tv[-1]) / (Tv[-1] - Tv[-2])
            else:
                j1 = np.searchsorted(Tv, tT)
                j0 = j1 - 1
                w = (tT - Tv[j0]) / (Tv[j1] - Tv[j0])

            f0 = val_along_rho_at_row(j0, r)
            f1 = val_along_rho_at_row(j1, r)
            result[k] = f0 + (f1 - f0) * w

        return result

    def _dP_drho_num(self, rho: float, T: float, eps_rel: float = 1e-6) -> float:
        """Numerical ∂P/∂ρ at fixed T using a symmetric difference (with edge guards)."""
        r = float(rho)
        scale = max(abs(r), (self.rho_vals[-1] - self.rho_vals[0]) / max(1, (len(self.rho_vals) - 1)))
        h = eps_rel * max(scale, 1e-12)
        r_lo = r - h
        r_hi = r + h
        P_lo = float(self.get_p_rhot(r_lo, T))
        P_hi = float(self.get_p_rhot(r_hi, T))
        return (P_hi - P_lo) / (2.0 * h)

    def _dS_dT_num(self, P: float, T: float, eps_rel: float = 1e-6) -> float:
        """Numerical ∂S/∂T at fixed P using a symmetric difference."""
        t = float(T)
        scale = max(abs(t), (self.T_vals[-1] - self.T_vals[0]) / max(1, (len(self.T_vals) - 1)))
        h = eps_rel * max(scale, 1e-12)
        T_lo = t - h
        T_hi = t + h
        S_lo = float(self.get_s_pt(P, T_lo))
        S_hi = float(self.get_s_pt(P, T_hi))
        return (S_hi - S_lo) / (2.0 * h)

    def _initial_rho_guess(self, P_target: float, T: float) -> float:
        """Pick a robust starting guess: the rho on the global grid whose P(rho,T) is closest to target."""
        P_line = self.get_p_rhot(self.rho_vals, T)  # shape (nR,)
        j = int(np.nanargmin(np.abs(P_line - P_target)))
        return float(self.rho_vals[j])

    def _initial_T_guess(self, S_target: float, P: float) -> float:
        """Pick a starting guess: the T on the table grid whose S(P,T) is closest to target."""
        # Evaluate S across the tabulated T grid at fixed P (vectorized)
        S_line = self.get_s_pt(P, self.T_vals)  # shape (nT,)
        j = int(np.nanargmin(np.abs(S_line - S_target)))
        return float(self.T_vals[j])

    @staticmethod
    def _as_arrays(a: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A.copy(), B.copy()

    @staticmethod
    def _auto_colmap(columns) -> Dict[str, str]:
        cols = list(columns)
        low = [c.lower() for c in cols]

        def pick(parts):
            for name, lname in zip(cols, low):
                if all(p in lname for p in parts):
                    return name
            return None

        return {
            "P":   pick(["p", "gpa"]) or pick(["pressure"]) or pick(["p ("]),
            "T":   pick(["t", "(k"]) or pick(["temp"]),
            "rho": pick(["rho"]) or pick(["density"]),
            "U":   pick(["u("]) or pick(["u "]) or pick(["internal", "energy"]),
            "S":   pick(["s ("]) or pick(["entropy"]),
        }


