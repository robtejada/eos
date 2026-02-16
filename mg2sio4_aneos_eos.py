"""
mg2sio4_aneos_eos.py

Mg2SiO4 (forsterite) ANEOS wrapper with an API aligned to mgsio3_comb_eos.py,
but without explicit solid/liquid/vapor phase blending. ANEOS tables are
treated as already phase-smoothed across transitions.

Current status:
- Uses legacy rho-T ANEOS tables as the always-available backend.
- Includes explicit placeholders for regenerated P-T and S-P tables.
- When new P-T/S-P tables are added, tabulated paths are auto-enabled.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import os
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import newton, brentq, least_squares, root_scalar

from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

ArrayLike = Union[float, int, np.ndarray]

@dataclass
class Domain:
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float

class MG2SIO4_ANEOS_EOS:
    """
    Mg2SiO4 (forsterite) ANEOS EOS wrapper.

    Pressure units: GPa
    Temperature units: K
    Density units: g/cm^3
    Entropy units:
      - table native: MJ/kg/K
      - public getters: erg/g/K
    Internal energy units:
      - table native: MJ/kg
      - public getters: erg/g
    """

    def __init__(
        self,
        *,
        # rhot_arrays_path: str = "eos/aneos/forsterite_eos_arrays.npy",
        # rhot_grids_path: str = "eos/aneos/forsterite_eos_grids.npy",

        rhot_table_path: str = "eos/rock_eos/mg2sio4_aneos_RHOT.npz",
        # ------------------------------------------------------------------
        # TODO (user): replace these placeholders with regenerated PT/SP grids
        # after inverting/saving updated forsterite ANEOS tables.
        # ------------------------------------------------------------------
        pt_table_path: str = "eos/rock_eos/mg2sio4_aneos_PT.npz",
        sp_table_path: str = "eos/rock_eos/mg2sio4_aneos_SP.npz",
        melt_transition_width_K: float = 50.0,
    ):
        self.L = 7.322e5 * (u.J/u.kg).to('erg/g')  # latent heat of fusion of the mantle
        # Unit conversions
        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.MJkg_to_ergg = float((1.0 * u.MJ / u.kg).to(u.erg / u.g).value)
        self.MJkgK_to_erggK = float((1.0 * u.MJ / u.kg / u.K).to(u.erg / u.g / u.K).value)
        self.ergg_to_MJkg = 1.0 / self.MJkg_to_ergg
        self.erggK_to_MJkgK = 1.0 / self.MJkgK_to_erggK

        self.dyn_to_Pa = (u.dyn / u.cm**2).to("Pa")
        self.dyn_to_GPa = (u.dyn / u.cm**2).to("GPa")
        self.GPa_to_dyncm2 = u.GPa.to("dyn/cm^2")
        self.melt_transition_width_K = float(melt_transition_width_K)
        if self.melt_transition_width_K <= 0.0:
            raise ValueError("melt_transition_width_K must be > 0.")

        # Always-available rho-T ANEOS backend

        self.rhot_data = np.load(rhot_table_path)

        self.rhovals, self.tvals = self.rhot_data["rhovals"], self.rhot_data["tvals"]
        self.rhovals = np.asarray(self.rhovals, dtype=float) # g/cc
        self.tvals = np.asarray(self.tvals, dtype=float) # K
        self.logrhovals = np.log10(np.clip(self.rhovals, 1e-300, None))
        self.logtvals = np.log10(np.clip(self.tvals, 1e-300, None))
        self.domain = Domain(
            rho_min=float(np.min(self.rhovals)),
            rho_max=float(np.max(self.rhovals)),
            T_min=float(np.min(self.tvals)),
            T_max=float(np.max(self.tvals)),
        )

        s_grid_MJkgK, p_grid_GPa, u_grid_MJkg = self.rhot_data["s_grid"], self.rhot_data["p_grid"], self.rhot_data["u_grid"]
        cv_grid_MJkgK = self.rhot_data["cv_grid"]
        self.s_grid_MJkgK = self._match_grid_shape(
            np.asarray(s_grid_MJkgK, dtype=float),
            nx=self.tvals.size,
            ny=self.rhovals.size,
            name="s_grid",
        )
        self.p_grid_GPa = self._match_grid_shape(
            np.asarray(p_grid_GPa, dtype=float),
            nx=self.tvals.size,
            ny=self.rhovals.size,
            name="p_grid",
        )
        self.u_grid_MJkg = self._match_grid_shape(
            np.asarray(u_grid_MJkg, dtype=float),
            nx=self.tvals.size,
            ny=self.rhovals.size,
            name="u_grid",
        )
        self.cv_grid_MJkgK = self._match_grid_shape(
            np.asarray(cv_grid_MJkgK, dtype=float),
            nx=self.tvals.size,
            ny=self.rhovals.size,
            name="cv_grid",
        )

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._p_rgi_rhot = RGI((self.tvals, self.rhovals), self.p_grid_GPa, **rgi_kwargs)
        self._s_rgi_rhot = RGI((self.tvals, self.rhovals), self.s_grid_MJkgK, **rgi_kwargs)
        self._u_rgi_rhot = RGI((self.tvals, self.rhovals), self.u_grid_MJkg, **rgi_kwargs)
        self._cv_rgi_rhot = RGI((self.tvals, self.rhovals), self.cv_grid_MJkgK, **rgi_kwargs)

        # Optional P-T and S-P tables (placeholders by default).
        self._has_pt_table = True
        self._has_sp_table = True

        self.pt_table_path = pt_table_path
        self.sp_table_path = sp_table_path

        if os.path.exists(self.pt_table_path):
            self._load_pt_table(self.pt_table_path)
        else:
            warnings.warn(
                f"PT table not found at '{self.pt_table_path}'. Falling back to inversion backend.",
                RuntimeWarning,
            )

        if os.path.exists(self.sp_table_path):
            self._load_sp_table(self.sp_table_path)
        else:
            warnings.warn(
                f"SP table not found at '{self.sp_table_path}'. Falling back to inversion backend.",
                RuntimeWarning,
            )

    # ---------- compatibility helpers ----------

    @staticmethod
    def _as_arrays(a, b):
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A, B

    @staticmethod
    def _match_grid_shape(arr: np.ndarray, *, nx: int, ny: int, name: str) -> np.ndarray:
        if arr.shape == (nx, ny):
            return arr
        if arr.shape == (ny, nx):
            return arr.T
        raise ValueError(f"{name} has shape {arr.shape}, expected {(nx, ny)} or {(ny, nx)}.")

    @staticmethod
    def _as_array_single(x):
        return np.array(x, ndmin=1, dtype=float)

    @staticmethod
    def _broadcast(P, Q):
        scalar = np.isscalar(P) and np.isscalar(Q)
        P_arr = np.array(P, ndmin=1, dtype=float)
        Q_arr = np.array(Q, ndmin=1, dtype=float)
        if P_arr.shape != Q_arr.shape:
            P_arr, Q_arr = np.broadcast_arrays(P_arr, Q_arr)
        return scalar, P_arr, Q_arr

    @staticmethod
    def _entropy_eps(s_ref):
        return 1e-12 * np.maximum(1.0, np.abs(s_ref))

    def _interp(self, rgi, P_arr, Q_arr):
        pts = np.stack((P_arr.ravel(), Q_arr.ravel()), axis=-1)
        return rgi(pts).reshape(P_arr.shape)

    def _entropy_to_cgs(self, s_in, *, s_units: str = "kbbar"):
        su = str(s_units).lower()
        s_arr = np.asarray(s_in, dtype=float)
        if su in ("kbbar", "kb/baryon", "kbperbaryon"):
            return s_arr / self.erg_to_kbbar
        if su in ("cgs", "erg/g/k", "erg/g/kelvin"):
            return s_arr
        if su in ("native", "mjkgk", "mj/kg/k"):
            return s_arr * self.MJkgK_to_erggK
        raise ValueError("s_units must be one of {'kbbar','cgs','native'}")

    def _entropy_cgs_to_sp_table(self, s_cgs):
        return np.asarray(s_cgs, dtype=float) * self.erggK_to_MJkgK

    def _entropy_to_sp_table(self, s_in, *, s_units: str = "kbbar"):
        return self._entropy_cgs_to_sp_table(self._entropy_to_cgs(s_in, s_units=s_units))

    # ---------- optional PT/SP loaders ----------

    def _load_pt_table(self, path: str):
        data = np.load(path)
        self.P_vals_pt = np.asarray(data["pvals_pt"], dtype=float)
        self.T_vals_pt = np.asarray(data["tvals_pt"], dtype=float)
        self.rho_vals_pt = self._match_grid_shape(
            np.asarray(data["rho_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="rho_grid_pt",
        )
        self.u_vals_pt = self._match_grid_shape(
            np.asarray(data["u_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="u_grid_pt",
        )
        self.s_vals_pt = self._match_grid_shape(
            np.asarray(data["s_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="s_grid_pt",
        )

        self.cv_vals_pt = self._match_grid_shape(
            np.asarray(data["cv_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="cv_grid_pt",       
        )

        self.cp_vals_pt = self._match_grid_shape(
            np.asarray(data["cp_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="cp_grid_pt",       
        )

        self.alpha_vals_pt = self._match_grid_shape(
            np.asarray(data["alpha_grid_pt"], dtype=float),
            nx=self.P_vals_pt.size,
            ny=self.T_vals_pt.size,
            name="alpha_grid_pt",       
        )

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._rho_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.rho_vals_pt, **rgi_kwargs)
        self._u_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.u_vals_pt, **rgi_kwargs)
        self._s_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.s_vals_pt, **rgi_kwargs)
        self._cv_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.cv_vals_pt, **rgi_kwargs)
        self._cp_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.cp_vals_pt, **rgi_kwargs)
        self._alpha_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.alpha_vals_pt, **rgi_kwargs)

    def _load_sp_table(self, path: str):
        data = np.load(path)
        self.S_vals_sp = np.asarray(data["svals_sp"], dtype=float)
        self.P_vals_sp = np.asarray(data["pvals_sp"], dtype=float)
        self.T_vals_sp = self._match_grid_shape(
            np.asarray(data["t_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="t_grid_sp",
        )
        self.rho_vals_sp = self._match_grid_shape(
            np.asarray(data["rho_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="rho_grid_sp",
        )
        self.u_vals_sp = self._match_grid_shape(
            np.asarray(data["u_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="u_grid_sp",
        )

        self.cv_vals_sp = self._match_grid_shape(
            np.asarray(data["cv_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="cv_grid_sp",
        )

        self.cp_vals_sp = self._match_grid_shape(
            np.asarray(data["cp_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="cp_grid_sp",
        )

        self.alpha_vals_sp = self._match_grid_shape(
            np.asarray(data["alpha_grid_sp"], dtype=float),
            nx=self.S_vals_sp.size,
            ny=self.P_vals_sp.size,
            name="alpha_grid_sp",
        )

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._t_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.T_vals_sp, **rgi_kwargs)
        self._rho_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.rho_vals_sp, **rgi_kwargs)
        self._u_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.u_vals_sp, **rgi_kwargs)

        self._cp_rgi_sp = None
        self._cv_rgi_sp = None
        self._alpha_rgi_sp = None
        if "cp_grid_sp" in data.files:
            cp_grid = self._match_grid_shape(
                np.asarray(data["cp_grid_sp"], dtype=float),
                nx=self.S_vals_sp.size,
                ny=self.P_vals_sp.size,
                name="cp_grid_sp",
            )
            self._cp_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), cp_grid, **rgi_kwargs)
        if "cv_grid_sp" in data.files:
            cv_grid = self._match_grid_shape(
                np.asarray(data["cv_grid_sp"], dtype=float),
                nx=self.S_vals_sp.size,
                ny=self.P_vals_sp.size,
                name="cv_grid_sp",
            )
            self._cv_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), cv_grid, **rgi_kwargs)
        if "alpha_grid_sp" in data.files:
            alpha_grid = self._match_grid_shape(
                np.asarray(data["alpha_grid_sp"], dtype=float),
                nx=self.S_vals_sp.size,
                ny=self.P_vals_sp.size,
                name="alpha_grid_sp",
            )
            self._alpha_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), alpha_grid, **rgi_kwargs)

        self._has_sp_table = True

    # ---------- rho-T native getters ----------

    def get_p_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        vals = self._p_rgi_rhot(np.column_stack((T_arr.ravel(), rho_arr.ravel()))).reshape(rho_arr.shape)
        return float(vals) if scalar else vals

    def get_s_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        vals = self._s_rgi_rhot(np.column_stack((T_arr.ravel(), rho_arr.ravel()))).reshape(rho_arr.shape)
        return float(vals) if scalar else vals

    def get_u_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        vals = self._u_rgi_rhot(np.column_stack((T_arr.ravel(), rho_arr.ravel()))).reshape(rho_arr.shape)
        return float(vals) if scalar else vals
    
    def get_cv_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        vals = self._cv_rgi_rhot(np.column_stack((T_arr.ravel(), rho_arr.ravel()))).reshape(rho_arr.shape)
        return float(vals) if scalar else vals

    def get_p_rhot(self, rho, T):
        return self.get_p_rhot_tab(rho, T)

    def get_s_rhot(self, rho, T):
        vals = np.asarray(self.get_s_rhot_tab(rho, T), dtype=float) * self.MJkgK_to_erggK
        return float(vals) if np.isscalar(rho) and np.isscalar(T) else vals

    def get_u_rhot(self, rho, T):
        vals = np.asarray(self.get_u_rhot_tab(rho, T), dtype=float) * self.MJkg_to_ergg
        return float(vals) if np.isscalar(rho) and np.isscalar(T) else vals

    def get_cv_rhot(self, rho, T):
        vals = np.asarray(self.get_cv_rhot_tab(rho, T), dtype=float) * self.MJkgK_to_erggK
        return float(vals) if np.isscalar(rho) and np.isscalar(T) else vals

    # ---------- optional PT/SP table getters (native table units) ----------

    def get_rho_pt_tab(self, P, T):
        if not self._has_pt_table:
            raise RuntimeError("PT table is not loaded.")
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._rho_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
        return float(vals) if scalar else vals

    def get_s_pt_tab(self, P, T):
        if not self._has_pt_table:
            raise RuntimeError("PT table is not loaded.")
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._s_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
        return float(vals) if scalar else vals

    def get_u_pt_tab(self, P, T):
        if not self._has_pt_table:
            raise RuntimeError("PT table is not loaded.")
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._u_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
        return float(vals) if scalar else vals

    def get_t_sp_tab(self, S_tab, P):
        if not self._has_sp_table:
            raise RuntimeError("SP table is not loaded.")
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        vals = self._t_rgi_sp(np.column_stack((S_arr.ravel(), P_arr.ravel()))).reshape(S_arr.shape)
        return float(vals) if scalar else vals

    def get_rho_sp_tab(self, S_tab, P):
        if not self._has_sp_table:
            raise RuntimeError("SP table is not loaded.")
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        vals = self._rho_rgi_sp(np.column_stack((S_arr.ravel(), P_arr.ravel()))).reshape(S_arr.shape)
        return float(vals) if scalar else vals

    def get_u_sp_tab(self, S_tab, P):
        if not self._has_sp_table:
            raise RuntimeError("SP table is not loaded.")
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        vals = self._u_rgi_sp(np.column_stack((S_arr.ravel(), P_arr.ravel()))).reshape(S_arr.shape)
        return float(vals) if scalar else vals

    # ---------- RHOT-derived thermodynamic derivatives ----------

    def _dP_dT_rhot_num(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        p_hi = self.get_p_rhot_tab(rho_arr, T_hi)
        p_lo = self.get_p_rhot_tab(rho_arr, T_lo)
        return (p_hi - p_lo) / (T_hi - T_lo)

    def _dP_drho_rhot_num(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        rho_hi = rho_arr * (1.0 + eps_rel)
        rho_lo = np.maximum(rho_arr * (1.0 - eps_rel), 1e-99)
        p_hi = self.get_p_rhot_tab(rho_hi, T_arr)
        p_lo = self.get_p_rhot_tab(rho_lo, T_arr)
        return (p_hi - p_lo) / (rho_hi - rho_lo)

    def get_alpha_rhot(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        dP_dT = self._dP_dT_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
        dP_drho = self._dP_drho_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
        alpha = dP_dT / np.maximum(rho_arr * dP_drho, 1e-99)
        return float(alpha) if np.isscalar(rho) and np.isscalar(T) else alpha

    def get_cp_rhot(self, rho, T, eps_rel=1e-3, method="cv_relation"):
        rho_arr, T_arr = self._as_arrays(rho, T)
        method_l = str(method).lower()

        if method_l == "cv_relation":
            cv = self.get_cv_rhot(rho_arr, T_arr)
            alpha = self.get_alpha_rhot(rho_arr, T_arr, eps_rel=eps_rel)
            dP_drho_gpa = self._dP_drho_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
            dP_drho_cgs = dP_drho_gpa * self.GPa_to_dyncm2
            cp = cv + T_arr * alpha * alpha * dP_drho_cgs
        elif method_l == "entropy":
            T_hi = T_arr * (1.0 + eps_rel)
            T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
            rho_hi = rho_arr * (1.0 + eps_rel)
            rho_lo = np.maximum(rho_arr * (1.0 - eps_rel), 1e-99)

            s_hi = self.get_s_rhot(rho_arr, T_hi)
            s_lo = self.get_s_rhot(rho_arr, T_lo)
            dS_dT_rho = (s_hi - s_lo) / (T_hi - T_lo)

            s_rho_hi = self.get_s_rhot(rho_hi, T_arr)
            s_rho_lo = self.get_s_rhot(rho_lo, T_arr)
            dS_drho_T = (s_rho_hi - s_rho_lo) / (rho_hi - rho_lo)

            dP_dT = self._dP_dT_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
            dP_drho = self._dP_drho_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
            dS_dT_P = dS_dT_rho - dS_drho_T * (dP_dT / np.maximum(dP_drho, 1e-99))
            cp = T_arr * dS_dT_P
        else:
            raise ValueError("method must be one of {'cv_relation', 'entropy'}")

        return float(cp) if np.isscalar(rho) and np.isscalar(T) else cp

    # ---------- melt-curve hooks ----------

    def _melt_fraction_PT(self, P_arr, T_arr):
        """
        Smooth melt-fraction proxy in [0, 1] across Tm(P):
            x_melt = 0.5 * [1 + tanh((T - Tm)/dT)]
        with dT = 50 K by default.
        """
        Tm = self.get_T_melt(P_arr)
        arg = (T_arr - Tm) / self.melt_transition_width_K
        x_melt = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(x_melt, 0.0, 1.0)

    def _get_rho_pt_safe(self, P_arr, T_arr):
        # Prefer PT table if loaded, otherwise always fall back to inversion.
        use_tab = bool(getattr(self, "_has_pt_table", False) and hasattr(self, "_rho_rgi_pt"))
        return self.get_rho_pt(P_arr, T_arr, tab=use_tab)

    def get_T_melt(self, P):
        """
        Forsterite melt curve from Presnall & Walter (1993), Eq. (12):
            P(GPa) = 2.44 * ((T/2171 K)^11.4 - 1)
        Rearranged to T(P):
            Tm(P) = 2171 K * (1 + P/2.44 GPa)^(1/11.4)
        """
        P_arr = np.asarray(P, dtype=float)
        base = np.maximum(1.0 + P_arr / 2.44, 1e-12)
        Tm = 2171.0 * np.power(base, 1.0 / 11.4)
        return float(Tm) if np.isscalar(P) else Tm

    def get_S_liq_at_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        out = np.full_like(P_arr, np.nan, dtype=float)
        return float(out[0]) if np.isscalar(P) else out

    def get_S_sol_at_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        out = np.full_like(P_arr, np.nan, dtype=float)
        return float(out[0]) if np.isscalar(P) else out

    # ---------- PT getters ----------

    def _initial_rho_guess(self, P, T):
        p = float(P)
        t = float(T)

        iT = int(np.argmin(np.abs(self.tvals - t)))
        p_row = np.asarray(self.p_grid_GPa[iT], dtype=float)
        rho_row = np.asarray(self.rhovals, dtype=float)

        if np.all(np.diff(p_row) >= 0.0):
            return float(np.interp(p, p_row, rho_row, left=rho_row[0], right=rho_row[-1]))
        if np.all(np.diff(p_row) <= 0.0):
            return float(np.interp(p, p_row[::-1], rho_row[::-1], left=rho_row[0], right=rho_row[-1]))

        j = int(np.argmin(np.abs(p_row - p)))
        return float(rho_row[j])

    def _dP_drho_num(self, rho, T, eps_rel=1e-6):
        r = float(rho)
        t = float(T)
        dr = max(abs(r) * float(eps_rel), 1e-12)
        r_lo = max(r - dr, 1e-20)
        r_hi = r + dr
        if r_hi <= r_lo:
            r_hi = r_lo * (1.0 + 1e-6)
        p_hi = float(self.get_p_rhot(r_hi, t))
        p_lo = float(self.get_p_rhot(r_lo, t))
        return (p_hi - p_lo) / (r_hi - r_lo)

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
          2) If Newton fails, fall back to Brent's method (brentq) on a positive bracket.
        """
        P_arr, T_arr = self._as_arrays(P, T)
        out = np.empty_like(P_arr, dtype=float)

        if rho0 is not None:
            rho0_arr, _ = self._as_arrays(rho0, T_arr)
        else:
            rho0_arr = None

        rho_min = max(1e-20, float(np.min(self.rhovals)))
        rho_max = max(float(np.max(self.rhovals)) * 10.0, 1000.0)

        for idx in np.ndindex(P_arr.shape):
            p = float(P_arr[idx])
            t = float(T_arr[idx])

            def f(r):
                return float(self.get_p_rhot(r, t) - p)

            def fprime(r):
                return float(self._dP_drho_num(r, t, eps_rel=dPdrho_eps_rel))

            r_guess = (
                float(rho0_arr[idx])
                if rho0_arr is not None
                else float(self._initial_rho_guess(p, t))
            )

            solved = False
            if newton_first:
                try:
                    r_newton = newton(f, r_guess, fprime=fprime, tol=tol, maxiter=maxiter)
                    if np.isfinite(r_newton) and r_newton > 0.0:
                        out[idx] = float(r_newton)
                        solved = True
                except Exception:
                    solved = False

            if not solved:
                a = rho_min
                b = rho_max
                fa = f(a)
                fb = f(b)
                if np.isnan(fa) or np.isnan(fb) or fa * fb > 0.0:
                    raise ValueError(
                        f"brentq: target P={p:g} at T={t:g} K not bracketed on "
                        f"[rho_min={a:g}, rho_max={b:g}]. "
                        f"f(rho_min)={fa:g}, f(rho_max)={fb:g}."
                    )
                r_brent = brentq(f, a, b, xtol=tol, maxiter=maxiter)
                out[idx] = float(r_brent)

        return out.reshape(P_arr.shape)

    def get_rho_pt(self, P, T, tab=True, **kwargs):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            rho = self.get_rho_pt_tab(P_arr, T_arr)
        else:
            rho = self.get_rho_pt_inv(P_arr, T_arr, **kwargs)

        return float(rho) if np.isscalar(P) and np.isscalar(T) else rho

    def get_s_pt(self, P, T, tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            s_cgs = self.get_s_pt_tab(P_arr, T_arr)
            #s_cgs = s_MJkgK * self.MJkgK_to_erggK
            #return float(s_cgs) if np.isscalar(P) and np.isscalar(T) else s_cgs
            return s_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        #s_MJkgK = self.get_s_rhot_tab(np.asarray(rho, dtype=float), T_arr)
        s_cgs = self.get_s_rhot(np.asarray(rho, dtype=float), T_arr)
        return float(s_cgs) if np.isscalar(P) and np.isscalar(T) else s_cgs

    def get_u_pt(self, P, T, tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            u_cgs = self.get_u_pt_tab(P_arr, T_arr)
            return float(u_cgs) if np.isscalar(P) and np.isscalar(T) else u_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        #u_MJkg = self.get_u_rhot_tab(np.asarray(rho, dtype=float), T_arr)
        u_cgs = self.get_u_rhot(np.asarray(rho, dtype=float), T_arr)
        return float(u_cgs) if np.isscalar(P) and np.isscalar(T) else u_cgs

    # ---------- derived PT properties ----------

    def get_alpha_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._alpha_rgi_pt is not None:
            vals = self._alpha_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            return float(vals) if np.isscalar(P) and np.isscalar(T) else vals

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        alpha = self.get_alpha_rhot(rho, T_arr, eps_rel=eps_rel)
        return float(alpha) if np.isscalar(P) and np.isscalar(T) else alpha

    def get_cp_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._cp_rgi_pt is not None:
            #cp_MJkgK = self._cp_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            cp_cgs = self._cp_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            return float(cp_cgs) if np.isscalar(P) and np.isscalar(T) else cp_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        cp = self.get_cp_rhot(rho, T_arr, eps_rel=eps_rel)
        return float(cp) if np.isscalar(P) and np.isscalar(T) else cp

    def get_cv_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._cv_rgi_pt is not None:
            #cv_MJkgK = self._cv_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            cv_cgs = self._cv_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            return float(cv_cgs) if np.isscalar(P) and np.isscalar(T) else cv_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        cv = self.get_cv_rhot(rho, T_arr)
        return float(cv) if np.isscalar(P) and np.isscalar(T) else cv

    def get_alpha_x(self, P, T, rho, eps_rel=1e-3, dx_floor=1e-6, x_floor=1e-4, dT_melt=None):
        """
        Effective compositional expansivity-like coefficient for melt fraction:
            alpha_x = rho * (1/rho_melt - 1/rho_solid)

        For Mg2SiO4 ANEOS we do not have separate solid/liquid EOS objects, so
        we estimate phase-endpoint densities using two PT states around the melt
        curve at fixed pressure:
            rho_solid ~ rho(P, Tm(P) - dT)
            rho_melt  ~ rho(P, Tm(P) + dT)

        This avoids the unstable division by dx/dT when x_melt saturates at 0 or 1.
        """
        del dx_floor, x_floor  # retained only for backward-compatible call signatures
        scalar, P_arr, T_arr = self._broadcast(P, T)

        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.shape != P_arr.shape:
            rho_arr = np.broadcast_to(rho_arr, P_arr.shape)

        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)

        if dT_melt is None:
            # Physically motivated default probe width: latent-heat temperature scale.
            cp_tm = np.asarray(self.get_cp_pt(P_arr, np.clip(Tm, self.domain.T_min, self.domain.T_max), tab=True), dtype=float)
            dT = self.L / np.maximum(cp_tm, 1e-30)
            dT = np.clip(
                dT,
                2.0 * self.melt_transition_width_K,
                3000.0,
            )
            dT = np.maximum(dT, eps_rel * np.maximum(Tm, 1.0))
        else:
            dT = np.asarray(dT_melt, dtype=float)
            if dT.shape == ():
                dT = np.full(P_arr.shape, float(dT), dtype=float)
            elif dT.shape != P_arr.shape:
                dT = np.broadcast_to(dT, P_arr.shape)
            dT = np.maximum(dT, 1.0)

        Tmin = float(self.domain.T_min)
        Tmax = float(self.domain.T_max)
        T_solid = np.clip(Tm - dT, Tmin, Tmax)
        T_melt = np.clip(Tm + dT, Tmin, Tmax)

        rho_solid = self._get_rho_pt_safe(P_arr, T_solid)
        rho_melt = self._get_rho_pt_safe(P_arr, T_melt)

        alpha_x = rho_arr * (
            1.0 / np.maximum(np.asarray(rho_melt, dtype=float), 1e-99)
            - 1.0 / np.maximum(np.asarray(rho_solid, dtype=float), 1e-99)
        )

        return alpha_x if not scalar else float(alpha_x)

    # ---------- SP inversion ----------

    def get_t_sp_inv(
        self,
        S_target,
        P_target,
        *,
        s_units="kbbar",
        T_guess=None,
        bounds_T: Optional[Tuple[float, float]] = None,
        secant_maxiter=30,
        secant_rtol=1e-10,
        dy0=1e-3,
        bracket_factor=1.3,
        expand_steps=12,
        expand_factor=2.0,
        brent_rtol=1e-10,
        brent_xtol=1e-6,
        brent_maxiter=200,
        return_diagnostics=False,
        fail_value=np.nan,
    ):
        S_arr = np.asarray(S_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        shape = S_arr.shape

        S_goal = self._entropy_to_cgs(S_arr, s_units=s_units)

        if bounds_T is None:
            T_lo = max(1e-12, float(self.domain.T_min))
            T_hi = float(self.domain.T_max)
        else:
            T_lo, T_hi = map(float, bounds_T)
        if T_lo <= 0 or T_hi <= T_lo:
            raise ValueError("bounds_T must satisfy 0 < T_lo < T_hi")
        y_min = np.log(T_lo)
        y_max = np.log(T_hi)

        Tout = np.full(shape, fail_value, dtype=float)

        diag = None
        if return_diagnostics:
            diag = {
                "success": np.zeros(shape, dtype=bool),
                "method": np.empty(shape, dtype=object),
                "message": np.empty(shape, dtype=object),
                "n_expand": np.zeros(shape, dtype=int),
                "nfev": np.zeros(shape, dtype=int),
                "iterations": np.zeros(shape, dtype=int),
            }

        def g_of_y(P, y, Sg):
            y = float(np.clip(y, y_min, y_max))
            T = float(np.exp(y))
            Sm = self.get_s_pt(P, T, tab=False)
            Sm = float(np.asarray(Sm).reshape(-1)[0])
            if not np.isfinite(Sm):
                return np.nan
            return Sm - Sg

        def nudge_to_finite(P, Sg, y_guess):
            y = float(np.clip(y_guess, y_min, y_max))
            v = g_of_y(P, y, Sg)
            if np.isfinite(v):
                return y, v
            for step in (0.2, -0.2, 0.5, -0.5, 1.0, -1.0):
                yt = float(np.clip(y + step, y_min, y_max))
                vt = g_of_y(P, yt, Sg)
                if np.isfinite(vt):
                    return yt, vt
            return y, np.nan

        if T_guess is None:
            T_guess_arr = np.full(shape, np.sqrt(T_lo * T_hi), dtype=float)
        else:
            T_guess_arr = np.asarray(T_guess, dtype=float)
            T_guess_arr, _ = np.broadcast_arrays(T_guess_arr, P_arr)
            T_guess_arr = np.clip(T_guess_arr, T_lo, T_hi)

        T_prev = None

        for idx in np.ndindex(shape):
            P = float(P_arr[idx])
            Sg = float(S_goal[idx])

            if not (np.isfinite(P) and np.isfinite(Sg)) or P <= 0:
                if return_diagnostics:
                    diag["method"][idx] = "none"
                    diag["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            Tg = float(T_guess_arr[idx] if (T_prev is None or not np.isfinite(T_prev)) else T_prev)
            y0 = float(np.log(np.clip(Tg, T_lo, T_hi)))
            y0, g0 = nudge_to_finite(P, Sg, y0)

            secant_ok = False
            if np.isfinite(g0):
                y1 = float(np.clip(y0 + dy0, y_min, y_max))
                y1, g1 = nudge_to_finite(P, Sg, y1)
                if np.isfinite(g1):
                    try:
                        sol = root_scalar(
                            lambda yy: g_of_y(P, yy, Sg),
                            method="secant",
                            x0=y0,
                            x1=y1,
                            rtol=secant_rtol,
                            maxiter=int(secant_maxiter),
                        )
                        if sol.converged and np.isfinite(sol.root):
                            y_sol = float(np.clip(sol.root, y_min, y_max))
                            T_sol = float(np.exp(y_sol))
                            gres = g_of_y(P, y_sol, Sg)
                            if np.isfinite(gres):
                                Tout[idx] = T_sol
                                T_prev = T_sol
                                secant_ok = True
                                if return_diagnostics:
                                    diag["success"][idx] = True
                                    diag["method"][idx] = "secant(logT)"
                                    diag["message"][idx] = "OK"
                                    diag["iterations"][idx] = getattr(sol, "iterations", 0)
                                    diag["nfev"][idx] = getattr(sol, "function_calls", 0)
                    except Exception as e:
                        if return_diagnostics:
                            diag["method"][idx] = "secant(logT)"
                            diag["message"][idx] = f"Secant exception: {e}"

            if secant_ok:
                continue

            try:
                yc = float(np.log(np.clip(Tg, T_lo, T_hi)))
                gc = g_of_y(P, yc, Sg)
                if not np.isfinite(gc):
                    yc, gc = nudge_to_finite(P, Sg, yc)

                if not np.isfinite(gc):
                    raise RuntimeError("Could not find finite residual near initial guess.")

                yl = yr = yc
                gl = gr = gc
                n_expand = 0
                bracket_ok = False
                step = np.log(bracket_factor)

                for k in range(int(expand_steps)):
                    n_expand = k + 1
                    yl = float(max(y_min, yc - step))
                    yr = float(min(y_max, yc + step))
                    gl = g_of_y(P, yl, Sg)
                    gr = g_of_y(P, yr, Sg)
                    if np.isfinite(gl) and np.isfinite(gr) and gl * gr <= 0.0:
                        bracket_ok = True
                        break
                    step *= expand_factor

                if not bracket_ok:
                    glb = g_of_y(P, y_min, Sg)
                    grb = g_of_y(P, y_max, Sg)
                    if np.isfinite(glb) and np.isfinite(grb) and glb * grb <= 0.0:
                        yl, yr, gl, gr = y_min, y_max, glb, grb
                        bracket_ok = True

                if not bracket_ok:
                    raise RuntimeError(
                        "Failed to bracket root in logT. "
                        f"P={P:.6g}, Sg={Sg:.6g}, g(ymin)={g_of_y(P, y_min, Sg)}, g(ymax)={g_of_y(P, y_max, Sg)}"
                    )

                solb = root_scalar(
                    lambda yy: g_of_y(P, yy, Sg),
                    method="brentq",
                    bracket=(yl, yr),
                    rtol=brent_rtol,
                    xtol=brent_xtol,
                    maxiter=int(brent_maxiter),
                )
                if not solb.converged or not np.isfinite(solb.root):
                    raise RuntimeError("Brent solver did not converge.")

                y_root = float(np.clip(solb.root, y_min, y_max))
                T_root = float(np.exp(y_root))

                Tout[idx] = T_root
                T_prev = T_root

                if return_diagnostics:
                    diag["success"][idx] = True
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = "OK"
                    diag["n_expand"][idx] = n_expand
                    diag["iterations"][idx] = getattr(solb, "iterations", 0)
                    diag["nfev"][idx] = getattr(solb, "function_calls", 0)

            except Exception as e:
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = f"Brent exception: {e}"

        if return_diagnostics:
            return Tout, diag
        return Tout

    # ---------- SP getters ----------

    def get_rhot_sp_2d_inv(
        self,
        s_target,
        P_target,
        *,
        s_units: str = "kbbar",          # "kbbar", "cgs", or "native" (MJ/kg/K)
        guess="auto",                    # "auto" or (rho_guess, T_guess) in (g/cm^3, K)
        T_guess0: Optional[float] = None,
        bounds_rho: Optional[Tuple[float, float]] = None,   # (g/cm^3, g/cm^3)
        bounds_T:   Optional[Tuple[float, float]] = None,   # (K, K)
        xtol: float = 1e-10,
        ftol: float = 1e-10,
        gtol: float = 1e-10,
        max_nfev: int = 200,
        fail_value=np.nan,
        return_diagnostics: bool = False,
    ):
        """
        Coupled 2-D inversion using the native RHOT basis:
            P(rho, T) = P_target   [GPa]
            S(rho, T) = s_target   [units set by s_units]

        Warm-start behavior for array inputs:
        - First element uses `guess` (or "auto")
        - Each subsequent element uses the previous converged (rho, T) as the initial guess

        Parameters
        ----------
        s_target, P_target : float or array-like
            Target entropy and pressure.
        s_units : {"kbbar","cgs","native"}
            - "kbbar": k_B/baryon
            - "cgs"  : erg/g/K
            - "native": MJ/kg/K
        guess : "auto" or tuple
            If tuple, interpreted as (rho_guess[g/cm^3], T_guess[K]).
            If "auto", try SP tables for a seed (if present), then fallback to rho(P,Tseed).
        T_guess0 : float, optional
            Seed temperature used for fallback auto-guess if SP-table seed fails.
        bounds_rho, bounds_T : optional
            Bounds for rho and T. Defaults to your table domains with a small cushion.

        Returns
        -------
        rho_sol, T_sol : ndarray
            Broadcasted to the shape of inputs.
        diagnostics : dict (optional)
            Only if return_diagnostics=True.
        """

        # ---------------- broadcast inputs ----------------
        s_arr = np.asarray(s_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)
        shape = s_arr.shape

        # ---------------- convert entropy target -> cgs (erg/g/K) ----------------
        s_cgs = self._entropy_to_cgs(s_arr, s_units=s_units)

        # ---------------- default bounds ----------------
        if bounds_rho is None:
            rho_lo = max(1e-12, float(self.domain.rho_min) - 0.05 * abs(self.domain.rho_min))
            rho_hi = float(self.domain.rho_max) + 0.05 * abs(self.domain.rho_max)
            bounds_rho = (rho_lo, rho_hi)
        if bounds_T is None:
            T_lo = max(1e-12, float(self.domain.T_min) - 0.05 * abs(self.domain.T_min))
            T_hi = float(self.domain.T_max) + 0.05 * abs(self.domain.T_max)
            bounds_T = (T_lo, T_hi)

        rho_lo, rho_hi = map(float, bounds_rho)
        T_lo,   T_hi   = map(float, bounds_T)
        if rho_lo <= 0 or T_lo <= 0:
            raise ValueError("bounds_rho and bounds_T must be > 0 on the low end.")

        # log-space bounds (ensures positivity)
        lb = np.array([np.log(rho_lo), np.log(T_lo)], dtype=float)
        ub = np.array([np.log(rho_hi), np.log(T_hi)], dtype=float)

        rho_out = np.full(shape, fail_value, dtype=float)
        T_out   = np.full(shape, fail_value, dtype=float)

        info = None
        if return_diagnostics:
            info = {
                "success": np.zeros(shape, dtype=bool),
                "cost": np.full(shape, np.nan),
                "nfev": np.full(shape, np.nan),
                "resid_P_frac": np.full(shape, np.nan),
                "resid_S_frac": np.full(shape, np.nan),
                "message": np.empty(shape, dtype=object),
            }

        # ---------------- choose initial seed for the first element ----------------
        if guess == "auto":
            # First try to seed from your SP tables (fast + usually very close)
            rho_guess_cur = None
            T_guess_cur   = None

            # fallback temperature seed
            Tseed = float(np.sqrt(T_lo * T_hi) if T_guess0 is None else T_guess0)
            Tseed = min(max(Tseed, T_lo), T_hi)

        else:
            rho_seed, T_seed = map(float, guess)
            rho_guess_cur = min(max(rho_seed, rho_lo), rho_hi)
            T_guess_cur   = min(max(T_seed,   T_lo),   T_hi)
            Tseed = None  # unused

        # ---------------- iterate with warm-start continuation ----------------
        for idx in np.ndindex(shape):
            Pt = float(P_arr[idx])
            St = float(s_cgs[idx])

            if not (np.isfinite(Pt) and np.isfinite(St)) or Pt <= 0:
                if return_diagnostics:
                    info["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            # AUTO guess logic for the *first* unsolved element (and also if previous was None)
            if guess == "auto" and (rho_guess_cur is None or T_guess_cur is None):
                seeded = False

                # 1) Seed from SP tables if possible (does NOT evaluate residuals from SP tables;
                #    it’s only used as an initial guess)
                if self._has_sp_table:
                    try:
                        St_tab = float(self._entropy_cgs_to_sp_table(St))
                        T_try = float(self.get_t_sp_tab(St_tab, Pt))
                        rho_try = float(self.get_rho_sp_tab(St_tab, Pt))
                        if np.isfinite(T_try) and np.isfinite(rho_try) and (T_try > 0) and (rho_try > 0):
                            rho_guess_cur = rho_try
                            T_guess_cur   = T_try
                            seeded = True
                    except Exception:
                        seeded = False

                # 2) Fallback: seed rho from P at a chosen Tseed, keep T=Tseed
                if not seeded:
                    try:
                        T_try = float(self.get_t_sp_inv(St, Pt, s_units="cgs", bounds_T=(T_lo, T_hi)))
                        rho_try = float(self.get_rho_pt(Pt, T_try, tab=False))
                        if np.isfinite(T_try) and np.isfinite(rho_try) and (T_try > 0) and (rho_try > 0):
                            rho_guess_cur = rho_try
                            T_guess_cur = T_try
                            seeded = True
                    except Exception:
                        seeded = False

                # 3) Last fallback: seed rho from P at Tseed, keep T=Tseed
                if not seeded:
                    try:
                        rho_guess_cur = float(self.get_rho_pt(Pt, Tseed, tab=False))
                    except Exception:
                        rho_guess_cur = 0.5 * (rho_lo + rho_hi)
                    T_guess_cur = Tseed

            # Clamp guesses
            rho_guess_cur = min(max(float(rho_guess_cur), rho_lo), rho_hi)
            T_guess_cur   = min(max(float(T_guess_cur),   T_lo),   T_hi)

            x0 = np.array([np.log(rho_guess_cur), np.log(T_guess_cur)], dtype=float)

            # Scaling to keep equations comparable
            P_scale = max(abs(Pt), 1e-12)   # GPa
            S_scale = max(abs(St), 1e-30)   # erg/g/K

            def residuals(x):
                rho = float(np.exp(x[0]))
                T   = float(np.exp(x[1]))

                Pm = float(self.get_p_rhot(rho, T))   # GPa
                Sm = float(self.get_s_rhot(rho, T))   # erg/g/K

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
                    r = residuals(sol.x)
                    good_solution = bool(np.all(np.isfinite(r)) and np.max(np.abs(r)) < 1e-4)

                    if good_solution:
                        rho_out[idx] = rho_sol
                        T_out[idx]   = T_sol

                        # Warm-start continuation: update guesses ONLY on success
                        rho_guess_cur = rho_sol
                        T_guess_cur   = T_sol

                        if return_diagnostics:
                            info["success"][idx] = True
                            info["cost"][idx] = sol.cost
                            info["nfev"][idx] = sol.nfev
                            info["resid_P_frac"][idx] = abs(r[0])
                            info["resid_S_frac"][idx] = abs(r[1])
                            info["message"][idx] = sol.message
                    elif return_diagnostics:
                        info["message"][idx] = "least_squares converged to a high-residual point"
                else:
                    if return_diagnostics:
                        info["message"][idx] = getattr(sol, "message", "least_squares failed")
                    # keep last good guess
            except Exception as e:
                if return_diagnostics:
                    info["message"][idx] = f"Exception: {e}"
                # keep last good guess

            if not np.isfinite(rho_out[idx]) or not np.isfinite(T_out[idx]):
                try:
                    T_fb = float(self.get_t_sp_inv(St, Pt, s_units="cgs", bounds_T=(T_lo, T_hi)))
                    rho_fb = float(self.get_rho_pt(Pt, T_fb, tab=False))
                    if np.isfinite(T_fb) and np.isfinite(rho_fb) and (T_fb > 0) and (rho_fb > 0):
                        rho_out[idx] = rho_fb
                        T_out[idx] = T_fb
                        rho_guess_cur = rho_fb
                        T_guess_cur = T_fb
                        if return_diagnostics:
                            info["success"][idx] = True
                            info["message"][idx] = "Fallback via 1-D SP/PT inversion"
                except Exception:
                    pass

        if return_diagnostics:
            return rho_out, T_out, info
        return rho_out, T_out

    def get_rhot_sp_inv(self, S, P, s_units="kbbar", **inv_kwargs):
        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        T = self.get_t_sp_inv(S, P, **inv_kwargs)
        rho = self.get_rho_pt(P, T, tab=False)
        return rho, T

    def get_rho_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals = self._interp(self._rho_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, _ = self.get_rhot_sp_2d_inv(S_arr, P_arr, s_units=s_units, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_t_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals = self._interp(self._t_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(T) if scalar else T

    def get_u_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals_MJkg = self._interp(self._u_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkg * self.MJkg_to_ergg
            return float(vals_cgs) if scalar else vals_cgs

        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_cp_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._cp_rgi_sp is not None:
            #S_tab = self._entropy_to_sp_table(S_arr, s_units=s_units)
            vals_MJkgK = self._interp(self._cp_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkgK * self.MJkgK_to_erggK
            return float(vals_cgs) if scalar else vals_cgs

        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cp_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_cv_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._cv_rgi_sp is not None:
            #S_tab = self._entropy_to_sp_table(S_arr, s_units=s_units)
            vals_MJkgK = self._interp(self._cv_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkgK * self.MJkgK_to_erggK
            return float(vals_cgs) if scalar else vals_cgs

        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cv_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_alpha_sp(self, S, P, tab=True, s_units="kbbar", **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._alpha_rgi_sp is not None:
            #S_tab = self._entropy_to_sp_table(S_arr, s_units=s_units)
            vals = self._interp(self._alpha_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        if "s_units" not in inv_kwargs:
            inv_kwargs["s_units"] = s_units
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals
