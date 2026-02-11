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

import os
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root_scalar

from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u


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
        rhot_arrays_path: str = "eos/aneos/forsterite_eos_arrays.npy",
        rhot_grids_path: str = "eos/aneos/forsterite_eos_grids.npy",
        # ------------------------------------------------------------------
        # TODO (user): replace these placeholders with regenerated PT/SP grids
        # after inverting/saving updated forsterite ANEOS tables.
        # ------------------------------------------------------------------
        pt_table_path: str = "eos/rock_eos/mg2sio4_aneos_PT.npz",
        sp_table_path: str = "eos/rock_eos/mg2sio4_aneos_SP.npz",
    ):
        # Unit conversions
        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.MJkg_to_ergg = float((1.0 * u.MJ / u.kg).to(u.erg / u.g).value)
        self.MJkgK_to_erggK = float((1.0 * u.MJ / u.kg / u.K).to(u.erg / u.g / u.K).value)

        self.dyn_to_Pa = (u.dyn / u.cm**2).to("Pa")
        self.dyn_to_GPa = (u.dyn / u.cm**2).to("GPa")

        # Always-available rho-T ANEOS backend
        self.logrhovals, self.logtvals = np.load(rhot_arrays_path, allow_pickle=True)
        self.logrhovals = np.asarray(self.logrhovals, dtype=float)
        self.logtvals = np.asarray(self.logtvals, dtype=float)

        s_grid_MJkgK, p_grid_GPa, u_grid_MJkg = np.load(rhot_grids_path, allow_pickle=True)
        self.s_grid_MJkgK = np.asarray(s_grid_MJkgK, dtype=float)
        self.p_grid_GPa = np.asarray(p_grid_GPa, dtype=float)
        self.u_grid_MJkg = np.asarray(u_grid_MJkg, dtype=float)

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._p_rgi_rhot = RGI((self.logtvals, self.logrhovals), self.p_grid_GPa, **rgi_kwargs)
        self._s_rgi_rhot = RGI((self.logtvals, self.logrhovals), self.s_grid_MJkgK, **rgi_kwargs)
        self._u_rgi_rhot = RGI((self.logtvals, self.logrhovals), self.u_grid_MJkg, **rgi_kwargs)

        # Optional P-T and S-P tables (placeholders by default).
        self._has_pt_table = False
        self._has_sp_table = False

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

    # ---------- optional PT/SP loaders ----------

    def _load_pt_table(self, path: str):
        data = np.load(path)
        self.P_vals_pt = np.asarray(data["P_grid"], dtype=float)
        self.T_vals_pt = np.asarray(data["T_grid"], dtype=float)
        self.rho_vals_pt = np.asarray(data["rho_grid"], dtype=float)
        self.u_vals_pt = np.asarray(data["u_grid"], dtype=float)
        self.s_vals_pt = np.asarray(data["s_grid"], dtype=float)

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._rho_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.rho_vals_pt, **rgi_kwargs)
        self._u_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.u_vals_pt, **rgi_kwargs)
        self._s_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), self.s_vals_pt, **rgi_kwargs)

        # Optional derivative tables (if present)
        self._alpha_rgi_pt = None
        self._cp_rgi_pt = None
        self._cv_rgi_pt = None
        if "alpha_grid" in data.files:
            self._alpha_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), np.asarray(data["alpha_grid"], dtype=float), **rgi_kwargs)
        if "cp_grid" in data.files:
            self._cp_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), np.asarray(data["cp_grid"], dtype=float), **rgi_kwargs)
        if "cv_grid" in data.files:
            self._cv_rgi_pt = RGI((self.P_vals_pt, self.T_vals_pt), np.asarray(data["cv_grid"], dtype=float), **rgi_kwargs)

        self._has_pt_table = True

    def _load_sp_table(self, path: str):
        data = np.load(path)
        self.S_vals_sp = np.asarray(data["S_grid"], dtype=float)
        self.P_vals_sp = np.asarray(data["P_grid"], dtype=float)
        self.T_vals_sp = np.asarray(data["T_grid"], dtype=float)
        self.rho_vals_sp = np.asarray(data["rho_grid"], dtype=float)
        self.u_vals_sp = np.asarray(data["u_grid"], dtype=float)

        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self._t_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.T_vals_sp, **rgi_kwargs)
        self._rho_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.rho_vals_sp, **rgi_kwargs)
        self._u_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), self.u_vals_sp, **rgi_kwargs)

        self._cp_rgi_sp = None
        self._cv_rgi_sp = None
        self._alpha_rgi_sp = None
        if "cp_grid" in data.files:
            self._cp_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), np.asarray(data["cp_grid"], dtype=float), **rgi_kwargs)
        if "cv_grid" in data.files:
            self._cv_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), np.asarray(data["cv_grid"], dtype=float), **rgi_kwargs)
        if "alpha_grid" in data.files:
            self._alpha_rgi_sp = RGI((self.S_vals_sp, self.P_vals_sp), np.asarray(data["alpha_grid"], dtype=float), **rgi_kwargs)

        self._has_sp_table = True

    # ---------- rho-T native getters ----------

    def get_p_rhot_tab(self, logrho, logt):
        if np.isscalar(logrho) and np.isscalar(logt):
            return float(self._p_rgi_rhot(np.array([[logt, logrho]], dtype=float)))
        logt_arr, logrho_arr = self._as_arrays(logt, logrho)
        pts = np.column_stack((logt_arr.ravel(), logrho_arr.ravel()))
        return self._p_rgi_rhot(pts).reshape(logt_arr.shape)

    def get_s_rhot_tab(self, logrho, logt):
        if np.isscalar(logrho) and np.isscalar(logt):
            return float(self._s_rgi_rhot(np.array([[logt, logrho]], dtype=float)))
        logt_arr, logrho_arr = self._as_arrays(logt, logrho)
        pts = np.column_stack((logt_arr.ravel(), logrho_arr.ravel()))
        return self._s_rgi_rhot(pts).reshape(logt_arr.shape)

    def get_u_rhot_tab(self, logrho, logt):
        if np.isscalar(logrho) and np.isscalar(logt):
            return float(self._u_rgi_rhot(np.array([[logt, logrho]], dtype=float)))
        logt_arr, logrho_arr = self._as_arrays(logt, logrho)
        pts = np.column_stack((logt_arr.ravel(), logrho_arr.ravel()))
        return self._u_rgi_rhot(pts).reshape(logt_arr.shape)

    # ---------- no phase-transition-specific hooks (compatibility only) ----------

    def get_T_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        out = np.full_like(P_arr, np.nan, dtype=float)
        return float(out[0]) if np.isscalar(P) else out

    def get_S_liq_at_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        out = np.full_like(P_arr, np.nan, dtype=float)
        return float(out[0]) if np.isscalar(P) else out

    def get_S_sol_at_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        out = np.full_like(P_arr, np.nan, dtype=float)
        return float(out[0]) if np.isscalar(P) else out

    # ---------- PT getters ----------

    def _solve_logrho_pt_single(self, p_gpa, t_k, *, maxiter=100, rtol=1e-10):
        if not (np.isfinite(p_gpa) and np.isfinite(t_k) and t_k > 0):
            return np.nan

        logt = np.log10(t_k)
        lr_min = float(np.min(self.logrhovals))
        lr_max = float(np.max(self.logrhovals))

        # Initial guess from closest-T P(logrho) row
        iT = int(np.argmin(np.abs(self.logtvals - logt)))
        p_row = np.asarray(self.p_grid_GPa[iT], dtype=float)
        lr_row = np.asarray(self.logrhovals, dtype=float)

        if np.all(np.diff(p_row) >= 0):
            lr_guess = np.interp(p_gpa, p_row, lr_row, left=lr_min, right=lr_max)
        elif np.all(np.diff(p_row) <= 0):
            lr_guess = np.interp(p_gpa, p_row[::-1], lr_row[::-1], left=lr_min, right=lr_max)
        else:
            j = int(np.argmin(np.abs(p_row - p_gpa)))
            lr_guess = float(lr_row[j])

        def f(lr):
            return float(self.get_p_rhot_tab(lr, logt) - p_gpa)

        # Try secant first
        try:
            x0 = float(np.clip(lr_guess, lr_min, lr_max))
            x1 = float(np.clip(x0 + 1e-3, lr_min, lr_max))
            sol = root_scalar(f, method="secant", x0=x0, x1=x1, maxiter=maxiter, rtol=rtol)
            if sol.converged and np.isfinite(sol.root):
                return float(np.clip(sol.root, lr_min, lr_max))
        except Exception:
            pass

        # Bracketed fallback
        fa = f(lr_min)
        fb = f(lr_max)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0.0:
            sol = root_scalar(f, method="brentq", bracket=(lr_min, lr_max), maxiter=maxiter, rtol=rtol)
            if sol.converged and np.isfinite(sol.root):
                return float(np.clip(sol.root, lr_min, lr_max))

        # Last-resort grid projection
        f_row = np.abs(p_row - p_gpa)
        j = int(np.argmin(f_row))
        return float(lr_row[j])

    def get_rho_pt(self, P, T, tab=True, **inv_kwargs):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            rho = self._rho_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            return float(rho) if np.isscalar(P) and np.isscalar(T) else rho

        rho = np.empty_like(P_arr, dtype=float)
        it = np.ndindex(P_arr.shape)
        for idx in it:
            lr = self._solve_logrho_pt_single(float(P_arr[idx]), float(T_arr[idx]), **inv_kwargs)
            rho[idx] = 10.0 ** lr
        return float(rho) if np.isscalar(P) and np.isscalar(T) else rho

    def get_s_pt(self, P, T, tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            s_MJkgK = self._s_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            s_cgs = s_MJkgK * self.MJkgK_to_erggK
            return float(s_cgs) if np.isscalar(P) and np.isscalar(T) else s_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        logrho = np.log10(np.asarray(rho, dtype=float))
        logt = np.log10(T_arr)
        s_MJkgK = self.get_s_rhot_tab(logrho, logt)
        s_cgs = s_MJkgK * self.MJkgK_to_erggK
        return float(s_cgs) if np.isscalar(P) and np.isscalar(T) else s_cgs

    def get_u_pt(self, P, T, tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table:
            u_MJkg = self._u_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            u_cgs = u_MJkg * self.MJkg_to_ergg
            return float(u_cgs) if np.isscalar(P) and np.isscalar(T) else u_cgs

        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        logrho = np.log10(np.asarray(rho, dtype=float))
        logt = np.log10(T_arr)
        u_MJkg = self.get_u_rhot_tab(logrho, logt)
        u_cgs = u_MJkg * self.MJkg_to_ergg
        return float(u_cgs) if np.isscalar(P) and np.isscalar(T) else u_cgs

    # ---------- derived PT properties ----------

    def get_alpha_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._alpha_rgi_pt is not None:
            vals = self._alpha_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            return float(vals) if np.isscalar(P) and np.isscalar(T) else vals

        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        rho_hi = self.get_rho_pt(P_arr, T_hi, tab=False)
        rho_lo = self.get_rho_pt(P_arr, T_lo, tab=False)
        drho_dT = (rho_hi - rho_lo) / (T_hi - T_lo)
        alpha = -(drho_dT / np.maximum(rho, 1e-99))
        return float(alpha) if np.isscalar(P) and np.isscalar(T) else alpha

    def get_cp_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._cp_rgi_pt is not None:
            cp_MJkgK = self._cp_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            cp_cgs = cp_MJkgK * self.MJkgK_to_erggK
            return float(cp_cgs) if np.isscalar(P) and np.isscalar(T) else cp_cgs

        # Cp = T * (dS/dT)_P
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        s_hi = self.get_s_pt(P_arr, T_hi, tab=False)
        s_lo = self.get_s_pt(P_arr, T_lo, tab=False)
        ds_dT = (s_hi - s_lo) / (T_hi - T_lo)
        cp = T_arr * ds_dT
        return float(cp) if np.isscalar(P) and np.isscalar(T) else cp

    def get_cv_pt(self, P, T, tab=True, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)

        if tab and self._has_pt_table and self._cv_rgi_pt is not None:
            cv_MJkgK = self._cv_rgi_pt(np.column_stack((P_arr.ravel(), T_arr.ravel()))).reshape(P_arr.shape)
            cv_cgs = cv_MJkgK * self.MJkgK_to_erggK
            return float(cv_cgs) if np.isscalar(P) and np.isscalar(T) else cv_cgs

        # Cv = (dU/dT)_rho (computed at local rho(P,T))
        rho = self.get_rho_pt(P_arr, T_arr, tab=False)
        logrho = np.log10(np.asarray(rho, dtype=float))
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        u_hi = self.get_u_rhot_tab(logrho, np.log10(T_hi)) * self.MJkg_to_ergg
        u_lo = self.get_u_rhot_tab(logrho, np.log10(T_lo)) * self.MJkg_to_ergg
        cv = (u_hi - u_lo) / (T_hi - T_lo)
        return float(cv) if np.isscalar(P) and np.isscalar(T) else cv

    def get_alpha_x(self, P, T, rho, x):
        P_arr, T_arr = self._as_arrays(P, T)
        out = np.zeros_like(P_arr, dtype=float)
        return float(out) if np.isscalar(P) and np.isscalar(T) else out

    # ---------- SP inversion ----------

    def get_t_sp_inv(
        self,
        S_target,
        P_target,
        *,
        s_units="kbbar",
        T_guess=None,
        bounds_T=(1.0, 2e5),
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

        if str(s_units).lower() == "kbbar":
            S_goal = S_arr / float(self.erg_to_kbbar)
        else:
            S_goal = S_arr

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

    def get_rhot_sp_inv(self, S, P, **inv_kwargs):
        T = self.get_t_sp_inv(S, P, **inv_kwargs)
        rho = self.get_rho_pt(P, T, tab=False)
        return rho, T

    def get_rho_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals = self._interp(self._rho_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, _ = self.get_rhot_sp_inv(S_arr, P_arr, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_t_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals = self._interp(self._t_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        return float(T) if scalar else T

    def get_u_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table:
            vals_MJkg = self._interp(self._u_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkg * self.MJkg_to_ergg
            return float(vals_cgs) if scalar else vals_cgs

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_cp_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._cp_rgi_sp is not None:
            vals_MJkgK = self._interp(self._cp_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkgK * self.MJkgK_to_erggK
            return float(vals_cgs) if scalar else vals_cgs

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cp_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_cv_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._cv_rgi_sp is not None:
            vals_MJkgK = self._interp(self._cv_rgi_sp, S_arr, P_arr)
            vals_cgs = vals_MJkgK * self.MJkgK_to_erggK
            return float(vals_cgs) if scalar else vals_cgs

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cv_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

    def get_alpha_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab and self._has_sp_table and self._alpha_rgi_sp is not None:
            vals = self._interp(self._alpha_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_pt(P_arr, T, tab=False)
        return float(vals) if scalar else vals

