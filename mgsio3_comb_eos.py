"""
mgsio3_comb_eos.py

Phase-mixed MgSiO3 EOS.
Selects / blends solid and liquid MgSiO3 EOS using a smooth transition
around the melting curve T_melt(P).

Usage:
    from eos import mgsio3_liquid_eos, mgsio3_solid_eos
    from mgsio3_combo_eos import MGSIO3_COMBINED_EOS

    combo = MGSIO3_COMBINED_EOS()

Public API (all return cgs / same units as the underlying classes):
    get_rho_pt(P, T)
    get_s_pt(P, T)
    get_u_pt(P, T)
    get_alpha_pt(P, T)
    get_cp_pt(P, T)
    get_cv_pt(P, T)

    get_t_sp(S, P)
    get_rho_sp(S, P)
    get_u_sp(S, P)

Assumptions:
- mgsio3_liquid_eos.MGSIO3_LIQUID_EOS and
  mgsio3_solid_eos.MGSIO3_SOLID_EOS expose:
    get_rho_pt(P, T)
    get_s_pt(P, T)
    get_u_pt(P, T)
    get_alpha_pt(P, T)
    get_cp_pt(P, T)
    get_cv_pt(P, T)
    get_t_sp(S, P)
    get_rho_sp(S, P)
    get_u_sp(S, P)
- Units match between solid and liquid implementations:
  P in GPa
  T in K
  rho in g/cm^3
  S in erg/g/K
  U in erg/g (or consistent cgs per-mass energy)
  Cp, Cv in erg/g/K
  alpha in 1/K
If the solid class uses different output units you must reconcile upstream.
"""

from __future__ import annotations
import numpy as np
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from scipy.optimize import brenth, root_scalar
from scipy.interpolate import RegularGridInterpolator as RGI

from eos import mgsio3_liquid_eos, mgsio3_solid_eos


class MGSIO3_COMBINED_EOS:
    """
    Wrapper that smoothly blends solid and liquid EOS at T_melt(P).

    Define weight w in [0,1] where w=0 is solid and w=1 is liquid.
    We use a tanh switch with width dT = frac_width * T_melt.

    For PT queries:
        X = (1-w)*X_solid + w*X_liquid

    For SP queries:
        1. Compute T_solid = solid.get_t_sp(S,P)
           and T_liquid = liquid.get_t_sp(S,P).
        2. Compare both to T_melt(P). Use same smooth switch logic,
           evaluated using T_avg = 0.5*(T_solid+T_liquid).
        3. Blend all SP quantities with that weight.
    """

    def __init__(self, frac_width: float = 0.05, dT=50):
        """
        frac_width sets the smoothing half-width as a fraction of T_melt.
        dT = frac_width * T_melt.
        """
        self.solid = mgsio3_solid_eos.MGSIO3_SOLID_EOS()
        self.liquid = mgsio3_liquid_eos.MGSIO3_LIQUID_EOS()
        self.frac_width = float(frac_width)
        self.dT = dT

        self.erg_to_kbbar = float((u.erg/u.Kelvin/u.gram).to(k_B/amu))  # (erg/g/K) -> (kB/baryon)

        self.dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion

        self.dyn_to_GPa = (u.dyn/u.cm**2).to('GPa') # dyn/cm² to GPa conversion

        self.L = 7.322e5 * (u.J/u.kg).to('erg/g')  # latent heat of fusion of the mantle

        #### Load SP data ####

        self.data_sp = np.load(f'eos/rock_eos/MgSiO3_combined_SP_new.npz')

        self.svals_sp = np.asarray(self.data_sp['svals_sp'], dtype=float)  # kb/baryon
        self.pvals_sp = np.asarray(self.data_sp['pvals_sp'], dtype=float)  # Pa

        self.rho_grid_sp = np.asarray(self.data_sp['rho_grid_sp'], dtype=float) # kg/m^3
        self.t_grid_sp = np.asarray(self.data_sp['t_grid_sp'], dtype=float)  # K
        self.u_grid_sp = np.asarray(self.data_sp['u_grid_sp'], dtype=float)  # erg/g
        self.cp_grid_sp = np.asarray(self.data_sp['cp_grid_sp'], dtype=float)  # erg/g/K
        self.cv_grid_sp = np.asarray(self.data_sp['cv_grid_sp'], dtype=float)  # erg/g/K
        self.alpha_grid_sp = np.asarray(self.data_sp['alpha_grid_sp'], dtype=float)  # 1/K

        # Interpolators: input points are (S, P)
        rgi_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
        self.rho_rgi_sp = RGI((self.svals_sp, self.pvals_sp), self.rho_grid_sp, **rgi_kwargs)
        self.t_rgi_sp = RGI((self.svals_sp, self.pvals_sp), self.t_grid_sp, **rgi_kwargs)
        self.u_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.u_grid_sp, **rgi_kwargs)
        self.cp_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.cp_grid_sp, **rgi_kwargs)
        self.cv_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.cv_grid_sp, **rgi_kwargs)
        self.alpha_rgi_sp   = RGI((self.svals_sp, self.pvals_sp), self.alpha_grid_sp, **rgi_kwargs)

    # ---------- helpers ----------

    @staticmethod
    def _as_arrays(a, b):
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A, B

    @staticmethod
    def _as_array_single(x):
        return np.array(x, ndmin=1, dtype=float)

    def get_T_melt(self, P):
        """
        Fei et al. (2021).
        P in GPa.
        Return T_melt(P) in K.
        """
        P_arr = np.array(P, ndmin=1, dtype=float)
        return 6295.0 * (P_arr / 140.0) ** 0.317
    
    def _solid_entropy_offset_at_melt(self, P_arr):
        """
        Compute an offset (>=0) to subtract from solid entropy so that at Tm(P):
            S_liq - S_sol = L / Tm
        If already satisfied, offset = 0.

        Assumes:
        - P in GPa
        - get_s_pt returns erg/g/K
        - L in erg/g
        - Tm in K
        """
        Tm = self.get_T_melt(P_arr)
        Tm_safe = np.maximum(Tm, 1.0)

        s_sol_m = self.solid.get_s_pt(P_arr, Tm)
        s_liq_m = self.liquid.get_s_pt(P_arr, Tm)

        dS_target = self.L / Tm_safe  # erg/g/K

        # Want: s_sol_corr = s_liq_m - dS_target
        # If s_sol_m is higher than that, subtract the excess.
        offset = s_sol_m - (s_liq_m - dS_target)

        # Keep only positive, finite corrections
        offset = np.where(np.isfinite(offset) & (offset > 0.0), offset, 0.0)
        return offset


    @staticmethod
    def _entropy_eps(s_ref):
        # tiny separation to enforce strict inequality without affecting anything macroscopic
        return 1e-12 * np.maximum(1.0, np.abs(s_ref))


    def get_S_liq_at_melt(self, P):
        """
        Return S_liq(P, T_melt) in erg/g/K.
        """
        P_arr = self._as_array_single(P)
        Tm = self.get_T_melt(P_arr)
        S_liq = self.liquid.get_s_pt(P_arr, Tm)
        return S_liq.reshape(P_arr.shape)

    def get_S_sol_at_melt(self, P):
        """
        Return S_sol(P, T_melt) in erg/g/K.
        """
        P_arr = self._as_array_single(P)
        Tm = self.get_T_melt(P_arr)
        S_sol = self.solid.get_s_pt(P_arr, Tm)
        return S_sol.reshape(P_arr.shape)

    def _blend_weight_PT(self, P_arr, T_arr):
        """
        Compute w_liq(P,T) using tanh around T_melt(P).

        w_liq = 0.5 * [1 + tanh((T - Tm)/dT)]
        with dT = frac_width * Tm.
        Clamp dT to at least 1 K to avoid zero-width at low P.
        """
        Tm = self.get_T_melt(P_arr)
        #dT = np.maximum(self.frac_width * Tm, 1.0)
        arg = (T_arr - Tm) / self.dT
        w = 0.5 * (1.0 + np.tanh(arg))
        # numerical safety
        return np.clip(w, 0.0, 1.0)

    def _blend_weight_given_T(self, P_arr, T_trial_arr):
        """
        Same as _blend_weight_PT but caller supplies T_trial_arr directly.
        Used for SP-based quantities where we estimate T first.
        """
        Tm = self.get_T_melt(P_arr)
        #dT = np.maximum(self.frac_width * Tm, 1.0)
        arg = (T_trial_arr - Tm) / self.dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)
    
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

    # ---------- PT interface ----------

    def get_rho_pt(self, P, T):
        P_arr, T_arr = self._as_arrays(P, T)

        rho_s = self.solid.get_rho_pt(P_arr, T_arr)
        rho_l = self.liquid.get_rho_pt(P_arr, T_arr)

        w = self._blend_weight_PT(P_arr, T_arr)
        rho_mix = 1 / ((1.0 - w) * 1/rho_s + w * 1/rho_l)
        return rho_mix.reshape(P_arr.shape)

    @staticmethod
    def _entropy_eps(s_ref):
        # tiny separation to enforce strict inequality without affecting anything macroscopic
        return 1e-12 * np.maximum(1.0, np.abs(s_ref))

    def get_s_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        # --- Get raw phase entropies ---
        # Use tab kwarg if your phase objects support it; otherwise drop those kwargs.
        try:
            s_s = self.solid.get_s_pt(P_arr, T_arr, tab=solid_tab)
        except TypeError:
            s_s = self.solid.get_s_pt(P_arr, T_arr)

        try:
            s_l = self.liquid.get_s_pt(P_arr, T_arr, tab=liquid_tab)
        except TypeError:
            s_l = self.liquid.get_s_pt(P_arr, T_arr)

        # --- Hard clamp: solids must not exceed liquids at same (P,T) ---
        eps = self._entropy_eps(s_l)
        deltaS = s_s - s_l
        # If s_s >= s_l - eps, subtract enough to make s_s <= s_l - eps
        corr = np.where(np.isfinite(deltaS) & (deltaS > -eps), deltaS + eps, 0.0)
        s_s = s_s - corr

        # --- Blend ---
        w = self._blend_weight_PT(P_arr, T_arr)
        s_mix = (1.0 - w) * s_s + w * s_l
        return s_mix.reshape(P_arr.shape)

    def get_u_pt(self, P, T):
        P_arr, T_arr = self._as_arrays(P, T)

        u_s = self.solid.get_u_pt(P_arr, T_arr)
        u_l = self.liquid.get_u_pt(P_arr, T_arr)

        w = self._blend_weight_PT(P_arr, T_arr)
        u_mix = (1.0 - w) * u_s + w * u_l
        return u_mix.reshape(P_arr.shape)

    def get_alpha_pt(self, P, T):
        P_arr, T_arr = self._as_arrays(P, T)

        a_s = self.solid.get_alpha_pt(P_arr, T_arr)
        a_l = self.liquid.get_alpha_pt(P_arr, T_arr)

        w = self._blend_weight_PT(P_arr, T_arr)
        a_mix = (1.0 - w) * a_s + w * a_l
        return a_mix.reshape(P_arr.shape)

    def get_cp_pt(self, P, T):
        P_arr, T_arr = self._as_arrays(P, T)

        cp_s = self.solid.get_cp_pt(P_arr, T_arr)
        cp_l = self.liquid.get_cp_pt(P_arr, T_arr)

        w = self._blend_weight_PT(P_arr, T_arr)
        cp_mix = (1.0 - w) * cp_s + w * cp_l
        return cp_mix.reshape(P_arr.shape)

    def get_cv_pt(self, P, T):
        P_arr, T_arr = self._as_arrays(P, T)

        cv_s = self.solid.get_cv_pt(P_arr, T_arr)
        cv_l = self.liquid.get_cv_pt(P_arr, T_arr)

        w = self._blend_weight_PT(P_arr, T_arr)
        cv_mix = (1.0 - w) * cv_s + w * cv_l
        return cv_mix.reshape(P_arr.shape)

    def get_alpha_x(self, P, T, rho, x):
        P_arr, T_arr = self._as_arrays(P, T)

        rho_s = self.solid.get_rho_pt(P_arr, T_arr)
        rho_l = self.liquid.get_rho_pt(P_arr, T_arr)

        return rho * ((1 / rho_l) - ((1 / rho_s)))

# ---------- SP interface (thermodynamically consistent) ----------

    def get_t_sp_inv(
        self,
        S_target,
        P_target,
        *,
        s_units="kbbar",              # "kbbar" or "cgs" (erg/g/K)
        T_guess=None,                 # optional scalar/array initial guess for FIRST element
        bounds_T=(1.0, 2e5),          # K
        # warm-start secant controls
        secant_maxiter=30,
        secant_rtol=1e-10,
        dy0=1e-3,                     # initial secant separation in ln(T)
        # fallback bracketing controls
        bracket_factor=1.3,
        expand_steps=12,
        expand_factor=2.0,
        brent_rtol=1e-10,
        brent_xtol=1e-6,
        brent_maxiter=200,
        return_diagnostics=False,
        fail_value=np.nan,
    ):
        """
        Invert the *mixed* entropy law:
            S_mix(P, T) = S_target

        where S_mix is computed by self.get_s_pt(P,T) (which already applies your
        smooth phase switch w(P,T)).

        Warm-start:
        - first element uses T_guess (or a default)
        - subsequent elements use previous converged T as the guess
        """
        # ---- broadcast inputs ----
        S_arr = np.asarray(S_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        shape = S_arr.shape

        # ---- convert target entropy to cgs (erg/g/K), because get_s_pt returns cgs ----
        if str(s_units).lower() == "kbbar":
            S_goal = S_arr / float(self.erg_to_kbbar)  # kbbar -> cgs
        else:
            S_goal = S_arr

        # ---- bounds in y = ln T ----
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

        # ---- residual in logT-space ----
        def g_of_y(P, y, Sg):
            y = float(np.clip(y, y_min, y_max))
            T = float(np.exp(y))
            Sm = self.get_s_pt(P, T)  # <-- mixed entropy law, cgs
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

        # ---- initial guess array for FIRST element ----
        if T_guess is None:
            # a decent physics-ish default: near melt, but clipped to bounds
            T_guess_arr = np.clip(self.get_T_melt(P_arr), T_lo, T_hi)
        else:
            T_guess_arr = np.asarray(T_guess, dtype=float)
            T_guess_arr, _ = np.broadcast_arrays(T_guess_arr, P_arr)
            T_guess_arr = np.clip(T_guess_arr, T_lo, T_hi)

        T_prev = None  # warm-start carrier

        for idx in np.ndindex(shape):
            P = float(P_arr[idx])
            Sg = float(S_goal[idx])

            if not (np.isfinite(P) and np.isfinite(Sg)) or P <= 0:
                if return_diagnostics:
                    diag["method"][idx] = "none"
                    diag["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            # warm-start guess
            Tg = float(T_guess_arr[idx] if (T_prev is None or not np.isfinite(T_prev)) else T_prev)
            y0 = float(np.log(np.clip(Tg, T_lo, T_hi)))
            y0, g0 = nudge_to_finite(P, Sg, y0)

            # 1) secant in logT
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
                            # sanity: must be finite residual
                            if np.isfinite(g_of_y(P, y_sol, Sg)):
                                Tout[idx] = T_sol
                                T_prev = T_sol
                                secant_ok = True
                                if return_diagnostics:
                                    diag["success"][idx] = True
                                    diag["method"][idx] = "secant(logT)"
                                    diag["message"][idx] = "OK"
                                    diag["nfev"][idx] = getattr(sol, "function_calls", 0) or 0
                                    diag["iterations"][idx] = getattr(sol, "iterations", 0) or 0
                    except Exception as e:
                        if return_diagnostics:
                            diag["method"][idx] = "secant(logT)"
                            diag["message"][idx] = f"Secant exception: {e}"

            if secant_ok:
                continue

            # 2) bracket-expand + brenth in logT
            w = np.log(float(bracket_factor))
            y_lo = float(np.clip(y0 - w, y_min, y_max))
            y_hi = float(np.clip(y0 + w, y_min, y_max))
            f_lo = g_of_y(P, y_lo, Sg)
            f_hi = g_of_y(P, y_hi, Sg)

            n_expand = 0
            while (
                (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0)
                and n_expand < int(expand_steps)
            ):
                n_expand += 1
                w *= float(expand_factor)
                y_lo = float(np.clip(y0 - w, y_min, y_max))
                y_hi = float(np.clip(y0 + w, y_min, y_max))
                f_lo = g_of_y(P, y_lo, Sg)
                f_hi = g_of_y(P, y_hi, Sg)

            if return_diagnostics:
                diag["n_expand"][idx] = n_expand

            if (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0):
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = "Failed to bracket root."
                continue

            try:
                y_root = brenth(
                    lambda yy: g_of_y(P, yy, Sg),
                    y_lo, y_hi,
                    xtol=float(brent_xtol),
                    rtol=float(brent_rtol),
                    maxiter=int(brent_maxiter),
                    disp=False,
                )
                y_root = float(np.clip(y_root, y_min, y_max))
                T_root = float(np.exp(y_root))

                Tout[idx] = T_root
                T_prev = T_root

                if return_diagnostics:
                    diag["success"][idx] = True
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = "OK"
            except Exception as e:
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = f"Brenth exception: {e}"

        if return_diagnostics:
            return Tout, diag
        return Tout


    # -------------------------
    # SP table API (override)
    # -------------------------

    def get_rho_sp(self, S, P, tab=True, **inv_kwargs):
        """
        rho(S,P) in kg/m^3.
        If tab=False, uses Fe_EOS.get_rho_sp_inv.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.rho_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        rho, T = self.get_rhot_sp_inv(S_arr, P_arr, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_t_sp(self, S, P, tab=True, **inv_kwargs):
        """
        T(S,P) in K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.t_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
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

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_pt(P_arr, T)  # erg/g (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_cp_sp(self, S, P, tab=True, **inv_kwargs):
        """
        cp(S,P) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.cp_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cp_pt(P_arr, T)  # erg/g/K (your Fe_EOS convention)
        return float(vals) if scalar else vals

    def get_cv_sp(self, S, P, tab=True, **inv_kwargs):
        """
        cv(S,P) in erg/g/K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv then analytic getter.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.cv_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_cv_pt(P_arr, T)  # erg/g/K (your Fe_EOS convention)
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

        T = self.get_t_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_pt(P_arr, T)  # 1/K (your Fe_EOS convention)
        return float(vals) if scalar else vals