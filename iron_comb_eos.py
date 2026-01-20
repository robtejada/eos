"""
iron_eos_comb.py

Phase-mixed iron EOS.
Selects / blends solid and liquid iron EOS using a smooth transition
around the melting curve T_melt(P).

Liquid: ichikawa_iron_eos.Fe_EOS
Solid : dorogo_iron_eos.Fe_EOS(phase="hcp")

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
- Underlying classes expose:
    get_rho_pt(P, T)
    get_s_pt(P, T)
    get_u_pt(P, T)
    get_alpha_pt(P, T)
    get_CP_pt(P, T)
    get_CV_pt(P, T)
    get_T_sp(S, P)
    get_rho_sp(S, P)
    get_u_sp(S, P)
- Units match between solid and liquid implementations:
    P in Pa
    T in K
    rho in kg/m^3
    S in erg/g/K
    U in erg/g
    Cp, Cv in erg/g/K
    alpha in 1/K
"""

from __future__ import annotations

import numpy as np

from eos import dorogo_iron_eos
from scipy.optimize import brenth, least_squares, root_scalar
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

class Fe_COMBINED_EOS:
    """
    Wrapper that smoothly blends solid and liquid EOS at T_melt(P).

    Define weight w in [0,1] where w=0 is solid and w=1 is liquid.
    We use a tanh switch with width dT = frac_width * T_melt.

    For PT queries:
        X = (1-w)*X_solid + w*X_liquid

    For SP queries:
        1. Compute T_solid = solid.get_T_sp(S,P)
           and T_liquid = liquid.get_T_sp(S,P).
        2. Compare both to T_melt(P). Use same smooth switch logic,
           evaluated using T_avg = 0.5*(T_solid+T_liquid).
        3. Blend all SP quantities with that weight.
    """

    erg_to_kbbar: float = ERG_GK_TO_KBBAR
    dyn_to_Pa: float = DYNCM2_TO_PA
    dyn_to_GPa: float = DYNCM2_TO_GPA
    L: float = 1.2e6 * (u.J/u.kg).to('erg/g')  # latent heat of fusion of Fe-Si alloy in erg/g (Anderson & Duba 1997)
    kb: float = k_B.to('erg/K') # ergs/K

    def __init__(self, frac_width: float = 0.05, dT: float = 50.0):
        """
        frac_width sets the smoothing half-width as a fraction of T_melt.
        dT = frac_width * T_melt.
        """
        self.solid = dorogo_iron_eos.Fe_EOS(phase="hcp")
        # self.liquid = ichikawa_iron_eos.Fe_EOS()
        self.liquid = dorogo_iron_eos.Fe_EOS(phase="liquid")
        self.frac_width = float(frac_width)
        self.dT = float(dT)

        #### Load SP data ####

        self.data_sp = np.load(f'eos/dorogokupets_iron_eos/iron_eos_SP_comb.npz')

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
    
    def _solid_entropy_offset_at_melt(self, P_arr, solid_tab=True, liquid_tab=True):
        """
        Compute an offset (>=0) to subtract from solid entropy so that at Tm(P):
            S_liq - S_sol = L / Tm
        If the EOS already satisfies this, offset = 0.
        """
        Tm = self.get_T_melt(P_arr)
        Tm_safe = np.maximum(Tm, 1.0)  # avoid divide-by-zero silliness

        s_sol_m = self.solid.get_s_pt(P_arr, Tm, tab=solid_tab)
        s_liq_m = self.liquid.get_s_pt(P_arr, Tm, tab=liquid_tab)

        dS_target = self.L / Tm_safe  # erg/g/K

        # We want: s_sol_corr = s_liq_m - dS_target
        # If s_sol_m is higher than that, subtract the excess.
        offset = s_sol_m - (s_liq_m - dS_target)

        # Only keep positive, finite corrections
        offset = np.where(np.isfinite(offset) & (offset > 0.0), offset, 0.0)
        return offset

    @staticmethod
    def _entropy_eps(s_ref):
        # tiny separation to enforce strict inequality without affecting anything macroscopic
        return 1e-12 * np.maximum(1.0, np.abs(s_ref))


    def get_T_melt(self, P):
        """
        Zhang et al. (2015) melt curve used in ichikawa_iron_eos.
        P in Pa. Return T_melt(P) in K.
        """
        P_arr = self._as_array_single(P)
        Tm = self.liquid.get_T_melt(P_arr)
        return Tm.reshape(P_arr.shape)

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
        dT = self.dT
        arg = (T_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)

    def _blend_weight_given_T(self, P_arr, T_trial_arr):
        """
        Same as _blend_weight_PT but caller supplies T_trial_arr directly.
        Used for SP-based quantities where we estimate T first.
        """
        Tm = self.get_T_melt(P_arr)
        dT = self.dT
        arg = (T_trial_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)
    
    def _blend_weight_given_rhoT(self, P_arr, rho_trial_arr, T_trial_arr):
        """
        Same as _blend_weight_PT but caller supplies rho_trial_arr and T_trial_arr directly.
        Used for SP-based quantities where we estimate T first.
        """
        Tm = self.get_T_melt(P_arr)
        dT = self.dT
        arg = (T_trial_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)

    # ---------- PT interface ----------

    def get_rho_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        rho_s = self.solid.get_rho_pt(P_arr, T_arr, tab=solid_tab)
        rho_l = self.liquid.get_rho_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        rho_mix = 1.0 / ((1.0 - w)/rho_s + w / rho_l)
        return rho_mix.reshape(P_arr.shape)

    # def get_s_pt(self, P, T, solid_tab=True, liquid_tab=True):
    #     P_arr, T_arr = self._as_arrays(P, T)

    #     s_s = self.solid.get_s_pt(P_arr, T_arr, tab=solid_tab)
    #     s_l = self.liquid.get_s_pt(P_arr, T_arr, tab=liquid_tab)
    #     w = self._blend_weight_PT(P_arr, T_arr)
    #     # deltaS = s_s - s_l # need to account for entropy mixing here?
    #     # s_s[deltaS > 0] += s_s
    #     s_mix = (1.0 - w) * s_s + w * s_l
    #     return s_mix.reshape(P_arr.shape)

    def get_s_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        s_s_raw = self.solid.get_s_pt(P_arr, T_arr, tab=solid_tab)
        s_l     = self.liquid.get_s_pt(P_arr, T_arr, tab=liquid_tab)

        # --- Step 1: pressure-dependent reference fix at the melt curve ---
        offset = self._solid_entropy_offset_at_melt(P_arr, solid_tab=solid_tab, liquid_tab=liquid_tab)
        s_s = s_s_raw - offset

        # --- Step 2: hard clamp: solids must not exceed liquids at same (P,T) ---
        # If deltaS_raw > 0, subtract it (plus a tiny epsilon) so s_s <= s_l.
        eps = self._entropy_eps(s_l)
        deltaS = s_s - s_l
        corr = np.where(np.isfinite(deltaS) & (deltaS > -eps), deltaS + eps, 0.0)
        s_s = s_s - corr

        # --- Blend ---
        w = self._blend_weight_PT(P_arr, T_arr)
        s_mix = (1.0 - w) * s_s + w * s_l
        return s_mix.reshape(P_arr.shape)


    def get_u_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        u_s = self.solid.get_u_pt(P_arr, T_arr, tab=solid_tab)
        u_l = self.liquid.get_u_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        u_mix = (1.0 - w) * u_s + w * u_l
        return u_mix.reshape(P_arr.shape)

    def get_alpha_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        a_s = self.solid.get_alpha_pt(P_arr, T_arr, tab=solid_tab)
        a_l = self.liquid.get_alpha_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        a_mix = (1.0 - w) * a_s + w * a_l
        return a_mix.reshape(P_arr.shape)

    def get_cp_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        cp_s = self.solid.get_CP_pt(P_arr, T_arr, tab=solid_tab)
        cp_l = self.liquid.get_CP_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        cp_mix = (1.0 - w) * cp_s + w * cp_l
        return cp_mix.reshape(P_arr.shape)

    def get_CP_pt(self, P, T, solid_tab=True, liquid_tab=True):
        return self.get_cp_pt(P, T, solid_tab=solid_tab, liquid_tab=liquid_tab)

    def get_cv_pt(self, P, T, solid_tab=True, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        cv_s = self.solid.get_CV_pt(P_arr, T_arr, tab=solid_tab)
        cv_l = self.liquid.get_CV_pt(P_arr, T_arr, tab=liquid_tab)

        w = self._blend_weight_PT(P_arr, T_arr)
        cv_mix = (1.0 - w) * cv_s + w * cv_l
        return cv_mix.reshape(P_arr.shape)

    def get_CV_pt(self, P, T, solid_tab=True, liquid_tab=True):
        return self.get_cv_pt(P, T, solid_tab=solid_tab, liquid_tab=liquid_tab)
    
    # ---------- SP interface ----------

    def get_T_sp_inv(
        self,
        S_target,
        P_target,
        *,
        s_units="kbbar",              # "cgs" or "kbbar"
        T_guess=None,                 # optional scalar/array initial guess for FIRST element
        bounds_T=(1.0, 2e5),          # K
        # fast continuation controls
        secant_maxiter=30,            # few iters is usually enough when warm-started
        secant_rtol=1e-10,
        dy0=1e-3,                     # initial secant separation in ln(T)
        # fallback bracketing controls (rarely used after first element)
        bracket_factor=1.3,           # initial bracket window around guess
        expand_steps=12,
        expand_factor=2.0,
        brent_rtol=1e-10,
        brent_xtol=1e-6,              # MUST be > 0 (your SciPy requires this)
        brent_maxiter=200,
        solid_tab=True,
        liquid_tab=True,
        return_diagnostics=False,
        fail_value=np.nan,
        fill_nans=True,              # fill interior NaN holes + edge extrap
        fill_axis=-1,                # axis along which P varies (for 1D, irrelevant)
        extrapolate_edges=True,      # extrapolate missing low/high-P edges

    ):
        """
        Warm-start inversion for T(S,P) using previous element's solution as guess.

        Strategy per element (in broadcast order):
        1) Try fast secant on y=ln T initialized from previous successful solution.
        2) If secant fails, fall back to bracket expansion + brenth (on y=ln T).
        """

        # ---- broadcast inputs ----
        S_arr = np.asarray(S_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        shape = S_arr.shape

        # ---- convert entropy targets to cgs if needed ----
        if str(s_units).lower() == "kbbar":
            S_cgs = S_arr / float(self.erg_to_kbbar)  # kbbar -> cgs
        else:
            S_cgs = S_arr

        # ---- bounds in y = ln T ----
        T_lo, T_hi = map(float, bounds_T)
        if T_lo <= 0 or T_hi <= T_lo:
            raise ValueError("bounds_T must satisfy 0 < T_lo < T_hi")
        y_min = np.log(T_lo)
        y_max = np.log(T_hi)

        def _fill_1d_loglog(Pv, Tv):
            """
            Fill NaNs in a 1D T(P) curve.
            - Interpolate lnT vs lnP across interior gaps.
            - Extrapolate linearly in lnT vs lnP at edges (optional).
            """
            Pv = np.asarray(Pv, dtype=float)
            Tv = np.asarray(Tv, dtype=float)

            good = np.isfinite(Pv) & (Pv > 0) & np.isfinite(Tv) & (Tv > 0)
            if good.all():
                return Tv

            # If nothing converged, return a harmless constant (geometric mid of bounds)
            if np.count_nonzero(good) == 0:
                return np.full_like(Tv, np.sqrt(T_lo * T_hi), dtype=float)

            x = np.log(Pv)
            order = np.argsort(x)
            xo = x[order]
            To = Tv[order]

            goodo = np.isfinite(To) & (To > 0) & np.isfinite(xo)
            if np.count_nonzero(goodo) == 0:
                return np.full_like(Tv, np.sqrt(T_lo * T_hi), dtype=float)

            xg = xo[goodo]
            yg = np.log(np.clip(To[goodo], T_lo, T_hi))

            # baseline: interior interpolation + constant edge fill
            ynew = np.interp(xo, xg, yg)

            # edge extrapolation in log-log (optional)
            if extrapolate_edges and xg.size >= 2:
                # left slope from first two good points
                mL = (yg[1] - yg[0]) / (xg[1] - xg[0])
                left = xo < xg[0]
                ynew[left] = yg[0] + mL * (xo[left] - xg[0])

                # right slope from last two good points
                mR = (yg[-1] - yg[-2]) / (xg[-1] - xg[-2])
                right = xo > xg[-1]
                ynew[right] = yg[-1] + mR * (xo[right] - xg[-1])

            ynew = np.clip(ynew, y_min, y_max)
            Tfilled_o = np.exp(ynew)

            # restore original ordering
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            Tfilled = Tfilled_o[inv]

            # keep original finite values exactly
            out = Tv.copy()
            bad = ~good
            out[bad] = Tfilled[bad]
            return np.clip(out, T_lo, T_hi)

        T_out = np.full(shape, fail_value, dtype=float)

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

        # ---- safe scalar entropy residual in y-space ----
        def g_of_y(P, y, Sgoal):
            # clamp to bounds to avoid exp overflow / negative T
            y = float(np.clip(y, y_min, y_max))
            T = float(np.exp(y))
            Sm = self.get_s_pt(P, T, solid_tab=solid_tab, liquid_tab=liquid_tab)
            Sm = float(np.asarray(Sm).reshape(-1)[0])
            if not np.isfinite(Sm):
                return np.nan
            return Sm - Sgoal

        # ---- helper: try to get a finite residual near a guess ----
        def nudge_to_finite(P, Sgoal, y_guess):
            y = float(np.clip(y_guess, y_min, y_max))
            val = g_of_y(P, y, Sgoal)
            if np.isfinite(val):
                return y, val

            # nudge around multiplicatively in T (additively in y)
            for step in (0.2, -0.2, 0.5, -0.5, 1.0, -1.0):
                y_try = float(np.clip(y + step, y_min, y_max))
                v_try = g_of_y(P, y_try, Sgoal)
                if np.isfinite(v_try):
                    return y_try, v_try

            return y, np.nan

        # ---- initial guess for the FIRST element ----
        if T_guess is None:
            T_guess_arr = np.full(shape, 6000.0, dtype=float)
        else:
            T_guess_arr = np.asarray(T_guess, dtype=float)
            T_guess_arr, _ = np.broadcast_arrays(T_guess_arr, S_cgs)

        # continuation state (updated after each success)
        T_prev = None  # last successful temperature

        for idx in np.ndindex(shape):
            P = float(P_arr[idx])
            Sgoal = float(S_cgs[idx])

            if not (np.isfinite(P) and np.isfinite(Sgoal)) or P <= 0:
                if return_diagnostics:
                    diag["message"][idx] = "Invalid target (non-finite or P<=0)."
                    diag["method"][idx] = "none"
                continue

            # warm-start guess:
            if T_prev is None or not np.isfinite(T_prev):
                Tg = float(T_guess_arr[idx])
            else:
                Tg = float(T_prev)

            Tg = float(np.clip(Tg, T_lo, T_hi))
            y0 = float(np.log(Tg))

            # ensure g(y0) is finite (otherwise secant is doomed)
            y0, g0 = nudge_to_finite(P, Sgoal, y0)

            # -------------------------
            # 1) Fast secant on y=lnT
            # -------------------------
            secant_ok = False
            nfev = 0
            iters = 0

            if np.isfinite(g0):
                # second point for secant
                y1 = float(np.clip(y0 + dy0, y_min, y_max))
                y1, g1 = nudge_to_finite(P, Sgoal, y1)

                if np.isfinite(g1):
                    try:
                        sol = root_scalar(
                            lambda yy: g_of_y(P, yy, Sgoal),
                            method="secant",
                            x0=y0,
                            x1=y1,
                            rtol=secant_rtol,
                            maxiter=int(secant_maxiter),
                        )
                        nfev = getattr(sol, "function_calls", 0) or 0
                        iters = getattr(sol, "iterations", 0) or 0

                        if sol.converged and np.isfinite(sol.root):
                            y_sol = float(np.clip(sol.root, y_min, y_max))
                            T_sol = float(np.exp(y_sol))
                            # verify
                            if np.isfinite(g_of_y(P, y_sol, Sgoal)):
                                T_out[idx] = T_sol
                                T_prev = T_sol
                                secant_ok = True
                                if return_diagnostics:
                                    diag["success"][idx] = True
                                    diag["method"][idx] = "secant(logT)"
                                    diag["message"][idx] = "OK"
                                    diag["nfev"][idx] = nfev
                                    diag["iterations"][idx] = iters
                    except Exception as e:
                        secant_ok = False
                        if return_diagnostics:
                            diag["method"][idx] = "secant(logT)"
                            diag["message"][idx] = f"Secant exception: {e}"

            if secant_ok:
                continue

            # ---------------------------------------
            # 2) Fallback: bracket + brenth in logT
            # ---------------------------------------
            # Build a bracket around y0
            w = np.log(float(bracket_factor))
            y_lo = float(np.clip(y0 - w, y_min, y_max))
            y_hi = float(np.clip(y0 + w, y_min, y_max))

            f_lo = g_of_y(P, y_lo, Sgoal)
            f_hi = g_of_y(P, y_hi, Sgoal)

            n_expand = 0
            while (
                (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0)
                and n_expand < int(expand_steps)
            ):
                n_expand += 1
                w *= float(expand_factor)
                y_lo = float(np.clip(y0 - w, y_min, y_max))
                y_hi = float(np.clip(y0 + w, y_min, y_max))
                f_lo = g_of_y(P, y_lo, Sgoal)
                f_hi = g_of_y(P, y_hi, Sgoal)

            if return_diagnostics:
                diag["n_expand"][idx] = n_expand

            if (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0):
                # ---- No bracket: likely Sgoal outside achievable S(P,T) within bounds_T,
                # or EOS weirdness. For interpolation scaffolding, clamp to the closer bound. ----
                yb_lo, fb_lo = nudge_to_finite(P, Sgoal, y_min)
                yb_hi, fb_hi = nudge_to_finite(P, Sgoal, y_max)

                if np.isfinite(fb_lo) and np.isfinite(fb_hi):
                    # choose the endpoint that minimizes |residual|
                    y_best = yb_lo if abs(fb_lo) <= abs(fb_hi) else yb_hi
                    T_best = float(np.exp(float(np.clip(y_best, y_min, y_max))))

                    T_out[idx] = T_best
                    T_prev = T_best  # keep warm-start alive

                    if return_diagnostics:
                        diag["success"][idx] = True
                        diag["method"][idx] = "clamp(bounds_T)"
                        diag["message"][idx] = (
                            "No bracket; clamped to nearest bound in residual."
                        )
                    continue

                # If even bounds give NaNs, leave fail_value and move on
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = (
                        "Failed to bracket root and could not evaluate bounds finitely."
                    )
                continue

            try:
                y_root = brenth(
                    lambda yy: g_of_y(P, yy, Sgoal),
                    y_lo, y_hi,
                    xtol=float(brent_xtol),     # MUST be > 0
                    rtol=float(brent_rtol),
                    maxiter=int(brent_maxiter),
                    disp=False,
                )
                y_root = float(np.clip(y_root, y_min, y_max))
                T_root = float(np.exp(y_root))

                T_out[idx] = T_root
                T_prev = T_root  # warm-start seed for next element

                if return_diagnostics:
                    diag["success"][idx] = True
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = "OK"
            except Exception as e:
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = f"Brenth exception: {e}"

        # ---- Post-fill any remaining NaNs (holes or missing edges) along P axis ----
        if fill_nans:
            # move fill_axis to the last axis
            Tout = np.moveaxis(T_out, fill_axis, -1)
            Parr = np.moveaxis(P_arr, fill_axis, -1)

            for sl in np.ndindex(Tout.shape[:-1]):
                Tout[sl] = _fill_1d_loglog(Parr[sl], Tout[sl])

            T_out = np.moveaxis(Tout, -1, fill_axis)

        if return_diagnostics:
            return T_out, diag
        return T_out

    def get_rhot_sp_inv(
        self,
        S_target,
        P_target,
        *,
        s_units="kbbar",
        **kwargs,
    ):
        """
        Convenience wrapper:
        1) invert T from S(P,T)
        2) compute rho(P,T) with the mixed rho rule you specified
        """
        T = self.get_T_sp_inv(S_target, P_target, s_units=s_units, **kwargs)
        rho = self.get_rho_pt(P_target, T, solid_tab=kwargs.get("solid_tab", True), liquid_tab=kwargs.get("liquid_tab", True))
        return rho, T

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

    def get_T_sp(self, S, P, tab=True, **inv_kwargs):
        """
        T(S,P) in K.
        If tab=False, uses Fe_EOS.get_rho_sp_inv.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)

        if tab:
            vals = self._interp(self.t_rgi_sp, S_arr, P_arr)
            return float(vals) if scalar else vals

        T = self.get_T_sp_inv(S_arr, P_arr, **inv_kwargs)
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

        T = self.get_T_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_pt(P_arr, T)  # erg/g (your Fe_EOS convention)
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

        T = self.get_T_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CP_pt(P_arr, T)  # erg/g/K (your Fe_EOS convention)
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

        T = self.get_T_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CV_pt(P_arr, T)  # erg/g/K (your Fe_EOS convention)
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

        T = self.get_T_sp_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_pt(P_arr, T)  # 1/K (your Fe_EOS convention)
        return float(vals) if scalar else vals