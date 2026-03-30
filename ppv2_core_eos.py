"""
ppv2_core_eos.py

Mantle-style wrapper around the ppv2_eos (MgSiO3 post-perovskite) tables.

Exposes the same API used by mgsio3_comb_eos.py, mg2sio4_aneos_eos.py,
and aqua_core_eos.py so that this class can be dropped into the mantle-
evolution code as a mantle_eos option.

Units:
    P   : GPa           (input / output)
    T   : K
    rho : g/cm^3
    S   : erg/g/K       (PT outputs)
          k_B/baryon    (SP inputs, matching the AQUA/MgSiO3 convention)
    U   : erg/g
    Cp  : erg/g/K
    Cv  : erg/g/K
    alpha : 1/K
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.optimize import root_scalar, brenth

try:
    from eos import ppv2_eos
except ImportError:
    import ppv2_eos


class PPV2_CORE_EOS:
    """
    MgSiO3 post-perovskite wrapper with the linear-unit interface used by
    the mantle EOSes.

    The underlying ppv2_eos module stores tables in log-space:
        - Pressure axis : log10(P [dyn/cm^2])
        - Temperature   : log10(T [K])
        - Density output: log10(rho [g/cm^3])
    This wrapper converts between GPa/K/g-cm^3 and those internal coords.
    """

    def __init__(self):
        # ---- unit conversions ----
        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.kbbar_to_erg = 1.0 / self.erg_to_kbbar
        self.GPa_to_dyn = float(u.GPa.to("dyn/cm^2"))
        self.dyn_to_GPa = float((u.dyn / u.cm**2).to("GPa"))
        self.dyn_to_Pa = float((u.dyn / u.cm**2).to("Pa"))

        # Latent heat of fusion for MgSiO3 (same as mgsio3_comb_eos)
        self.L = float(7.322e5 * (u.J / u.kg).to("erg/g"))

        # ---- grab the underlying RGIs from ppv2_eos ----
        # PT table
        self._logrho_rgi_pt = ppv2_eos.rho_rgi_pt
        self._s_rgi_pt = ppv2_eos.s_rgi_pt
        self._logu_rgi_pt = ppv2_eos.u_rgi_pt

        # SP table
        self._logrho_rgi_sp = ppv2_eos.rho_rgi
        self._logt_rgi_sp = ppv2_eos.t_rgi
        self._logu_rgi_sp = ppv2_eos.u_rgi

        # Axis ranges (in log10 dyn/cm^2)
        self._logp_min_pt = float(ppv2_eos.logpvals_pt.min())
        self._logp_max_pt = float(ppv2_eos.logpvals_pt.max())
        self._logt_min_pt = float(ppv2_eos.logtvals_pt.min())
        self._logt_max_pt = float(ppv2_eos.logtvals_pt.max())

    # ------------------------------------------------------------------ #
    #                            helpers                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast(a, b):
        scalar = np.isscalar(a) and np.isscalar(b)
        a_arr = np.array(a, ndmin=1, dtype=float)
        b_arr = np.array(b, ndmin=1, dtype=float)
        if a_arr.shape != b_arr.shape:
            a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
        return scalar, a_arr, b_arr

    @staticmethod
    def _maybe_scalar(scalar, vals):
        vals = np.asarray(vals, dtype=float)
        return float(vals.ravel()[0]) if scalar else vals

    @staticmethod
    def _interp(rgi, x_arr, y_arr):
        pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
        return rgi(pts).reshape(x_arr.shape)

    def _to_logp(self, P_GPa):
        """P [GPa] -> log10(P [dyn/cm^2])."""
        P_arr = np.clip(np.asarray(P_GPa, dtype=float), 1e-300, None)
        return np.log10(P_arr * self.GPa_to_dyn)

    @staticmethod
    def _to_logt(T):
        """T [K] -> log10(T)."""
        return np.log10(np.clip(np.asarray(T, dtype=float), 1e-300, None))

    # ------------------------------------------------------------------ #
    #                         PT interface                                 #
    # ------------------------------------------------------------------ #

    def get_rho_pt(self, P, T):
        """Density from (P, T).  P in GPa, T in K.  Returns rho in g/cm^3."""
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logp = self._to_logp(P_arr)
        logt = self._to_logt(T_arr)
        logrho = self._interp(self._logrho_rgi_pt, logp, logt)
        return self._maybe_scalar(scalar, 10.0 ** logrho)

    def get_s_pt(self, P, T):
        """Entropy from (P, T).  P in GPa, T in K.  Returns S in erg/g/K."""
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logp = self._to_logp(P_arr)
        logt = self._to_logt(T_arr)
        vals = self._interp(self._s_rgi_pt, logp, logt)
        return self._maybe_scalar(scalar, vals)

    def get_u_pt(self, P, T):
        """Internal energy from (P, T).  P in GPa, T in K.  Returns U in erg/g."""
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logp = self._to_logp(P_arr)
        logt = self._to_logt(T_arr)
        logu = self._interp(self._logu_rgi_pt, logp, logt)
        return self._maybe_scalar(scalar, 10.0 ** logu)

    def get_cp_pt(self, P, T, dT_frac=1e-3):
        """
        Isobaric heat capacity from (P, T).
        Cp = T * (dS/dT)_P  via central finite difference.
        P in GPa, T in K.  Returns Cp in erg/g/K.
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)
        dT = np.maximum(T_arr * dT_frac, 1.0)
        s_hi = np.asarray(self.get_s_pt(P_arr, T_arr + dT), dtype=float)
        s_lo = np.asarray(self.get_s_pt(P_arr, T_arr - dT), dtype=float)
        cp = T_arr * (s_hi - s_lo) / (2.0 * dT)
        return self._maybe_scalar(scalar, cp)

    def get_alpha_pt(self, P, T, dT_frac=1e-3):
        """
        Thermal expansion from (P, T).
        alpha = -(1/rho) * (drho/dT)_P  via central finite difference.
        P in GPa, T in K.  Returns alpha in 1/K.
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)
        dT = np.maximum(T_arr * dT_frac, 1.0)
        rho_hi = np.asarray(self.get_rho_pt(P_arr, T_arr + dT), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_arr, T_arr - dT), dtype=float)
        rho_mid = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        alpha = -(1.0 / rho_mid) * (rho_hi - rho_lo) / (2.0 * dT)
        return self._maybe_scalar(scalar, alpha)

    def get_cv_pt(self, P, T, dT_frac=1e-3, dP_frac=1e-3):
        """
        Isochoric heat capacity from (P, T).
        Cv = Cp - T * alpha^2 * KT / rho
        where KT = rho * (dP/drho)_T is computed via (drho/dP)_T.
        P in GPa, T in K.  Returns Cv in erg/g/K.
        """
        scalar, P_arr, T_arr = self._broadcast(P, T)

        cp = np.asarray(self.get_cp_pt(P_arr, T_arr, dT_frac=dT_frac), dtype=float)
        alpha = np.asarray(self.get_alpha_pt(P_arr, T_arr, dT_frac=dT_frac), dtype=float)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)

        # KT via (drho/dP)_T
        dP = np.maximum(P_arr * dP_frac, 1e-3)  # GPa
        rho_hi = np.asarray(self.get_rho_pt(P_arr + dP, T_arr), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_arr - dP, T_arr), dtype=float)
        drho_dP = (rho_hi - rho_lo) / (2.0 * dP)  # g/cm^3 / GPa

        # KT = rho / (drho/dP)  in GPa
        safe_drdP = np.where(np.abs(drho_dP) > 1e-30, drho_dP, 1e-30)
        KT_GPa = rho / safe_drdP  # GPa

        # Cv = Cp - T * alpha^2 * KT / rho
        # Need consistent units: KT in erg/cm^3 = GPa * 1e10
        KT_cgs = KT_GPa * 1e10  # dyn/cm^2 = erg/cm^3
        cv = cp - T_arr * alpha ** 2 * KT_cgs / rho
        cv = np.maximum(cv, 0.0)
        return self._maybe_scalar(scalar, cv)

    def get_alpha_x(self, P, T, rho, **kwargs):
        """
        Density contrast at the melting curve.
        For a single-phase EOS this is effectively zero, but we provide
        a finite-difference estimate across T_melt.
        """
        P_arr, T_arr = np.atleast_1d(P).astype(float), np.atleast_1d(T).astype(float)
        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)
        dT = 50.0
        rho_lo = np.asarray(self.get_rho_pt(P_arr, np.maximum(Tm - dT, 1.0)), dtype=float)
        rho_hi = np.asarray(self.get_rho_pt(P_arr, Tm + dT), dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        return rho_arr * ((1.0 / rho_hi) - (1.0 / rho_lo))

    # Uppercase aliases for core_eos compatibility
    def get_CP_pt(self, P, T, **kwargs):
        """Alias for get_cp_pt (core_eos convention)."""
        return self.get_cp_pt(P, T, **kwargs)

    def get_CV_pt(self, P, T, **kwargs):
        """Alias for get_cv_pt (core_eos convention)."""
        return self.get_cv_pt(P, T, **kwargs)

    # ------------------------------------------------------------------ #
    #                       Melting curve                                   #
    # ------------------------------------------------------------------ #

    def get_T_melt(self, P):
        """
        Melting curve for MgSiO3 post-perovskite.
        Fei et al. (2021).  P in GPa, returns T_melt in K.
        """
        P_arr = np.array(P, ndmin=1, dtype=float)
        return 6295.0 * (P_arr / 140.0) ** 0.317

    # ------------------------------------------------------------------ #
    #                       Entropy helpers                                 #
    # ------------------------------------------------------------------ #

    def get_S_liq_at_melt(self, P):
        """Entropy of the liquid just above T_melt.  Returns erg/g/K."""
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        Tm = self.get_T_melt(P_arr)
        return np.asarray(self.get_s_pt(P_arr, Tm + 50.0), dtype=float)

    def get_S_sol_at_melt(self, P):
        """Entropy of the solid just below T_melt.  Returns erg/g/K."""
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        Tm = self.get_T_melt(P_arr)
        return np.asarray(self.get_s_pt(P_arr, np.maximum(Tm - 50.0, 1.0)), dtype=float)

    # ------------------------------------------------------------------ #
    #                         SP interface                                  #
    # ------------------------------------------------------------------ #

    def get_t_sp(self, S, P, **kwargs):
        """
        Temperature from (S, P).
        S in k_B/baryon, P in GPa.  Returns T in K.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)
        logp = self._to_logp(P_arr)
        logt = self._interp(self._logt_rgi_sp, S_arr, logp)
        return self._maybe_scalar(scalar, 10.0 ** logt)

    # Uppercase alias for core_eos compatibility
    def get_T_sp(self, S, P, **kwargs):
        """Alias for get_t_sp (core_eos convention)."""
        return self.get_t_sp(S, P, **kwargs)

    def get_rho_sp(self, S, P, **kwargs):
        """
        Density from (S, P).
        S in k_B/baryon, P in GPa.  Returns rho in g/cm^3.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)
        logp = self._to_logp(P_arr)
        logrho = self._interp(self._logrho_rgi_sp, S_arr, logp)
        return self._maybe_scalar(scalar, 10.0 ** logrho)

    def get_u_sp(self, S, P, **kwargs):
        """
        Internal energy from (S, P).
        S in k_B/baryon, P in GPa.  Returns U in erg/g.
        """
        scalar, S_arr, P_arr = self._broadcast(S, P)
        logp = self._to_logp(P_arr)
        logu = self._interp(self._logu_rgi_sp, S_arr, logp)
        return self._maybe_scalar(scalar, 10.0 ** logu)

    def get_cp_sp(self, S, P, **kwargs):
        """Cp from (S, P).  S in k_B/baryon, P in GPa.  Returns erg/g/K."""
        T = self.get_t_sp(S, P)
        return self.get_cp_pt(P, T)

    def get_cv_sp(self, S, P, **kwargs):
        """Cv from (S, P).  S in k_B/baryon, P in GPa.  Returns erg/g/K."""
        T = self.get_t_sp(S, P)
        return self.get_cv_pt(P, T)

    def get_alpha_sp(self, S, P, **kwargs):
        """alpha from (S, P).  S in k_B/baryon, P in GPa.  Returns 1/K."""
        T = self.get_t_sp(S, P)
        return self.get_alpha_pt(P, T)

    # ------------------------------------------------------------------ #
    #                      SP inversion                                     #
    # ------------------------------------------------------------------ #

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
        """
        Invert S(P, T) = S_target for T.

        Parameters
        ----------
        S_target : entropy target.
        P_target : pressure in GPa.
        s_units  : 'kbbar' (k_B/baryon, default) or 'cgs' (erg/g/K).
        """
        S_arr = np.atleast_1d(np.asarray(S_target, dtype=float))
        P_arr = np.atleast_1d(np.asarray(P_target, dtype=float))
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        shape = S_arr.shape

        if str(s_units).lower() in ("kbbar", "kb/baryon"):
            S_goal = S_arr * self.kbbar_to_erg
        else:
            S_goal = S_arr.copy()

        T_lo, T_hi = map(float, bounds_T)
        y_min, y_max = np.log(T_lo), np.log(T_hi)

        Tout = np.full(shape, fail_value, dtype=float)

        def g_of_y(P, y, Sg):
            y = float(np.clip(y, y_min, y_max))
            T = float(np.exp(y))
            Sm = float(np.asarray(self.get_s_pt(P, T)).ravel()[0])
            if not np.isfinite(Sm):
                return np.nan
            return Sm - Sg

        if T_guess is None:
            T_guess_arr = np.clip(self.get_T_melt(P_arr), T_lo, T_hi)
        else:
            T_guess_arr = np.broadcast_arrays(np.asarray(T_guess, dtype=float), P_arr)[0]
            T_guess_arr = np.clip(T_guess_arr, T_lo, T_hi)

        T_prev = None

        for idx in np.ndindex(shape):
            P = float(P_arr[idx])
            Sg = float(S_goal[idx])
            if not (np.isfinite(P) and np.isfinite(Sg)) or P <= 0:
                continue

            Tg = float(T_guess_arr[idx] if (T_prev is None or not np.isfinite(T_prev)) else T_prev)
            y0 = float(np.log(np.clip(Tg, T_lo, T_hi)))

            # secant
            try:
                sol = root_scalar(
                    lambda yy: g_of_y(P, yy, Sg),
                    method="secant",
                    x0=y0,
                    x1=y0 + dy0,
                    rtol=secant_rtol,
                    maxiter=int(secant_maxiter),
                )
                if sol.converged and np.isfinite(sol.root):
                    T_sol = float(np.exp(np.clip(sol.root, y_min, y_max)))
                    Tout[idx] = T_sol
                    T_prev = T_sol
                    continue
            except Exception:
                pass

            # bracket + brenth fallback
            w = np.log(float(bracket_factor))
            y_lo_b = float(np.clip(y0 - w, y_min, y_max))
            y_hi_b = float(np.clip(y0 + w, y_min, y_max))
            for _ in range(int(expand_steps)):
                f_lo = g_of_y(P, y_lo_b, Sg)
                f_hi = g_of_y(P, y_hi_b, Sg)
                if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
                    break
                w *= float(expand_factor)
                y_lo_b = float(np.clip(y0 - w, y_min, y_max))
                y_hi_b = float(np.clip(y0 + w, y_min, y_max))
            else:
                continue

            try:
                y_root = brenth(
                    lambda yy: g_of_y(P, yy, Sg),
                    y_lo_b, y_hi_b,
                    xtol=float(brent_xtol),
                    rtol=float(brent_rtol),
                    maxiter=int(brent_maxiter),
                )
                T_root = float(np.exp(np.clip(y_root, y_min, y_max)))
                Tout[idx] = T_root
                T_prev = T_root
            except Exception:
                pass

        if return_diagnostics:
            return Tout, {}
        return Tout
