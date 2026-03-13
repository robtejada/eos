"""
aqua_core_eos.py

Mantle-style wrapper around the legacy AQUA water EOS tables.

This exposes the same broad API used by the rock/core EOS wrappers:
    - P in GPa
    - T in K
    - rho in g/cm^3
    - S outputs from PT/RHOT/RHOU getters in erg/g/K
    - S inputs to SP getters in k_B / baryon by default
    - U in erg/g
    - Cp, Cv in erg/g/K
    - alpha in 1/K

The direct thermodynamic surfaces come from the existing AQUA tables:
    - PT   : rho(P, T), S(P, T), U(P, T)
    - RHOT : P(rho, T), S(rho, T), U(rho, T)
    - RHOU : P(rho, U), S(rho, U), T(rho, U)
    - SP   : rho(S, P), T(S, P) with S in k_B / baryon

Derived quantities are evaluated numerically using the RHOT/RHOU tables so
that this class can be dropped into mantle-evolution code that expects the
Mg-silicate EOS interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.optimize import root_scalar

try:
    from eos import aqua_eos
except ImportError:  # pragma: no cover - convenient for local direct imports
    import aqua_eos


ArrayLike = Union[float, int, np.ndarray]


@dataclass
class Domain:
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float
    P_min: float
    P_max: float


class AQUA_CORE_EOS:
    """
    AQUA wrapper with the same linear-unit interface used by the mantle EOSes.

    Notes
    -----
    - `get_s_pt`, `get_s_rhot`, and `get_s_rhou` return cgs entropy
      (erg/g/K).
    - `get_t_sp`, `get_rho_sp`, and related SP getters expect entropy in
      k_B / baryon by default, matching the existing AQUA SP tables.
    - AQUA already provides direct PT and SP tables, so `tab=False` is kept as
      a compatibility argument and aliases the same direct table-backed path.
    """

    def __init__(self):
        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.kbbar_to_erg = 1.0 / self.erg_to_kbbar
        self.GPa_to_dyn = float(u.GPa.to("dyn/cm^2"))
        self.dyn_to_GPa = float((u.dyn / u.cm**2).to("GPa"))

        self._has_pt_table = True
        self._has_sp_table = True
        self._has_rhou_table = True

        self.logpvals_pt = np.asarray(aqua_eos.logpvals_pt, dtype=float)
        self.logtvals_pt = np.asarray(aqua_eos.logtvals, dtype=float)
        self.logrhovals_rhot = np.asarray(aqua_eos.logrhovals_rhot, dtype=float)
        self.logtvals_rhot = np.asarray(aqua_eos.logtvals_rhot, dtype=float)
        self.loguvals_rhou = np.asarray(aqua_eos.loguvals_rhou, dtype=float)

        self.svals_sp = np.asarray(aqua_eos.svals_sp, dtype=float)  # kb/baryon
        self.logpvals_sp = np.asarray(aqua_eos.logpvals_sp, dtype=float)

        self.svals_srho = np.asarray(aqua_eos.svals_srho, dtype=float)  # kb/baryon
        self.logrhovals_srho = np.asarray(aqua_eos.logrhovals_srho, dtype=float)

        self.domain = Domain(
            rho_min=float(10.0 ** np.min(self.logrhovals_rhot)),
            rho_max=float(10.0 ** np.max(self.logrhovals_rhot)),
            T_min=float(10.0 ** np.min(self.logtvals_rhot)),
            T_max=float(10.0 ** np.max(self.logtvals_rhot)),
            P_min=float((10.0 ** np.min(self.logpvals_pt)) * self.dyn_to_GPa),
            P_max=float((10.0 ** np.max(self.logpvals_pt)) * self.dyn_to_GPa),
        )

        self._logrho_rgi_pt = aqua_eos.rho_rgi
        self._s_rgi_pt = aqua_eos.s_rgi_pt
        self._logu_rgi_pt = aqua_eos.u_rgi_pt

        self._logp_rgi_rhot = aqua_eos.p_rgi_rhot
        self._s_rgi_rhot = aqua_eos.s_rgi_rhot
        self._logu_rgi_rhot = aqua_eos.u_rgi_rhot

        self._s_rgi_rhou = aqua_eos.s_rgi_rhou
        self._logp_rgi_rhou = aqua_eos.p_rgi_rhou
        self._logt_rgi_rhou = aqua_eos.t_rgi_rhou

        self._logrho_rgi_sp = aqua_eos.get_rho_rgi_sp
        self._logt_rgi_sp = aqua_eos.get_t_rgi_sp

        self._logp_rgi_srho = aqua_eos.get_p_rgi_srho
        self._logt_rgi_srho = aqua_eos.get_t_rgi_srho

        self._build_phase_transition_profiles()

    # ---------- helpers ----------

    @staticmethod
    def _broadcast(a, b):
        scalar = np.isscalar(a) and np.isscalar(b)
        a_arr = np.array(a, ndmin=1, dtype=float)
        b_arr = np.array(b, ndmin=1, dtype=float)
        if a_arr.shape != b_arr.shape:
            a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
        return scalar, a_arr, b_arr

    @staticmethod
    def _maybe_scalar(scalar: bool, vals):
        vals = np.asarray(vals, dtype=float)
        return float(vals.reshape(-1)[0]) if scalar else vals

    @staticmethod
    def _as_arrays(a, b):
        a_arr = np.array(a, ndmin=1, dtype=float)
        b_arr = np.array(b, ndmin=1, dtype=float)
        a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
        return a_arr, b_arr

    @staticmethod
    def _interp(rgi, x_arr, y_arr):
        pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
        return rgi(pts).reshape(x_arr.shape)

    @staticmethod
    def _clip_positive(vals, floor=1e-300):
        return np.clip(np.asarray(vals, dtype=float), floor, None)

    def _interp_loglog_profile(self, P, logp_knots, logt_knots, left=np.nan, right="hold"):
        scalar, P_arr, _ = self._broadcast(P, P)
        logp = np.log10(self._clip_positive(P_arr))
        vals = np.full(P_arr.shape, np.nan, dtype=float)

        in_range = (logp >= logp_knots[0]) & (logp <= logp_knots[-1])
        if np.any(in_range):
            vals[in_range] = 10.0 ** np.interp(logp[in_range], logp_knots, logt_knots)

        below = logp < logp_knots[0]
        if np.any(below) and np.isfinite(left):
            vals[below] = float(left)

        above = logp > logp_knots[-1]
        if np.any(above):
            if right == "hold":
                vals[above] = 10.0 ** logt_knots[-1]
            elif right == "extrap":
                slope = (logt_knots[-1] - logt_knots[-2]) / (logp_knots[-1] - logp_knots[-2])
                vals[above] = 10.0 ** (logt_knots[-1] + slope * (logp[above] - logp_knots[-1]))
            elif np.isfinite(right):
                vals[above] = float(right)
            else:
                vals[above] = np.nan

        return self._maybe_scalar(scalar, vals)

    def _to_logp(self, P):
        P_arr = self._clip_positive(P)
        return np.log10(P_arr * self.GPa_to_dyn)

    @staticmethod
    def _to_logt(T):
        T_arr = np.clip(np.asarray(T, dtype=float), 1e-300, None)
        return np.log10(T_arr)

    @staticmethod
    def _to_logrho(rho):
        rho_arr = np.clip(np.asarray(rho, dtype=float), 1e-300, None)
        return np.log10(rho_arr)

    @staticmethod
    def _to_logu(U):
        U_arr = np.clip(np.asarray(U, dtype=float), 1e-300, None)
        return np.log10(U_arr)

    def _from_logp(self, logp):
        return (10.0 ** np.asarray(logp, dtype=float)) * self.dyn_to_GPa

    @staticmethod
    def _from_logt(logt):
        return 10.0 ** np.asarray(logt, dtype=float)

    @staticmethod
    def _from_logrho(logrho):
        return 10.0 ** np.asarray(logrho, dtype=float)

    @staticmethod
    def _from_logu(logu):
        return 10.0 ** np.asarray(logu, dtype=float)

    def _entropy_to_cgs(self, s_in, *, s_units: str = "kbbar"):
        su = str(s_units).lower()
        s_arr = np.asarray(s_in, dtype=float)
        if su in ("kbbar", "kb/baryon", "kbperbaryon", "native"):
            return s_arr * self.kbbar_to_erg
        if su in ("cgs", "erg/g/k", "erg/g/kelvin"):
            return s_arr
        raise ValueError("s_units must be one of {'kbbar', 'cgs', 'native'}")

    def _entropy_to_sp_table(self, s_in, *, s_units: str = "kbbar"):
        return self._entropy_to_cgs(s_in, s_units=s_units) * self.erg_to_kbbar

    def _build_phase_transition_profiles(self):
        """
        Build phase-transition profiles directly from the AQUA PT phase map.

        AQUA phase IDs from the table header:
            ice-VII: -7
            ice-X  : -10
            supercritical+superionic: 5

        The full solidus (max T among all solid phases at fixed P) is retained
        as a helper profile. The strict `ice_x` branch is kept on the native
        AQUA phase-5 -> ice-X boundary, while the `superionic` branch is a
        digitized fit to the broad experimental melting line in Fig. 5 of
        Millot et al. (2018).
        """
        pt_df = aqua_eos.aqua_reader("pt")
        press_pa = np.asarray(pt_df["press"], dtype=float)
        press_gpa = press_pa * float(u.Pa.to("GPa"))
        temp_k = np.asarray(pt_df["temp"], dtype=float)
        s_cgs = np.asarray(pt_df["s"], dtype=float)
        phase = np.asarray(pt_df["phase"], dtype=float)

        solid_mask = phase < 0.0
        solid_pressures = np.unique(press_gpa[solid_mask])
        solidus_t = np.empty_like(solid_pressures)
        for i, p_gpa in enumerate(solid_pressures):
            mask = solid_mask & (press_gpa == p_gpa)
            solidus_t[i] = np.max(temp_k[mask])

        self._solidus_pressures_gpa = solid_pressures
        self._solidus_logp_gpa = np.log10(np.clip(solid_pressures, 1e-300, None))
        self._solidus_t = solidus_t

        icex_mask = phase == -10.0
        icex_pressures = np.unique(press_gpa[icex_mask])
        icex_t = np.empty_like(icex_pressures)
        icex_latent_ergg = np.full_like(icex_pressures, np.nan, dtype=float)

        for i, p_gpa in enumerate(icex_pressures):
            solid_here = (press_gpa == p_gpa) & (phase == -10.0)
            fluid_here = (press_gpa == p_gpa) & (phase == 5.0)
            t_solid = np.max(temp_k[solid_here])
            icex_t[i] = t_solid

            if np.any(fluid_here):
                fluid_t = temp_k[fluid_here]
                fluid_s = s_cgs[fluid_here]
                above = fluid_t > t_solid
                if np.any(above):
                    j = np.argmin(fluid_t[above])
                    s_solid = s_cgs[solid_here & (temp_k == t_solid)][0]
                    s_fluid = fluid_s[above][j]
                    t_mid = 0.5 * (t_solid + fluid_t[above][j])
                    icex_latent_ergg[i] = t_mid * (s_fluid - s_solid)

        self._icex_pressures_gpa = icex_pressures
        self._icex_logp_gpa = np.log10(np.clip(icex_pressures, 1e-300, None))
        self._icex_t = icex_t
        self._icex_latent_ergg = icex_latent_ergg
        self.P_ice_x_min_GPa = float(icex_pressures[0]) if icex_pressures.size else np.nan

        rep_mask = (
            np.isfinite(icex_latent_ergg)
            & (icex_pressures >= self.P_ice_x_min_GPa)
            & (icex_pressures <= 100.0)
        )
        if not np.any(rep_mask):
            rep_mask = np.isfinite(icex_latent_ergg)
        self.L = float(np.nanmedian(icex_latent_ergg[rep_mask])) if np.any(rep_mask) else 0.0
        self.L_aqua_ice_x = self.L
        self.L_ice_x = self.L_aqua_ice_x
        self.P_ice_x_min_GPa_aqua = self.P_ice_x_min_GPa

        # Approximate broad white experimental melting line digitized from
        # Fig. 5 of Millot et al. (2018, Nat. Phys.). This is the superionic
        # fluid boundary requested by the user.
        self.P_superionic_min_GPa = 10.0
        self.P_superionic_max_data_GPa = 340.0
        self._superionic_pressures_gpa = np.array(
            [10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 340.0],
            dtype=float,
        )
        self._superionic_t = np.array(
            [700.0, 950.0, 1300.0, 1800.0, 2300.0, 3000.0, 4000.0, 5000.0, 5700.0, 6300.0, 6800.0],
            dtype=float,
        )
        self._superionic_logp_gpa = np.log10(self._superionic_pressures_gpa)
        self._superionic_logt = np.log10(self._superionic_t)

        self.L_superionic = self._estimate_latent_heat_from_curve(
            self._superionic_pressures_gpa, phase="superionic"
        )
        self.L = self.L_ice_x

    @staticmethod
    def _normalize_transition_phase(phase: Optional[str]) -> str:
        if phase is None:
            raise ValueError("phase must be specified explicitly: use 'superionic' or 'ice_x'.")
        key = str(phase).strip().lower()
        if key in ("icex", "ice_x", "ice-x", "ice x"):
            return "ice_x"
        if key in ("superionic", "superion", "si"):
            return "superionic"
        raise ValueError("phase must be one of {'superionic', 'ice_x'}.")

    def _estimate_latent_heat_from_curve(self, pressures_gpa, phase: str):
        p_arr = np.asarray(pressures_gpa, dtype=float)
        if p_arr.size == 0:
            return np.nan

        t_melt = np.asarray(self.get_T_melt(p_arr, phase=phase), dtype=float)
        dT = np.maximum(25.0, 0.02 * t_melt)
        t_lo = np.clip(t_melt - dT, self.domain.T_min, self.domain.T_max)
        t_hi = np.clip(t_melt + dT, self.domain.T_min, self.domain.T_max)

        s_sol = np.asarray(self.get_s_pt(p_arr, t_lo), dtype=float)
        s_liq = np.asarray(self.get_s_pt(p_arr, t_hi), dtype=float)
        latent = t_melt * (s_liq - s_sol)
        latent = latent[np.isfinite(latent) & (latent > 0.0)]
        if latent.size == 0:
            return np.nan
        return float(np.nanmedian(latent))

    def _dT_dS_sp_num(self, s_tab, P, eps_rel=1e-3):
        s_cgs = self._entropy_to_cgs(s_tab, s_units="kbbar")
        s_hi_cgs = s_cgs * (1.0 + eps_rel)
        s_lo_cgs = np.maximum(s_cgs * (1.0 - eps_rel), 1e-30)
        s_hi_tab = self._entropy_to_sp_table(s_hi_cgs, s_units="cgs")
        s_lo_tab = self._entropy_to_sp_table(s_lo_cgs, s_units="cgs")
        T_hi = self.get_t_sp_tab(s_hi_tab, P)
        T_lo = self.get_t_sp_tab(s_lo_tab, P)
        return (T_hi - T_lo) / (s_hi_cgs - s_lo_cgs)

    def _dT_dS_srho_num(self, s_tab, rho, eps_rel=1e-3):
        s_cgs = self._entropy_to_cgs(s_tab, s_units="kbbar")
        s_hi_cgs = s_cgs * (1.0 + eps_rel)
        s_lo_cgs = np.maximum(s_cgs * (1.0 - eps_rel), 1e-30)
        s_hi_tab = self._entropy_to_sp_table(s_hi_cgs, s_units="cgs")
        s_lo_tab = self._entropy_to_sp_table(s_lo_cgs, s_units="cgs")
        T_hi = self.get_t_srho_tab(s_hi_tab, rho)
        T_lo = self.get_t_srho_tab(s_lo_tab, rho)
        return (T_hi - T_lo) / (s_hi_cgs - s_lo_cgs)

    # ---------- native PT getters ----------

    def get_rho_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logrho = self._interp(self._logrho_rgi_pt, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logrho(logrho))

    def get_s_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._interp(self._s_rgi_pt, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, vals)

    def get_u_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logu = self._interp(self._logu_rgi_pt, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logu(logu))

    # ---------- native RHOT getters ----------

    def get_p_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        logp = self._interp(self._logp_rgi_rhot, self._to_logrho(rho_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logp(logp))

    def get_s_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        vals = self._interp(self._s_rgi_rhot, self._to_logrho(rho_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, vals)

    def get_u_rhot_tab(self, rho, T):
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        logu = self._interp(self._logu_rgi_rhot, self._to_logrho(rho_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logu(logu))

    # ---------- native RHOU getters ----------

    def get_s_rhou_tab(self, rho, U):
        scalar, rho_arr, U_arr = self._broadcast(rho, U)
        vals = self._interp(self._s_rgi_rhou, self._to_logrho(rho_arr), self._to_logu(U_arr))
        return self._maybe_scalar(scalar, vals)

    def get_p_rhou_tab(self, rho, U):
        scalar, rho_arr, U_arr = self._broadcast(rho, U)
        logp = self._interp(self._logp_rgi_rhou, self._to_logrho(rho_arr), self._to_logu(U_arr))
        return self._maybe_scalar(scalar, self._from_logp(logp))

    def get_t_rhou_tab(self, rho, U):
        scalar, rho_arr, U_arr = self._broadcast(rho, U)
        logt = self._interp(self._logt_rgi_rhou, self._to_logrho(rho_arr), self._to_logu(U_arr))
        return self._maybe_scalar(scalar, self._from_logt(logt))

    # ---------- native SP/SRHO getters ----------

    def get_rho_sp_tab(self, S_tab, P):
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        logrho = self._interp(self._logrho_rgi_sp, S_arr, self._to_logp(P_arr))
        return self._maybe_scalar(scalar, self._from_logrho(logrho))

    def get_t_sp_tab(self, S_tab, P):
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        logt = self._interp(self._logt_rgi_sp, S_arr, self._to_logp(P_arr))
        return self._maybe_scalar(scalar, self._from_logt(logt))

    def get_u_sp_tab(self, S_tab, P):
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        T = self.get_t_sp_tab(S_arr, P_arr)
        vals = self.get_u_pt_tab(P_arr, T)
        return self._maybe_scalar(scalar, vals)

    def get_p_srho_tab(self, S_tab, rho):
        scalar, S_arr, rho_arr = self._broadcast(S_tab, rho)
        logp = self._interp(self._logp_rgi_srho, S_arr, self._to_logrho(rho_arr))
        return self._maybe_scalar(scalar, self._from_logp(logp))

    def get_t_srho_tab(self, S_tab, rho):
        scalar, S_arr, rho_arr = self._broadcast(S_tab, rho)
        logt = self._interp(self._logt_rgi_srho, S_arr, self._to_logrho(rho_arr))
        return self._maybe_scalar(scalar, self._from_logt(logt))

    # ---------- public RHOT/RHOU getters ----------

    def get_p_rhot(self, rho, T):
        return self.get_p_rhot_tab(rho, T)

    def get_s_rhot(self, rho, T):
        return self.get_s_rhot_tab(rho, T)

    def get_u_rhot(self, rho, T):
        return self.get_u_rhot_tab(rho, T)

    def get_s_rhou(self, rho, U):
        return self.get_s_rhou_tab(rho, U)

    def get_p_rhou(self, rho, U):
        return self.get_p_rhou_tab(rho, U)

    def get_t_rhou(self, rho, U):
        return self.get_t_rhou_tab(rho, U)

    # ---------- RHOT/RHOU-derived derivatives ----------

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
        rho_lo = np.maximum(rho_arr * (1.0 - eps_rel), 1e-30)
        p_hi = self.get_p_rhot_tab(rho_hi, T_arr)
        p_lo = self.get_p_rhot_tab(rho_lo, T_arr)
        return (p_hi - p_lo) / (rho_hi - rho_lo)

    def get_alpha_rhot(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        dP_dT = self._dP_dT_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
        dP_drho = self._dP_drho_rhot_num(rho_arr, T_arr, eps_rel=eps_rel)
        alpha = dP_dT / np.maximum(rho_arr * dP_drho, 1e-99)
        P_arr = np.asarray(self.get_p_rhot(rho_arr, T_arr), dtype=float)
        alpha_pt = np.asarray(self.get_alpha_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        bad = (~np.isfinite(alpha)) | (np.abs(alpha) > 10.0 * np.maximum(np.abs(alpha_pt), 1e-12))
        alpha = np.where(bad, alpha_pt, alpha)
        scalar = np.isscalar(rho) and np.isscalar(T)
        return self._maybe_scalar(scalar, alpha)

    def _get_cv_rhot_rhou(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        U_arr = np.asarray(self.get_u_rhot(rho_arr, T_arr), dtype=float)
        U_hi = U_arr * (1.0 + eps_rel)
        U_lo = np.maximum(U_arr * (1.0 - eps_rel), 1e-30)
        T_hi = np.asarray(self.get_t_rhou_tab(rho_arr, U_hi), dtype=float)
        T_lo = np.asarray(self.get_t_rhou_tab(rho_arr, U_lo), dtype=float)
        dT_dU = (T_hi - T_lo) / (U_hi - U_lo)
        return 1.0 / np.maximum(dT_dU, 1e-99)

    def _get_cv_rhot_srho(self, rho, T, eps_rel=1e-3):
        rho_arr, T_arr = self._as_arrays(rho, T)
        s_cgs = np.asarray(self.get_s_rhot(rho_arr, T_arr), dtype=float)
        s_tab = self._entropy_to_sp_table(s_cgs, s_units="cgs")
        dT_dS = self._dT_dS_srho_num(s_tab, rho_arr, eps_rel=eps_rel)
        return T_arr / np.maximum(dT_dS, 1e-99)

    def get_cv_rhot(self, rho, T, eps_rel=1e-3, method="rhou"):
        method_l = str(method).lower()
        if method_l == "rhou":
            cv = self._get_cv_rhot_rhou(rho, T, eps_rel=eps_rel)
        elif method_l == "srho":
            cv = self._get_cv_rhot_srho(rho, T, eps_rel=eps_rel)
        else:
            raise ValueError("method must be one of {'rhou', 'srho'}")
        scalar = np.isscalar(rho) and np.isscalar(T)
        return self._maybe_scalar(scalar, cv)

    def get_cp_rhot(self, rho, T, eps_rel=1e-3, method="cv_relation"):
        rho_arr, T_arr = self._as_arrays(rho, T)
        method_l = str(method).lower()

        if method_l == "cv_relation":
            cv = np.asarray(self.get_cv_rhot(rho_arr, T_arr, eps_rel=eps_rel, method="rhou"), dtype=float)
            P_arr = np.asarray(self.get_p_rhot(rho_arr, T_arr), dtype=float)
            alpha = np.asarray(self.get_alpha_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
            dP_drho = self._dP_drho_rhot_num(rho_arr, T_arr, eps_rel=eps_rel) * self.GPa_to_dyn
            cp = cv + T_arr * alpha * alpha * dP_drho
        elif method_l == "entropy":
            P_arr = np.asarray(self.get_p_rhot(rho_arr, T_arr), dtype=float)
            s_cgs = np.asarray(self.get_s_rhot(rho_arr, T_arr), dtype=float)
            s_tab = self._entropy_to_sp_table(s_cgs, s_units="cgs")
            dT_dS = self._dT_dS_sp_num(s_tab, P_arr, eps_rel=eps_rel)
            cp = T_arr / np.maximum(dT_dS, 1e-99)
        else:
            raise ValueError("method must be one of {'cv_relation', 'entropy'}")

        scalar = np.isscalar(rho) and np.isscalar(T)
        return self._maybe_scalar(scalar, cp)

    # ---------- PT getters ----------

    def get_rho_pt_inv(self, P, T, **kwargs):
        del kwargs
        return self.get_rho_pt_tab(P, T)

    def get_rho_pt(self, P, T, tab=True, **kwargs):
        del tab, kwargs
        return self.get_rho_pt_tab(P, T)

    def get_s_pt(self, P, T, tab=True):
        del tab
        return self.get_s_pt_tab(P, T)

    def get_u_pt(self, P, T, tab=True):
        del tab
        return self.get_u_pt_tab(P, T)

    def get_alpha_pt_tab(self, P, T, eps_rel=1e-3):
        return self.get_alpha_pt(P, T, eps_rel=eps_rel)

    def get_cp_pt_tab(self, P, T, eps_rel=1e-3, method="cv_relation"):
        return self.get_cp_pt(P, T, eps_rel=eps_rel, method=method)

    def get_cv_pt_tab(self, P, T, eps_rel=1e-3, method="rhou"):
        return self.get_cv_pt(P, T, eps_rel=eps_rel, method=method)

    def get_alpha_pt(self, P, T, tab=True, eps_rel=1e-3):
        del tab
        P_arr, T_arr = self._as_arrays(P, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        rho_hi = np.asarray(self.get_rho_pt(P_arr, T_hi), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_arr, T_lo), dtype=float)
        drho_dT = (rho_hi - rho_lo) / (T_hi - T_lo)
        alpha = -(drho_dT / np.maximum(rho, 1e-99))
        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, alpha)

    def get_cp_pt(self, P, T, tab=True, eps_rel=1e-3, method="cv_relation"):
        del tab
        P_arr, T_arr = self._as_arrays(P, T)
        rho = self.get_rho_pt(P_arr, T_arr)
        cp = self.get_cp_rhot(rho, T_arr, eps_rel=eps_rel, method=method)
        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, cp)

    def get_cv_pt(self, P, T, tab=True, eps_rel=1e-3, method="rhou"):
        del tab
        P_arr, T_arr = self._as_arrays(P, T)
        rho = self.get_rho_pt(P_arr, T_arr)
        cv = self.get_cv_rhot(rho, T_arr, eps_rel=eps_rel, method=method)
        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, cv)

    # ---------- compatibility hooks ----------

    def get_T_melt_ice_x(self, P):
        scalar, P_arr, _ = self._broadcast(P, P)
        logp = np.log10(self._clip_positive(P_arr))

        # Match the earlier mantle-facing behavior: below the onset of the
        # explicit Ice X branch, continue along the AQUA solidus so callers
        # still see a finite freezing curve; above it, follow the phase-5 ->
        # ice-X boundary directly.
        vals = np.interp(
            logp,
            self._solidus_logp_gpa,
            self._solidus_t,
            left=self._solidus_t[0],
            right=self._solidus_t[-1],
        )

        icex_mask = P_arr >= self.P_ice_x_min_GPa
        if np.any(icex_mask):
            vals[icex_mask] = np.interp(
                logp[icex_mask],
                self._icex_logp_gpa,
                self._icex_t,
                left=np.nan,
                right=self._icex_t[-1],
            )

        return self._maybe_scalar(scalar, vals)

    def get_T_melt_superionic(self, P):
        return self._interp_loglog_profile(
            P,
            self._superionic_logp_gpa,
            self._superionic_logt,
            left=np.nan,
            right="extrap",
        )

    def get_T_solidus(self, P):
        return self._interp_loglog_profile(
            P,
            self._solidus_logp_gpa,
            np.log10(self._solidus_t),
            left=self._solidus_t[0],
            right="hold",
        )

    def get_T_melt(self, P, phase=None):
        phase_key = self._normalize_transition_phase(phase)
        if phase_key == "ice_x":
            return self.get_T_melt_ice_x(P)
        return self.get_T_melt_superionic(P)

    def get_S_liq_at_melt(self, P, phase="ice_x"):
        scalar, P_arr, _ = self._broadcast(P, P)
        Tm = np.asarray(self.get_T_melt(P_arr, phase=phase), dtype=float)
        dT = np.maximum(25.0, 0.02 * Tm)
        T_liq = np.clip(Tm + dT, self.domain.T_min, self.domain.T_max)
        vals = self.get_s_pt(P_arr, T_liq, tab=True)
        return self._maybe_scalar(scalar, vals)

    def get_S_sol_at_melt(self, P, phase="ice_x"):
        scalar, P_arr, _ = self._broadcast(P, P)
        Tm = np.asarray(self.get_T_melt(P_arr, phase=phase), dtype=float)
        dT = np.maximum(25.0, 0.02 * Tm)
        T_sol = np.clip(Tm - dT, self.domain.T_min, self.domain.T_max)
        vals = self.get_s_pt(P_arr, T_sol, tab=True)
        return self._maybe_scalar(scalar, vals)

    def get_alpha_x(self, P, T, rho, **kwargs):
        del T
        phase = kwargs.pop("phase", "ice_x")
        dT_melt = kwargs.pop("dT_melt", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        scalar, P_arr, rho_arr = self._broadcast(P, rho)
        Tm = np.asarray(self.get_T_melt(P_arr, phase=phase), dtype=float)
        if dT_melt is None:
            dT = np.maximum(25.0, 0.02 * Tm)
        else:
            dT = np.asarray(dT_melt, dtype=float)
            if dT.shape == ():
                dT = np.full(P_arr.shape, float(dT), dtype=float)
            else:
                dT = np.broadcast_to(dT, P_arr.shape)
            dT = np.maximum(dT, 1.0)

        T_sol = np.clip(Tm - dT, self.domain.T_min, self.domain.T_max)
        T_liq = np.clip(Tm + dT, self.domain.T_min, self.domain.T_max)
        rho_sol = np.asarray(self.get_rho_pt(P_arr, T_sol, tab=True), dtype=float)
        rho_liq = np.asarray(self.get_rho_pt(P_arr, T_liq, tab=True), dtype=float)
        alpha_x = rho_arr * (
            1.0 / np.maximum(rho_liq, 1e-99)
            - 1.0 / np.maximum(rho_sol, 1e-99)
        )
        return self._maybe_scalar(scalar, alpha_x)

    # ---------- SP getters ----------

    def get_t_sp_inv(self, S, P, s_units="kbbar", **kwargs):
        T_guess = kwargs.pop("T_guess", None)
        bounds_T = kwargs.pop("bounds_T", None)
        secant_maxiter = int(kwargs.pop("secant_maxiter", 30))
        secant_rtol = float(kwargs.pop("secant_rtol", 1e-10))
        dy0 = float(kwargs.pop("dy0", 1e-3))
        bracket_factor = float(kwargs.pop("bracket_factor", 1.3))
        expand_steps = int(kwargs.pop("expand_steps", 12))
        expand_factor = float(kwargs.pop("expand_factor", 2.0))
        brent_rtol = float(kwargs.pop("brent_rtol", 1e-10))
        brent_xtol = float(kwargs.pop("brent_xtol", 1e-6))
        brent_maxiter = int(kwargs.pop("brent_maxiter", 200))
        return_diagnostics = bool(kwargs.pop("return_diagnostics", False))
        fail_value = kwargs.pop("fail_value", np.nan)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        S_arr = np.asarray(S, dtype=float)
        P_arr = np.asarray(P, dtype=float)
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        shape = S_arr.shape

        S_goal = self._entropy_to_cgs(S_arr, s_units=s_units)

        if bounds_T is None:
            T_lo = max(1.0, float(self.domain.T_min))
            T_hi = float(self.domain.T_max)
        else:
            T_lo, T_hi = map(float, bounds_T)
        if T_lo <= 0.0 or T_hi <= T_lo:
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

        def g_of_y(P_val, y_val, S_cgs):
            y_val = float(np.clip(y_val, y_min, y_max))
            T_val = float(np.exp(y_val))
            Sm = self.get_s_pt(P_val, T_val, tab=True)
            Sm = float(np.asarray(Sm).reshape(-1)[0])
            if not np.isfinite(Sm):
                return np.nan
            return Sm - S_cgs

        def nudge_to_finite(P_val, S_cgs, y_guess):
            y_cur = float(np.clip(y_guess, y_min, y_max))
            g_cur = g_of_y(P_val, y_cur, S_cgs)
            if np.isfinite(g_cur):
                return y_cur, g_cur
            for step in (0.2, -0.2, 0.5, -0.5, 1.0, -1.0):
                y_try = float(np.clip(y_cur + step, y_min, y_max))
                g_try = g_of_y(P_val, y_try, S_cgs)
                if np.isfinite(g_try):
                    return y_try, g_try
            return y_cur, np.nan

        if T_guess is None:
            T_guess_arr = np.full(shape, np.sqrt(T_lo * T_hi), dtype=float)
            if self._has_sp_table:
                try:
                    S_tab_guess = self._entropy_to_sp_table(S_arr, s_units=s_units)
                    T_guess_arr = np.asarray(self.get_t_sp_tab(S_tab_guess, P_arr), dtype=float)
                except Exception:
                    pass
        else:
            T_guess_arr = np.asarray(T_guess, dtype=float)
            T_guess_arr, _ = np.broadcast_arrays(T_guess_arr, P_arr)
        T_guess_arr = np.clip(T_guess_arr, T_lo, T_hi)

        T_prev = None

        for idx in np.ndindex(shape):
            P_val = float(P_arr[idx])
            S_cgs = float(S_goal[idx])

            if not (np.isfinite(P_val) and np.isfinite(S_cgs)) or P_val <= 0.0:
                if return_diagnostics:
                    diag["method"][idx] = "none"
                    diag["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            Tg = float(T_guess_arr[idx] if (T_prev is None or not np.isfinite(T_prev)) else T_prev)
            y0 = float(np.log(np.clip(Tg, T_lo, T_hi)))
            y0, g0 = nudge_to_finite(P_val, S_cgs, y0)

            secant_ok = False
            if np.isfinite(g0):
                y1 = float(np.clip(y0 + dy0, y_min, y_max))
                y1, g1 = nudge_to_finite(P_val, S_cgs, y1)
                if np.isfinite(g1):
                    try:
                        sol = root_scalar(
                            lambda yy: g_of_y(P_val, yy, S_cgs),
                            method="secant",
                            x0=y0,
                            x1=y1,
                            rtol=secant_rtol,
                            maxiter=secant_maxiter,
                        )
                        if sol.converged and np.isfinite(sol.root):
                            y_sol = float(np.clip(sol.root, y_min, y_max))
                            T_sol = float(np.exp(y_sol))
                            g_sol = g_of_y(P_val, y_sol, S_cgs)
                            if np.isfinite(g_sol):
                                Tout[idx] = T_sol
                                T_prev = T_sol
                                secant_ok = True
                                if return_diagnostics:
                                    diag["success"][idx] = True
                                    diag["method"][idx] = "secant(logT)"
                                    diag["message"][idx] = "OK"
                                    diag["iterations"][idx] = getattr(sol, "iterations", 0)
                                    diag["nfev"][idx] = getattr(sol, "function_calls", 0)
                    except Exception as exc:
                        if return_diagnostics:
                            diag["method"][idx] = "secant(logT)"
                            diag["message"][idx] = f"Secant exception: {exc}"

            if secant_ok:
                continue

            try:
                yc = float(np.log(np.clip(Tg, T_lo, T_hi)))
                gc = g_of_y(P_val, yc, S_cgs)
                if not np.isfinite(gc):
                    yc, gc = nudge_to_finite(P_val, S_cgs, yc)

                if not np.isfinite(gc):
                    raise RuntimeError("Could not find a finite entropy residual near the initial guess.")

                yl = yr = yc
                n_expand = 0
                bracket_ok = False
                step = np.log(bracket_factor)

                for k in range(expand_steps):
                    n_expand = k + 1
                    yl = float(max(y_min, yc - step))
                    yr = float(min(y_max, yc + step))
                    gl = g_of_y(P_val, yl, S_cgs)
                    gr = g_of_y(P_val, yr, S_cgs)
                    if np.isfinite(gl) and np.isfinite(gr) and gl * gr <= 0.0:
                        bracket_ok = True
                        break
                    step *= expand_factor

                if not bracket_ok:
                    glb = g_of_y(P_val, y_min, S_cgs)
                    grb = g_of_y(P_val, y_max, S_cgs)
                    if np.isfinite(glb) and np.isfinite(grb) and glb * grb <= 0.0:
                        yl, yr = y_min, y_max
                        bracket_ok = True

                if not bracket_ok:
                    raise RuntimeError(
                        "Failed to bracket logT root. "
                        f"P={P_val:.6g} GPa, S={S_cgs:.6g} erg/g/K"
                    )

                solb = root_scalar(
                    lambda yy: g_of_y(P_val, yy, S_cgs),
                    method="brentq",
                    bracket=(yl, yr),
                    rtol=brent_rtol,
                    xtol=brent_xtol,
                    maxiter=brent_maxiter,
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

            except Exception as exc:
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brenth(logT)"
                    diag["message"][idx] = f"Brent exception: {exc}"

        if return_diagnostics:
            return Tout, diag
        return Tout

    def get_rhot_sp_2d_inv(self, S, P, s_units="kbbar", **kwargs):
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        rho = self.get_rho_pt(P, T, tab=True)
        return rho, T

    def get_rhot_sp_inv(self, S, P, s_units="kbbar", **kwargs):
        return self.get_rhot_sp_2d_inv(S, P, s_units=s_units, **kwargs)

    def get_rho_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if not tab:
            rho, _ = self.get_rhot_sp_2d_inv(S, P, s_units=s_units, **kwargs)
            return rho
        S_tab = self._entropy_to_sp_table(S, s_units=s_units)
        return self.get_rho_sp_tab(S_tab, P)

    def get_t_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if not tab:
            return self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        S_tab = self._entropy_to_sp_table(S, s_units=s_units)
        return self.get_t_sp_tab(S_tab, P)

    def get_u_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if not tab:
            scalar, S_arr, P_arr = self._broadcast(S, P)
            T = self.get_t_sp_inv(S_arr, P_arr, s_units=s_units, **kwargs)
            vals = self.get_u_pt(P_arr, T, tab=True)
            return self._maybe_scalar(scalar, vals)
        S_tab = self._entropy_to_sp_table(S, s_units=s_units)
        return self.get_u_sp_tab(S_tab, P)

    def get_cp_sp_tab(self, S_tab, P, eps_rel=1e-3, method="cv_relation"):
        s_cgs = self._entropy_to_cgs(S_tab, s_units="kbbar")
        return self.get_cp_sp(s_cgs, P, s_units="cgs", eps_rel=eps_rel, method=method)

    def get_cv_sp_tab(self, S_tab, P, eps_rel=1e-3, method="rhou"):
        s_cgs = self._entropy_to_cgs(S_tab, s_units="kbbar")
        return self.get_cv_sp(s_cgs, P, s_units="cgs", eps_rel=eps_rel, method=method)

    def get_alpha_sp_tab(self, S_tab, P, eps_rel=1e-3):
        s_cgs = self._entropy_to_cgs(S_tab, s_units="kbbar")
        return self.get_alpha_sp(s_cgs, P, s_units="cgs", eps_rel=eps_rel)

    def get_cp_sp(self, S, P, tab=True, s_units="kbbar", eps_rel=1e-3, method="cv_relation", **kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        T = self.get_t_sp(S_arr, P_arr, tab=tab, s_units=s_units, **kwargs)
        vals = self.get_cp_pt(P_arr, T, eps_rel=eps_rel, method=method)
        return self._maybe_scalar(scalar, vals)

    def get_cv_sp(self, S, P, tab=True, s_units="kbbar", eps_rel=1e-3, method="rhou", **kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        T = self.get_t_sp(S_arr, P_arr, tab=tab, s_units=s_units, **kwargs)
        vals = self.get_cv_pt(P_arr, T, eps_rel=eps_rel, method=method)
        return self._maybe_scalar(scalar, vals)

    def get_alpha_sp(self, S, P, tab=True, s_units="kbbar", eps_rel=1e-3, **kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        T = self.get_t_sp(S_arr, P_arr, tab=tab, s_units=s_units, **kwargs)
        vals = self.get_alpha_pt(P_arr, T, eps_rel=eps_rel)
        return self._maybe_scalar(scalar, vals)
