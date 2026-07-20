"""
aquarock_core_eos.py

Mantle-style wrapper around a WATER/ROCK mixture EOS, the water/rock analogue
of ``aqua_revised_core_eos.AQUA_REVISED_CORE_EOS`` (which is pure water).

The metal here is a Volume Addition Law (VAL) mixture of revised-AQUA water
(Cano Amoros et al.; entropies from Mazevet et al. 2021, superionic data from
French et al. 2016) and Mg2SiO4 forsterite (Stewart et al. 2020 ANEOS), at a
fixed rock mass fraction ``f_rock`` within the metal budget:

    1/rho = (1 - f_rock)/rho_water + f_rock/rho_rock          (VAL)
    S     = (1 - f_rock)*S_water + f_rock*S_rock + S_mix_id   (ideal mixing)
    U     = (1 - f_rock)*U_water + f_rock*U_rock

The PT forward model is built once at construction by evaluating the VAL
mixture (via ``eos.eos_class.z_eos_val_mixtures``) on the AQUA P-T grid, so any
``f_rock`` in [0, 1] works without a precomputed mixture table.  At f_rock=0 it
reduces to pure water (matching ``AQUA_REVISED_CORE_EOS``); at f_rock=1 it is
pure mg2sio4.

Melt curve: the Mg2SiO4 (forsterite) melt curve of Presnall & Walter (1993),
``Tm(P) = 2171 K * (1 + P/2.44 GPa)^(1/11.4)`` — i.e. the rock melt curve, not
water's (the mixture's mantle melt physics is rock-dominated).

The same broad API as the other mantle/core EOS wrappers:
    - P in GPa, T in K, rho in g/cm^3
    - S outputs from PT getters in erg/g/K
    - S inputs to SP getters in k_B / baryon by default
    - U in erg/g, Cp/Cv in erg/g/K, alpha in 1/K

Because the VAL mixture ships no tabulated thermodynamic derivatives, Cp, Cv,
alpha and the adiabatic gradient are obtained by finite-differencing the P-T
surfaces (the ``finite_diff`` path of the AQUA wrapper, made the default here).

Inversion tables (the slow part) are NOT built here.  Compute the S-P and
rho-T basis variants separately and drop them next to this file; the loaders
(``_load_sp_table`` / ``_load_rhot_table``, auto-invoked from ``__init__`` when
the .npz files exist) then make the ``*_sp`` / ``*_rhot`` getters read the
table instead of inverting on the fly.  Until then those getters fall back to
Newton/brentq inversion of the PT mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root_scalar

try:
    from eos.eos_class import z_eos_val_mixtures
except ImportError:  # pragma: no cover - allow direct-run imports
    from eos_class import z_eos_val_mixtures


ArrayLike = Union[float, int, np.ndarray]

CURR_DIR = Path(__file__).resolve().parent
# Default on-disk locations for the (user-computed) inversion tables.  The
# rock fraction is encoded in the filename so several mixtures can coexist,
# e.g. aquarock_core_eos_sp_frock0.50.npz.
_SP_TABLE_DIR = CURR_DIR / "aqua" / "aquarock"
_RHOT_TABLE_DIR = CURR_DIR / "aqua" / "aquarock"


@dataclass
class Domain:
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float
    P_min: float
    P_max: float


class AQUAROCK_CORE_EOS:
    """Water/rock VAL-mixture mantle EOS with a linear-unit interface matching
    the other mantle EOSes (``AQUA_REVISED_CORE_EOS``, ``MG2SIO4_ANEOS_EOS``).

    Parameters
    ----------
    f_rock : float
        Rock (mg2sio4) mass fraction within the metal, in [0, 1].  0 = pure
        water, 1 = pure rock, 0.5 = 50/50.  The VAL mixture is built on the
        AQUA P-T grid at construction.
    smooth_low_pt : bool
        Forwarded to ``z_eos_val_mixtures`` (per-component smoothing of the
        low-P/low-T tables before mixing).
    sp_table_path, rhot_table_path : str or Path or None
        Optional explicit paths to pre-computed inversion tables.  If None,
        the default ``f_rock``-tagged paths are tried; if those do not exist,
        the class inverts the PT mixture on demand.
    """

    # SI -> CGS conversion factors (shared with z_eos / aqua_revised_core_eos)
    _Pa_to_dyn = 10.0          # 1 Pa = 10 dyn/cm^2
    _kgm3_to_gcm3 = 1e-3       # 1 kg/m^3 = 1e-3 g/cm^3
    _J_kgK_to_erg_gK = 1e4     # 1 J/(kg*K) = 1e4 erg/(g*K)
    _J_kg_to_erg_g = 1e4       # 1 J/kg = 1e4 erg/g

    def __init__(
        self,
        f_rock: float = 0.5,
        smooth_low_pt: bool = False,
        sp_table_path: Optional[Union[str, Path]] = None,
        rhot_table_path: Optional[Union[str, Path]] = None,
    ):
        self.f_rock = float(f_rock)
        if not (0.0 <= self.f_rock <= 1.0):
            raise ValueError(f"f_rock must be in [0, 1] (got {self.f_rock}).")

        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.kbbar_to_erg = 1.0 / self.erg_to_kbbar
        self.GPa_to_dyn = float(u.GPa.to("dyn/cm^2"))
        self.dyn_to_GPa = float((u.dyn / u.cm**2).to("GPa"))
        self.dyn_to_Pa = float((u.dyn / u.cm**2).to("Pa"))
        # The VAL mixture U is already in erg/g, so no extra conversion.
        self.U_conv_cgs = 1.0
        # Latent heat of fusion [erg/g] used by the mantle latent-heat term.
        # Adopt the Mg2SiO4 (forsterite) value, consistent with the rock melt
        # curve used below (Stewart et al. 2020 mantle).
        self.L = float(7.322e5 * (u.J / u.kg).to("erg/g"))

        self._has_pt_table = True
        self._has_sp_table = False
        self._has_rhot_table = False

        # Build the VAL mixture P-T forward model on the AQUA grid.
        self._build_mixture_pt(smooth_low_pt)
        self._build_rgi()

        self.domain = Domain(
            rho_min=float(10.0 ** np.nanmin(self.logrho_pt)),
            rho_max=float(10.0 ** np.nanmax(self.logrho_pt)),
            T_min=float(10.0 ** np.min(self.logtvals_pt)),
            T_max=float(10.0 ** np.max(self.logtvals_pt)),
            P_min=float((10.0 ** np.min(self.logpvals_pt)) * self.dyn_to_GPa),
            P_max=float((10.0 ** np.max(self.logpvals_pt)) * self.dyn_to_GPa),
        )

        # Mg2SiO4 forsterite melt curve (Presnall & Walter 1993) parameters.
        self.melt_transition_width_K = 50.0
        self._melt_T0 = 2171.0   # K
        self._melt_P0 = 2.44     # GPa
        self._melt_exp = 11.4

        # --- Inversion-table hooks (user-computed; auto-loaded if present) ---
        self._t_rgi_sp = None
        self._rho_rgi_sp = None
        self._u_rgi_sp = None
        self._cp_rgi_sp = None
        self._cv_rgi_sp = None
        self._alpha_rgi_sp = None

        self._p_rgi_rhot = None
        self._s_rgi_rhot = None
        self._u_rgi_rhot = None

        tag = f"frock{self.f_rock:.2f}"
        self.sp_table_path = (Path(sp_table_path) if sp_table_path is not None
                              else _SP_TABLE_DIR / f"aquarock_core_eos_sp_{tag}.npz")
        self.rhot_table_path = (Path(rhot_table_path) if rhot_table_path is not None
                                else _RHOT_TABLE_DIR / f"aquarock_core_eos_rhot_{tag}.npz")
        if self.sp_table_path.exists():
            self._load_sp_table(self.sp_table_path)
        if self.rhot_table_path.exists():
            self._load_rhot_table(self.rhot_table_path)

    # =================================================================
    # PT mixture construction
    # =================================================================
    def _build_mixture_pt(self, smooth_low_pt: bool):
        """Evaluate the water/rock VAL mixture on the AQUA P-T grid.

        Uses ``z_eos_val_mixtures`` (water_revised + mg2sio4); its NaN-corner
        interpolation/extrapolation keeps the entropy finite in the AQUA
        superionic corner.  Density and U are mass/volume-weighted; entropy
        includes the ideal entropy of mixing.
        """
        zmix = z_eos_val_mixtures(species_list=["water_revised", "mg2sio4"],
                                  smooth_z=smooth_low_pt, fill_z_nans=True)
        self._zmix = zmix

        water = zmix.z["water"]
        # AQUA P-T grid (logp in dyn/cm^2, logt in K) — the native mantle grid.
        self.logpvals_pt = np.asarray(water.logpvals_pt, dtype=float)
        self.logtvals_pt = np.asarray(water.logtvals_pt, dtype=float)

        LP, LT = np.meshgrid(self.logpvals_pt, self.logtvals_pt, indexing="ij")
        shape = LP.shape
        lp_flat, lt_flat = LP.ravel(), LT.ravel()
        fr = self.f_rock

        logrho = np.asarray(zmix.get_logrho_z(lp_flat, lt_flat, _zr=fr), dtype=float)
        s_cgs = np.asarray(zmix.get_s_pt(lp_flat, lt_flat, _zr=fr), dtype=float)
        u_cgs = np.asarray(zmix.get_u_z(lp_flat, lt_flat, _zr=fr), dtype=float)

        self.logrho_pt = logrho.reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.logs_pt = np.where(s_cgs > 0, np.log10(s_cgs), np.nan).reshape(shape)
            self.logu_pt = np.where(u_cgs > 0, np.log10(u_cgs), np.nan).reshape(shape)

    @staticmethod
    def _fill_nans_rows(grid: np.ndarray) -> np.ndarray:
        """Linear fill of NaN cells along the second axis of a 2-D grid."""
        filled = np.asarray(grid, dtype=float).copy()
        for i in range(filled.shape[0]):
            row = filled[i]
            nans = np.isnan(row)
            if not nans.any() or nans.all():
                continue
            valid = ~nans
            filled[i] = np.interp(np.arange(len(row)), np.where(valid)[0], row[valid])
        return filled

    def _build_rgi(self):
        axes = (self.logpvals_pt, self.logtvals_pt)
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        self.logrho_pt_rgi = RGI(axes, self._fill_nans_rows(self.logrho_pt), **kw)
        self.logs_pt_rgi = RGI(axes, self._fill_nans_rows(self.logs_pt), **kw)
        self.logu_pt_rgi = RGI(axes, self._fill_nans_rows(self.logu_pt), **kw)

    # =================================================================
    # Shape / unit helpers (mirror aqua_revised_core_eos.py)
    # =================================================================
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
        return np.broadcast_arrays(a_arr, b_arr)

    @staticmethod
    def _interp(rgi, x_arr, y_arr):
        pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
        return rgi(pts).reshape(x_arr.shape)

    @staticmethod
    def _clip_positive(vals, floor=1e-300):
        return np.clip(np.asarray(vals, dtype=float), floor, None)

    def _to_logp(self, P):
        return np.log10(self._clip_positive(P) * self.GPa_to_dyn)

    @staticmethod
    def _to_logt(T):
        return np.log10(np.clip(np.asarray(T, dtype=float), 1e-300, None))

    def _from_logp(self, logp):
        return (10.0 ** np.asarray(logp, dtype=float)) * self.dyn_to_GPa

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

    def _entropy_to_kbbar(self, s_cgs):
        return np.asarray(s_cgs, dtype=float) * self.erg_to_kbbar

    def _entropy_to_sp_table(self, s_in, *, s_units: str = "kbbar"):
        su = str(s_units).lower()
        s_arr = np.asarray(s_in, dtype=float)
        if su in ("kbbar", "kb/baryon", "kbperbaryon", "native"):
            return s_arr
        if su in ("cgs", "erg/g/k", "erg/g/kelvin"):
            return s_arr * self.erg_to_kbbar
        raise ValueError("s_units must be one of {'kbbar', 'cgs', 'native'}")

    # =================================================================
    # Native PT getters (table-backed)
    # =================================================================
    def get_rho_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logrho = self._interp(self.logrho_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logrho(logrho))

    def get_s_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logs = self._interp(self.logs_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, 10.0 ** logs)

    def get_u_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logu = self._interp(self.logu_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr))
        return self._maybe_scalar(scalar, self._from_logu(logu))

    # Public forward getters ------------------------------------------------
    def get_rho_pt(self, P, T, tab=True, **kwargs):
        del tab, kwargs
        return self.get_rho_pt_tab(P, T)

    def get_s_pt(self, P, T, tab=True, **kwargs):
        del tab, kwargs
        return self.get_s_pt_tab(P, T)

    def get_u_pt(self, P, T, tab=True, **kwargs):
        del tab, kwargs
        return self.get_u_pt_tab(P, T)

    # =================================================================
    # Derived PT quantities (finite-difference; the VAL mixture ships no
    # tabulated thermodynamic derivatives)
    # =================================================================
    def get_cp_pt(self, P, T, tab=True, eps_rel=1e-3, **kwargs):
        """Cp = T (dS/dT)|_P  [erg/(g*K)] via central difference in T."""
        del tab, kwargs
        P_arr, T_arr = self._as_arrays(P, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        s_hi = np.asarray(self.get_s_pt(P_arr, T_hi), dtype=float)
        s_lo = np.asarray(self.get_s_pt(P_arr, T_lo), dtype=float)
        cp = T_arr * (s_hi - s_lo) / (T_hi - T_lo)
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), cp)

    def get_alpha_pt(self, P, T, tab=True, eps_rel=1e-3, **kwargs):
        """alpha = -(1/rho)(drho/dT)|_P  [1/K] via central difference in T."""
        del tab, kwargs
        P_arr, T_arr = self._as_arrays(P, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        rho_hi = np.asarray(self.get_rho_pt(P_arr, T_hi), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_arr, T_lo), dtype=float)
        drho_dT = (rho_hi - rho_lo) / (T_hi - T_lo)
        alpha = -(drho_dT / np.maximum(rho, 1e-99))
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), alpha)

    def get_cv_pt(self, P, T, tab=True, eps_rel=1e-3, **kwargs):
        """Cv = Cp - T V alpha^2 / kappa_T, with kappa_T = (1/rho)(drho/dP)|_T."""
        del tab, kwargs
        P_arr, T_arr = self._as_arrays(P, T)
        cp = np.asarray(self.get_cp_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        alpha = np.asarray(self.get_alpha_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)

        logp = self._to_logp(P_arr)
        P_hi = (10.0 ** (logp + eps_rel)) * self.dyn_to_GPa
        P_lo = (10.0 ** (logp - eps_rel)) * self.dyn_to_GPa
        rho_hi = np.asarray(self.get_rho_pt(P_hi, T_arr), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_lo, T_arr), dtype=float)
        dlnrho_dlnP = (np.log(rho_hi) - np.log(rho_lo)) / (2.0 * np.log(10.0) * eps_rel)
        P_cgs = P_arr * self.GPa_to_dyn
        kappa_T = dlnrho_dlnP / np.maximum(P_cgs, 1e-99)
        V = 1.0 / np.maximum(rho, 1e-99)
        cv = cp - T_arr * V * alpha * alpha / np.maximum(kappa_T, 1e-99)
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), cv)

    def get_dlnS_dlnT_P_pt(self, P, T, eps_rel=1e-3):
        """dlnS/dlnT|_P = Cp / S."""
        P_arr, T_arr = self._as_arrays(P, T)
        S = np.asarray(self.get_s_pt(P_arr, T_arr), dtype=float)
        cp = np.asarray(self.get_cp_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        vals = cp / np.maximum(S, 1e-99)
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), vals)

    def get_dlnS_dlnP_T_pt(self, P, T):
        """dlnS/dlnP|_T = -(P alpha)/(rho S)  (Maxwell: (dS/dP)_T = -alpha/rho)."""
        P_arr, T_arr = self._as_arrays(P, T)
        S = np.asarray(self.get_s_pt(P_arr, T_arr), dtype=float)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        alpha = np.asarray(self.get_alpha_pt(P_arr, T_arr), dtype=float)
        P_cgs = P_arr * self.GPa_to_dyn
        vals = -(P_cgs * alpha) / np.maximum(rho * S, 1e-99)
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), vals)

    def get_ad_grad_pt(self, P, T, eps_rel=1e-3):
        """nabla_ad = (dlnT/dlnP)_S = P alpha / (rho Cp T)."""
        P_arr, T_arr = self._as_arrays(P, T)
        cp = np.asarray(self.get_cp_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        alpha = np.asarray(self.get_alpha_pt(P_arr, T_arr, eps_rel=eps_rel), dtype=float)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        P_cgs = P_arr * self.GPa_to_dyn
        vals = P_cgs * alpha / np.maximum(rho * cp * T_arr, 1e-99)
        return self._maybe_scalar(np.isscalar(P) and np.isscalar(T), vals)

    # =================================================================
    # Melt curve (Mg2SiO4 forsterite; Presnall & Walter 1993)
    # =================================================================
    def get_T_melt(self, P, phase: str = "forsterite"):
        """Forsterite melt curve: Tm(P) = T0 * (1 + P/P0)^(1/n)."""
        del phase
        P_arr = np.asarray(P, dtype=float)
        base = np.maximum(1.0 + P_arr / self._melt_P0, 1e-12)
        Tm = self._melt_T0 * np.power(base, 1.0 / self._melt_exp)
        return float(Tm) if np.isscalar(P) else Tm

    # Aliases so callers expecting the AQUA phase names still resolve to the
    # (rock-dominated) mixture melt curve.
    def get_T_solidus(self, P):
        return self.get_T_melt(P)

    def get_T_melt_superionic(self, P):
        return self.get_T_melt(P)

    def get_T_melt_ice_x(self, P):
        return self.get_T_melt(P)

    def _melt_fraction_PT(self, P, T):
        """Smooth melt-fraction proxy x = 0.5[1 + tanh((T - Tm)/dT)] in [0, 1]."""
        scalar, P_arr, T_arr = self._broadcast(P, T)
        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)
        x = 0.5 * (1.0 + np.tanh((T_arr - Tm) / self.melt_transition_width_K))
        return self._maybe_scalar(scalar, np.clip(x, 0.0, 1.0))

    def _S_at_melt_offset(self, P, sign, dT_melt=None):
        """Mixture entropy a half-window below (sign<0) or above (sign>0) Tm."""
        P_arr = np.array(P, ndmin=1, dtype=float)
        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)
        if dT_melt is None:
            dT = np.maximum(2.0 * self.melt_transition_width_K, 0.02 * Tm)
        else:
            dT = np.broadcast_to(np.asarray(dT_melt, dtype=float), P_arr.shape)
        T = np.clip(Tm + np.sign(sign) * dT, self.domain.T_min, self.domain.T_max)
        return P_arr, np.asarray(self.get_s_pt(P_arr, T), dtype=float)

    def get_S_liq_at_melt(self, P, dT_melt=None):
        """Mixture entropy on the liquid side of the melt curve [erg/g/K]."""
        P_arr, s = self._S_at_melt_offset(P, +1, dT_melt=dT_melt)
        return float(s[0]) if np.isscalar(P) else s

    def get_S_sol_at_melt(self, P, dT_melt=None):
        """Mixture entropy on the solid side of the melt curve [erg/g/K]."""
        P_arr, s = self._S_at_melt_offset(P, -1, dT_melt=dT_melt)
        return float(s[0]) if np.isscalar(P) else s

    def get_latent_heat(self, P, dT_melt=None):
        """Latent heat L = Tm * (S_liq - S_sol) across the melt curve [erg/g]."""
        scalar = np.isscalar(P)
        P_arr = np.array(P, ndmin=1, dtype=float)
        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)
        s_liq = np.asarray(self.get_S_liq_at_melt(P_arr, dT_melt=dT_melt), dtype=float)
        s_sol = np.asarray(self.get_S_sol_at_melt(P_arr, dT_melt=dT_melt), dtype=float)
        L = Tm * (s_liq - s_sol)
        return float(L[0]) if scalar else L

    def get_alpha_x(self, P, T, rho, dT_melt=None, **kwargs):
        """Effective melt expansivity alpha_x = rho (1/rho_liq - 1/rho_sol)."""
        kwargs.pop("phase", None)
        scalar, P_arr, rho_arr = self._broadcast(P, rho)
        Tm = np.asarray(self.get_T_melt(P_arr), dtype=float)
        if dT_melt is None:
            dT = np.maximum(2.0 * self.melt_transition_width_K, 0.02 * Tm)
        else:
            dT = np.broadcast_to(np.asarray(dT_melt, dtype=float), P_arr.shape)
        T_sol = np.clip(Tm - dT, self.domain.T_min, self.domain.T_max)
        T_liq = np.clip(Tm + dT, self.domain.T_min, self.domain.T_max)
        rho_sol = np.asarray(self.get_rho_pt(P_arr, T_sol), dtype=float)
        rho_liq = np.asarray(self.get_rho_pt(P_arr, T_liq), dtype=float)
        alpha_x = rho_arr * (1.0 / np.maximum(rho_liq, 1e-99)
                             - 1.0 / np.maximum(rho_sol, 1e-99))
        return self._maybe_scalar(scalar, alpha_x)

    # =================================================================
    # Inversions (PT mixture root-finding; used when no SP table is loaded)
    # =================================================================
    def get_rho_pt_inv(self, P, T, **kwargs):
        """PT is the native basis, so this aliases the table lookup."""
        del kwargs
        return self.get_rho_pt_tab(P, T)

    def get_t_sp_inv(self, S, P, s_units="kbbar", T_guess=None, **kwargs):
        """Solve S(P, T) = S_target for T by per-point root-finding.

        Newton in logT with an analytic-ish derivative from dlnS/dlnT|_P,
        falling back to a brentq bracket over the table's T range.  ``S`` is
        in k_B/baryon by default (``s_units``).
        """
        newton_maxiter = int(kwargs.pop("newton_maxiter", 40))
        tol = float(kwargs.pop("tol", 1e-8))
        # Upper logT bound for the inversion search.  Default 6.0 (1e6 K) lets T
        # extrapolate ABOVE the underlying AQUA water table ceiling
        # (logT = 4.77, ~58,884 K): a hot-Jupiter compact core thermally tied to
        # the envelope base runs hotter than that, and the PT-grid RGI
        # (fill_value=None) extrapolates entropy linearly above the ceiling.
        # This matches AQUA_REVISED_CORE_EOS.get_t_sp_inv (T_hi = 1e6 K).
        # Without it the SP table saturates at 58,884 K and extrapolating off
        # that flat edge produces a spurious core temperature inversion.
        hi_logt_cap = float(kwargs.pop("hi_logt", 6.0))
        if kwargs:
            raise TypeError("Unexpected kwargs: " + ", ".join(sorted(kwargs)))

        scalar = np.isscalar(S) and np.isscalar(P)
        S_arr = np.array(S, ndmin=1, dtype=float)
        P_arr = np.array(P, ndmin=1, dtype=float)
        S_arr, P_arr = np.broadcast_arrays(S_arr, P_arr)
        s_target_cgs = self._entropy_to_cgs(S_arr, s_units=s_units)

        lo_logt = float(np.min(self.logtvals_pt))
        hi_logt = hi_logt_cap
        out = np.full(S_arr.shape, np.nan, dtype=float)
        prev = None
        for idx in np.ndindex(S_arr.shape):
            s_i = float(s_target_cgs[idx])
            p_i = float(P_arr[idx])

            def f(logt, _p=p_i, _s=s_i):
                return float(self.get_s_pt_tab(_p, 10.0 ** logt)) - _s

            guess = (np.log10(T_guess) if (T_guess is not None and prev is None)
                     else (prev if prev is not None else 0.5 * (lo_logt + hi_logt)))
            sol = np.nan
            x = float(np.clip(guess, lo_logt, hi_logt))
            for _ in range(newton_maxiter):
                fx = f(x)
                if not np.isfinite(fx):
                    break
                if abs(fx) < tol * max(abs(s_i), 1.0):
                    sol = x
                    break
                h = 1e-4
                dfx = (f(x + h) - f(x - h)) / (2.0 * h)
                if abs(dfx) < 1e-30:
                    break
                x_new = float(np.clip(x - fx / dfx, lo_logt, hi_logt))
                if abs(x_new - x) < 1e-10:
                    sol = x_new
                    break
                x = x_new
            if not np.isfinite(sol):
                fa, fb = f(lo_logt), f(hi_logt)
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    try:
                        sol = root_scalar(f, bracket=(lo_logt, hi_logt),
                                          method="brentq", xtol=1e-8).root
                    except (ValueError, RuntimeError):
                        sol = np.nan
            if np.isfinite(sol):
                out[idx] = 10.0 ** sol
                prev = sol
        return float(out.reshape(-1)[0]) if scalar else out

    # =================================================================
    # Pre-computed S-P basis table (optional; user-computed)
    # =================================================================
    def _load_sp_table(self, path: Union[str, Path]):
        """Load a pre-computed S-P basis table built from this PT mixture.

        Expected keys (grids shaped ``(n_S, n_P)``):
            svals_sp      : S axis, k_B/baryon
            pvals_sp      : P axis, GPa
            t_grid_sp     : T, K
            rho_grid_sp   : rho, g/cm^3
            u_grid_sp     : U, erg/g
            cp_grid_sp    : Cp, erg/g/K     (optional)
            cv_grid_sp    : Cv, erg/g/K     (optional)
            alpha_grid_sp : alpha, 1/K      (optional)
        """
        data = np.load(path)
        self.svals_sp = np.asarray(data["svals_sp"], dtype=float)
        self.pvals_sp = np.asarray(data["pvals_sp"], dtype=float)
        expected = (self.svals_sp.size, self.pvals_sp.size)

        def _grid(name, optional=False):
            if name not in data:
                if optional:
                    return None
                raise KeyError(f"SP table {path} missing required key '{name}'.")
            arr = np.asarray(data[name], dtype=float)
            if arr.shape != expected:
                if arr.T.shape == expected:
                    arr = arr.T
                else:
                    raise ValueError(
                        f"{name} has shape {arr.shape}, expected {expected} "
                        f"(svals_sp x pvals_sp).")
            return arr

        rgi_kw = dict(method="linear", bounds_error=False, fill_value=None)
        axes = (self.svals_sp, self.pvals_sp)
        self.t_grid_sp = _grid("t_grid_sp")
        self.rho_grid_sp = _grid("rho_grid_sp")
        self.u_grid_sp = _grid("u_grid_sp")
        self._t_rgi_sp = RGI(axes, self.t_grid_sp, **rgi_kw)
        self._rho_rgi_sp = RGI(axes, self.rho_grid_sp, **rgi_kw)
        self._u_rgi_sp = RGI(axes, self.u_grid_sp, **rgi_kw)
        for name, attr in (("cp_grid_sp", "_cp_rgi_sp"),
                           ("cv_grid_sp", "_cv_rgi_sp"),
                           ("alpha_grid_sp", "_alpha_rgi_sp")):
            g = _grid(name, optional=True)
            if g is not None:
                setattr(self, name, g)
                setattr(self, attr, RGI(axes, g, **rgi_kw))
        self._has_sp_table = True

    def _sp_tab_lookup(self, rgi, S_tab, P):
        if rgi is None:
            raise RuntimeError("Requested SP-table quantity is not loaded.")
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        pts = np.column_stack((S_arr.ravel(), P_arr.ravel()))
        return self._maybe_scalar(scalar, rgi(pts).reshape(S_arr.shape))

    def get_t_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._t_rgi_sp, S_tab, P)

    def get_rho_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._rho_rgi_sp, S_tab, P)

    def get_u_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._u_rgi_sp, S_tab, P)

    def get_cp_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._cp_rgi_sp, S_tab, P)

    def get_cv_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._cv_rgi_sp, S_tab, P)

    def get_alpha_sp_tab(self, S_tab, P):
        return self._sp_tab_lookup(self._alpha_rgi_sp, S_tab, P)

    # SP convenience getters: read the table when loaded, else invert the PT
    # mixture on the fly.
    def get_t_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            return self.get_t_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)

    def get_rho_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            return self.get_rho_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_rho_pt(P, self.get_t_sp_inv(S, P, s_units=s_units, **kwargs))

    def get_u_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            return self.get_u_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_u_pt(P, self.get_t_sp_inv(S, P, s_units=s_units, **kwargs))

    def get_cp_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table and self._cp_rgi_sp is not None:
            return self.get_cp_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_cp_pt(P, self.get_t_sp_inv(S, P, s_units=s_units, **kwargs))

    def get_cv_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table and self._cv_rgi_sp is not None:
            return self.get_cv_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_cv_pt(P, self.get_t_sp_inv(S, P, s_units=s_units, **kwargs))

    def get_alpha_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table and self._alpha_rgi_sp is not None:
            return self.get_alpha_sp_tab(self._entropy_to_sp_table(S, s_units=s_units), P)
        return self.get_alpha_pt(P, self.get_t_sp_inv(S, P, s_units=s_units, **kwargs))

    # =================================================================
    # Pre-computed rho-T basis table (optional; user-computed)
    # =================================================================
    def _load_rhot_table(self, path: Union[str, Path]):
        """Load a pre-computed rho-T basis table built from this PT mixture.

        Expected keys (grids shaped ``(n_rho, n_T)``):
            rhovals_rhot : rho axis, g/cm^3
            tvals_rhot   : T axis, K
            p_grid_rhot  : P, GPa
            s_grid_rhot  : S, erg/g/K       (optional)
            u_grid_rhot  : U, erg/g         (optional)
        """
        data = np.load(path)
        self.rhovals_rhot = np.asarray(data["rhovals_rhot"], dtype=float)
        self.tvals_rhot = np.asarray(data["tvals_rhot"], dtype=float)
        expected = (self.rhovals_rhot.size, self.tvals_rhot.size)

        def _grid(name, optional=False):
            if name not in data:
                if optional:
                    return None
                raise KeyError(f"rho-T table {path} missing required key '{name}'.")
            arr = np.asarray(data[name], dtype=float)
            if arr.shape != expected and arr.T.shape == expected:
                arr = arr.T
            return arr

        rgi_kw = dict(method="linear", bounds_error=False, fill_value=None)
        axes = (self.rhovals_rhot, self.tvals_rhot)
        self.p_grid_rhot = _grid("p_grid_rhot")
        self._p_rgi_rhot = RGI(axes, self.p_grid_rhot, **rgi_kw)
        for name, attr in (("s_grid_rhot", "_s_rgi_rhot"),
                           ("u_grid_rhot", "_u_rgi_rhot")):
            g = _grid(name, optional=True)
            if g is not None:
                setattr(self, name, g)
                setattr(self, attr, RGI(axes, g, **rgi_kw))
        self._has_rhot_table = True

    def _rhot_tab_lookup(self, rgi, rho, T):
        if rgi is None:
            raise RuntimeError("Requested rho-T-table quantity is not loaded.")
        scalar, rho_arr, T_arr = self._broadcast(rho, T)
        pts = np.column_stack((rho_arr.ravel(), T_arr.ravel()))
        return self._maybe_scalar(scalar, rgi(pts).reshape(rho_arr.shape))

    def get_p_rhot_tab(self, rho, T):
        return self._rhot_tab_lookup(self._p_rgi_rhot, rho, T)

    def get_s_rhot_tab(self, rho, T):
        return self._rhot_tab_lookup(self._s_rgi_rhot, rho, T)

    def get_u_rhot_tab(self, rho, T):
        return self._rhot_tab_lookup(self._u_rgi_rhot, rho, T)

    def get_p_rhot(self, rho, T, tab=True, **kwargs):
        """P(rho, T): from the rho-T table if loaded, else invert rho(P, T)."""
        if tab and self._has_rhot_table:
            return self.get_p_rhot_tab(rho, T)
        return self.get_p_rhot_inv(rho, T, **kwargs)

    def get_p_rhot_inv(self, rho, T, P_guess=None, **kwargs):
        """Solve rho(P, T) = rho_target for P by per-point root-finding."""
        newton_maxiter = int(kwargs.pop("newton_maxiter", 40))
        tol = float(kwargs.pop("tol", 1e-8))
        if kwargs:
            raise TypeError("Unexpected kwargs: " + ", ".join(sorted(kwargs)))
        scalar = np.isscalar(rho) and np.isscalar(T)
        rho_arr = np.array(rho, ndmin=1, dtype=float)
        T_arr = np.array(T, ndmin=1, dtype=float)
        rho_arr, T_arr = np.broadcast_arrays(rho_arr, T_arr)
        lo_lp = float(np.min(self.logpvals_pt))
        hi_lp = float(np.max(self.logpvals_pt))
        out = np.full(rho_arr.shape, np.nan, dtype=float)
        prev = None
        for idx in np.ndindex(rho_arr.shape):
            lr_t = float(np.log10(max(rho_arr[idx], 1e-300)))
            t_i = float(T_arr[idx])

            def f(logp_dyn, _t=t_i, _lr=lr_t):
                P_gpa = (10.0 ** logp_dyn) * self.dyn_to_GPa
                return float(np.log10(max(self.get_rho_pt_tab(P_gpa, _t), 1e-300))) - _lr

            guess = (np.log10(P_guess * self.GPa_to_dyn)
                     if (P_guess is not None and prev is None)
                     else (prev if prev is not None else 0.5 * (lo_lp + hi_lp)))
            sol = np.nan
            fa, fb = f(lo_lp), f(hi_lp)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                try:
                    sol = root_scalar(f, bracket=(lo_lp, hi_lp), x0=float(np.clip(guess, lo_lp, hi_lp)),
                                      method="brentq", xtol=1e-8).root
                except (ValueError, RuntimeError):
                    sol = np.nan
            if np.isfinite(sol):
                out[idx] = (10.0 ** sol) * self.dyn_to_GPa
                prev = sol
        return float(out.reshape(-1)[0]) if scalar else out


class ROCKWATER_INTERP_CORE_EOS:
    """Rock+water core EOS at ARBITRARY rock fraction, interpolating between
    the two validated endpoint core EOSes:

        f_rock = 0  ->  AQUA_REVISED_CORE_EOS  (mantle_comp='h2o', 'revised')
        f_rock = 1  ->  MG2SIO4_ANEOS_EOS      (mantle_comp='mg2sio4')

    Mixing laws (ideal, matching the VAL construction of AQUAROCK_CORE_EOS):
      - density:  additive volume, 1/rho = (1-f)/rho_w + f/rho_r
      - S, U, Cp, Cv: mass-weighted specific quantities
      - alpha:    volume-weighted, alpha = rho_mix * [(1-f) a_w/rho_w + f a_r/rho_r]
      - T(S, P):  vectorized bisection on the mass-weighted S(P, T)
      - melt curve / latent heat / alpha_x: delegated to the ROCK endpoint
        (forsterite curve), mirroring AQUAROCK_CORE_EOS's convention.

    Exists because the cached AQUAROCK S-P inversion table only ships for
    f_rock = 0.50; this class needs no per-f_rock tables. Same call surface
    and unit conventions as the other mantle EOSes (P in GPa, S in kb/baryon).
    """

    def __init__(self, f_rock=0.5):
        from eos.aqua_revised_core_eos import AQUA_REVISED_CORE_EOS
        from eos.mg2sio4_aneos_eos import MG2SIO4_ANEOS_EOS
        f = float(f_rock)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"f_rock must be in [0, 1] (got {f}).")
        self.f_rock = f
        self.water = AQUA_REVISED_CORE_EOS()
        self.rock = MG2SIO4_ANEOS_EOS()

        # Shared unit conventions (identical constants on both endpoints).
        self.dyn_to_GPa = self.water.dyn_to_GPa
        self.dyn_to_Pa = getattr(self.water, 'dyn_to_Pa',
                                 self.dyn_to_GPa * 1e9)
        self.erg_to_kbbar = self.water.erg_to_kbbar
        self.kbbar_to_erg = getattr(self.water, 'kbbar_to_erg',
                                    1.0 / self.erg_to_kbbar)
        self._Uw = getattr(self.water, 'U_conv_cgs', 1.0)
        self._Ur = getattr(self.rock, 'U_conv_cgs', 1.0)
        self.U_conv_cgs = 1.0
        # Forsterite melt curve / latent heat (AQUAROCK convention).
        self.L = self.rock.L

        # T bracket for the S(P,T) inversion: intersection of endpoint
        # domains where exposed, else generous fixed bounds.
        tlo, thi = 160.0, 5.0e4
        for obj in (self.water, self.rock):
            dom = getattr(obj, 'domain', None)
            if dom is not None:
                tlo = max(tlo, float(getattr(dom, 'T_min', tlo)))
                thi = min(thi, float(getattr(dom, 'T_max', thi)))
        self._logT_lo = np.log10(tlo * 1.001)
        self._logT_hi = np.log10(thi * 0.999)

    # ---------------- P-T basis (mixing laws) ----------------
    def get_s_pt(self, P, T, **kwargs):
        f = self.f_rock
        return ((1.0 - f) * np.asarray(self.water.get_s_pt(P, T))
                + f * np.asarray(self.rock.get_s_pt(P, T)))

    def get_rho_pt(self, P, T, **kwargs):
        f = self.f_rock
        rw = np.asarray(self.water.get_rho_pt(P, T), dtype=float)
        rr = np.asarray(self.rock.get_rho_pt(P, T), dtype=float)
        return 1.0 / ((1.0 - f) / np.maximum(rw, 1e-99)
                      + f / np.maximum(rr, 1e-99))

    def get_u_pt(self, P, T, **kwargs):
        f = self.f_rock
        return ((1.0 - f) * np.asarray(self.water.get_u_pt(P, T)) * self._Uw
                + f * np.asarray(self.rock.get_u_pt(P, T)) * self._Ur)

    def get_cp_pt(self, P, T, **kwargs):
        f = self.f_rock
        return ((1.0 - f) * np.asarray(self.water.get_cp_pt(P, T))
                + f * np.asarray(self.rock.get_cp_pt(P, T)))

    def get_cv_pt(self, P, T, **kwargs):
        f = self.f_rock
        return ((1.0 - f) * np.asarray(self.water.get_cv_pt(P, T))
                + f * np.asarray(self.rock.get_cv_pt(P, T)))

    def get_alpha_pt(self, P, T, **kwargs):
        f = self.f_rock
        rw = np.asarray(self.water.get_rho_pt(P, T), dtype=float)
        rr = np.asarray(self.rock.get_rho_pt(P, T), dtype=float)
        aw = np.asarray(self.water.get_alpha_pt(P, T), dtype=float)
        ar = np.asarray(self.rock.get_alpha_pt(P, T), dtype=float)
        rho = 1.0 / ((1.0 - f) / np.maximum(rw, 1e-99)
                     + f / np.maximum(rr, 1e-99))
        return rho * ((1.0 - f) * aw / np.maximum(rw, 1e-99)
                      + f * ar / np.maximum(rr, 1e-99))

    # ---------------- S-P basis (bisection inversion) ----------------
    def get_t_sp(self, S, P, **kwargs):
        """T such that the mass-weighted S(P, T) equals the target.
        Vectorized bisection in log10 T (monotone S(T) at fixed P).
        S is in k_B/baryon (core-EOS convention: get_s_pt returns cgs,
        get_t_sp takes k_B/baryon; callers bridge with erg_to_kbbar)."""
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        S_b, P_b = np.broadcast_arrays(S_arr, P_arr)
        lo = np.full(S_b.shape, self._logT_lo)
        hi = np.full(S_b.shape, self._logT_hi)
        # Clamp targets into the achievable range at the brackets.
        s_lo = self.get_s_pt(P_b, 10.0 ** lo)
        s_hi = self.get_s_pt(P_b, 10.0 ** hi)
        tgt = np.clip(S_b * self.kbbar_to_erg,
                      np.minimum(s_lo, s_hi), np.maximum(s_lo, s_hi))
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            s_mid = self.get_s_pt(P_b, 10.0 ** mid)
            go_up = s_mid < tgt
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        out = 10.0 ** (0.5 * (lo + hi))
        return float(out.reshape(-1)[0]) if (np.isscalar(S) and np.isscalar(P)) else out

    def get_rho_sp(self, S, P, **kwargs):
        return self.get_rho_pt(P, self.get_t_sp(S, P))

    def get_u_sp(self, S, P, **kwargs):
        return self.get_u_pt(P, self.get_t_sp(S, P))

    def get_cp_sp(self, S, P, **kwargs):
        return self.get_cp_pt(P, self.get_t_sp(S, P))

    def get_alpha_sp(self, S, P, **kwargs):
        return self.get_alpha_pt(P, self.get_t_sp(S, P))

    # ---------------- melt curve / latent heat (rock endpoint) --------
    def get_T_melt(self, P, **kwargs):
        return self.rock.get_T_melt(P, **kwargs)

    def get_T_solidus(self, P, **kwargs):
        fn = getattr(self.rock, 'get_T_solidus', None)
        return fn(P, **kwargs) if fn is not None else self.rock.get_T_melt(P)

    def get_S_liq_at_melt(self, P, **kwargs):
        return self.rock.get_S_liq_at_melt(P, **kwargs)

    def get_S_sol_at_melt(self, P, **kwargs):
        return self.rock.get_S_sol_at_melt(P, **kwargs)

    def get_alpha_x(self, P, T, rho, **kwargs):
        return self.rock.get_alpha_x(P, T, rho, **kwargs)
