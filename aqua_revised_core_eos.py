"""
aqua_revised_core_eos.py

Mantle-style wrapper around the revised AQUA water EOS of Cano Amoros et al.
(entropies from Mazevet et al. 2021, superionic data from French et al. 2016).

This exposes the same broad API used by the rock/core EOS wrappers:
    - P in GPa
    - T in K
    - rho in g/cm^3
    - S outputs from PT getters in erg/g/K
    - S inputs to SP getters in k_B / baryon by default
    - U in erg/g
    - Cp, Cv in erg/g/K
    - alpha in 1/K

The revised AQUA release ships only a P-T basis table (plus the derivatives
dlnS/dlnP|_T, dlnS/dlnT|_P, and the adiabatic gradient). Cp and alpha are
therefore obtained analytically from those tabulated derivatives instead of
finite-differencing the underlying surfaces. For entropy inversions the class
seeds Newton's method with an initial guess from the legacy AQUA SP table
(via eos.aqua_core_eos.AQUA_CORE_EOS), falling back to secant + brentq when
Newton fails. Ice-X / superionic / solidus phase profiles are reused verbatim
from the legacy class since they describe the same water substance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar

try:
    from eos import aqua_core_eos as _legacy_aqua_module
except ImportError:  # pragma: no cover - allow direct-run imports
    import aqua_core_eos as _legacy_aqua_module


ArrayLike = Union[float, int, np.ndarray]

CURR_DIR = Path(__file__).resolve().parent
_REVISED_CSV = CURR_DIR / "aqua" / "aqua_revised_amoros" / "AQUA_revised_eos_pt.csv"
_SP_TABLE_NPZ = CURR_DIR / "aqua" / "aqua_revised_amoros" / "AQUA_revised_eos_sp.npz"


@dataclass
class Domain:
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float
    P_min: float
    P_max: float


class AQUA_REVISED_CORE_EOS:
    """
    Revised AQUA wrapper with a linear-unit interface matching the mantle EOSes.

    Parameters
    ----------
    smooth_low_pt : bool
        If True, apply a tanh-masked 2D Gaussian smoother at low P / low T to
        tame the minor phase-boundary kinks carried over from the raw tables.
        Mirrors the behaviour of ``z_eos._smooth_aqua_lowp_lowt``.
    """

    # SI -> CGS conversion factors (shared with z_eos._load_aqua_revised)
    _Pa_to_dyn = 10.0          # 1 Pa = 10 dyn/cm^2
    _kgm3_to_gcm3 = 1e-3       # 1 kg/m^3 = 1e-3 g/cm^3
    _J_kgK_to_erg_gK = 1e4     # 1 J/(kg*K) = 1e4 erg/(g*K)
    _J_kg_to_erg_g = 1e4       # 1 J/kg = 1e4 erg/g

    def __init__(
        self,
        smooth_low_pt: bool = False,
        sp_table_path: Optional[Union[str, Path]] = None,
    ):
        self.erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
        self.kbbar_to_erg = 1.0 / self.erg_to_kbbar
        self.GPa_to_dyn = float(u.GPa.to("dyn/cm^2"))
        self.dyn_to_GPa = float((u.dyn / u.cm**2).to("GPa"))

        self._has_pt_table = True
        self._has_sp_table = False
        self._has_rhou_table = False

        self._load_revised_csv()

        if smooth_low_pt:
            self._smooth_low_pt()

        self._build_rgi()

        self.domain = Domain(
            rho_min=float(10.0 ** np.nanmin(self.logrho_pt)),
            rho_max=float(10.0 ** np.nanmax(self.logrho_pt)),
            T_min=float(10.0 ** np.min(self.logtvals_pt)),
            T_max=float(10.0 ** np.max(self.logtvals_pt)),
            P_min=float((10.0 ** np.min(self.logpvals_pt)) * self.dyn_to_GPa),
            P_max=float((10.0 ** np.max(self.logpvals_pt)) * self.dyn_to_GPa),
        )

        self._legacy_singleton: Optional["_legacy_aqua_module.AQUA_CORE_EOS"] = None
        self._build_phase_transition_profiles()

        # Optional pre-computed S-P basis table. When present, get_*_sp getters
        # with tab=True short-circuit the Newton inversion and read from the
        # table directly (style: mg2sio4_aneos_eos.py).
        self._t_rgi_sp = None
        self._rho_rgi_sp = None
        self._u_rgi_sp = None
        self._cp_rgi_sp = None
        self._cv_rgi_sp = None
        self._alpha_rgi_sp = None

        self.sp_table_path = Path(sp_table_path) if sp_table_path is not None else _SP_TABLE_NPZ
        if self.sp_table_path.exists():
            self._load_sp_table(self.sp_table_path)

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------
    def _load_revised_csv(self):
        pt_df = pd.read_csv(_REVISED_CSV)
        pt_df.rename(columns={
            "pressure_Pa": "press",
            "temperature_K": "temp",
            "density_kg_m3": "rho",
            "entropy_J_kgK": "s",
            "internal_energy_J_kg": "u",
        }, inplace=True)

        # Sentinel entropy = -1 means "undefined"; mask to NaN.
        pt_df.loc[pt_df["s"] == -1, "s"] = np.nan

        pt_df["logp"] = np.log10(pt_df["press"] * self._Pa_to_dyn)
        pt_df["logt"] = np.log10(pt_df["temp"])
        pt_df["logrho"] = np.log10(pt_df["rho"] * self._kgm3_to_gcm3)
        pt_df["logu"] = np.log10(pt_df["u"] * self._J_kg_to_erg_g)

        s_cgs = pt_df["s"].values * self._J_kgK_to_erg_gK
        with np.errstate(invalid="ignore", divide="ignore"):
            pt_df["logs"] = np.where(s_cgs > 0, np.log10(s_cgs), np.nan)

        n_p = pt_df["logp"].nunique()
        shape = (n_p, -1)

        self.logpvals_pt = np.reshape(pt_df["logp"].values, shape)[:, 0]
        self.logtvals_pt = np.reshape(pt_df["logt"].values, shape)[0, :]

        self.logrho_pt = np.reshape(pt_df["logrho"].values, shape)
        self.logs_pt = np.reshape(pt_df["logs"].values, shape)
        self.logu_pt = np.reshape(pt_df["logu"].values, shape)

        self.phase_pt = np.reshape(pt_df["phase"].values, shape)
        self.flag_pt = np.reshape(pt_df["flag"].values, shape)
        self.ad_grad_pt = np.reshape(pt_df["ad_grad"].values, shape)
        self.dlnS_dlnP_T_pt = np.reshape(pt_df["dlnS_dlnP_T"].values, shape)
        self.dlnS_dlnT_P_pt = np.reshape(pt_df["dlnS_dlnT_P"].values, shape)

    def _smooth_low_pt(self):
        logp_2d = self.logpvals_pt[:, np.newaxis]
        logt_2d = self.logtvals_pt[np.newaxis, :]
        mask_p = 0.5 * (1.0 - np.tanh((logp_2d - 8.0) / 1.5))
        mask_t = 0.5 * (1.0 - np.tanh((logt_2d - 3.0) / 0.3))
        mask = mask_p * mask_t

        for attr in ("logrho_pt", "logs_pt", "logu_pt"):
            grid = getattr(self, attr)
            filled = grid.copy()
            nan_mask = np.isnan(filled)
            if nan_mask.any():
                for i in range(filled.shape[0]):
                    row = filled[i]
                    nans = np.isnan(row)
                    if nans.all():
                        continue
                    valid = ~nans
                    filled[i] = np.interp(
                        np.arange(len(row)), np.where(valid)[0], row[valid]
                    )

            smoothed = gaussian_filter(filled, sigma=[3.0, 3.0], mode="nearest")
            blended = (1.0 - mask) * filled + mask * smoothed
            blended[nan_mask] = np.nan
            setattr(self, attr, blended)

    @staticmethod
    def _fill_nans_rows(grid: np.ndarray) -> np.ndarray:
        """Nearest-neighbour fill along the second axis of a 2-D grid."""
        filled = grid.copy()
        for i in range(filled.shape[0]):
            row = filled[i]
            nans = np.isnan(row)
            if not nans.any() or nans.all():
                continue
            valid = ~nans
            filled[i] = np.interp(
                np.arange(len(row)), np.where(valid)[0], row[valid]
            )
        return filled

    def _build_rgi(self):
        axes = (self.logpvals_pt, self.logtvals_pt)
        kw = dict(method="linear", bounds_error=False, fill_value=None)

        self.logrho_pt_rgi = RGI(axes, self._fill_nans_rows(self.logrho_pt), **kw)
        self.logs_pt_rgi = RGI(axes, self._fill_nans_rows(self.logs_pt), **kw)
        self.logu_pt_rgi = RGI(axes, self._fill_nans_rows(self.logu_pt), **kw)

        self.ad_grad_pt_rgi = RGI(
            axes, self._fill_nans_rows(np.asarray(self.ad_grad_pt, dtype=float)), **kw
        )
        self.dlnS_dlnT_P_pt_rgi = RGI(
            axes,
            self._fill_nans_rows(np.asarray(self.dlnS_dlnT_P_pt, dtype=float)),
            **kw,
        )
        self.dlnS_dlnP_T_pt_rgi = RGI(
            axes,
            self._fill_nans_rows(np.asarray(self.dlnS_dlnP_T_pt, dtype=float)),
            **kw,
        )

    # -----------------------------------------------------------------
    # Shape / unit helpers (mirror aqua_core_eos.py)
    # -----------------------------------------------------------------
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

    def _to_logp(self, P):
        P_arr = self._clip_positive(P)
        return np.log10(P_arr * self.GPa_to_dyn)

    @staticmethod
    def _to_logt(T):
        T_arr = np.clip(np.asarray(T, dtype=float), 1e-300, None)
        return np.log10(T_arr)

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

    def _entropy_to_kbbar(self, s_cgs):
        return np.asarray(s_cgs, dtype=float) * self.erg_to_kbbar

    def _entropy_to_sp_table(self, s_in, *, s_units: str = "kbbar"):
        """Convert an entropy input (kbbar or cgs) to the k_B/baryon axis used
        by the SP table."""
        su = str(s_units).lower()
        s_arr = np.asarray(s_in, dtype=float)
        if su in ("kbbar", "kb/baryon", "kbperbaryon", "native"):
            return s_arr
        if su in ("cgs", "erg/g/k", "erg/g/kelvin"):
            return s_arr * self.erg_to_kbbar
        raise ValueError("s_units must be one of {'kbbar', 'cgs', 'native'}")

    # -----------------------------------------------------------------
    # Pre-computed S-P table (optional)
    # -----------------------------------------------------------------
    def _load_sp_table(self, path: Union[str, Path]):
        """Load a pre-computed S-P basis table produced from this class's
        ``get_t_sp_inv`` + PT-table getters.

        Expected keys (shape ``(n_S, n_P)`` for the grids):
            svals_sp    : S axis, k_B/baryon
            pvals_sp    : P axis, GPa
            t_grid_sp   : T, K
            rho_grid_sp : rho, g/cm^3
            u_grid_sp   : U, erg/g
            cp_grid_sp  : Cp, erg/g/K
            cv_grid_sp  : Cv, erg/g/K
            alpha_grid_sp : alpha, 1/K
        """
        data = np.load(path)
        self.svals_sp = np.asarray(data["svals_sp"], dtype=float)
        self.pvals_sp = np.asarray(data["pvals_sp"], dtype=float)

        expected = (self.svals_sp.size, self.pvals_sp.size)

        def _grid(name: str):
            arr = np.asarray(data[name], dtype=float)
            if arr.shape != expected:
                if arr.T.shape == expected:
                    arr = arr.T
                else:
                    raise ValueError(
                        f"{name} has shape {arr.shape}, expected {expected} "
                        f"(matching svals_sp x pvals_sp)."
                    )
            return arr

        self.t_grid_sp = _grid("t_grid_sp")
        self.rho_grid_sp = _grid("rho_grid_sp")
        self.u_grid_sp = _grid("u_grid_sp")
        self.cp_grid_sp = _grid("cp_grid_sp")
        self.cv_grid_sp = _grid("cv_grid_sp")
        self.alpha_grid_sp = _grid("alpha_grid_sp")

        rgi_kw = dict(method="linear", bounds_error=False, fill_value=None)
        axes = (self.svals_sp, self.pvals_sp)
        self._t_rgi_sp = RGI(axes, self.t_grid_sp, **rgi_kw)
        self._rho_rgi_sp = RGI(axes, self.rho_grid_sp, **rgi_kw)
        self._u_rgi_sp = RGI(axes, self.u_grid_sp, **rgi_kw)
        self._cp_rgi_sp = RGI(axes, self.cp_grid_sp, **rgi_kw)
        self._cv_rgi_sp = RGI(axes, self.cv_grid_sp, **rgi_kw)
        self._alpha_rgi_sp = RGI(axes, self.alpha_grid_sp, **rgi_kw)

        self._has_sp_table = True

    # -----------------------------------------------------------------
    # Legacy AQUA (used for Newton initial guesses)
    # -----------------------------------------------------------------
    def _legacy_aqua(self):
        if self._legacy_singleton is None:
            self._legacy_singleton = _legacy_aqua_module.AQUA_CORE_EOS()
        return self._legacy_singleton

    # -----------------------------------------------------------------
    # Phase transition profiles (ported from aqua_core_eos.py)
    # -----------------------------------------------------------------
    def _build_phase_transition_profiles(self):
        """Reuse the legacy AQUA phase profiles (ice-X, superionic, solidus).

        The underlying water substance is the same, so the legacy ice-X and
        superionic boundaries are valid here. We simply copy the profile
        arrays from the legacy wrapper rather than duplicating the phase-map
        logic.
        """
        legacy = self._legacy_aqua()

        self._solidus_pressures_gpa = legacy._solidus_pressures_gpa.copy()
        self._solidus_logp_gpa = legacy._solidus_logp_gpa.copy()
        self._solidus_t = legacy._solidus_t.copy()

        self._icex_pressures_gpa = legacy._icex_pressures_gpa.copy()
        self._icex_logp_gpa = legacy._icex_logp_gpa.copy()
        self._icex_t = legacy._icex_t.copy()
        self._icex_latent_ergg = legacy._icex_latent_ergg.copy()
        self.P_ice_x_min_GPa = float(legacy.P_ice_x_min_GPa)

        self._superionic_pressures_gpa = legacy._superionic_pressures_gpa.copy()
        self._superionic_t = legacy._superionic_t.copy()
        self._superionic_logp_gpa = legacy._superionic_logp_gpa.copy()
        self._superionic_logt = legacy._superionic_logt.copy()
        self.P_superionic_min_GPa = float(legacy.P_superionic_min_GPa)
        self.P_superionic_max_data_GPa = float(legacy.P_superionic_max_data_GPa)

        # Re-estimate the latent heats against the *revised* S(P, T) surface,
        # so that get_S_liq_at_melt / get_S_sol_at_melt are self-consistent.
        self.L_ice_x = self._estimate_latent_heat_from_curve(
            self._icex_pressures_gpa, phase="ice_x"
        )
        self.L_superionic = self._estimate_latent_heat_from_curve(
            self._superionic_pressures_gpa, phase="superionic"
        )
        # Preserve legacy aliases.
        self.L_aqua_ice_x = self.L_ice_x
        self.L = self.L_ice_x

    @staticmethod
    def _normalize_transition_phase(phase: Optional[str]) -> str:
        if phase is None:
            raise ValueError("phase must be specified explicitly: 'superionic' or 'ice_x'.")
        key = str(phase).strip().lower()
        if key in ("icex", "ice_x", "ice-x", "ice x"):
            return "ice_x"
        if key in ("superionic", "superion", "si"):
            return "superionic"
        raise ValueError("phase must be one of {'superionic', 'ice_x'}.")

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

    def get_T_melt_ice_x(self, P):
        scalar, P_arr, _ = self._broadcast(P, P)
        logp = np.log10(self._clip_positive(P_arr))

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

    def get_T_melt(self, P, phase: str = "ice_x"):
        phase_key = self._normalize_transition_phase(phase)
        if phase_key == "ice_x":
            return self.get_T_melt_ice_x(P)
        return self.get_T_melt_superionic(P)

    def get_S_liq_at_melt(self, P, phase: str = "ice_x"):
        scalar, P_arr, _ = self._broadcast(P, P)
        Tm = np.asarray(self.get_T_melt(P_arr, phase=phase), dtype=float)
        dT = np.maximum(25.0, 0.02 * Tm)
        T_liq = np.clip(Tm + dT, self.domain.T_min, self.domain.T_max)
        vals = self.get_s_pt(P_arr, T_liq)
        return self._maybe_scalar(scalar, vals)

    def get_S_sol_at_melt(self, P, phase: str = "ice_x"):
        scalar, P_arr, _ = self._broadcast(P, P)
        Tm = np.asarray(self.get_T_melt(P_arr, phase=phase), dtype=float)
        dT = np.maximum(25.0, 0.02 * Tm)
        T_sol = np.clip(Tm - dT, self.domain.T_min, self.domain.T_max)
        vals = self.get_s_pt(P_arr, T_sol)
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
        rho_sol = np.asarray(self.get_rho_pt(P_arr, T_sol), dtype=float)
        rho_liq = np.asarray(self.get_rho_pt(P_arr, T_liq), dtype=float)
        alpha_x = rho_arr * (
            1.0 / np.maximum(rho_liq, 1e-99) - 1.0 / np.maximum(rho_sol, 1e-99)
        )
        return self._maybe_scalar(scalar, alpha_x)

    # -----------------------------------------------------------------
    # Native PT getters (table-backed)
    # -----------------------------------------------------------------
    def get_rho_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logrho = self._interp(
            self.logrho_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, self._from_logrho(logrho))

    def get_s_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logs = self._interp(
            self.logs_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, 10.0 ** logs)

    def get_u_pt_tab(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        logu = self._interp(
            self.logu_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, self._from_logu(logu))

    def get_dlnS_dlnT_P_pt(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._interp(
            self.dlnS_dlnT_P_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, vals)

    def get_dlnS_dlnP_T_pt(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._interp(
            self.dlnS_dlnP_T_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, vals)

    def get_ad_grad_pt(self, P, T):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._interp(
            self.ad_grad_pt_rgi, self._to_logp(P_arr), self._to_logt(T_arr)
        )
        return self._maybe_scalar(scalar, vals)

    # Public forward getters -------------------------------------------------
    def get_rho_pt(self, P, T, tab=True, **kwargs):
        del tab, kwargs
        return self.get_rho_pt_tab(P, T)

    def get_s_pt(self, P, T, tab=True):
        del tab
        return self.get_s_pt_tab(P, T)

    def get_u_pt(self, P, T, tab=True):
        del tab
        return self.get_u_pt_tab(P, T)

    # -----------------------------------------------------------------
    # Derived PT quantities (analytical where table provides derivatives)
    # -----------------------------------------------------------------
    def _get_alpha_pt_derivative(self, P, T):
        """alpha = -rho * S / P_cgs * dlnS/dlnP|_T  (Maxwell relation)."""
        P_arr, T_arr = self._as_arrays(P, T)
        S = np.asarray(self.get_s_pt(P_arr, T_arr), dtype=float)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        dlnS_dlnP_T = np.asarray(self.get_dlnS_dlnP_T_pt(P_arr, T_arr), dtype=float)
        P_cgs = np.asarray(P_arr, dtype=float) * self.GPa_to_dyn
        return -rho * S * dlnS_dlnP_T / np.maximum(P_cgs, 1e-99)

    def _get_alpha_pt_finite_diff(self, P, T, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)
        rho_hi = np.asarray(self.get_rho_pt(P_arr, T_hi), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_arr, T_lo), dtype=float)
        drho_dT = (rho_hi - rho_lo) / (T_hi - T_lo)
        return -(drho_dT / np.maximum(rho, 1e-99))

    def get_alpha_pt(self, P, T, tab=True, method="derivative_table", eps_rel=1e-3):
        del tab
        method_l = str(method).lower()
        if method_l in ("derivative_table", "table", "derivative"):
            alpha = self._get_alpha_pt_derivative(P, T)
        elif method_l in ("finite_diff", "fd", "numeric"):
            alpha = self._get_alpha_pt_finite_diff(P, T, eps_rel=eps_rel)
        else:
            raise ValueError(
                "method must be one of {'derivative_table', 'finite_diff'}"
            )
        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, alpha)

    def _get_cp_pt_derivative(self, P, T):
        """Cp = S * dlnS/dlnT|_P."""
        P_arr, T_arr = self._as_arrays(P, T)
        S = np.asarray(self.get_s_pt(P_arr, T_arr), dtype=float)
        dlnS_dlnT_P = np.asarray(self.get_dlnS_dlnT_P_pt(P_arr, T_arr), dtype=float)
        return S * dlnS_dlnT_P

    def _get_cp_pt_finite_diff(self, P, T, eps_rel=1e-3):
        P_arr, T_arr = self._as_arrays(P, T)
        T_hi = T_arr * (1.0 + eps_rel)
        T_lo = np.maximum(T_arr * (1.0 - eps_rel), 1.0)
        s_hi = np.asarray(self.get_s_pt(P_arr, T_hi), dtype=float)
        s_lo = np.asarray(self.get_s_pt(P_arr, T_lo), dtype=float)
        dS_dT = (s_hi - s_lo) / (T_hi - T_lo)
        return T_arr * dS_dT

    def get_cp_pt(self, P, T, tab=True, method="derivative_table", eps_rel=1e-3):
        del tab
        method_l = str(method).lower()
        if method_l in ("derivative_table", "table", "derivative"):
            cp = self._get_cp_pt_derivative(P, T)
        elif method_l in ("finite_diff", "fd", "numeric"):
            cp = self._get_cp_pt_finite_diff(P, T, eps_rel=eps_rel)
        else:
            raise ValueError(
                "method must be one of {'derivative_table', 'finite_diff'}"
            )
        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, cp)

    def get_cv_pt(self, P, T, tab=True, eps_rel=1e-3, method="derivative_table"):
        """Cv from the thermodynamic identity Cp - Cv = T V alpha^2 / kappa_T.

        kappa_T = (1/rho)(drho/dP)|_T is finite-differenced on logP at fixed T.
        """
        del tab
        P_arr, T_arr = self._as_arrays(P, T)

        cp = np.asarray(
            self.get_cp_pt(P_arr, T_arr, method=method, eps_rel=eps_rel), dtype=float
        )
        alpha = np.asarray(
            self.get_alpha_pt(P_arr, T_arr, method=method, eps_rel=eps_rel), dtype=float
        )
        rho = np.asarray(self.get_rho_pt(P_arr, T_arr), dtype=float)

        logp = self._to_logp(P_arr)
        dlogp = eps_rel  # in log10 P (roughly a 0.1% P change for eps_rel=1e-3)
        logp_hi = logp + dlogp
        logp_lo = logp - dlogp
        P_hi = (10.0 ** logp_hi) * self.dyn_to_GPa
        P_lo = (10.0 ** logp_lo) * self.dyn_to_GPa
        rho_hi = np.asarray(self.get_rho_pt(P_hi, T_arr), dtype=float)
        rho_lo = np.asarray(self.get_rho_pt(P_lo, T_arr), dtype=float)
        # dlnrho/dlnP|_T; convert to kappa_T (1/Pa-CGS) via /P_cgs.
        dlnrho_dlnP = (np.log(rho_hi) - np.log(rho_lo)) / (2.0 * np.log(10.0) * dlogp)
        P_cgs = P_arr * self.GPa_to_dyn
        kappa_T = dlnrho_dlnP / np.maximum(P_cgs, 1e-99)  # 1/(dyn/cm^2)

        V = 1.0 / np.maximum(rho, 1e-99)  # cm^3/g
        cv = cp - T_arr * V * alpha * alpha / np.maximum(kappa_T, 1e-99)

        scalar = np.isscalar(P) and np.isscalar(T)
        return self._maybe_scalar(scalar, cv)

    # -----------------------------------------------------------------
    # Inversions
    # -----------------------------------------------------------------
    def get_rho_pt_inv(self, P, T, **kwargs):
        """PT is the native basis, so this is just an alias for the table lookup."""
        del kwargs
        return self.get_rho_pt_tab(P, T)

    def get_t_sp_inv(self, S, P, s_units="kbbar", **kwargs):
        """Solve S(P, T) = S_target for T.

        Strategy (per element of the input array):
            1. Newton's method in log T with analytical derivative
               f'(T) = (dS/dT)|_P = S * dlnS/dlnT|_P / T.
               Seeded from a warm-started previous solution, an explicit
               ``T_guess`` kwarg, or the legacy AQUA SP table.
            2. Secant method (scipy.root_scalar, method='secant') as fallback.
            3. brentq on an expanding logT bracket as final fallback.
        """
        T_guess = kwargs.pop("T_guess", None)
        bounds_T = kwargs.pop("bounds_T", None)
        newton_maxiter = int(kwargs.pop("newton_maxiter", 30))
        newton_dy_tol = float(kwargs.pop("newton_dy_tol", 1e-8))
        newton_res_tol = float(kwargs.pop("newton_res_tol", 1e-10))
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
            T_hi = 1e6
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

        # Pre-compute legacy-AQUA seed guesses (vectorised, O(1) per element).
        if T_guess is None:
            T_guess_arr = np.full(shape, np.sqrt(T_lo * T_hi), dtype=float)
            try:
                legacy = self._legacy_aqua()
                S_tab_guess = self._entropy_to_kbbar(S_goal)
                T_seed = np.asarray(
                    legacy.get_t_sp_tab(S_tab_guess, P_arr), dtype=float
                )
                finite = np.isfinite(T_seed) & (T_seed > 0.0)
                T_guess_arr = np.where(finite, T_seed, T_guess_arr)
            except Exception:
                pass
        else:
            T_guess_arr = np.asarray(T_guess, dtype=float)
            T_guess_arr, _ = np.broadcast_arrays(T_guess_arr, P_arr)
        T_guess_arr = np.clip(T_guess_arr, T_lo, T_hi)

        def f_of_T(P_val, T_val, S_cgs):
            S_here = self.get_s_pt(P_val, T_val)
            S_here = float(np.asarray(S_here).reshape(-1)[0])
            if not np.isfinite(S_here):
                return np.nan, np.nan
            resid = S_here - S_cgs
            dlnS_dlnT_P = self.get_dlnS_dlnT_P_pt(P_val, T_val)
            dlnS_dlnT_P = float(np.asarray(dlnS_dlnT_P).reshape(-1)[0])
            if not np.isfinite(dlnS_dlnT_P):
                return resid, np.nan
            dS_dT = S_here * dlnS_dlnT_P / T_val
            return resid, dS_dT

        def g_of_y(P_val, y_val, S_cgs):
            y_val = float(np.clip(y_val, y_min, y_max))
            T_val = float(np.exp(y_val))
            Sm = self.get_s_pt(P_val, T_val)
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

        T_prev = None
        ln10 = float(np.log(10.0))

        for idx in np.ndindex(shape):
            P_val = float(P_arr[idx])
            S_cgs = float(S_goal[idx])

            if not (np.isfinite(P_val) and np.isfinite(S_cgs)) or P_val <= 0.0:
                if return_diagnostics:
                    diag["method"][idx] = "none"
                    diag["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            Tg_raw = T_prev if (T_prev is not None and np.isfinite(T_prev)) else T_guess_arr[idx]
            Tg = float(np.clip(Tg_raw, T_lo, T_hi))

            # ---- Newton (primary) ----
            newton_ok = False
            T_cur = Tg
            nfev = 0
            niter = 0
            last_T_finite = Tg
            last_T_prev_finite = None
            for niter in range(1, newton_maxiter + 1):
                resid, dS_dT = f_of_T(P_val, T_cur, S_cgs)
                nfev += 1
                if not np.isfinite(resid) or not np.isfinite(dS_dT) or dS_dT == 0.0:
                    break
                last_T_prev_finite = last_T_finite
                last_T_finite = T_cur

                S_scale = max(abs(S_cgs), 1e-30)
                if abs(resid) / S_scale < newton_res_tol:
                    newton_ok = True
                    break

                # Update in log10 T space: dlogT = -resid / (dS_dT * T * ln10)
                logT_cur = np.log10(T_cur)
                dlogT = -resid / (dS_dT * T_cur * ln10)
                # Limit step so we stay inside the table and behave for large residuals.
                dlogT = float(np.clip(dlogT, -0.5, 0.5))
                logT_new = logT_cur + dlogT
                # Keep inside temperature bounds.
                logT_new = float(np.clip(logT_new, np.log10(T_lo), np.log10(T_hi)))
                T_new = 10.0 ** logT_new

                if abs(logT_new - logT_cur) * ln10 < newton_dy_tol:
                    T_cur = T_new
                    # One more residual evaluation to confirm.
                    resid, _ = f_of_T(P_val, T_cur, S_cgs)
                    nfev += 1
                    if np.isfinite(resid) and abs(resid) / S_scale < max(newton_res_tol, 1e-6):
                        newton_ok = True
                    break
                T_cur = T_new

            if newton_ok and np.isfinite(T_cur) and T_cur > 0.0:
                Tout[idx] = T_cur
                T_prev = T_cur
                if return_diagnostics:
                    diag["success"][idx] = True
                    diag["method"][idx] = "newton(logT)"
                    diag["message"][idx] = "OK"
                    diag["iterations"][idx] = niter
                    diag["nfev"][idx] = nfev
                continue

            # ---- Secant fallback ----
            secant_ok = False
            # Seed from the last two Newton iterates when available.
            if last_T_prev_finite is not None and last_T_prev_finite != last_T_finite:
                y0 = float(np.log(np.clip(last_T_prev_finite, T_lo, T_hi)))
                y1 = float(np.log(np.clip(last_T_finite, T_lo, T_hi)))
            else:
                y0 = float(np.log(np.clip(Tg, T_lo, T_hi)))
                y1 = float(np.clip(y0 + dy0, y_min, y_max))

            y0, g0 = nudge_to_finite(P_val, S_cgs, y0)
            if np.isfinite(g0):
                y1, g1 = nudge_to_finite(P_val, S_cgs, y1)
                if np.isfinite(g1) and y1 != y0:
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
                                    diag["nfev"][idx] = nfev + getattr(sol, "function_calls", 0)
                    except Exception as exc:
                        if return_diagnostics:
                            diag["method"][idx] = "secant(logT)"
                            diag["message"][idx] = f"Secant exception: {exc}"

            if secant_ok:
                continue

            # ---- brentq fallback with expanding bracket ----
            try:
                yc = float(np.log(np.clip(Tg, T_lo, T_hi)))
                gc = g_of_y(P_val, yc, S_cgs)
                if not np.isfinite(gc):
                    yc, gc = nudge_to_finite(P_val, S_cgs, yc)
                if not np.isfinite(gc):
                    raise RuntimeError("No finite residual near the initial guess.")

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
                        f"Failed to bracket logT root. P={P_val:.6g} GPa, S={S_cgs:.6g} erg/g/K"
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
                    diag["method"][idx] = "brentq(logT)"
                    diag["message"][idx] = "OK"
                    diag["n_expand"][idx] = n_expand
                    diag["iterations"][idx] = getattr(solb, "iterations", 0)
                    diag["nfev"][idx] = nfev + getattr(solb, "function_calls", 0)

            except Exception as exc:
                if return_diagnostics:
                    diag["success"][idx] = False
                    diag["method"][idx] = "brentq(logT)"
                    diag["message"][idx] = f"Brent exception: {exc}"

        if return_diagnostics:
            return Tout, diag
        return Tout

    def get_rhot_sp_2d_inv(self, S, P, s_units="kbbar", **kwargs):
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        rho = self.get_rho_pt(P, T)
        return rho, T

    def get_rhot_sp_inv(self, S, P, s_units="kbbar", **kwargs):
        return self.get_rhot_sp_2d_inv(S, P, s_units=s_units, **kwargs)

    # -----------------------------------------------------------------
    # SP table getters (native table units; expect S in k_B/baryon, P in GPa)
    # -----------------------------------------------------------------
    def _sp_tab_lookup(self, rgi, S_tab, P):
        if rgi is None:
            raise RuntimeError("SP table is not loaded.")
        scalar, S_arr, P_arr = self._broadcast(S_tab, P)
        pts = np.column_stack((S_arr.ravel(), P_arr.ravel()))
        vals = rgi(pts).reshape(S_arr.shape)
        return self._maybe_scalar(scalar, vals)

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

    # -----------------------------------------------------------------
    # SP convenience getters
    #
    # When an SP table has been loaded (default), ``tab=True`` short-circuits
    # the Newton inversion and reads from the table directly. ``tab=False``
    # (or a missing table) routes through ``get_t_sp_inv`` for inversion-based
    # evaluation.
    # -----------------------------------------------------------------
    def get_t_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_t_sp_tab(S_tab, P)
        return self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)

    def get_rho_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_rho_sp_tab(S_tab, P)
        rho, _ = self.get_rhot_sp_2d_inv(S, P, s_units=s_units, **kwargs)
        return rho

    def get_u_sp(self, S, P, tab=True, s_units="kbbar", **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_u_sp_tab(S_tab, P)
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        return self.get_u_pt(P, T)

    def get_cp_sp(self, S, P, tab=True, s_units="kbbar", method="derivative_table",
                   eps_rel=1e-3, **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_cp_sp_tab(S_tab, P)
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        return self.get_cp_pt(P, T, method=method, eps_rel=eps_rel)

    def get_cv_sp(self, S, P, tab=True, s_units="kbbar", method="derivative_table",
                   eps_rel=1e-3, **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_cv_sp_tab(S_tab, P)
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        return self.get_cv_pt(P, T, method=method, eps_rel=eps_rel)

    def get_alpha_sp(self, S, P, tab=True, s_units="kbbar", method="derivative_table",
                      eps_rel=1e-3, **kwargs):
        if tab and self._has_sp_table:
            S_tab = self._entropy_to_sp_table(S, s_units=s_units)
            return self.get_alpha_sp_tab(S_tab, P)
        T = self.get_t_sp_inv(S, P, s_units=s_units, **kwargs)
        return self.get_alpha_pt(P, T, method=method, eps_rel=eps_rel)
