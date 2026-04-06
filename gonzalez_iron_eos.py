"""
gonzalez_iron_eos.py

Combined solid+liquid iron EOS from Gonzalez-Cataldo & Militzer (2023).
Smoothly blends solid and liquid phases around the melting curve using
a tanh transition.

Modeled after iron_comb_eos.py.

Public API (all return cgs units):
    get_rho_pt(P, T)       -> g/cm^3
    get_s_pt(P, T)         -> erg/g/K
    get_u_pt(P, T)         -> erg/g
    get_alpha_pt(P, T)     -> 1/K
    get_cp_pt(P, T)        -> erg/g/K
    get_cv_pt(P, T)        -> erg/g/K

    get_t_sp(S, P)         -> K
    get_rho_sp(S, P)       -> g/cm^3
    get_u_sp(S, P)         -> erg/g

    get_T_melt(P)          -> K

    plot(...)               -> matplotlib Figure

Units:
    P   : GPa
    T   : K
    S   : kB/baryon (SP interface) or erg/g/K (PT interface output)
    rho : g/cm^3
    U   : erg/g

Reference:
    Gonzalez-Cataldo, F. and Militzer, B. (2023).
    "Ab initio determination of iron melting at terapascal pressures
     and Super-Earths core crystallization."
    Physical Review Research 5, 033194.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import brenth

from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

from eos import iron_gonzalez_eos

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
ERG_GK_TO_KBBAR = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
DYNCM2_TO_GPA = float((u.dyn / u.cm**2).to("GPa"))
U_CONV_CGS = float((u.J / u.kg).to("erg/g"))
S_CONV_CGS = float((u.J / u.kg / u.K).to("erg/(g * K)"))


# ---------------------------------------------------------------------------
# Thin subclass to fix data-file names on disk
# ---------------------------------------------------------------------------
class _Fe_EOS_GC(iron_gonzalez_eos.Fe_EOS):
    """Fe_EOS with correct filenames for the Gonzalez-Cataldo data tables."""

    _PHASE_FILES = {
        "solid": "Fe_EOS_solid_felipe_10_all_solids_noFrenkelcorrection.txt",
        "liquid": "Fe_EOS_liquid_felipe_06_all_liquids.txt",
    }


# ---------------------------------------------------------------------------
# Combined EOS
# ---------------------------------------------------------------------------
class Fe_GONZALEZ_EOS:
    """
    Combined Gonzalez-Cataldo & Militzer iron EOS.

    Wraps separate solid and liquid Fe_EOS instances and blends them
    with a smooth tanh weight around the melting curve Tm(P).

    Parameters
    ----------
    dT : float
        Half-width of the tanh blending region in K (default 50).
    apply_reference_offsets : bool
        Passed through to the per-phase Fe_EOS loaders.
    data_dir : str or Path, optional
        Override the data directory for the raw tables.
    """

    erg_to_kbbar: float = ERG_GK_TO_KBBAR
    _DEFAULT_TABLE = "gonzalez_iron_combined_pt.npz"

    def __init__(
        self,
        dT: float = 50.0,
        apply_reference_offsets: bool = False,
        data_dir=None,
        tab: bool = False,
        tab_path: Optional[Union[str, Path]] = None,
        **fe_eos_kwargs,
    ):
        self.dT = float(dT)

        common = dict(
            data_dir=data_dir,
            apply_reference_offsets=apply_reference_offsets,
            **fe_eos_kwargs,
        )
        self.solid = _Fe_EOS_GC(phase="solid", **common)
        self.liquid = _Fe_EOS_GC(phase="liquid", **common)

        self.P_min = min(self.solid.P_min, self.liquid.P_min)
        self.P_max = max(self.solid.P_max, self.liquid.P_max)
        self.T_min = min(self.solid.T_min, self.liquid.T_min)
        self.T_max = max(self.solid.T_max, self.liquid.T_max)

        # --- Pre-computed combined table ---
        self._has_tab = False
        if tab:
            self._load_table(tab_path)

    def _load_table(self, tab_path=None):
        """Load the pre-computed combined PT table and build RGI interpolators."""
        if tab_path is None:
            this_dir = Path(__file__).resolve().parent
            candidates = [
                this_dir / "gonzalez_iron_eos" / self._DEFAULT_TABLE,
                Path.cwd() / "eos" / "gonzalez_iron_eos" / self._DEFAULT_TABLE,
            ]
            for cand in candidates:
                if cand.is_file():
                    tab_path = cand
                    break
            if tab_path is None:
                raise FileNotFoundError(
                    f"Could not find {self._DEFAULT_TABLE}. "
                    "Generate it first or pass tab_path explicitly."
                )
        else:
            tab_path = Path(tab_path)

        data = np.load(tab_path)
        self._tab_p = np.asarray(data["p_vals"], dtype=float)  # GPa
        self._tab_t = np.asarray(data["t_vals"], dtype=float)  # K

        rgi_kw = dict(method="linear", bounds_error=False, fill_value=None)
        self._tab_rho_rgi = RGI((self._tab_p, self._tab_t), data["rho_grid"], **rgi_kw)
        self._tab_s_rgi = RGI((self._tab_p, self._tab_t), data["s_grid"], **rgi_kw)
        self._tab_u_rgi = RGI((self._tab_p, self._tab_t), data["u_grid"], **rgi_kw)
        self._has_tab = True

    def _query_tab(self, rgi, P_arr, T_arr):
        """Evaluate a tabulated RGI interpolator at (P, T) points."""
        pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
        return rgi(pts).reshape(P_arr.shape)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_arrays(a, b):
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A, B

    @staticmethod
    def _as_array(x):
        return np.array(x, ndmin=1, dtype=float)

    # ------------------------------------------------------------------
    # Melt curve
    # ------------------------------------------------------------------

    def get_T_melt(self, P):
        """
        Gonzalez-Cataldo & Militzer (2023) iron melting curve.

            Tm(K) = 6469 * (1 + (P_GPa - 300) / 434.822) ^ 0.54369

        Parameters
        ----------
        P : float or array
            Pressure in GPa.

        Returns
        -------
        Tm : ndarray
            Melting temperature in K.
        """
        P_arr = self._as_array(P)
        arg = 1.0 + (P_arr - 300.0) / 434.822
        with np.errstate(invalid="ignore"):
            Tm = 6469.0 * np.power(arg, 0.54369)
        return Tm

    # ------------------------------------------------------------------
    # Blend weight
    # ------------------------------------------------------------------

    def _blend_weight(self, P_arr, T_arr):
        """
        Liquid fraction w in [0, 1].
        w = 0.5 * (1 + tanh((T - Tm) / dT))
        """
        Tm = self.get_T_melt(P_arr)
        arg = (T_arr - Tm) / self.dT
        return np.clip(0.5 * (1.0 + np.tanh(arg)), 0.0, 1.0)

    # ------------------------------------------------------------------
    # PT interface
    # ------------------------------------------------------------------

    def get_rho_pt(self, P, T, tab=None):
        """Density [g/cm^3] via harmonic (volume-weighted) blend."""
        P_arr, T_arr = self._as_arrays(P, T)
        if (tab is True) or (tab is None and self._has_tab):
            return self._query_tab(self._tab_rho_rgi, P_arr, T_arr)
        rho_s = self.solid.get_rho_pt(P_arr, T_arr)
        rho_l = self.liquid.get_rho_pt(P_arr, T_arr)
        w = self._blend_weight(P_arr, T_arr)
        return 1.0 / ((1.0 - w) / rho_s + w / rho_l)

    def get_s_pt(self, P, T, tab=None):
        """Entropy [erg/g/K] with solid entropy clamped below liquid."""
        P_arr, T_arr = self._as_arrays(P, T)
        if (tab is True) or (tab is None and self._has_tab):
            return self._query_tab(self._tab_s_rgi, P_arr, T_arr)
        s_s = self.solid.get_s_pt(P_arr, T_arr)
        s_l = self.liquid.get_s_pt(P_arr, T_arr)

        # Hard clamp: ensure s_solid <= s_liquid
        eps = 1e-12 * np.maximum(1.0, np.abs(s_l))
        excess = s_s - (s_l - eps)
        s_s = np.where(excess > 0, s_l - eps, s_s)

        w = self._blend_weight(P_arr, T_arr)
        return (1.0 - w) * s_s + w * s_l

    def get_u_pt(self, P, T, tab=None):
        """Internal energy [erg/g]."""
        P_arr, T_arr = self._as_arrays(P, T)
        if (tab is True) or (tab is None and self._has_tab):
            return self._query_tab(self._tab_u_rgi, P_arr, T_arr)
        u_s = self.solid.get_u_pt(P_arr, T_arr)
        u_l = self.liquid.get_u_pt(P_arr, T_arr)
        w = self._blend_weight(P_arr, T_arr)
        return (1.0 - w) * u_s + w * u_l

    def get_alpha_pt(self, P, T):
        """Thermal expansivity [1/K]."""
        P_arr, T_arr = self._as_arrays(P, T)
        a_s = self.solid.get_alpha_pt(P_arr, T_arr)
        a_l = self.liquid.get_alpha_pt(P_arr, T_arr)
        w = self._blend_weight(P_arr, T_arr)
        return (1.0 - w) * a_s + w * a_l

    def get_cp_pt(self, P, T):
        """Heat capacity at constant pressure [erg/g/K]."""
        P_arr, T_arr = self._as_arrays(P, T)
        cp_s = self.solid.get_CP_pt(P_arr, T_arr)
        cp_l = self.liquid.get_CP_pt(P_arr, T_arr)
        w = self._blend_weight(P_arr, T_arr)
        return (1.0 - w) * cp_s + w * cp_l

    def get_CP_pt(self, P, T):
        return self.get_cp_pt(P, T)

    def get_cv_pt(self, P, T):
        """Heat capacity at constant volume [erg/g/K]."""
        P_arr, T_arr = self._as_arrays(P, T)
        cv_s = self.solid.get_CV_pt(P_arr, T_arr)
        cv_l = self.liquid.get_CV_pt(P_arr, T_arr)
        w = self._blend_weight(P_arr, T_arr)
        return (1.0 - w) * cv_s + w * cv_l

    def get_CV_pt(self, P, T):
        return self.get_cv_pt(P, T)

    # ------------------------------------------------------------------
    # SP interface  (S in kB/baryon, P in GPa)
    # ------------------------------------------------------------------

    def get_t_sp(
        self,
        S,
        P,
        s_units="kbbar",
        bounds_T=(100.0, 2e5),
        secant_maxiter=30,
        secant_rtol=1e-10,
    ):
        """
        Invert S(P,T) to find T(S,P).

        Parameters
        ----------
        S : float or array
            Entropy target. Units set by *s_units*.
        P : float or array
            Pressure in GPa.
        s_units : {"kbbar", "cgs"}
            "kbbar" means kB/baryon (default); "cgs" means erg/g/K.
        bounds_T : tuple
            (T_min, T_max) in K for the root bracket.

        Returns
        -------
        T : ndarray
            Temperature in K.
        """
        S_arr, P_arr = self._as_arrays(S, P)

        if str(s_units).lower() == "kbbar":
            S_cgs = S_arr / self.erg_to_kbbar  # -> erg/g/K
        else:
            S_cgs = S_arr.copy()

        T_lo, T_hi = map(float, bounds_T)
        y_lo, y_hi = np.log(T_lo), np.log(T_hi)

        out = np.full(S_arr.shape, np.nan, dtype=float)
        y_prev = 0.5 * (y_lo + y_hi)  # initial guess

        for idx in np.ndindex(S_arr.shape):
            s_target = float(S_cgs[idx])
            p_val = float(P_arr[idx])

            def residual(y):
                T_try = np.exp(y)
                s_try = float(self.get_s_pt(p_val, T_try))
                return s_try - s_target

            # --- secant warm-start ---
            converged = False
            y0 = y_prev
            dy = 1e-3
            y1 = y0 + dy
            try:
                f0 = residual(y0)
                f1 = residual(y1)
                for _ in range(secant_maxiter):
                    if abs(f1) < secant_rtol * max(abs(s_target), 1e-30):
                        converged = True
                        break
                    if abs(f1 - f0) < 1e-50:
                        break
                    y_new = y1 - f1 * (y1 - y0) / (f1 - f0)
                    y_new = np.clip(y_new, y_lo, y_hi)
                    y0, f0 = y1, f1
                    y1 = y_new
                    f1 = residual(y1)
                if converged:
                    out[idx] = np.exp(y1)
                    y_prev = y1
            except Exception:
                converged = False

            # --- brenth fallback ---
            if not converged:
                try:
                    fa = residual(y_lo)
                    fb = residual(y_hi)
                    if fa * fb > 0:
                        # expand bracket
                        for _ in range(10):
                            y_lo_try = y_lo - 0.5
                            y_hi_try = y_hi + 0.5
                            fa = residual(y_lo_try)
                            fb = residual(y_hi_try)
                            y_lo, y_hi = y_lo_try, y_hi_try
                            if fa * fb <= 0:
                                break
                    if fa * fb <= 0:
                        y_root = brenth(residual, y_lo, y_hi, xtol=1e-6, rtol=1e-10)
                        out[idx] = np.exp(y_root)
                        y_prev = y_root
                except Exception:
                    pass

        return out

    def get_rho_sp(self, S, P, **kwargs):
        """Density [g/cm^3] at given (S, P)."""
        T = self.get_t_sp(S, P, **kwargs)
        _, P_arr = self._as_arrays(S, P)
        return self.get_rho_pt(P_arr, T)

    def get_u_sp(self, S, P, **kwargs):
        """Internal energy [erg/g] at given (S, P)."""
        T = self.get_t_sp(S, P, **kwargs)
        _, P_arr = self._as_arrays(S, P)
        return self.get_u_pt(P_arr, T)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        properties=("rho", "S"),
        P_range_GPa=(50, 1500),
        T_range_K=(2000, 30000),
        n_P=200,
        n_T=200,
        figsize=None,
        cmap="viridis",
    ):
        """
        Plot blended EOS properties as heatmaps with the melt curve overlaid.

        Parameters
        ----------
        properties : tuple of str
            Which properties to plot. Options: "rho", "S", "U".
        P_range_GPa : tuple
            (P_min, P_max) in GPa for the plot axes.
        T_range_K : tuple
            (T_min, T_max) in K for the plot axes.
        n_P, n_T : int
            Grid resolution.
        figsize : tuple, optional
            Figure size. Defaults to (6*n_panels, 5).
        cmap : str
            Colormap name.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        getter_map = {
            "rho": (self.get_rho_pt, r"$\rho$ [g/cm$^3$]"),
            "S": (self.get_s_pt, r"$S$ [erg/g/K]"),
            "U": (self.get_u_pt, r"$U$ [erg/g]"),
        }

        n_panels = len(properties)
        if figsize is None:
            figsize = (6 * n_panels, 5)

        P_1d = np.linspace(P_range_GPa[0], P_range_GPa[1], n_P)
        T_1d = np.linspace(T_range_K[0], T_range_K[1], n_T)
        T_grid, P_grid = np.meshgrid(T_1d, P_1d)

        # Melt curve
        Tm = self.get_T_melt(P_1d)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        for ax, prop_key in zip(axes, properties):
            if prop_key not in getter_map:
                raise ValueError(f"Unknown property '{prop_key}'. Choose from {list(getter_map)}")
            getter, label = getter_map[prop_key]
            Z = getter(P_grid, T_grid)

            pcm = ax.pcolormesh(T_1d, P_1d, Z, shading="auto", cmap=cmap)
            ax.plot(Tm, P_1d, "w--", linewidth=2, label="Melt curve")

            # Label phases
            ax.text(
                0.15, 0.85, "Solid", transform=ax.transAxes,
                fontsize=12, color="white", fontweight="bold",
                ha="center", va="center",
            )
            ax.text(
                0.85, 0.15, "Liquid", transform=ax.transAxes,
                fontsize=12, color="white", fontweight="bold",
                ha="center", va="center",
            )

            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("Pressure [GPa]")
            ax.set_title(label)
            fig.colorbar(pcm, ax=ax, pad=0.02)
            ax.legend(loc="upper right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Thermodynamic consistency
    # ------------------------------------------------------------------

    def test_thermodynamic_consistency(
        self,
        P_range_GPa=(50, 1500),
        T_range_K=(3500, 25000),
        n_P=100,
        n_T=100,
        fd_step=0.01,
        tab=False,
    ):
        """
        Test first-law and Maxwell-relation consistency on a (logP, logT) grid.

        Uses centered finite differences in log10 space.

        First law at constant P:
            dU/dlogT|_P = T * dS/dlogT|_P + (P/rho^2) * drho/dlogT|_P

        First law at constant T:
            dU/dlogP|_T = T * dS/dlogP|_T + (P/rho^2) * drho/dlogP|_T

        Maxwell relation (from dG = VdP - SdT):
            (P/rho^2) * drho/dlogT|_P = T * dS/dlogP|_T

        Parameters
        ----------
        P_range_GPa : tuple
            (P_min, P_max) in GPa.
        T_range_K : tuple
            (T_min, T_max) in K.
        n_P, n_T : int
            Grid resolution.
        fd_step : float
            Finite-difference step in log10 space.
        tab : bool
            If True, use the pre-computed table for S, rho, U.

        Returns
        -------
        dict with keys 'first_law_constP', 'first_law_constT', 'maxwell',
        plus 'P_1d', 'T_1d' arrays.
        """
        lgp_1d = np.linspace(np.log10(P_range_GPa[0]), np.log10(P_range_GPa[1]), n_P)
        lgt_1d = np.linspace(np.log10(T_range_K[0]), np.log10(T_range_K[1]), n_T)
        LGT, LGP = np.meshgrid(lgt_1d, lgp_1d)

        h = fd_step

        # Evaluate S, rho, U and their FD derivatives in log10 space.
        # P is in GPa throughout; convert to dyn/cm^2 for the PdV term.
        GPA_TO_DYNCM2 = 1e10  # 1 GPa = 1e10 dyn/cm^2

        def _eval(lgp, lgt):
            P_gpa = 10.0**lgp
            T_K = 10.0**lgt
            rho = self.get_rho_pt(P_gpa, T_K, tab=tab)
            s = self.get_s_pt(P_gpa, T_K, tab=tab)
            u_val = self.get_u_pt(P_gpa, T_K, tab=tab)
            return rho, s, u_val

        rho_c, s_c, u_c = _eval(LGP, LGT)

        # d/dlogT at constant P
        _, s_Tp, u_Tp = _eval(LGP, LGT + h)
        _, s_Tm, u_Tm = _eval(LGP, LGT - h)
        rho_Tp_arr, _, _ = _eval(LGP, LGT + h)
        rho_Tm_arr, _, _ = _eval(LGP, LGT - h)

        dS_dlogT = (s_Tp - s_Tm) / (2.0 * h)
        drho_dlogT = (rho_Tp_arr - rho_Tm_arr) / (2.0 * h)
        dU_dlogT = (u_Tp - u_Tm) / (2.0 * h)

        # d/dlogP at constant T
        rho_Pp, s_Pp, u_Pp = _eval(LGP + h, LGT)
        rho_Pm, s_Pm, u_Pm = _eval(LGP - h, LGT)

        dS_dlogP = (s_Pp - s_Pm) / (2.0 * h)
        drho_dlogP = (rho_Pp - rho_Pm) / (2.0 * h)
        dU_dlogP = (u_Pp - u_Pm) / (2.0 * h)

        T = 10.0**LGT
        P_cgs = 10.0**LGP * GPA_TO_DYNCM2  # dyn/cm^2

        with np.errstate(invalid="ignore", divide="ignore"):
            P_over_rho2 = P_cgs / (rho_c**2)

        def _frac_err(lhs, rhs):
            denom = 0.5 * (np.abs(lhs) + np.abs(rhs))
            with np.errstate(invalid="ignore", divide="ignore"):
                err = np.abs(lhs - rhs) / (denom + 1e-30)
            err[denom < 1e-30] = 0.0
            return err

        # Test A: first law at constant P
        lhs_A = dU_dlogT
        rhs_A = T * dS_dlogT + P_over_rho2 * drho_dlogT
        err_A = _frac_err(lhs_A, rhs_A)

        # Test B: first law at constant T
        lhs_B = dU_dlogP
        rhs_B = T * dS_dlogP + P_over_rho2 * drho_dlogP
        err_B = _frac_err(lhs_B, rhs_B)

        # Test C: Maxwell relation
        lhs_C = P_over_rho2 * drho_dlogT
        rhs_C = T * dS_dlogP
        err_C = _frac_err(lhs_C, rhs_C)

        results = {
            "P_1d_GPa": 10.0**lgp_1d,
            "T_1d_K": 10.0**lgt_1d,
            "first_law_constP": err_A,
            "first_law_constT": err_B,
            "maxwell": err_C,
        }

        for name in ["first_law_constP", "first_law_constT", "maxwell"]:
            finite = results[name][np.isfinite(results[name])]
            if len(finite) > 0:
                print(f"  {name:22s}  median={np.median(finite):.2e}  "
                      f"p95={np.percentile(finite, 95):.2e}  "
                      f"max={np.max(finite):.2e}")

        return results

    def plot_thermodynamic_consistency(
        self,
        results=None,
        figsize=(20, 5),
        save_path=None,
        **test_kwargs,
    ):
        """
        Run thermodynamic consistency tests and plot heatmaps.

        Parameters
        ----------
        results : dict, optional
            Pre-computed results from test_thermodynamic_consistency().
            If None, runs the test with **test_kwargs.
        save_path : str, optional
            If given, save figure to this path.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if results is None:
            results = self.test_thermodynamic_consistency(**test_kwargs)

        P_1d = results["P_1d_GPa"]
        T_1d = results["T_1d_K"]
        Tm = self.get_T_melt(P_1d)

        test_names = ["first_law_constP", "first_law_constT", "maxwell"]
        titles = [
            r"First law (const $P$): $dU/d\log T = T\,dS/d\log T + (P/\rho^2)\,d\rho/d\log T$",
            r"First law (const $T$): $dU/d\log P = T\,dS/d\log P + (P/\rho^2)\,d\rho/d\log P$",
            r"Maxwell: $(P/\rho^2)\,d\rho/d\log T|_P = T\,dS/d\log P|_T$",
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, tname, title in zip(axes, test_names, titles):
            err = results[tname]
            log_err = np.log10(np.clip(err, 1e-16, None))

            pcm = ax.pcolormesh(T_1d, P_1d, log_err, shading="auto",
                                cmap="inferno", vmin=-6, vmax=0)
            ax.plot(Tm, P_1d, "w--", linewidth=1.5, label="Melt curve")
            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("Pressure [GPa]")
            ax.set_title(title, fontsize=9)
            cb = fig.colorbar(pcm, ax=ax, pad=0.02)
            cb.set_label(r"$\log_{10}$ fractional error")
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle("Gonzalez Iron EOS — Thermodynamic Consistency",
                     fontsize=13, y=1.02)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig
