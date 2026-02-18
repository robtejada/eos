from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brenth, least_squares

from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

ArrayLike = Union[float, np.ndarray]

# --- scalar conversion factors (plain floats) ---
ERG_GK_TO_KBBAR = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))  # (erg/g/K) -> (kB/baryon)
DYNCM2_TO_PA = float((u.dyn / u.cm**2).to("Pa"))
DYNCM2_TO_GPA = float((u.dyn / u.cm**2).to("GPa"))
U_CONV_CGS = float((u.J / u.kg).to("erg/g"))  # J/kg -> erg/g
S_CONV_CGS = float((u.J / u.kg / u.K).to("erg/(g * K)"))  # J/kg/K -> erg/g/K


class GonzalezRectGridBuilder:
    """
    Build a rectangular (rho, T) grid from irregular EOS samples.

    Two-step construction (both linear with extrapolation):
      1) For each unique table T, interpolate property vs rho onto regular rho axis.
      2) For each regular rho, interpolate property vs T onto regular T axis.
    """

    def __init__(
        self,
        rho_raw: np.ndarray,
        T_raw: np.ndarray,
        fields: Dict[str, np.ndarray],
        *,
        n_rho: Optional[int] = None,
        n_T: Optional[int] = None,
        rho_bounds: Optional[Tuple[float, float]] = None,
        T_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.rho_raw = np.asarray(rho_raw, dtype=float)
        self.T_raw = np.asarray(T_raw, dtype=float)
        self.fields = {k: np.asarray(v, dtype=float) for k, v in fields.items()}

        if self.rho_raw.size == 0 or self.T_raw.size == 0:
            raise ValueError("Empty raw rho/T arrays.")

        if not all(v.size == self.rho_raw.size for v in self.fields.values()):
            raise ValueError("All field arrays must have same length as rho_raw/T_raw.")

        rho_lo_raw = float(np.min(self.rho_raw))
        rho_hi_raw = float(np.max(self.rho_raw))
        T_lo_raw = float(np.min(self.T_raw))
        T_hi_raw = float(np.max(self.T_raw))

        if rho_bounds is None:
            rho_lo, rho_hi = rho_lo_raw, rho_hi_raw
        else:
            rho_lo, rho_hi = map(float, rho_bounds)

        if T_bounds is None:
            T_lo, T_hi = T_lo_raw, T_hi_raw
        else:
            T_lo, T_hi = map(float, T_bounds)

        if not (rho_hi > rho_lo > 0):
            raise ValueError("rho bounds must satisfy 0 < rho_lo < rho_hi.")
        if not (T_hi > T_lo > 0):
            raise ValueError("T bounds must satisfy 0 < T_lo < T_hi.")

        rho_unique = np.unique(self.rho_raw)
        T_unique = np.unique(self.T_raw)

        if n_rho is None:
            n_rho = int(rho_unique.size)
        if n_T is None:
            # keep regular temperature grid meaningfully sampled for derivatives
            n_T = int(max(T_unique.size, 300))

        self.n_rho = int(n_rho)
        self.n_T = int(n_T)
        if self.n_rho < 2 or self.n_T < 2:
            raise ValueError("n_rho and n_T must be >= 2.")

        self.rho_axis = np.linspace(rho_lo, rho_hi, self.n_rho)
        self.T_axis = np.linspace(T_lo, T_hi, self.n_T)
        self.T_unique = np.array(sorted(np.unique(self.T_raw)), dtype=float)

    @staticmethod
    def _dedupe_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x)
        xs = np.asarray(x[order], dtype=float)
        ys = np.asarray(y[order], dtype=float)

        xu, inv = np.unique(xs, return_inverse=True)
        if xu.size == xs.size:
            return xs, ys

        yu = np.zeros_like(xu, dtype=float)
        counts = np.zeros_like(xu, dtype=float)
        for i, g in enumerate(inv):
            yu[g] += ys[i]
            counts[g] += 1.0
        yu /= np.maximum(counts, 1.0)
        return xu, yu

    @classmethod
    def _interp1d_linear_extrap(cls, x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        xs, ys = cls._dedupe_sorted_xy(x, y)
        if xs.size == 1:
            return np.full_like(x_new, ys[0], dtype=float)

        f = interp1d(
            xs,
            ys,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return np.asarray(f(x_new), dtype=float)

    def _build_one_field(self, vals_raw: np.ndarray) -> np.ndarray:
        # Step 1: for each table T, interpolate/extrapolate along rho onto regular rho axis
        vals_rho_on_Traw = np.empty((self.T_unique.size, self.n_rho), dtype=float)

        for i, T0 in enumerate(self.T_unique):
            mask = self.T_raw == T0
            rho_slice = self.rho_raw[mask]
            v_slice = vals_raw[mask]
            vals_rho_on_Traw[i, :] = self._interp1d_linear_extrap(rho_slice, v_slice, self.rho_axis)

        # Step 2: for each regular rho, interpolate/extrapolate along T onto regular T axis
        grid_rhot = np.empty((self.n_rho, self.n_T), dtype=float)
        for j in range(self.n_rho):
            v_Tslice = vals_rho_on_Traw[:, j]
            grid_rhot[j, :] = self._interp1d_linear_extrap(self.T_unique, v_Tslice, self.T_axis)

        return grid_rhot

    def build(self) -> Dict[str, np.ndarray]:
        out = {
            "rho_axis": self.rho_axis.copy(),
            "T_axis": self.T_axis.copy(),
        }

        for key, vals_raw in self.fields.items():
            out[key] = self._build_one_field(vals_raw)

        return out


class GonzalezRegularGridSurface:
    """
    RegularGridInterpolator wrapper for a single thermodynamic surface.
    Uses linear interpolation and linear extrapolation (fill_value=None).
    """

    def __init__(self, rho_axis: np.ndarray, T_axis: np.ndarray, grid_rhot: np.ndarray) -> None:
        self._rgi = RegularGridInterpolator(
            (np.asarray(rho_axis, dtype=float), np.asarray(T_axis, dtype=float)),
            np.asarray(grid_rhot, dtype=float),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        rho_arr = np.asarray(rho, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        rho_arr, T_arr = np.broadcast_arrays(rho_arr, T_arr)
        pts = np.column_stack((rho_arr.ravel(), T_arr.ravel()))
        vals = np.asarray(self._rgi(pts), dtype=float).reshape(rho_arr.shape)
        return vals


class Fe_EOS:
    """
    Gonzalez-Cataldo & Militzer iron EOS reader/interpolator.

    Loads either solid or liquid Fe text tables and builds interpolators for:
      P(rho, T), S(rho, T), U(rho, T), G(rho, T)

    Units:
      Inputs:
        rho : kg/m^3
        T   : K
        P   : Pa

      Outputs:
        P : Pa
        S : erg/g/K
        U : erg/g
        G : erg/g
        Cp, Cv : erg/g/K
        alpha : 1/K
    """

    erg_to_kbbar: float = ERG_GK_TO_KBBAR
    dyn_to_Pa: float = DYNCM2_TO_PA
    dyn_to_GPa: float = DYNCM2_TO_GPA
    kb: float = k_B.to("erg/K")

    _FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    _REGEX = {
        "rho_gcc": re.compile(r"rho\[g/cc\]=\s*" + _FLOAT),
        "T_K": re.compile(r"T\[K\]=\s*" + _FLOAT),
        "P_GPa": re.compile(r"P\[GPa\]=\s*" + _FLOAT),
        "E_Jkg": re.compile(r"E\[J/kg\]=\s*" + _FLOAT),
        "G_Jkg": re.compile(r"G_DFT\[J/kg\]=\s*" + _FLOAT),
        "S_JkgK": re.compile(r"S\[J/kg/K\]=\s*" + _FLOAT),
    }

    _PHASE_FILES = {
        "solid": "Fe_EOS_solid.txt",
        "liquid": "Fe_EOS_liquid.txt",
    }
    _GLOBAL_BOUNDS_CACHE: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    def __init__(
        self,
        phase: str = "solid",
        data_dir: Optional[Union[str, Path]] = None,
        diff_rel_T: float = 5e-2,
        diff_abs_T: float = 100.0,
        diff_rel_P: float = 1e-2,
        diff_abs_P: float = 1e9,
        grid_n_rho: Optional[int] = None,
        grid_n_T: Optional[int] = None,
        grid_rho_bounds: Optional[Tuple[float, float]] = None,
        grid_T_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        phase : {"solid", "liquid"}
            Which Gonzalez table to load.
        grid_n_rho, grid_n_T : int, optional
            Number of regular rho/T samples for the rectangularized EOS grid.
        grid_rho_bounds, grid_T_bounds : tuple, optional
            Explicit (min, max) bounds for the regularized grid axes.
        """
        self.phase = str(phase).lower()
        if self.phase not in self._PHASE_FILES:
            raise ValueError("phase must be 'solid' or 'liquid'")

        self.diff_rel_T = float(diff_rel_T)
        self.diff_abs_T = float(diff_abs_T)
        self.diff_rel_P = float(diff_rel_P)
        self.diff_abs_P = float(diff_abs_P)

        self._data_dir = self._resolve_data_dir(data_dir)
        self._file_path = self._data_dir / self._PHASE_FILES[self.phase]

        if grid_rho_bounds is None or grid_T_bounds is None:
            rho_bounds_all, T_bounds_all = self._global_phase_bounds(self._data_dir)
            if grid_rho_bounds is None:
                grid_rho_bounds = rho_bounds_all
            if grid_T_bounds is None:
                grid_T_bounds = T_bounds_all

        table = self._read_table(self._file_path)

        self.rho_table = table["rho_kgm3"]
        self.T_table = table["T_K"]
        self.P_table = table["P_Pa"]
        self.S_table_si = table["S_JkgK"]
        self.U_table_si = table["U_Jkg"]
        self.G_table_si = table["G_Jkg"]

        rect_builder = GonzalezRectGridBuilder(
            self.rho_table,
            self.T_table,
            {
                "P": self.P_table,
                "S": self.S_table_si,
                "U": self.U_table_si,
                "G": self.G_table_si,
            },
            n_rho=grid_n_rho,
            n_T=grid_n_T,
            rho_bounds=grid_rho_bounds,
            T_bounds=grid_T_bounds,
        )
        rect = rect_builder.build()

        self.rho_vals_rect = np.asarray(rect["rho_axis"], dtype=float)
        self.T_vals_rect = np.asarray(rect["T_axis"], dtype=float)
        self.P_grid_rect = np.asarray(rect["P"], dtype=float)
        self.S_grid_rect_si = np.asarray(rect["S"], dtype=float)
        self.U_grid_rect_si = np.asarray(rect["U"], dtype=float)
        self.G_grid_rect_si = np.asarray(rect["G"], dtype=float)

        self.rho_min = float(self.rho_vals_rect[0])
        self.rho_max = float(self.rho_vals_rect[-1])
        self.T_min = float(self.T_vals_rect[0])
        self.T_max = float(self.T_vals_rect[-1])
        self.P_min = float(np.min(self.P_table))
        self.P_max = float(np.max(self.P_table))

        self._surf: Dict[str, GonzalezRegularGridSurface] = {
            "P": GonzalezRegularGridSurface(self.rho_vals_rect, self.T_vals_rect, self.P_grid_rect),
            "S": GonzalezRegularGridSurface(self.rho_vals_rect, self.T_vals_rect, self.S_grid_rect_si),
            "U": GonzalezRegularGridSurface(self.rho_vals_rect, self.T_vals_rect, self.U_grid_rect_si),
            "G": GonzalezRegularGridSurface(self.rho_vals_rect, self.T_vals_rect, self.G_grid_rect_si),
        }

        self._build_isotherm_seed_index()

    # -------------------------
    # Setup helpers
    # -------------------------

    @staticmethod
    def _resolve_data_dir(data_dir: Optional[Union[str, Path]]) -> Path:
        this_dir = Path(__file__).resolve().parent

        candidates = []
        if data_dir is not None:
            candidates.append(Path(data_dir))

        candidates.extend(
            [
                this_dir / "gonzales_iron_eos",
                this_dir / "eos" / "gonzales_iron_eos",
                Path.cwd() / "gonzales_iron_eos",
                Path.cwd() / "eos" / "gonzales_iron_eos",
            ]
        )

        for cand in candidates:
            if cand.is_dir():
                return cand

        raise FileNotFoundError(
            "Could not locate 'gonzales_iron_eos' directory. "
            "Pass data_dir explicitly."
        )

    @classmethod
    def _global_phase_bounds(cls, data_dir: Path) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        key = str(Path(data_dir).resolve())
        if key in cls._GLOBAL_BOUNDS_CACHE:
            return cls._GLOBAL_BOUNDS_CACHE[key]

        rho_min = np.inf
        rho_max = -np.inf
        T_min = np.inf
        T_max = -np.inf

        for _, fname in cls._PHASE_FILES.items():
            path = Path(data_dir) / fname
            if not path.is_file():
                continue
            table = cls._read_table(path)
            rho_min = min(rho_min, float(np.min(table["rho_kgm3"])))
            rho_max = max(rho_max, float(np.max(table["rho_kgm3"])))
            T_min = min(T_min, float(np.min(table["T_K"])))
            T_max = max(T_max, float(np.max(table["T_K"])))

        if not np.isfinite(rho_min) or not np.isfinite(T_min):
            raise FileNotFoundError("Could not determine global bounds for Gonzalez iron tables.")

        out = ((rho_min, rho_max), (T_min, T_max))
        cls._GLOBAL_BOUNDS_CACHE[key] = out
        return out

    @classmethod
    def _extract_float(cls, pat: re.Pattern, text: str) -> Optional[float]:
        m = pat.search(text)
        if m is None:
            return None
        return float(m.group(1))

    @classmethod
    def _read_table(cls, path: Path) -> Dict[str, np.ndarray]:
        rho_gcc = []
        T_K = []
        P_GPa = []
        E_Jkg = []
        G_Jkg = []
        S_JkgK = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue

                rho = cls._extract_float(cls._REGEX["rho_gcc"], s)
                T = cls._extract_float(cls._REGEX["T_K"], s)
                P = cls._extract_float(cls._REGEX["P_GPa"], s)
                E = cls._extract_float(cls._REGEX["E_Jkg"], s)
                G = cls._extract_float(cls._REGEX["G_Jkg"], s)
                S = cls._extract_float(cls._REGEX["S_JkgK"], s)

                if None in (rho, T, P, E, G, S):
                    continue

                rho_gcc.append(rho)
                T_K.append(T)
                P_GPa.append(P)
                E_Jkg.append(E)
                G_Jkg.append(G)
                S_JkgK.append(S)

        if len(rho_gcc) == 0:
            raise ValueError(f"No valid rows parsed from {path}")

        rho_kgm3 = np.asarray(rho_gcc, dtype=float) * 1e3
        T_K = np.asarray(T_K, dtype=float)
        P_Pa = np.asarray(P_GPa, dtype=float) * 1e9
        U_Jkg = np.asarray(E_Jkg, dtype=float)
        G_Jkg = np.asarray(G_Jkg, dtype=float)
        S_JkgK = np.asarray(S_JkgK, dtype=float)

        # Drop exact duplicate (rho, T) rows (liquid table has a few exact duplicates).
        rt = np.column_stack((rho_kgm3, T_K))
        _, keep_idx = np.unique(rt, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)

        return {
            "rho_kgm3": rho_kgm3[keep_idx],
            "T_K": T_K[keep_idx],
            "P_Pa": P_Pa[keep_idx],
            "U_Jkg": U_Jkg[keep_idx],
            "G_Jkg": G_Jkg[keep_idx],
            "S_JkgK": S_JkgK[keep_idx],
        }

    def _build_isotherm_seed_index(self) -> None:
        self.tvals_iso = np.array(sorted(np.unique(self.T_table)), dtype=float)
        self._seed_by_T: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        for T in self.tvals_iso:
            mask = self.T_table == T
            P = self.P_table[mask]
            rho = self.rho_table[mask]

            order = np.argsort(P)
            P = P[order]
            rho = rho[order]

            P_u, idx = np.unique(P, return_index=True)
            rho_u = rho[idx]
            self._seed_by_T[float(T)] = (P_u, rho_u)

    # -------------------------
    # Utility helpers
    # -------------------------

    @staticmethod
    def _broadcast(a, b):
        scalar = np.isscalar(a) and np.isscalar(b)
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        if A.shape != B.shape:
            A, B = np.broadcast_arrays(A, B)
        return scalar, A, B

    @staticmethod
    def _as_float(x) -> float:
        return float(np.asarray(x))

    def _interp_property_si(self, key: str, rho_kgm3: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho_kgm3, dtype=float)
        T = np.asarray(T_K, dtype=float)

        if np.any(rho <= 0):
            raise ValueError("Density rho must be > 0 kg/m^3.")
        if np.any(T <= 0):
            raise ValueError("Temperature T must be > 0 K.")

        vals = self._surf[key](rho, T)
        vals = np.asarray(vals, dtype=float)
        return vals

    def _interp_property_si_scalar(self, key: str, rho_kgm3: float, T_K: float) -> float:
        if (not np.isfinite(rho_kgm3)) or (rho_kgm3 <= 0):
            return np.nan
        if (not np.isfinite(T_K)) or (T_K <= 0):
            return np.nan
        return float(self._surf[key](float(rho_kgm3), float(T_K)))

    def _rho_seed_from_isotherms(self, P_Pa: float, T_K: float) -> float:
        Ts = self.tvals_iso
        if Ts.size == 0:
            return np.sqrt(self.rho_min * self.rho_max)

        def rho_at_T(Tref: float) -> float:
            P_grid, rho_grid = self._seed_by_T[float(Tref)]
            return float(np.interp(P_Pa, P_grid, rho_grid, left=rho_grid[0], right=rho_grid[-1]))

        if T_K <= Ts[0]:
            return rho_at_T(Ts[0])
        if T_K >= Ts[-1]:
            return rho_at_T(Ts[-1])

        i_hi = int(np.searchsorted(Ts, T_K))
        i_lo = i_hi - 1
        T_lo = float(Ts[i_lo])
        T_hi = float(Ts[i_hi])
        if T_hi == T_lo:
            return rho_at_T(T_lo)

        rho_lo = rho_at_T(T_lo)
        rho_hi = rho_at_T(T_hi)
        w = (T_K - T_lo) / (T_hi - T_lo)
        return (1.0 - w) * rho_lo + w * rho_hi

    @staticmethod
    def _finite_diff_scalar(
        f,
        x: float,
        x_min: float,
        x_max: float,
        rel_step: float,
        abs_step: float,
    ) -> float:
        if not np.isfinite(x):
            return np.nan

        h = max(abs_step, rel_step * max(abs(x), 1.0))
        if h <= 0:
            return np.nan

        x_lo = x - h
        x_hi = x + h

        can_lo = x_lo >= x_min
        can_hi = x_hi <= x_max

        if can_lo and can_hi:
            f_lo = f(x_lo)
            f_hi = f(x_hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi):
                return (f_hi - f_lo) / (2.0 * h)
            return np.nan

        if can_hi:
            f0 = f(x)
            f1 = f(x_hi)
            if np.isfinite(f0) and np.isfinite(f1):
                return (f1 - f0) / h
            return np.nan

        if can_lo:
            f0 = f(x)
            f1 = f(x_lo)
            if np.isfinite(f0) and np.isfinite(f1):
                return (f0 - f1) / h
            return np.nan

        return np.nan

    # -------------------------
    # Core RHOT interpolation API
    # -------------------------

    def get_p_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        vals = self._interp_property_si("P", rho_arr, T_arr)
        return float(vals) if scalar else vals

    def get_s_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        vals_si = self._interp_property_si("S", rho_arr, T_arr)
        vals = vals_si * S_CONV_CGS
        return float(vals) if scalar else vals

    def get_u_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        vals_si = self._interp_property_si("U", rho_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    def get_g_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        vals_si = self._interp_property_si("G", rho_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    # -------------------------
    # Derivative-based thermo
    # -------------------------

    def _cv_si_scalar(self, rho: float, T: float) -> float:
        def u_of_T(Ti: float) -> float:
            return self._interp_property_si_scalar("U", rho, Ti)

        return self._finite_diff_scalar(
            u_of_T,
            T,
            self.T_min,
            self.T_max,
            self.diff_rel_T,
            self.diff_abs_T,
        )

    def _s_pt_si_scalar(self, P: float, T: float, rho0: Optional[float] = None) -> float:
        if (not np.isfinite(P)) or (not np.isfinite(T)) or (P <= 0) or (T <= 0):
            return np.nan
        rho_guess = rho0 if (rho0 is not None and np.isfinite(rho0) and rho0 > 0) else self._rho_seed_from_isotherms(P, T)
        rho = self.get_rho_pt_inv(P, T, rho0=rho_guess, on_fail="nan")
        rho = float(np.asarray(rho))
        if not np.isfinite(rho):
            return np.nan
        return self._interp_property_si_scalar("S", rho, T)

    def _g_pt_si_scalar(self, P: float, T: float, rho0: Optional[float] = None) -> float:
        if (not np.isfinite(P)) or (not np.isfinite(T)) or (P <= 0) or (T <= 0):
            return np.nan
        rho_guess = rho0 if (rho0 is not None and np.isfinite(rho0) and rho0 > 0) else self._rho_seed_from_isotherms(P, T)
        rho = self.get_rho_pt_inv(P, T, rho0=rho_guess, on_fail="nan")
        rho = float(np.asarray(rho))
        if not np.isfinite(rho):
            return np.nan
        return self._interp_property_si_scalar("G", rho, T)

    def _cp_pt_si_scalar(self, P: float, T: float, rho0: Optional[float] = None) -> float:
        def s_of_T(Ti: float) -> float:
            return self._s_pt_si_scalar(P, Ti, rho0=rho0)

        dS_dT_P = self._finite_diff_scalar(
            s_of_T,
            T,
            self.T_min,
            self.T_max,
            self.diff_rel_T,
            self.diff_abs_T,
        )
        if not np.isfinite(dS_dT_P):
            return np.nan
        return T * dS_dT_P

    def _alpha_pt_scalar(self, P: float, T: float, rho0: Optional[float] = None) -> float:
        cache: Dict[Tuple[float, float], float] = {}

        def g_cached(Pi: float, Ti: float) -> float:
            key = (float(Pi), float(Ti))
            if key not in cache:
                cache[key] = self._g_pt_si_scalar(Pi, Ti, rho0=rho0)
            return cache[key]

        def v_of_T(Ti: float) -> float:
            def g_of_P(Pi: float) -> float:
                return g_cached(Pi, Ti)

            return self._finite_diff_scalar(
                g_of_P,
                P,
                self.P_min,
                self.P_max,
                self.diff_rel_P,
                self.diff_abs_P,
            )

        v = v_of_T(T)
        if not np.isfinite(v) or abs(v) <= 0.0:
            return np.nan

        dv_dT = self._finite_diff_scalar(
            v_of_T,
            T,
            self.T_min,
            self.T_max,
            self.diff_rel_T,
            self.diff_abs_T,
        )
        if not np.isfinite(dv_dT):
            return np.nan
        return dv_dT / v

    def get_CV_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        out = np.full(rho_arr.shape, np.nan, dtype=float)

        for idx in np.ndindex(rho_arr.shape):
            out[idx] = self._cv_si_scalar(float(rho_arr[idx]), float(T_arr[idx]))

        out *= S_CONV_CGS
        return float(out) if scalar else out

    def get_CP_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        out = np.full(rho_arr.shape, np.nan, dtype=float)

        P_arr = self.get_p_rhot(rho_arr, T_arr)
        for idx in np.ndindex(rho_arr.shape):
            out[idx] = self._cp_pt_si_scalar(
                float(P_arr[idx]),
                float(T_arr[idx]),
                rho0=float(rho_arr[idx]),
            )

        out *= S_CONV_CGS
        return float(out) if scalar else out

    def get_alpha_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        out = np.full(rho_arr.shape, np.nan, dtype=float)

        P_arr = self.get_p_rhot(rho_arr, T_arr)
        for idx in np.ndindex(rho_arr.shape):
            out[idx] = self._alpha_pt_scalar(
                float(P_arr[idx]),
                float(T_arr[idx]),
                rho0=float(rho_arr[idx]),
            )

        return float(out) if scalar else out

    # lowercase aliases used in a few EOS modules
    def get_cv_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        return self.get_CV_rhot(rho_kgm3, T_K)

    def get_cp_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        return self.get_CP_rhot(rho_kgm3, T_K)

    # -------------------------
    # Inversion rho(P,T)
    # -------------------------

    def get_rho_pt_inv(
        self,
        P: ArrayLike,
        T_K: ArrayLike,
        rho_bracket_kgm3: Optional[Tuple[float, float]] = None,
        max_iter: int = 200,
        rtol: float = 1e-10,
        rho0: Optional[ArrayLike] = None,
        use_lsq_first: bool = True,
        lsq_max_nfev: int = 60,
        bracket_expand_steps: int = 30,
        bracket_expand_factor: float = 1.6,
        on_fail: str = "nan",  # "nan" or "raise"
    ) -> np.ndarray:
        P_arr = np.asarray(P, dtype=float)
        T_arr = np.asarray(T_K, dtype=float)
        shape = np.broadcast(P_arr, T_arr).shape
        P_b = np.broadcast_to(P_arr, shape)
        T_b = np.broadcast_to(T_arr, shape)

        out = np.full(shape, np.nan, dtype=float)

        if rho_bracket_kgm3 is None:
            rho_min = self.rho_min
            rho_max = self.rho_max
        else:
            rho_min = float(rho_bracket_kgm3[0])
            rho_max = float(rho_bracket_kgm3[1])

        if rho_min <= 0 or rho_max <= rho_min:
            raise ValueError("rho_bracket_kgm3 must satisfy 0 < rho_min < rho_max")

        log_rho_lo = np.log(rho_min)
        log_rho_hi = np.log(rho_max)

        rho0_b = None
        if rho0 is not None:
            rho0_arr = np.asarray(rho0, dtype=float)
            rho0_b = np.broadcast_to(rho0_arr, shape)

        rho_prev = np.nan

        def P_of_rho_scalar(rho_val: float, Ti: float) -> float:
            if rho_val <= 0 or Ti <= 0:
                return np.nan
            return self._interp_property_si_scalar("P", rho_val, Ti)

        for idx in np.ndindex(shape):
            Pt = float(P_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(Pt)) or (not np.isfinite(Ti)) or Pt <= 0 or Ti <= 0:
                continue

            if np.isfinite(rho_prev):
                rho_guess = float(rho_prev)
            elif rho0_b is not None and np.isfinite(rho0_b[idx]) and rho0_b[idx] > 0:
                rho_guess = float(rho0_b[idx])
            else:
                rho_guess = self._rho_seed_from_isotherms(Pt, Ti)

            rho_guess = min(max(rho_guess, rho_min), rho_max)

            P_scale = max(abs(Pt), 1.0)

            def resid(logrho_vec):
                rho_try = float(np.exp(logrho_vec[0]))
                Pm = P_of_rho_scalar(rho_try, Ti)
                if not np.isfinite(Pm):
                    return np.array([1e30], dtype=float)
                return np.array([(Pm - Pt) / P_scale], dtype=float)

            rho_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(rho_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([log_rho_lo], [log_rho_hi]),
                        xtol=rtol,
                        ftol=rtol,
                        gtol=rtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        rho_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(rho_try)], dtype=float))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            rho_sol = rho_try
                except Exception:
                    pass

            if not np.isfinite(rho_sol):
                def f(rho_val):
                    return P_of_rho_scalar(rho_val, Ti) - Pt

                left = right = rho_guess
                f_left = f(right)

                if not np.isfinite(f_left):
                    rho_guess = np.sqrt(rho_min * rho_max)
                    left = right = rho_guess
                    f_left = f(right)

                if np.isfinite(f_left) and f_left == 0.0:
                    rho_sol = rho_guess
                else:
                    for _ in range(bracket_expand_steps):
                        left = max(rho_min, left / bracket_expand_factor)
                        right = min(rho_max, right * bracket_expand_factor)

                        f_l = f(left)
                        f_r = f(right)
                        if not (np.isfinite(f_l) and np.isfinite(f_r)):
                            continue

                        if f_l == 0.0:
                            rho_sol = left
                            break
                        if f_r == 0.0:
                            rho_sol = right
                            break
                        if f_l * f_r < 0:
                            try:
                                rho_sol = brenth(f, left, right, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                rho_sol = np.nan
                            break

                    if not np.isfinite(rho_sol):
                        fA = f(rho_min)
                        fB = f(rho_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                rho_sol = brenth(f, rho_min, rho_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                rho_sol = np.nan

            if np.isfinite(rho_sol):
                out[idx] = rho_sol
                rho_prev = rho_sol
            elif on_fail == "raise":
                raise RuntimeError(
                    f"Failed rho(P,T) inversion: P={Pt:.3e} Pa, T={Ti:.3f} K "
                    f"within [{rho_min}, {rho_max}] kg/m^3"
                )

        return float(out) if out.size == 1 else out

    def get_s_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        rho = self.get_rho_pt_inv(P, T, rho0=rho0, **inv_kwargs)
        return self.get_s_rhot(rho, T)

    def get_u_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        rho = self.get_rho_pt_inv(P, T, rho0=rho0, **inv_kwargs)
        return self.get_u_rhot(rho, T)

    def get_g_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        rho = self.get_rho_pt_inv(P, T, rho0=rho0, **inv_kwargs)
        return self.get_g_rhot(rho, T)

    # -------------------------
    # Inversion T(S,rho)
    # -------------------------

    def get_T_srho_inv(
        self,
        _s,
        _rho,
        bracket=(1.0, 200000.0),
        xtol=1e-10,
        maxiter=200,
        s_units="kbbar",
        T_guess0=None,
        use_lsq_first=True,
        lsq_max_nfev=80,
        bracket_expand_steps=30,
        bracket_expand_factor=1.6,
    ):
        s_arr = np.asarray(_s, dtype=float)
        rho_arr = np.asarray(_rho, dtype=float)
        s_arr, rho_arr = np.broadcast_arrays(s_arr, rho_arr)
        shape = s_arr.shape

        Tmin, Tmax = map(float, bracket)
        Tmin = max(Tmin, self.T_min)
        Tmax = min(Tmax, self.T_max)
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")
        if Tmax <= Tmin:
            raise ValueError("Temperature bracket does not overlap table range.")

        if str(s_units).lower() == "kbbar":
            s_target_cgs = s_arr / float(self.erg_to_kbbar)
        else:
            s_target_cgs = s_arr

        T_out = np.full(shape, np.nan, dtype=float)

        def S_cgs(rho_val, T_val) -> float:
            return float(np.asarray(self.get_s_rhot(rho_val, T_val)))

        logT_lo, logT_hi = np.log(Tmin), np.log(Tmax)

        if T_guess0 is None:
            T_seed_first = float(np.median(self.tvals_iso))
        else:
            T_seed_first = float(T_guess0)
        T_seed_first = min(max(T_seed_first, Tmin), Tmax)

        T_prev = None

        for idx in np.ndindex(shape):
            rho = float(rho_arr[idx])
            s_t = float(s_target_cgs[idx])

            if (not np.isfinite(rho)) or rho <= 0 or (not np.isfinite(s_t)):
                continue

            T_guess = T_seed_first if (T_prev is None or not np.isfinite(T_prev)) else float(T_prev)
            T_guess = min(max(T_guess, Tmin), Tmax)

            S_scale = max(abs(s_t), 1.0)

            def resid(logT_vec):
                T = float(np.exp(logT_vec[0]))
                Sm = S_cgs(rho, T)
                if not np.isfinite(Sm):
                    return np.array([1e30], dtype=float)
                return np.array([(Sm - s_t) / S_scale], dtype=float)

            T_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(T_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([logT_lo], [logT_hi]),
                        xtol=xtol,
                        ftol=xtol,
                        gtol=xtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        T_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(T_try)], dtype=float))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            T_sol = T_try
                except Exception:
                    pass

            if not np.isfinite(T_sol):
                def f(T):
                    Sm = S_cgs(rho, T)
                    if not np.isfinite(Sm):
                        return np.nan
                    return Sm - s_t

                left = right = T_guess
                f_left = f(right)

                if not np.isfinite(f_left):
                    T_guess = np.sqrt(Tmin * Tmax)
                    left = right = T_guess
                    f_left = f(right)

                if np.isfinite(f_left) and f_left == 0.0:
                    T_sol = T_guess
                else:
                    for _ in range(bracket_expand_steps):
                        left = max(Tmin, left / bracket_expand_factor)
                        right = min(Tmax, right * bracket_expand_factor)

                        f_l = f(left)
                        f_r = f(right)
                        if not (np.isfinite(f_l) and np.isfinite(f_r)):
                            continue

                        if f_l == 0.0:
                            T_sol = left
                            break
                        if f_r == 0.0:
                            T_sol = right
                            break
                        if f_l * f_r < 0:
                            try:
                                T_sol = brenth(f, left, right, xtol=xtol, maxiter=maxiter)
                            except Exception:
                                T_sol = np.nan
                            break

                    if not np.isfinite(T_sol):
                        fA = f(Tmin)
                        fB = f(Tmax)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                T_sol = brenth(f, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
                            except Exception:
                                T_sol = np.nan

            if np.isfinite(T_sol):
                T_out[idx] = T_sol
                T_prev = T_sol

        return float(T_out) if T_out.size == 1 else T_out

    # -------------------------
    # Inversion T(S,P)
    # -------------------------

    def get_T_sp_inv(self, _s, _P, bracket=(1.0, 200000.0), xtol=1e-8, maxiter=500, s_units="kbbar"):
        s_arr = np.atleast_1d(_s).astype(float)
        P_arr = np.atleast_1d(_P).astype(float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        if str(s_units).lower() == "kbbar":
            s_target_cgs = s_arr / self.erg_to_kbbar
        else:
            s_target_cgs = s_arr

        Tmin, Tmax = float(bracket[0]), float(bracket[1])
        Tmin = max(Tmin, self.T_min)
        Tmax = min(Tmax, self.T_max)
        if Tmin <= 0:
            raise ValueError("bracket[0] must be > 0 K.")
        if Tmax <= Tmin:
            raise ValueError("Temperature bracket does not overlap table range.")

        def _find_T(s_cgs, P_val):
            if P_val <= 0 or not np.isfinite(P_val) or not np.isfinite(s_cgs):
                return np.nan

            def err(T):
                return self._as_float(self.get_s_pt_inv(P_val, T)) - s_cgs

            try:
                f_lo = err(Tmin)
                f_hi = err(Tmax)
                if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
                    return np.nan
                if f_lo == 0.0:
                    return Tmin
                if f_hi == 0.0:
                    return Tmax
                if f_lo * f_hi > 0:
                    return np.nan
                return brenth(err, Tmin, Tmax, xtol=xtol, maxiter=maxiter)
            except (ValueError, FloatingPointError, RuntimeError):
                return np.nan

        T_roots = np.vectorize(_find_T)(s_target_cgs, P_arr)
        return float(T_roots) if T_roots.size == 1 else T_roots

    # -------------------------
    # 2-D inversion (S,P) -> (rho,T)
    # -------------------------

    def get_rhot_sp_2d_inv(
        self,
        s_target,
        P_target,
        *,
        s_units="kbbar",  # "cgs" or "kbbar"
        guess="auto",  # "auto" or (rho_guess, T_guess)
        T_guess0=None,
        bounds_rho: Optional[Tuple[float, float]] = None,
        bounds_T: Optional[Tuple[float, float]] = None,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
        fail_value=np.nan,
        return_diagnostics=False,
    ):
        s_arr = np.asarray(s_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)
        shape = s_arr.shape

        if str(s_units).lower() == "kbbar":
            s_cgs = s_arr / self.erg_to_kbbar
        else:
            s_cgs = s_arr

        rho_out = np.full(shape, fail_value, dtype=float)
        T_out = np.full(shape, fail_value, dtype=float)

        info = {
            "success": np.zeros(shape, dtype=bool),
            "cost": np.full(shape, np.nan),
            "nfev": np.full(shape, np.nan),
            "resid_P_frac": np.full(shape, np.nan),
            "resid_S_frac": np.full(shape, np.nan),
            "message": np.empty(shape, dtype=object),
        } if return_diagnostics else None

        if bounds_rho is None:
            rho_lo = self.rho_min
            rho_hi = self.rho_max
        else:
            rho_lo, rho_hi = map(float, bounds_rho)

        if bounds_T is None:
            T_lo = self.T_min
            T_hi = self.T_max
        else:
            T_lo, T_hi = map(float, bounds_T)

        if rho_lo <= 0 or T_lo <= 0:
            raise ValueError("Lower bounds for rho and T must be > 0.")

        lb = np.array([np.log(rho_lo), np.log(T_lo)], dtype=float)
        ub = np.array([np.log(rho_hi), np.log(T_hi)], dtype=float)

        if guess == "auto":
            T_seed = float(np.median(self.tvals_iso) if T_guess0 is None else T_guess0)
            T_seed = min(max(T_seed, T_lo), T_hi)
            rho_seed = None
        else:
            rho_seed, T_seed = map(float, guess)
            rho_seed = min(max(rho_seed, rho_lo), rho_hi)
            T_seed = min(max(T_seed, T_lo), T_hi)

        rho_guess_cur = rho_seed
        T_guess_cur = T_seed

        for idx in np.ndindex(shape):
            Pt = float(P_arr[idx])
            St = float(s_cgs[idx])

            if not (np.isfinite(Pt) and np.isfinite(St)) or Pt <= 0:
                if return_diagnostics:
                    info["message"][idx] = "Invalid target (non-finite or P<=0)."
                continue

            if guess == "auto" and rho_guess_cur is None:
                rho_guess_cur = float(
                    np.asarray(self.get_rho_pt_inv(Pt, T_guess_cur, on_fail="nan"))
                )
                if not np.isfinite(rho_guess_cur):
                    rho_guess_cur = self._rho_seed_from_isotherms(Pt, T_guess_cur)

            rho_guess_cur = min(max(float(rho_guess_cur), rho_lo), rho_hi)
            T_guess_cur = min(max(float(T_guess_cur), T_lo), T_hi)

            x0 = np.array([np.log(rho_guess_cur), np.log(T_guess_cur)], dtype=float)

            P_scale = max(abs(Pt), 1.0e9)
            S_scale = max(abs(St), 1.0)

            def residuals(x):
                rho = float(np.exp(x[0]))
                T = float(np.exp(x[1]))
                Pm = float(np.asarray(self.get_p_rhot(rho, T)))
                Sm = float(np.asarray(self.get_s_rhot(rho, T)))
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
                    T_sol = float(np.exp(sol.x[1]))
                    rho_out[idx] = rho_sol
                    T_out[idx] = T_sol

                    rho_guess_cur = rho_sol
                    T_guess_cur = T_sol

                    if return_diagnostics:
                        info["success"][idx] = True
                        info["cost"][idx] = sol.cost
                        info["nfev"][idx] = sol.nfev
                        r = residuals(sol.x)
                        info["resid_P_frac"][idx] = abs(r[0])
                        info["resid_S_frac"][idx] = abs(r[1])
                        info["message"][idx] = sol.message
                else:
                    if return_diagnostics:
                        info["message"][idx] = getattr(sol, "message", "least_squares failed")
            except Exception as e:
                if return_diagnostics:
                    info["message"][idx] = f"Exception: {e}"

        if return_diagnostics:
            return rho_out, T_out, info
        return rho_out, T_out

    # -------------------------
    # PT interface
    # -------------------------

    def get_rho_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self.get_rho_pt_inv(P_arr, T_arr, rho0=rho0, **inv_kwargs)
        return float(vals) if scalar else vals

    def get_s_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        rho = self.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs)
        vals = self.get_s_rhot(rho, T_arr)
        return float(vals) if scalar else vals

    def get_u_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        rho = self.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs)
        vals = self.get_u_rhot(rho, T_arr)
        return float(vals) if scalar else vals

    def get_g_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        rho = self.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs)
        vals = self.get_g_rhot(rho, T_arr)
        return float(vals) if scalar else vals

    def get_CP_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        out = np.full(P_arr.shape, np.nan, dtype=float)

        if rho0 is None:
            rho0_arr = None
        else:
            rho0_arr = np.broadcast_to(np.asarray(rho0, dtype=float), P_arr.shape)

        for idx in np.ndindex(P_arr.shape):
            rho_seed = float(rho0_arr[idx]) if rho0_arr is not None and np.isfinite(rho0_arr[idx]) else None
            out[idx] = self._cp_pt_si_scalar(float(P_arr[idx]), float(T_arr[idx]), rho0=rho_seed)

        out *= S_CONV_CGS
        return float(out) if scalar else out

    def get_CV_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        rho = self.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs)
        vals = self.get_CV_rhot(rho, T_arr)
        return float(vals) if scalar else vals

    def get_alpha_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        out = np.full(P_arr.shape, np.nan, dtype=float)

        if rho0 is None:
            rho0_arr = None
        else:
            rho0_arr = np.broadcast_to(np.asarray(rho0, dtype=float), P_arr.shape)

        for idx in np.ndindex(P_arr.shape):
            rho_seed = float(rho0_arr[idx]) if rho0_arr is not None and np.isfinite(rho0_arr[idx]) else None
            out[idx] = self._alpha_pt_scalar(float(P_arr[idx]), float(T_arr[idx]), rho0=rho_seed)

        return float(out) if scalar else out

    # lowercase aliases
    def get_cp_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        return self.get_CP_pt(P, T, tab=tab, rho0=rho0, **inv_kwargs)

    def get_cv_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        return self.get_CV_pt(P, T, tab=tab, rho0=rho0, **inv_kwargs)

    # -------------------------
    # SP interface
    # -------------------------

    def get_rho_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, _ = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_T_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(T) if scalar else T

    def get_u_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_u_rhot(rho, T)
        return float(vals) if scalar else vals

    def get_g_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_g_rhot(rho, T)
        return float(vals) if scalar else vals

    def get_CP_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CP_rhot(rho, T)
        return float(vals) if scalar else vals

    def get_CV_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_CV_rhot(rho, T)
        return float(vals) if scalar else vals

    def get_alpha_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        vals = self.get_alpha_rhot(rho, T)
        return float(vals) if scalar else vals

    # -------------------------
    # SRho interface
    # -------------------------

    def get_T_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        return float(T) if scalar else T

    def get_p_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_p_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    def get_u_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_u_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    def get_g_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_g_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    def get_CP_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_CP_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    def get_CV_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_CV_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    def get_alpha_srho(self, S, rho, tab=True, **inv_kwargs):
        scalar, S_arr, rho_arr = self._broadcast(S, rho)
        T = self.get_T_srho_inv(S_arr, rho_arr, **inv_kwargs)
        vals = self.get_alpha_rhot(rho_arr, T)
        return float(vals) if scalar else vals

    # lowercase aliases for compatibility with some callers
    def get_cp_srho(self, S, rho, tab=True, **inv_kwargs):
        return self.get_CP_srho(S, rho, tab=tab, **inv_kwargs)

    def get_cv_srho(self, S, rho, tab=True, **inv_kwargs):
        return self.get_CV_srho(S, rho, tab=tab, **inv_kwargs)

    # Optional convenience; same form used in dorogo_iron_eos.py docs.
    def get_T_melt(self, P):
        P_arr = np.array(P, ndmin=1, dtype=float)
        P_GPa = P_arr * 1e-9
        Tm = 6469.0 * np.power(1.0 + (P_GPa - 300.0) / 434.822, 1.839)
        return Tm.reshape(P_arr.shape)


# Alias for explicit naming
Fe_EOS_Gonzalez = Fe_EOS
