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
    Build a rectangular (P, T) grid from irregular EOS samples.

    Two-step construction (both linear in log10-space with extrapolation):
      1) For each unique table T, interpolate property vs log10(P) onto regular log10(P) axis.
      2) For each regular log10(P), interpolate property vs log10(T) onto regular log10(T) axis.
    """

    def __init__(
        self,
        P_raw: np.ndarray,
        T_raw: np.ndarray,
        fields: Dict[str, np.ndarray],
        *,
        n_P: Optional[int] = None,
        n_T: Optional[int] = None,
        P_bounds: Optional[Tuple[float, float]] = None,
        T_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.P_raw = np.asarray(P_raw, dtype=float)
        self.T_raw = np.asarray(T_raw, dtype=float)
        self.fields = {k: np.asarray(v, dtype=float) for k, v in fields.items()}

        if self.P_raw.size == 0 or self.T_raw.size == 0:
            raise ValueError("Empty raw P/T arrays.")
        if np.any(self.P_raw <= 0) or np.any(self.T_raw <= 0):
            raise ValueError("Raw P and T values must be strictly positive for log10 interpolation.")

        if not all(v.size == self.P_raw.size for v in self.fields.values()):
            raise ValueError("All field arrays must have same length as P_raw/T_raw.")

        P_lo_raw = float(np.min(self.P_raw))
        P_hi_raw = float(np.max(self.P_raw))
        T_lo_raw = float(np.min(self.T_raw))
        T_hi_raw = float(np.max(self.T_raw))

        if P_bounds is None:
            P_lo, P_hi = P_lo_raw, P_hi_raw
        else:
            P_lo, P_hi = map(float, P_bounds)

        if T_bounds is None:
            T_lo, T_hi = T_lo_raw, T_hi_raw
        else:
            T_lo, T_hi = map(float, T_bounds)

        if not (P_hi > P_lo > 0):
            raise ValueError("P bounds must satisfy 0 < P_lo < P_hi.")
        if not (T_hi > T_lo > 0):
            raise ValueError("T bounds must satisfy 0 < T_lo < T_hi.")

        P_unique = np.unique(self.P_raw)
        T_unique = np.unique(self.T_raw)

        if n_P is None:
            n_P = int(P_unique.size)
        if n_T is None:
            # keep regular temperature grid meaningfully sampled for derivatives
            n_T = int(max(T_unique.size, 300))

        self.n_P = int(n_P)
        self.n_T = int(n_T)
        if self.n_P < 2 or self.n_T < 2:
            raise ValueError("n_P and n_T must be >= 2.")

        self.logP_axis = np.linspace(np.log10(P_lo), np.log10(P_hi), self.n_P)
        self.logT_axis = np.linspace(np.log10(T_lo), np.log10(T_hi), self.n_T)
        self.P_axis = np.power(10.0, self.logP_axis)
        self.T_axis = np.power(10.0, self.logT_axis)
        self.T_unique = np.array(sorted(np.unique(self.T_raw)), dtype=float)
        self.logT_unique = np.log10(self.T_unique)

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
        # Step 1: for each table T, interpolate/extrapolate along log10(P) onto regular log10(P) axis
        vals_P_on_Traw = np.empty((self.T_unique.size, self.n_P), dtype=float)

        for i, T0 in enumerate(self.T_unique):
            mask = self.T_raw == T0
            P_slice = self.P_raw[mask]
            v_slice = vals_raw[mask]
            vals_P_on_Traw[i, :] = self._interp1d_linear_extrap(
                np.log10(P_slice),
                v_slice,
                self.logP_axis,
            )

        # Step 2: for each regular log10(P), interpolate/extrapolate along log10(T) onto regular log10(T) axis
        grid_pt = np.empty((self.n_P, self.n_T), dtype=float)
        for j in range(self.n_P):
            v_Tslice = vals_P_on_Traw[:, j]
            grid_pt[j, :] = self._interp1d_linear_extrap(self.logT_unique, v_Tslice, self.logT_axis)

        return grid_pt

    def build(self) -> Dict[str, np.ndarray]:
        out = {
            "logP_axis": self.logP_axis.copy(),
            "logT_axis": self.logT_axis.copy(),
            "P_axis": self.P_axis.copy(),
            "T_axis": self.T_axis.copy(),
        }

        for key, vals_raw in self.fields.items():
            out[key] = self._build_one_field(vals_raw)

        return out


class GonzalezRegularGridSurface:
    """
    RegularGridInterpolator wrapper for a single thermodynamic surface.
    Uses linear interpolation and linear extrapolation (fill_value=None)
    in log10(P), log10(T) coordinates.
    """

    def __init__(self, logP_axis: np.ndarray, logT_axis: np.ndarray, grid_pt: np.ndarray) -> None:
        self._rgi = RegularGridInterpolator(
            (np.asarray(logP_axis, dtype=float), np.asarray(logT_axis, dtype=float)),
            np.asarray(grid_pt, dtype=float),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        P_arr = np.asarray(P, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
        out = np.full(P_arr.shape, np.nan, dtype=float)

        good = np.isfinite(P_arr) & np.isfinite(T_arr) & (P_arr > 0.0) & (T_arr > 0.0)
        if np.any(good):
            pts = np.column_stack((np.log10(P_arr[good]), np.log10(T_arr[good])))
            out[good] = np.asarray(self._rgi(pts), dtype=float)
        return out


class Fe_EOS:
    """
    Gonzalez-Cataldo & Militzer iron EOS reader/interpolator.

    Loads either solid or liquid Fe text tables and builds native PT interpolators for:
      rho(P, T), S(P, T), U(P, T), G(P, T)

    RHOT quantities (e.g., P(rho,T), S(rho,T)) are obtained through inversion/wrappers.

    Reference-state handling:
      S and U outputs are shifted by constant phase-dependent offsets to avoid negative values.
      G is shifted thermodynamically consistently as:
        G_shifted = G_raw + dU - T*dS + dG0
      where dG0 is an additional constant reference shift (independent of P,T).

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
    _RECT_P_MIN_DEFAULT_PA = 1.0e9  # Build rectangular P-grid down to 1 GPa by default.
    _GLOBAL_BOUNDS_CACHE: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    def __init__(
        self,
        phase: str = "liquid",
        data_dir: Optional[Union[str, Path]] = None,
        diff_rel_T: float = 1e-1,
        diff_abs_T: float = 50.0,
        diff_rel_P: float = 1e-1,
        diff_abs_P: float = 1e8,
        grid_n_P: Optional[int] = None,
        grid_n_T: Optional[int] = None,
        grid_P_bounds: Optional[Tuple[float, float]] = None,
        grid_T_bounds: Optional[Tuple[float, float]] = None,
        apply_reference_offsets: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        phase : {"solid", "liquid"}
            Which Gonzalez table to load.
        grid_n_P, grid_n_T : int, optional
            Number of regular P/T samples for the rectangularized EOS grid.
        grid_P_bounds, grid_T_bounds : tuple, optional
            Explicit (min, max) bounds for the regularized grid axes.
        apply_reference_offsets : bool, optional
            If True, apply constant reference shifts to S/U (and consistent T-dependent
            shift to G). If False, return raw table-interpolated values.
        """
        self.phase = str(phase).lower()
        if self.phase not in self._PHASE_FILES:
            raise ValueError("phase must be 'solid' or 'liquid'")

        self.diff_rel_T = float(diff_rel_T)
        self.diff_abs_T = float(diff_abs_T)
        self.diff_rel_P = float(diff_rel_P)
        self.diff_abs_P = float(diff_abs_P)
        self.apply_reference_offsets = bool(apply_reference_offsets)

        self._data_dir = self._resolve_data_dir(data_dir)
        self._file_path = self._data_dir / self._PHASE_FILES[self.phase]

        if grid_P_bounds is None or grid_T_bounds is None:
            P_bounds_all, T_bounds_all = self._global_phase_bounds(self._data_dir)
            if grid_P_bounds is None:
                grid_P_bounds = (self._RECT_P_MIN_DEFAULT_PA, float(P_bounds_all[1]))
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
            self.P_table,
            self.T_table,
            {
                "rho": self.rho_table,
                "S": self.S_table_si,
                "U": self.U_table_si,
                "G": self.G_table_si,
            },
            n_P=grid_n_P,
            n_T=grid_n_T,
            P_bounds=grid_P_bounds,
            T_bounds=grid_T_bounds,
        )
        rect = rect_builder.build()

        self.logP_vals_rect = np.asarray(rect["logP_axis"], dtype=float)
        self.logT_vals_rect = np.asarray(rect["logT_axis"], dtype=float)
        self.P_vals_rect = np.asarray(rect["P_axis"], dtype=float)
        self.T_vals_rect = np.asarray(rect["T_axis"], dtype=float)
        self.rho_grid_rect = np.asarray(rect["rho"], dtype=float)
        self.S_grid_rect_si = np.asarray(rect["S"], dtype=float)
        self.U_grid_rect_si = np.asarray(rect["U"], dtype=float)
        self.G_grid_rect_si = np.asarray(rect["G"], dtype=float)

        # Phase-wise reference offsets (SI units) chosen to avoid negative S/U/G
        # on the rectangular PT grid while preserving thermodynamic derivatives.
        if self.apply_reference_offsets:
            self.S_offset_si = max(0.0, -float(np.nanmin(self.S_grid_rect_si)))
            self.U_offset_si = max(0.0, -float(np.nanmin(self.U_grid_rect_si)))
            G_shift_consistent = self.G_grid_rect_si + self.U_offset_si - self.T_vals_rect[None, :] * self.S_offset_si
            self.G_offset_si = max(0.0, -float(np.nanmin(G_shift_consistent)))
        else:
            self.S_offset_si = 0.0
            self.U_offset_si = 0.0
            self.G_offset_si = 0.0
        self.S_offset_cgs = self.S_offset_si * S_CONV_CGS
        self.U_offset_cgs = self.U_offset_si * U_CONV_CGS
        self.G_offset_cgs = self.G_offset_si * U_CONV_CGS

        # Convenience shifted table/grid views in SI units.
        self.S_table_si_shifted = self.S_table_si + self.S_offset_si
        self.U_table_si_shifted = self.U_table_si + self.U_offset_si
        self.G_table_si_shifted = self.G_table_si + self.U_offset_si - self.T_table * self.S_offset_si + self.G_offset_si
        self.S_grid_rect_si_shifted = self.S_grid_rect_si + self.S_offset_si
        self.U_grid_rect_si_shifted = self.U_grid_rect_si + self.U_offset_si
        self.G_grid_rect_si_shifted = (
            self.G_grid_rect_si + self.U_offset_si - self.T_vals_rect[None, :] * self.S_offset_si + self.G_offset_si
        )

        self.P_min = float(self.P_vals_rect[0])
        self.P_max = float(self.P_vals_rect[-1])
        self.T_min = float(self.T_vals_rect[0])
        self.T_max = float(self.T_vals_rect[-1])
        self.rho_min = float(np.nanmin(self.rho_grid_rect))
        self.rho_max = float(np.nanmax(self.rho_grid_rect))

        self._surf: Dict[str, GonzalezRegularGridSurface] = {
            "rho": GonzalezRegularGridSurface(self.logP_vals_rect, self.logT_vals_rect, self.rho_grid_rect),
            "S": GonzalezRegularGridSurface(self.logP_vals_rect, self.logT_vals_rect, self.S_grid_rect_si),
            "U": GonzalezRegularGridSurface(self.logP_vals_rect, self.logT_vals_rect, self.U_grid_rect_si),
            "G": GonzalezRegularGridSurface(self.logP_vals_rect, self.logT_vals_rect, self.G_grid_rect_si),
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

        P_min = np.inf
        P_max = -np.inf
        T_min = np.inf
        T_max = -np.inf

        for _, fname in cls._PHASE_FILES.items():
            path = Path(data_dir) / fname
            if not path.is_file():
                continue
            table = cls._read_table(path)
            P_min = min(P_min, float(np.min(table["P_Pa"])))
            P_max = max(P_max, float(np.max(table["P_Pa"])))
            T_min = min(T_min, float(np.min(table["T_K"])))
            T_max = max(T_max, float(np.max(table["T_K"])))

        if not np.isfinite(P_min) or not np.isfinite(T_min):
            raise FileNotFoundError("Could not determine global bounds for Gonzalez iron tables.")

        out = ((P_min, P_max), (T_min, T_max))
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

        # Drop exact duplicate (P, T) rows (liquid table has a few exact duplicates).
        pt = np.column_stack((P_Pa, T_K))
        _, keep_idx = np.unique(pt, axis=0, return_index=True)
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
        self._seed_by_T: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

        for T in self.tvals_iso:
            mask = self.T_table == T
            P = self.P_table[mask]
            rho = self.rho_table[mask]

            order = np.argsort(P)
            P = P[order]
            rho = rho[order]

            P_u, idxP = np.unique(P, return_index=True)
            rho_uP = rho[idxP]

            order_rho = np.argsort(rho)
            rho_sorted = rho[order_rho]
            P_sorted = P[order_rho]
            rho_u, idxR = np.unique(rho_sorted, return_index=True)
            P_uR = P_sorted[idxR]

            self._seed_by_T[float(T)] = {
                "rho_of_P": (P_u, rho_uP),
                "P_of_rho": (rho_u, P_uR),
            }

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

    def _interp_property_si(self, key: str, P_Pa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        P = np.asarray(P_Pa, dtype=float)
        T = np.asarray(T_K, dtype=float)

        if np.any(P <= 0):
            raise ValueError("Pressure P must be > 0 Pa.")
        if np.any(T <= 0):
            raise ValueError("Temperature T must be > 0 K.")

        vals = self._surf[key](P, T)
        if key == "S":
            vals = vals + self.S_offset_si
        elif key == "U":
            vals = vals + self.U_offset_si
        elif key == "G":
            vals = vals + self.U_offset_si - T * self.S_offset_si + self.G_offset_si
        vals = np.asarray(vals, dtype=float)
        return vals

    def _interp_property_si_scalar(self, key: str, P_Pa: float, T_K: float) -> float:
        if (not np.isfinite(P_Pa)) or (P_Pa <= 0):
            return np.nan
        if (not np.isfinite(T_K)) or (T_K <= 0):
            return np.nan
        val = float(self._surf[key](float(P_Pa), float(T_K)))
        if key == "S":
            val += self.S_offset_si
        elif key == "U":
            val += self.U_offset_si
        elif key == "G":
            val += self.U_offset_si - float(T_K) * self.S_offset_si + self.G_offset_si
        return val

    def _rho_seed_from_isotherms(self, P_Pa: float, T_K: float) -> float:
        Ts = self.tvals_iso
        if Ts.size == 0:
            return np.sqrt(self.rho_min * self.rho_max)

        def rho_at_T(Tref: float) -> float:
            P_grid, rho_grid = self._seed_by_T[float(Tref)]["rho_of_P"]
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

    def _p_seed_from_isotherms(self, rho_kgm3: float, T_K: float) -> float:
        Ts = self.tvals_iso
        if Ts.size == 0:
            return np.sqrt(self.P_min * self.P_max)

        def P_at_T(Tref: float) -> float:
            rho_grid, P_grid = self._seed_by_T[float(Tref)]["P_of_rho"]
            return float(np.interp(rho_kgm3, rho_grid, P_grid, left=P_grid[0], right=P_grid[-1]))

        if T_K <= Ts[0]:
            return P_at_T(Ts[0])
        if T_K >= Ts[-1]:
            return P_at_T(Ts[-1])

        i_hi = int(np.searchsorted(Ts, T_K))
        i_lo = i_hi - 1
        T_lo = float(Ts[i_lo])
        T_hi = float(Ts[i_hi])
        if T_hi == T_lo:
            return P_at_T(T_lo)

        P_lo = P_at_T(T_lo)
        P_hi = P_at_T(T_hi)
        w = (T_K - T_lo) / (T_hi - T_lo)
        return (1.0 - w) * P_lo + w * P_hi

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
    # Core API (native PT basis)
    # -------------------------

    def get_rho_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self._interp_property_si("rho", P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_s_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals_si = self._interp_property_si("S", P_arr, T_arr)
        vals = vals_si * S_CONV_CGS
        return float(vals) if scalar else vals

    def get_u_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals_si = self._interp_property_si("U", P_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    def get_g_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals_si = self._interp_property_si("G", P_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    # -------------------------
    # RHOT wrappers via PT inversion
    # -------------------------

    def get_p_rhot(
        self,
        rho_kgm3: ArrayLike,
        T_K: ArrayLike,
        P_bracket_Pa: Optional[Tuple[float, float]] = None,
        max_iter: int = 200,
        rtol: float = 1e-10,
        P0: Optional[ArrayLike] = None,
        use_lsq_first: bool = True,
        lsq_max_nfev: int = 60,
        bracket_expand_steps: int = 30,
        bracket_expand_factor: float = 1.6,
        on_fail: str = "nan",  # "nan" or "raise"
    ) -> np.ndarray:
        rho_arr = np.asarray(rho_kgm3, dtype=float)
        T_arr = np.asarray(T_K, dtype=float)
        shape = np.broadcast(rho_arr, T_arr).shape
        rho_b = np.broadcast_to(rho_arr, shape)
        T_b = np.broadcast_to(T_arr, shape)

        out = np.full(shape, np.nan, dtype=float)

        if P_bracket_Pa is None:
            P_min = self.P_min
            P_max = self.P_max
        else:
            P_min = float(P_bracket_Pa[0])
            P_max = float(P_bracket_Pa[1])
        if P_min <= 0 or P_max <= P_min:
            raise ValueError("P_bracket_Pa must satisfy 0 < P_min < P_max")

        logP_lo = np.log(P_min)
        logP_hi = np.log(P_max)

        P0_b = None
        if P0 is not None:
            P0_arr = np.asarray(P0, dtype=float)
            P0_b = np.broadcast_to(P0_arr, shape)

        P_prev = np.nan

        def rho_of_P_scalar(P_val: float, Ti: float) -> float:
            if P_val <= 0 or Ti <= 0:
                return np.nan
            return self._interp_property_si_scalar("rho", P_val, Ti)

        for idx in np.ndindex(shape):
            rho_t = float(rho_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(rho_t)) or (not np.isfinite(Ti)) or rho_t <= 0 or Ti <= 0:
                continue

            if np.isfinite(P_prev):
                P_guess = float(P_prev)
            elif P0_b is not None and np.isfinite(P0_b[idx]) and P0_b[idx] > 0:
                P_guess = float(P0_b[idx])
            else:
                P_guess = self._p_seed_from_isotherms(rho_t, Ti)
            P_guess = min(max(P_guess, P_min), P_max)

            rho_scale = max(abs(rho_t), 1.0)

            def resid(logP_vec):
                P_try = float(np.exp(logP_vec[0]))
                rho_m = rho_of_P_scalar(P_try, Ti)
                if not np.isfinite(rho_m):
                    return np.array([1e30], dtype=float)
                return np.array([(rho_m - rho_t) / rho_scale], dtype=float)

            P_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(P_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([logP_lo], [logP_hi]),
                        xtol=rtol,
                        ftol=rtol,
                        gtol=rtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        P_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(P_try)], dtype=float))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            P_sol = P_try
                except Exception:
                    pass

            if not np.isfinite(P_sol):
                def f(P_val):
                    return rho_of_P_scalar(P_val, Ti) - rho_t

                left = right = P_guess
                f_left = f(right)

                if not np.isfinite(f_left):
                    P_guess = np.sqrt(P_min * P_max)
                    left = right = P_guess
                    f_left = f(right)

                if np.isfinite(f_left) and f_left == 0.0:
                    P_sol = P_guess
                else:
                    for _ in range(bracket_expand_steps):
                        left = max(P_min, left / bracket_expand_factor)
                        right = min(P_max, right * bracket_expand_factor)

                        f_l = f(left)
                        f_r = f(right)
                        if not (np.isfinite(f_l) and np.isfinite(f_r)):
                            continue
                        if f_l == 0.0:
                            P_sol = left
                            break
                        if f_r == 0.0:
                            P_sol = right
                            break
                        if f_l * f_r < 0:
                            try:
                                P_sol = brenth(f, left, right, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                P_sol = np.nan
                            break

                    if not np.isfinite(P_sol):
                        fA = f(P_min)
                        fB = f(P_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                P_sol = brenth(f, P_min, P_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                P_sol = np.nan

            if np.isfinite(P_sol):
                out[idx] = P_sol
                P_prev = P_sol
            elif on_fail == "raise":
                raise RuntimeError(
                    f"Failed P(rho,T) inversion: rho={rho_t:.3e} kg/m^3, T={Ti:.3f} K "
                    f"within [{P_min:.3e}, {P_max:.3e}] Pa"
                )

        return float(out) if out.size == 1 else out

    def get_s_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals_si = self._interp_property_si("S", P_arr, T_arr)
        vals = vals_si * S_CONV_CGS
        return float(vals) if scalar else vals

    def get_u_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals_si = self._interp_property_si("U", P_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    def get_g_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals_si = self._interp_property_si("G", P_arr, T_arr)
        vals = vals_si * U_CONV_CGS
        return float(vals) if scalar else vals

    # -------------------------
    # Derivative-based thermo
    # -------------------------

    def _cv_si_scalar(self, rho: float, T: float) -> float:
        def u_of_T(Ti: float) -> float:
            P_i = self.get_p_rhot(rho, Ti)
            P_i = float(np.asarray(P_i))
            if not np.isfinite(P_i):
                return np.nan
            return self._interp_property_si_scalar("U", P_i, Ti)

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
        return self._interp_property_si_scalar("S", P, T)

    def _g_pt_si_scalar(self, P: float, T: float, rho0: Optional[float] = None) -> float:
        if (not np.isfinite(P)) or (not np.isfinite(T)) or (P <= 0) or (T <= 0):
            return np.nan
        return self._interp_property_si_scalar("G", P, T)

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
    # PT inversion wrappers
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
        vals = self.get_rho_pt_native(P, T_K)
        arr = np.asarray(vals, dtype=float)
        if on_fail == "raise" and np.any(~np.isfinite(arr)):
            raise RuntimeError("Failed rho(P,T) interpolation.")
        return vals

    def get_s_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_s_pt_native(P, T)

    def get_u_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_u_pt_native(P, T)

    def get_g_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_g_pt_native(P, T)

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

        T_out = np.asarray(
            self.get_T_sp_inv(s_arr, P_arr, bracket=bounds_T or (self.T_min, self.T_max), s_units=s_units),
            dtype=float,
        )
        rho_out = np.asarray(self.get_rho_pt(P_arr, T_out), dtype=float)

        bad = ~np.isfinite(T_out) | ~np.isfinite(rho_out)
        if np.any(bad):
            T_out = T_out.copy()
            rho_out = rho_out.copy()
            T_out[bad] = fail_value
            rho_out[bad] = fail_value

        if return_diagnostics:
            info = {
                "success": np.isfinite(rho_out) & np.isfinite(T_out),
                "cost": np.full(rho_out.shape, np.nan),
                "nfev": np.full(rho_out.shape, np.nan),
                "resid_P_frac": np.full(rho_out.shape, np.nan),
                "resid_S_frac": np.full(rho_out.shape, np.nan),
                "message": np.empty(rho_out.shape, dtype=object),
            }
            info["message"][~info["success"]] = "Failed SP->T inversion."
            info["message"][info["success"]] = "Solved via T(S,P) then rho(P,T)."
            return rho_out, T_out, info
        return rho_out, T_out

    # -------------------------
    # PT interface
    # -------------------------

    def get_rho_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self.get_rho_pt_native(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_s_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self.get_s_pt_native(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_u_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self.get_u_pt_native(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_g_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        vals = self.get_g_pt_native(P_arr, T_arr)
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
        out = np.full(P_arr.shape, np.nan, dtype=float)
        rho_arr = self.get_rho_pt_native(P_arr, T_arr)
        for idx in np.ndindex(P_arr.shape):
            out[idx] = self._cv_si_scalar(float(rho_arr[idx]), float(T_arr[idx]))
        vals = out * S_CONV_CGS
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


class Fe_COMBINED_EOS(Fe_EOS):
    """
    Combined Gonzalez iron EOS with smooth solid/liquid blending around T_melt(P).

    The class composes two phase-specific Fe_EOS instances and blends thermodynamic
    properties with a tanh weight:
        w_liq = 0.5 * (1 + tanh((T - Tm(P))/dT))
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        dT: float = 50.0,
        diff_rel_T: float = 1e-1,
        diff_abs_T: float = 50.0,
        diff_rel_P: float = 1e-1,
        diff_abs_P: float = 1e8,
        grid_n_P: Optional[int] = None,
        grid_n_T: Optional[int] = None,
        grid_P_bounds: Optional[Tuple[float, float]] = None,
        grid_T_bounds: Optional[Tuple[float, float]] = None,
        apply_reference_offsets: bool = False,
    ) -> None:
        self.phase = "combined"
        self.dT = float(dT)
        if self.dT <= 0.0:
            raise ValueError("dT must be > 0 K.")

        self.diff_rel_T = float(diff_rel_T)
        self.diff_abs_T = float(diff_abs_T)
        self.diff_rel_P = float(diff_rel_P)
        self.diff_abs_P = float(diff_abs_P)
        self.apply_reference_offsets = bool(apply_reference_offsets)

        common_kwargs = dict(
            data_dir=data_dir,
            diff_rel_T=diff_rel_T,
            diff_abs_T=diff_abs_T,
            diff_rel_P=diff_rel_P,
            diff_abs_P=diff_abs_P,
            grid_n_P=grid_n_P,
            grid_n_T=grid_n_T,
            grid_P_bounds=grid_P_bounds,
            grid_T_bounds=grid_T_bounds,
            apply_reference_offsets=apply_reference_offsets,
        )
        self.solid = Fe_EOS(phase="solid", **common_kwargs)
        self.liquid = Fe_EOS(phase="liquid", **common_kwargs)

        self.P_min = min(self.solid.P_min, self.liquid.P_min)
        self.P_max = max(self.solid.P_max, self.liquid.P_max)
        self.T_min = min(self.solid.T_min, self.liquid.T_min)
        self.T_max = max(self.solid.T_max, self.liquid.T_max)
        self.rho_min = min(self.solid.rho_min, self.liquid.rho_min)
        self.rho_max = max(self.solid.rho_max, self.liquid.rho_max)

    @staticmethod
    def _blend_linear(solid_vals: np.ndarray, liquid_vals: np.ndarray, w_liq: np.ndarray) -> np.ndarray:
        return (1.0 - w_liq) * solid_vals + w_liq * liquid_vals

    def _blend_weight_PT(self, P_Pa: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        Tm = self.get_T_melt(P_Pa)
        arg = (np.asarray(T_K, dtype=float) - np.asarray(Tm, dtype=float)) / self.dT
        return np.clip(0.5 * (1.0 + np.tanh(arg)), 0.0, 1.0)

    def get_T_melt(self, P):
        """
        Gonzalez et al. (2023) melt curve:
            Tm(K) = 6469 * (1 + (P_GPa - 300) / 434.822) ** 0.54369
        """
        P_arr = np.asarray(P, dtype=float)
        P_GPa = P_arr * 1e-9
        arg = 1.0 + (P_GPa - 300.0) / 434.822
        with np.errstate(invalid="ignore"):
            Tm = 6469.0 * np.power(arg, 0.54369)
        return Tm

    # -------------------------
    # PT interface
    # -------------------------

    def get_rho_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        rho_s = np.asarray(self.solid.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        rho_l = np.asarray(self.liquid.get_rho_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        w = self._blend_weight_PT(P_arr, T_arr)
        denom = (1.0 - w) / rho_s + w / rho_l
        vals = np.where(np.isfinite(denom) & (denom != 0.0), 1.0 / denom, np.nan)
        return float(vals) if scalar else vals

    def get_s_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        s_s = np.asarray(self.solid.get_s_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        s_l = np.asarray(self.liquid.get_s_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(s_s, s_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_u_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        u_s = np.asarray(self.solid.get_u_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        u_l = np.asarray(self.liquid.get_u_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(u_s, u_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_g_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        g_s = np.asarray(self.solid.get_g_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        g_l = np.asarray(self.liquid.get_g_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(g_s, g_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_CP_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        cp_s = np.asarray(self.solid.get_CP_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        cp_l = np.asarray(self.liquid.get_CP_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(cp_s, cp_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_CV_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        cv_s = np.asarray(self.solid.get_CV_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        cv_l = np.asarray(self.liquid.get_CV_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(cv_s, cv_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_alpha_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        scalar, P_arr, T_arr = self._broadcast(P, T)
        a_s = np.asarray(self.solid.get_alpha_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        a_l = np.asarray(self.liquid.get_alpha_pt(P_arr, T_arr, tab=tab, rho0=rho0, **inv_kwargs), dtype=float)
        vals = self._blend_linear(a_s, a_l, self._blend_weight_PT(P_arr, T_arr))
        return float(vals) if scalar else vals

    def get_cp_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        return self.get_CP_pt(P, T, tab=tab, rho0=rho0, **inv_kwargs)

    def get_cv_pt(self, P, T, tab=True, rho0=None, **inv_kwargs):
        return self.get_CV_pt(P, T, tab=tab, rho0=rho0, **inv_kwargs)

    # Native aliases (same units/conventions as Fe_EOS PT-native methods).
    def get_rho_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        return self.get_rho_pt(P, T)

    def get_s_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        return self.get_s_pt(P, T)

    def get_u_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        return self.get_u_pt(P, T)

    def get_g_pt_native(self, P: ArrayLike, T: ArrayLike) -> np.ndarray:
        return self.get_g_pt(P, T)

    def get_rho_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_rho_pt(P, T, rho0=rho0, **inv_kwargs)

    def get_s_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_s_pt(P, T, rho0=rho0, **inv_kwargs)

    def get_u_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_u_pt(P, T, rho0=rho0, **inv_kwargs)

    def get_g_pt_inv(self, P: ArrayLike, T: ArrayLike, rho0: Optional[ArrayLike] = None, **inv_kwargs):
        return self.get_g_pt(P, T, rho0=rho0, **inv_kwargs)

    # -------------------------
    # RHOT interface
    # -------------------------

    def get_p_rhot(
        self,
        rho_kgm3: ArrayLike,
        T_K: ArrayLike,
        P_bracket_Pa: Optional[Tuple[float, float]] = None,
        max_iter: int = 200,
        rtol: float = 1e-10,
        P0: Optional[ArrayLike] = None,
        use_lsq_first: bool = True,
        lsq_max_nfev: int = 60,
        bracket_expand_steps: int = 30,
        bracket_expand_factor: float = 1.6,
        on_fail: str = "nan",
    ) -> np.ndarray:
        rho_arr = np.asarray(rho_kgm3, dtype=float)
        T_arr = np.asarray(T_K, dtype=float)
        shape = np.broadcast(rho_arr, T_arr).shape
        rho_b = np.broadcast_to(rho_arr, shape)
        T_b = np.broadcast_to(T_arr, shape)
        out = np.full(shape, np.nan, dtype=float)

        if P_bracket_Pa is None:
            P_min = self.P_min
            P_max = self.P_max
        else:
            P_min = float(P_bracket_Pa[0])
            P_max = float(P_bracket_Pa[1])
        if P_min <= 0 or P_max <= P_min:
            raise ValueError("P_bracket_Pa must satisfy 0 < P_min < P_max")

        logP_lo = np.log(P_min)
        logP_hi = np.log(P_max)

        P0_b = None
        if P0 is not None:
            P0_arr = np.asarray(P0, dtype=float)
            P0_b = np.broadcast_to(P0_arr, shape)

        P_prev = np.nan

        def rho_of_P_scalar(P_val: float, Ti: float) -> float:
            if P_val <= 0 or Ti <= 0:
                return np.nan
            return float(np.asarray(self.get_rho_pt(P_val, Ti)))

        for idx in np.ndindex(shape):
            rho_t = float(rho_b[idx])
            Ti = float(T_b[idx])

            if (not np.isfinite(rho_t)) or (not np.isfinite(Ti)) or rho_t <= 0 or Ti <= 0:
                continue

            if np.isfinite(P_prev):
                P_guess = float(P_prev)
            elif P0_b is not None and np.isfinite(P0_b[idx]) and P0_b[idx] > 0:
                P_guess = float(P0_b[idx])
            else:
                Ps = float(np.asarray(self.solid.get_p_rhot(rho_t, Ti, on_fail="nan")))
                Pl = float(np.asarray(self.liquid.get_p_rhot(rho_t, Ti, on_fail="nan")))
                if np.isfinite(Ps) and np.isfinite(Pl):
                    P_guess = 0.5 * (Ps + Pl)
                elif np.isfinite(Ps):
                    P_guess = Ps
                elif np.isfinite(Pl):
                    P_guess = Pl
                else:
                    P_guess = np.sqrt(P_min * P_max)

            P_guess = min(max(P_guess, P_min), P_max)
            rho_scale = max(abs(rho_t), 1.0)

            def resid(logP_vec):
                P_try = float(np.exp(logP_vec[0]))
                rho_m = rho_of_P_scalar(P_try, Ti)
                if not np.isfinite(rho_m):
                    return np.array([1e30], dtype=float)
                return np.array([(rho_m - rho_t) / rho_scale], dtype=float)

            P_sol = np.nan

            if use_lsq_first:
                x0 = np.array([np.log(P_guess)], dtype=float)
                try:
                    sol = least_squares(
                        resid,
                        x0,
                        bounds=([logP_lo], [logP_hi]),
                        xtol=rtol,
                        ftol=rtol,
                        gtol=rtol,
                        max_nfev=lsq_max_nfev,
                        method="trf",
                    )
                    if sol.success and np.isfinite(sol.x[0]):
                        P_try = float(np.exp(sol.x[0]))
                        r = resid(np.array([np.log(P_try)], dtype=float))[0]
                        if np.isfinite(r) and abs(r) < 1e-8:
                            P_sol = P_try
                except Exception:
                    pass

            if not np.isfinite(P_sol):
                def f(P_val):
                    return rho_of_P_scalar(P_val, Ti) - rho_t

                left = right = P_guess
                f_guess = f(P_guess)

                if np.isfinite(f_guess) and f_guess == 0.0:
                    P_sol = P_guess
                else:
                    for _ in range(bracket_expand_steps):
                        left = max(P_min, left / bracket_expand_factor)
                        right = min(P_max, right * bracket_expand_factor)
                        f_l = f(left)
                        f_r = f(right)
                        if not (np.isfinite(f_l) and np.isfinite(f_r)):
                            continue
                        if f_l == 0.0:
                            P_sol = left
                            break
                        if f_r == 0.0:
                            P_sol = right
                            break
                        if f_l * f_r < 0:
                            try:
                                P_sol = brenth(f, left, right, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                P_sol = np.nan
                            break

                    if not np.isfinite(P_sol):
                        fA = f(P_min)
                        fB = f(P_max)
                        if np.isfinite(fA) and np.isfinite(fB) and (fA == 0.0 or fB == 0.0 or fA * fB < 0):
                            try:
                                P_sol = brenth(f, P_min, P_max, xtol=rtol, maxiter=max_iter)
                            except Exception:
                                P_sol = np.nan

            if np.isfinite(P_sol):
                out[idx] = P_sol
                P_prev = P_sol
            elif on_fail == "raise":
                raise RuntimeError(
                    f"Failed P(rho,T) inversion: rho={rho_t:.3e} kg/m^3, T={Ti:.3f} K "
                    f"within [{P_min:.3e}, {P_max:.3e}] Pa"
                )

        return float(out) if out.size == 1 else out

    def get_s_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_s_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_u_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_u_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_g_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_g_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_CP_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_CP_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_CV_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_CV_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_alpha_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        scalar, rho_arr, T_arr = self._broadcast(rho_kgm3, T_K)
        P_arr = self.get_p_rhot(rho_arr, T_arr)
        vals = self.get_alpha_pt(P_arr, T_arr)
        return float(vals) if scalar else vals

    def get_cp_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        return self.get_CP_rhot(rho_kgm3, T_K)

    def get_cv_rhot(self, rho_kgm3: ArrayLike, T_K: ArrayLike) -> np.ndarray:
        return self.get_CV_rhot(rho_kgm3, T_K)

    # -------------------------
    # SP interface
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
                return self._as_float(self.get_s_pt(P_val, T)) - s_cgs

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

    def get_rhot_sp_2d_inv(
        self,
        s_target,
        P_target,
        *,
        s_units="kbbar",
        bounds_T: Optional[Tuple[float, float]] = None,
        fail_value=np.nan,
        return_diagnostics=False,
        **kwargs,
    ):
        s_arr = np.asarray(s_target, dtype=float)
        P_arr = np.asarray(P_target, dtype=float)
        s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

        T_out = np.asarray(
            self.get_T_sp_inv(s_arr, P_arr, bracket=bounds_T or (self.T_min, self.T_max), s_units=s_units),
            dtype=float,
        )
        rho_out = np.asarray(self.get_rho_pt(P_arr, T_out), dtype=float)

        bad = ~np.isfinite(T_out) | ~np.isfinite(rho_out)
        if np.any(bad):
            T_out = T_out.copy()
            rho_out = rho_out.copy()
            T_out[bad] = fail_value
            rho_out[bad] = fail_value

        if return_diagnostics:
            info = {
                "success": np.isfinite(rho_out) & np.isfinite(T_out),
                "cost": np.full(rho_out.shape, np.nan),
                "nfev": np.full(rho_out.shape, np.nan),
                "resid_P_frac": np.full(rho_out.shape, np.nan),
                "resid_S_frac": np.full(rho_out.shape, np.nan),
                "message": np.empty(rho_out.shape, dtype=object),
            }
            info["message"][~info["success"]] = "Failed SP->T inversion."
            info["message"][info["success"]] = "Solved via T(S,P) then rho(P,T)."
            return rho_out, T_out, info
        return rho_out, T_out

    def get_rho_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        rho, _ = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(rho) if scalar else rho

    def get_T_sp(self, S, P, tab=True, **inv_kwargs):
        scalar, S_arr, P_arr = self._broadcast(S, P)
        _, T = self.get_rhot_sp_2d_inv(S_arr, P_arr, **inv_kwargs)
        return float(T) if scalar else T

    def get_t_sp(self, S, P, tab=True, **inv_kwargs):
        return self.get_T_sp(S, P, tab=tab, **inv_kwargs)

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

    def get_cp_sp(self, S, P, tab=True, **inv_kwargs):
        return self.get_CP_sp(S, P, tab=tab, **inv_kwargs)

    def get_cv_sp(self, S, P, tab=True, **inv_kwargs):
        return self.get_CV_sp(S, P, tab=tab, **inv_kwargs)

    # -------------------------
    # SRho interface
    # -------------------------

    def get_T_srho_inv(self, _s, _rho, bracket=(1.0, 200000.0), xtol=1e-8, maxiter=500, s_units="kbbar"):
        s_arr = np.atleast_1d(_s).astype(float)
        rho_arr = np.atleast_1d(_rho).astype(float)
        s_arr, rho_arr = np.broadcast_arrays(s_arr, rho_arr)

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

        def _find_T(s_cgs, rho_val):
            if rho_val <= 0 or not np.isfinite(rho_val) or not np.isfinite(s_cgs):
                return np.nan

            def err(T):
                return self._as_float(self.get_s_rhot(rho_val, T)) - s_cgs

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

        T_roots = np.vectorize(_find_T)(s_target_cgs, rho_arr)
        return float(T_roots) if T_roots.size == 1 else T_roots

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

    def get_cp_srho(self, S, rho, tab=True, **inv_kwargs):
        return self.get_CP_srho(S, rho, tab=tab, **inv_kwargs)

    def get_cv_srho(self, S, rho, tab=True, **inv_kwargs):
        return self.get_CV_srho(S, rho, tab=tab, **inv_kwargs)


# Alias for explicit naming
Fe_EOS_Gonzalez = Fe_EOS
