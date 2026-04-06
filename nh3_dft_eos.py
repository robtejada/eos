import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI, griddata
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from eos import ideal_eos

"""
    DFT Equation of State for Ammonia (NH3).

    Source: Bethkenhagen et al. (2017) DFT-MD simulations.
    Native basis: (rho, T) — density in g/cm³, temperature in K.

    Provides:
      - P(rho, T)       pressure in dyn/cm²
      - U(rho, T)       internal energy in erg/g (globally offset to be positive)
      - S(rho, T)       entropy in erg/g/K (from thermodynamic integration)
      - c_v(rho, T)     specific heat at constant volume in erg/g/K
      - dPdT(rho, T)    (dP/dT)_rho in dyn/cm²/K

    Authors: Roberto Tejada Arevalo
"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# --- Unit conversions ---
_GPa_to_dyn = 1.0e10       # 1 GPa = 1e10 dyn/cm²
_kJg_to_erg = 1.0e10       # 1 kJ/g = 1e10 erg/g

# Ideal EOS for inversion guesses (NH3: mu = 17.031)
_ideal = ideal_eos.IdealEOS(m=17.031)

# =================================================================
# Load raw DFT data
# =================================================================
_raw = np.loadtxt(f'{CURR_DIR}/methane_ammonia/DFT_EOS_ammonia.dat')
_rho_raw = _raw[:, 0]   # g/cm³
_T_raw   = _raw[:, 1]   # K
_P_raw   = _raw[:, 2]   # GPa
_U_raw   = _raw[:, 3]   # kJ/g

# Build sorted unique axes
rho_arr = np.sort(np.unique(_rho_raw))   # shape (n_rho,)
T_arr   = np.sort(np.unique(_T_raw))     # shape (n_T,)
n_rho, n_T = len(rho_arr), len(T_arr)

# Reshape into 2D grids: (n_rho, n_T)
_rho_idx = {v: i for i, v in enumerate(rho_arr)}
_T_idx   = {v: i for i, v in enumerate(T_arr)}

P_grid_GPa = np.full((n_rho, n_T), np.nan)
U_grid_kJg = np.full((n_rho, n_T), np.nan)
for row in _raw:
    ri, ti = _rho_idx[row[0]], _T_idx[row[1]]
    P_grid_GPa[ri, ti] = row[2]
    U_grid_kJg[ri, ti] = row[3]

# Convert to CGS
P_grid = P_grid_GPa * _GPa_to_dyn       # dyn/cm²
U_grid_raw = U_grid_kJg * _kJg_to_erg   # erg/g

# Global offset so U is strictly positive
U_OFFSET = -np.nanmin(U_grid_raw) + 1.0e8   # small margin in erg/g
U_grid = U_grid_raw + U_OFFSET

# =================================================================
# Fill NaN cells via interpolation (incomplete DFT grid)
# =================================================================
_rho_mesh, _T_mesh = np.meshgrid(rho_arr, T_arr, indexing='ij')
_valid = ~np.isnan(P_grid)
_pts_good = np.column_stack((_rho_mesh[_valid], _T_mesh[_valid]))
_pts_all  = np.column_stack((_rho_mesh.ravel(), _T_mesh.ravel()))

for _grid in [P_grid, U_grid]:
    _nan_mask = np.isnan(_grid)
    if _nan_mask.any():
        _vals_good = _grid[_valid]
        _filled = griddata(_pts_good, _vals_good, _pts_all, method='cubic')
        _filled_2d = _filled.reshape(n_rho, n_T)
        _still_nan = np.isnan(_filled_2d)
        if _still_nan.any():
            _nearest = griddata(_pts_good, _vals_good, _pts_all, method='nearest')
            _filled_2d[_still_nan] = _nearest.reshape(n_rho, n_T)[_still_nan]
        _grid[_nan_mask] = _filled_2d[_nan_mask]

# =================================================================
# Derived quantities via finite differences on the grid
# =================================================================

def _compute_cv_grid():
    """c_v = (dU/dT)_rho on the full grid, in erg/g/K."""
    return np.gradient(U_grid, T_arr, axis=1)  # shape (n_rho, n_T)

def _compute_dPdT_grid():
    """(dP/dT)_rho on the full grid, in dyn/cm²/K."""
    return np.gradient(P_grid, T_arr, axis=1)  # shape (n_rho, n_T)

cv_grid = _compute_cv_grid()
dPdT_grid = _compute_dPdT_grid()

# =================================================================
# Entropy via thermodynamic integration
# =================================================================
#
# Path: (rho_0, T_0) -> (rho_0, T) -> (rho, T)
#
# s(rho, T) = s_ref
#   + integral_{T_0}^{T} c_v(rho_0, T') / T' dT'          [leg 1: along T at rho_0]
#   - integral_{rho_0}^{rho} (1/rho'^2)(dP/dT)|_{rho',T} drho'  [leg 2: along rho at T]
#
# Reference point: (rho_0, T_0) = first grid point.
# S_REF is set to match Gao et al. (2023) at (rho_arr[0], T_arr[0]),
# plus a constant offset (~1.00e8 erg/g/K) that keeps all S values positive
# and matches the convention used by the ORCHARD production tables.
from eos import nh3_gao_eos as _gao
_S_OFFSET = 1.00e8   # erg/g/K — ensures all S > 0 in the planetary regime
S_REF = _gao.get_s_rhot(rho_arr[0], T_arr[0]) + _S_OFFSET

def _compute_entropy_grid():
    """Compute entropy on the full (rho, T) grid."""
    S = np.zeros((n_rho, n_T))

    # Leg 1: integrate along T at rho_0 (index 0)
    integrand_T = cv_grid[0, :] / T_arr   # c_v(rho_0, T') / T'
    S_along_T = np.zeros(n_T)
    S_along_T[1:] = cumulative_trapezoid(integrand_T, T_arr)
    S[0, :] = S_REF + S_along_T

    # Leg 2: for each T_j, integrate along rho from rho_0 to rho
    for j in range(n_T):
        integrand_rho = -dPdT_grid[:, j] / (rho_arr**2)
        S_along_rho = np.zeros(n_rho)
        S_along_rho[1:] = cumulative_trapezoid(integrand_rho, rho_arr)
        S[:, j] = S[0, j] + S_along_rho

    return S

S_grid = _compute_entropy_grid()

# =================================================================
# RGI interpolators for fast lookup
# =================================================================
_rgi_kw = dict(method='linear', bounds_error=False, fill_value=None)

P_rgi   = RGI((rho_arr, T_arr), P_grid,    **_rgi_kw)
U_rgi   = RGI((rho_arr, T_arr), U_grid,    **_rgi_kw)
S_rgi   = RGI((rho_arr, T_arr), S_grid,    **_rgi_kw)
cv_rgi  = RGI((rho_arr, T_arr), cv_grid,   **_rgi_kw)
dPdT_rgi = RGI((rho_arr, T_arr), dPdT_grid, **_rgi_kw)

# =================================================================
# Public API — all inputs in linear (rho [g/cm³], T [K])
# =================================================================

def _query(rgi, rho, T):
    """Evaluate an RGI, handling scalar and array inputs."""
    args = (rho, T)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rgi(pts)
    if all(np.isscalar(a) for a in args):
        return result.item()
    return result

def get_p_rhot(rho, T):
    """Pressure in dyn/cm²."""
    return _query(P_rgi, rho, T)

def get_u_rhot(rho, T):
    """Internal energy in erg/g (with global offset applied)."""
    return _query(U_rgi, rho, T)

def get_s_rhot(rho, T):
    """Entropy in erg/g/K."""
    return _query(S_rgi, rho, T)

def get_cv_rhot(rho, T):
    """Specific heat at constant volume in erg/g/K."""
    return _query(cv_rgi, rho, T)

def get_dPdT_rhot(rho, T):
    """(dP/dT)_rho in dyn/cm²/K."""
    return _query(dPdT_rgi, rho, T)

# =================================================================
# Save derived tables to .npz
# =================================================================

def save_rhot_table(outpath=None):
    """Compute and save all rho-T basis quantities to .npz.

    Saved arrays:
      rho_arr, T_arr          — 1-D grid axes (g/cm³, K)
      P_grid                  — pressure (dyn/cm²), shape (n_rho, n_T)
      U_grid                  — internal energy (erg/g, offset), shape (n_rho, n_T)
      U_OFFSET                — scalar offset applied to U (erg/g)
      S_grid                  — entropy (erg/g/K), shape (n_rho, n_T)
      cv_grid                 — c_v (erg/g/K), shape (n_rho, n_T)
      dPdT_grid               — (dP/dT)_rho (dyn/cm²/K), shape (n_rho, n_T)
    """
    if outpath is None:
        outpath = f'{CURR_DIR}/methane_ammonia/ammonia_dft_rhot.npz'
    np.savez(outpath,
             rho_arr=rho_arr, T_arr=T_arr,
             P_grid=P_grid, U_grid=U_grid, U_OFFSET=np.array(U_OFFSET),
             S_grid=S_grid, cv_grid=cv_grid, dPdT_grid=dPdT_grid)
    print(f'Saved rho-T table to {outpath}')


# =================================================================
# 1-D Newton + brentq inversion
# =================================================================

def _newton_brentq(err_func, guess, lo, hi, max_iter=30, tol=1e-8, h=1e-4):
    """Newton-Raphson with adaptive brentq fallback.

    Matches the pattern in eos_class.py z_eos._newton_1d_z.
    """
    x = float(np.clip(guess, lo, hi))

    for _ in range(max_iter):
        f_val = err_func(x)
        if not np.isfinite(f_val):
            break
        if abs(f_val) < tol:
            return x, True
        f_plus = err_func(x + h)
        f_minus = err_func(x - h)
        if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
            break
        fp = (f_plus - f_minus) / (2.0 * h)
        if abs(fp) < 1e-30:
            break
        step = f_val / fp
        if abs(step) > 1.0:
            step = np.sign(step) * 1.0
        x_new = np.clip(x - step, lo, hi)
        if abs(x_new - x) < 1e-12:
            f_new = err_func(x_new)
            return x_new, (np.isfinite(f_new) and abs(f_new) < tol * 100)
        x = x_new

    # Adaptive brentq fallback
    delta = 0.5
    factor = 2.0
    a = np.clip(x - delta, lo, hi)
    b = np.clip(x + delta, lo, hi)
    for _ in range(8):
        fa = err_func(a)
        fb = err_func(b)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            try:
                sol = brentq(err_func, a, b, xtol=1e-6, maxiter=100)
                return sol, True
            except (ValueError, RuntimeError):
                pass
        a = np.clip(a - delta * factor, lo, hi)
        b = np.clip(b + delta * factor, lo, hi)
        delta *= factor
        if a == lo and b == hi:
            fa, fb = err_func(a), err_func(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                try:
                    return brentq(err_func, a, b, xtol=1e-6, maxiter=100), True
                except (ValueError, RuntimeError):
                    pass
            return np.nan, False
    return np.nan, False


# =================================================================
# P-T basis table via inversion
# =================================================================

# logrho bounds for the inversion (log10 g/cm³)
_LOGRHO_LO = np.log10(rho_arr[0]) - 0.5
_LOGRHO_HI = np.log10(rho_arr[-1]) + 0.5

def build_pt_table(logp_grid=None, logt_grid=None, save=True, outpath=None):
    """Invert (rho, T) -> (P, T) to build a regular P-T table.

    For each (logP_i, logT_j), solve P_forward(10^logrho, 10^logT_j) = 10^logP_i
    for logrho using Newton + brentq fallback.

    Parameters
    ----------
    logp_grid : 1-D array, optional
        log10 P grid in dyn/cm² (default: spans P_grid range).
    logt_grid : 1-D array, optional
        log10 T grid in K (default: spans T_arr range).
    save : bool
        Save to .npz if True.
    outpath : str, optional
        Override output path.

    Returns
    -------
    logrho_pt, logs_pt, logu_pt : 2-D arrays shaped (n_P, n_T)
    """
    if logp_grid is None:
        logp_min = np.log10(np.nanmin(P_grid)) + 0.1
        logp_max = np.log10(np.nanmax(P_grid)) - 0.1
        logp_grid = np.linspace(logp_min, logp_max, 80)
    if logt_grid is None:
        logt_grid = np.log10(T_arr)

    n_P, n_T_inv = len(logp_grid), len(logt_grid)
    logrho_res = np.full((n_P, n_T_inv), np.nan)
    logs_res   = np.full((n_P, n_T_inv), np.nan)
    logu_res   = np.full((n_P, n_T_inv), np.nan)

    for j in range(n_T_inv):
        T_j = 10**logt_grid[j]
        prev_sol_col = None
        for i in range(n_P):
            P_target = 10**logp_grid[i]

            def err(lgrho, _P=P_target, _T=T_j):
                rho_test = 10**lgrho
                P_test = get_p_rhot(rho_test, _T)
                return (P_test / _P) - 1.0

            # Initial guess: ideal gas or warm-start from neighbor
            if prev_sol_col is not None:
                guess = prev_sol_col
            else:
                guess = float(np.clip(
                    _ideal.get_rho_pt(logp_grid[i], logt_grid[j], 0.0),
                    _LOGRHO_LO, _LOGRHO_HI))

            sol, ok = _newton_brentq(err, guess, _LOGRHO_LO, _LOGRHO_HI)
            if ok and np.isfinite(sol):
                logrho_res[i, j] = sol
                rho_sol = 10**sol
                S_val = get_s_rhot(rho_sol, T_j)
                U_val = get_u_rhot(rho_sol, T_j)
                with np.errstate(invalid='ignore', divide='ignore'):
                    logs_res[i, j] = np.log10(S_val) if S_val > 0 else np.nan
                    logu_res[i, j] = np.log10(U_val) if U_val > 0 else np.nan
                prev_sol_col = sol

    if save:
        if outpath is None:
            outpath = f'{CURR_DIR}/methane_ammonia/ammonia_dft_pt.npz'
        np.savez(outpath,
                 logP=logp_grid, logT=logt_grid,
                 logrho=logrho_res, logs=logs_res, logu=logu_res)
        print(f'Saved P-T table to {outpath}')

    return logrho_res, logs_res, logu_res


# =================================================================
# P-T basis lookups (load pre-built table if available)
# =================================================================

_pt_file = f'{CURR_DIR}/methane_ammonia/ammonia_dft_pt.npz'
if os.path.exists(_pt_file):
    _pt_data = np.load(_pt_file)
    _logP_pt = _pt_data['logP']
    _logT_pt = _pt_data['logT']
    _logrho_pt_rgi = RGI((_logP_pt, _logT_pt), _pt_data['logrho'], **_rgi_kw)
    _logs_pt_rgi   = RGI((_logP_pt, _logT_pt), _pt_data['logs'],   **_rgi_kw)
    _logu_pt_rgi   = RGI((_logP_pt, _logT_pt), _pt_data['logu'],   **_rgi_kw)

    def get_logrho_pt(lgp, lgt):
        """log10(rho [g/cm³]) from (logP [dyn/cm²], logT [K])."""
        return _query(_logrho_pt_rgi, lgp, lgt)

    def get_logs_pt(lgp, lgt):
        """log10(S [erg/g/K]) from (logP, logT)."""
        return _query(_logs_pt_rgi, lgp, lgt)

    def get_logu_pt(lgp, lgt):
        """log10(U [erg/g]) from (logP, logT)."""
        return _query(_logu_pt_rgi, lgp, lgt)
