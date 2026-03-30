import numpy as np
from eos import ch4_setzmann as setz
from eos import ch4_nh3_mandy as mandy
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import gaussian_filter1d
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
import pdb

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
erg_to_kJ = (u.erg/u.gram).to('kJ/g')

#np.log10((u.MJ/u.kg).to('erg/g'))


"""
This module contains the equation of state for methane (CH4) using
the analytic fits of Setzmann et al. (1991) and the DFT data from
Bethkenhagen et al. (2017). It interpolates smoothly between the two
methods using a logistic function. The pressure is calculated
using the analytic fit for low densities and the DFT data for high
densities.
"""
# ---------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------
def logistic(x, k=10.0):
    return 1.0 / (1.0 + np.exp(-k * x))

# ---------------------------------------------------------------------
# blended pressure
# ---------------------------------------------------------------------

def pressure_blend(rho, T,
                   rho_switch   = 0.60,  delta_rho = 0.001,
                   delta_T      = 5.0,  k        = 10.0,
                   Tmax_analytic= 500.0,
                   p_switch     = 1e10,  # 1 GPa  (dyn cm⁻²)
                   delta_p      = None,
                   sigma=3.0):
    """
    • Analytic EOS for ρ < rho_switch   (hard cut)
    • ρ–T logistic blend for rho>rho_switch, P<=p_switch
    • Force DFT EOS (smoothly) whenever BOTH ρ>rho_switch & P>p_switch
    """
    rho = np.atleast_1d(rho).astype(float)
    T   = np.atleast_1d(T).astype(float)

    p_a = pressure_analytic(rho, T)
    p_d = pressure_dft     (rho, T)

    # ---------- region 1 : low-density, always analytic ----------------
    out       = p_a.copy()
    mask_hi   = rho > rho_switch        # points where we MAY blend/override
    if not np.any(mask_hi):
        return out.squeeze()            # nothing to blend

    # work only on the high-density subset
    rho_hi, T_hi  = rho[mask_hi], T[mask_hi]
    p_a_hi, p_d_hi= p_a[mask_hi], p_d[mask_hi]

    # ---------- analytic EOS validity in T -----------------------------
    # find the last temperature where analytic EOS is positive
    mask_pos = (p_a_hi >= 0.0) & (T_hi <= Tmax_analytic)
    if np.any(mask_pos):
        T0     = T_hi[mask_pos].max()
        p_peak = p_a_hi[mask_pos].max()
    else:
        T0, p_peak = Tmax_analytic, 0.0

    # clamp analytic P above T0 to p_peak so the logistic has a plateau
    p_a_clamped = np.where(T_hi > T0, p_peak, p_a_hi)

    # ---------- logistic weights --------------------------------------
    w_r = logistic((rho_hi - rho_switch) / delta_rho, k=k)     # density switch
    w_t = logistic((T_hi   - T0        ) / delta_T,   k=k)     # temperature
    w_dt = w_r * w_t                                          # AND (ρ & T)

    if delta_p is None:
        delta_p = 0.1 * p_switch                              # 10 % window
    # pressure override applies ONLY when rho>rho_switch
    w_p  = w_r * logistic((p_a_clamped - p_switch) / delta_p, k=k)

    # final weight: smooth OR  -> max(w_dt, w_p)
    w = 1.0 - (1.0 - w_dt)*(1.0 - w_p)

    # ---------- blend & store back -------------------------------------
    out[mask_hi] = (1.0 - w)*p_a_clamped + w*p_d_hi
    return out.squeeze()

# assumes `logistic`, `pressure_analytic`, `pressure_dft`,
# energy_analytic/energy_dft, entropy_analytic/entropy_dft
# are already defined in scope

def _blend_scalar(quantity_analytic, quantity_dft,
                  rho, T,
                  rho_switch=0.60, delta_rho=0.01,
                  delta_T=5.0,  k=10.0,
                  Tmax_analytic=500.0,
                  p_switch=1e10, delta_p=None,
                  sigma=3.0):
    """
    Generic helper: blends `quantity_analytic` with `quantity_dft`
    using the same switches as pressure_blend.
    """
    rho = np.atleast_1d(rho).astype(float)
    T   = np.atleast_1d(T).astype(float)

    q_a = quantity_analytic(rho, T)
    q_d = quantity_dft     (rho, T)

    #pdb.set_trace()
    delta = q_d[rho >= rho_switch][0] - q_a[rho >= rho_switch][0]

    q_a += delta
    #q_a += 0.0

    # region 1 – always analytic
    out     = q_a.copy()
    mask_hi = rho > rho_switch
    if not np.any(mask_hi):
        return out.squeeze()

    # subset with rho > rho_switch
    rho_hi, T_hi = rho[mask_hi], T[mask_hi]
    q_a_hi, q_d_hi = q_a[mask_hi], q_d[mask_hi]

    # ---- prepare analytic pressure for the same points ---------------
    p_a_hi = pressure_analytic(rho_hi, T_hi)  # plain analytic P
    # clamp analytic P above its valid T range (for logistic stability)
    mask_pos = (p_a_hi >= 0.0) & (T_hi <= Tmax_analytic)
    if np.any(mask_pos):
        T0     = T_hi[mask_pos].max()
        p_peak = p_a_hi[mask_pos].max()
    else:
        T0, p_peak = Tmax_analytic, 0.0
    p_a_clamped = np.where(T_hi > T0, p_peak, p_a_hi)

    # logistic weights
    w_r  = logistic((rho_hi - rho_switch) / delta_rho, k=k)
    w_t  = logistic((T_hi   - T0       ) / delta_T,   k=k)
    w_dt = w_r * w_t                            # ρ AND T

    if delta_p is None:
        delta_p = 0.1 * p_switch
    w_p  = w_r * logistic((p_a_clamped - p_switch) / delta_p, k=k)

    # OR-style combination
    w = 1.0 - (1.0 - w_dt)*(1.0 - w_p)

    # blend and write back
    out[mask_hi] = (1.0 - w) * q_a_hi + w * q_d_hi
    return out.squeeze()

# ---------------------------------------------------------------------
# user-facing wrappers
# ---------------------------------------------------------------------
def energy_blend(rho, T, **kwargs):
    """
    Blended specific internal energy (erg g⁻¹) following the same
    rules as pressure_blend.
    """
    return _blend_scalar(energy_analytic, energy_dft, rho, T, **kwargs)


def entropy_blend(rho, T, **kwargs):
    """
    Blended specific entropy (erg g⁻¹ K⁻¹) following the same
    rules as pressure_blend.
    """
    return _blend_scalar(entropy_analytic, entropy_dft, rho, T, **kwargs)

def free_energy_blend(rho, T, **kwargs):
    """
    Blended specific entropy (erg g⁻¹ K⁻¹) following the same
    rules as pressure_blend.
    """
    return _blend_scalar(free_energy_analytic, free_energy_dft, rho, T, **kwargs)



def free_energy_analytic(rho, T):
    return setz.free_energy(rho, T)

def free_energy_dft(rho, T):
    return mandy.get_a_rhot(rho, T, molecule='methane')

def pressure_analytic(rho, T):
    return setz.get_p_rhot(rho, T)

def pressure_dft(rho, T):
    return mandy.get_p_rhot(rho, T, molecule='methane')

def energy_analytic(rho, T):
    return setz.get_u_rhot(rho, T)#+37880829214.13139#+52869137549.447876 # offset wrt DFT at 1000 K

def energy_dft(rho, T):
    return mandy.get_u_rhot(rho, T, molecule='methane')

def entropy_analytic(rho, T):
    return setz.get_s_rhot(rho, T)#+1.713e8

def entropy_dft(rho, T):
    return mandy.get_s_rhot(rho, T, molecule='methane')


# ---------------------------------------------------------------------
# P-T table smoothing
# ---------------------------------------------------------------------

def smooth_pt_tables(logrho, s, u, logT, logP):
    """Smooth high-P discontinuities in the CH4 P-T table.

    Targets the region where the analytic EOS breaks down (negative
    entropy at low T / high P) and the DFT blending boundary.

    Parameters
    ----------
    logrho, s, u : 2D arrays shaped (n_T, n_P)
    logT, logP : 1D coordinate arrays

    Returns
    -------
    logrho_s, s_s, u_s : smoothed 2D arrays
    """
    from scipy.ndimage import gaussian_filter

    logrho_s = logrho.copy()
    s_s = s.copy()
    u_s = u.copy()

    # Step 1: fix negative / extreme entropy by interpolating from
    # valid pressure neighbors along each isotherm
    for arr in [s_s, u_s, logrho_s]:
        for ti in range(arr.shape[0]):
            row = arr[ti, :]
            bad = ~np.isfinite(row)
            if arr is s_s:
                bad |= (row < 0)
            if not bad.any():
                continue
            valid = ~bad
            if valid.sum() < 2:
                continue
            arr[ti, bad] = np.interp(
                np.where(bad)[0], np.where(valid)[0], row[valid])

    # Step 2: targeted 2D Gaussian in the low-T / high-P blending region
    logT_2d = logT[:, np.newaxis]
    logP_2d = logP[np.newaxis, :]

    # Soft mask: strongest at low T, high P (where artifacts live)
    mask_t = 0.5 * (1.0 - np.tanh((logT_2d - 3.0) / 0.3))
    mask_p = 0.5 * (1.0 + np.tanh((logP_2d - 9.0) / 1.0))
    mask = mask_t * mask_p

    for arr in [logrho_s, s_s, u_s]:
        smoothed = gaussian_filter(arr.astype(float),
                                   sigma=[2.0, 3.0], mode='nearest')
        arr[:] = (1.0 - mask) * arr + mask * smoothed

    return logrho_s, s_s, u_s


# ---------------------------------------------------------------------
# P-T Combined CH4 EOS
# ---------------------------------------------------------------------

data_methane = np.load('eos/methane_ammonia/methane_eos_pt_extended.npz')

rgi_args = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

rho_pt_rgi_methane = RGI((data_methane['logT'], data_methane['logP']),
                            data_methane['logrho'], **rgi_args)
s_pt_rgi_methane = RGI((data_methane['logT'], data_methane['logP']),
                            data_methane['s'], **rgi_args)
u_pt_rgi_methane = RGI((data_methane['logT'], data_methane['logP']),
                            data_methane['u'], **rgi_args)

def get_logrho_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_pt_rgi_methane(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_pt_rgi_methane(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_u_pt(_lgp, _lgt):

    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_pt_rgi_methane(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result