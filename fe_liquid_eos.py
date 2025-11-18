import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.optimize import brentq, brenth
from scipy.integrate import quad
import pdb
import os

J_K_kg_to_erg_K_g = (u.J / (u.kg * u.K)).to('erg/(K*g)') # specific entropy conversion
J_kg_to_erg_g = (u.J / u.kg).to('erg/g') # specific energy conversion
dyn_to_Pa = (u.dyn/u.cm**2).to('Pa') # dyn/cm² to Pa conversion
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

####### Liquid and Iron EOS calls ########

P_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Pgrid_Fel.txt')
T_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Tgrid_Fel.txt')

s_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/s_Fel.txt') * J_K_kg_to_erg_K_g
rho_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/rho_Fel.txt')
u_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Eth_Fel.txt')* (u.J/u.kg).to('erg/g')
CP_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cP_Fel.txt') * J_K_kg_to_erg_K_g
CV_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cV_Fel.txt') * J_K_kg_to_erg_K_g

rgi_kwargs = {
    'method': 'linear',
    'bounds_error': False,
    'fill_value': None
}

alpha_grid_Fe_l = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/alpha_Fel.txt')

get_rho_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), rho_grid_Fe_l, **rgi_kwargs)
get_s_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), s_grid_Fe_l, **rgi_kwargs)
get_u_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), u_grid_Fe_l, **rgi_kwargs)

get_CP_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), CP_grid_Fe_l, **rgi_kwargs)
get_CV_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), CV_grid_Fe_l, **rgi_kwargs)
get_alpha_Fel = RGI((P_grid_Fe_l, T_grid_Fe_l), alpha_grid_Fe_l, **rgi_kwargs)

def get_rho_pt(P, T):
    """
    Get the density of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_rho_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_s_pt(P, T):
    """
    Get the entropy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_s_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_u_pt(P, T):
    """
    Get the internal energy of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_u_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CP_pt(P, T):
    """
    Get the isobaric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CP_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_CV_pt(P, T):
    """
    Get the isochoric heat capacity of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_CV_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

def get_alpha_pt(P, T):
    """
    Get the thermal expansion coefficient of Fe-Si alloy at pressure P and temperature T.
    P in Pa, T in K.
    """
    scalar = np.isscalar(P) and np.isscalar(T)
    P_arr = np.array(P, ndmin=1)
    T_arr = np.array(T, ndmin=1)
    if P_arr.shape != T_arr.shape:
        P_arr, T_arr = np.broadcast_arrays(P_arr, T_arr)
    pts = np.stack((P_arr.ravel(), T_arr.ravel()), axis=-1)
    vals = get_alpha_Fel(pts).reshape(P_arr.shape)
    return float(vals) if scalar else vals

T_min_Fe = 0
T_max_Fe = 200000

def get_T_sp_inv(_s, _P, xtol=1e-8, maxiter=500):
    """
    Invert s(P, T) → T, i.e. find T such that get_s_pt(P, T) == s.

    Parameters
    ----------
    _s : float or array_like
        Entropy value(s) in kB/baryon.
    _P : float or array_like
        Pressure value(s) in Pa.
    xtol : float, optional
        Tolerance on the temperature root (passed to brentq).
    maxiter : int, optional
        Maximum number of iterations for brentq.

    Returns
    -------
    T_sol : float or ndarray
        Temperature(s) in K.  If any root‐finding fails, the corresponding
        entry is set to np.nan.
    """
    s_arr = np.atleast_1d(_s)
    P_arr = np.atleast_1d(_P)
    s_arr, P_arr = np.broadcast_arrays(s_arr, P_arr)

    def _find_T(s_val, P_val):
        # The function whose root in T we seek:
        def err(T_val):
            return get_s_pt(P_val, T_val) * erg_to_kbbar / s_val - 1

        try:
            return brentq(err, T_min_Fe, T_max_Fe, xtol=xtol, maxiter=maxiter)
        except ValueError:
            # e.g. f(T_min)*f(T_max) ≥ 0 or NaN → no bracket
            return np.nan

    # vectorize over the pair (s_val, P_val)
    T_roots = np.vectorize(_find_T)(s_arr, P_arr)

    # return scalar if inputs were scalars
    if T_roots.size == 1:
        return float(T_roots)
    return T_roots

###### S, P basis ######

sp_data_Fel = np.load('eos/zhang_eos/zhang_multiphase/iron2_sp.npz')

svals_sp_Fel = sp_data_Fel['s_vals'] # kb/baryon
logpvals_sp_Fel = sp_data_Fel['logpvals'] # log dyn/cm^2

logrho_grid_sp_Fel = sp_data_Fel['logrho_sp'] # in g/cm^3
logt_grid_sp_Fel = sp_data_Fel['logt_sp'] # in K
logu_grid_sp_Fel = sp_data_Fel['logu_sp'] # in erg/g

logt_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logt_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)
logrho_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logrho_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_sp_Fel = RGI((svals_sp_Fel, logpvals_sp_Fel), logu_grid_sp_Fel, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp(S, P):
    """
    Get the density of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** (logrho_rgi_sp_Fel(pts).reshape(S_arr.shape) + 3) # kg/m^3
    return float(vals) if scalar else vals

def get_T_sp(S, P):
    """
    Get the temperature of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** logt_rgi_sp_Fel(pts).reshape(S_arr.shape) # K
    return float(vals) if scalar else vals

def get_u_sp(S, P):
    """
    Get the internal energy of liquid Fe at entropy S and pressure P.
    S in kb/baryon, P in Pa.
    """

    logP_cgs = np.log10(P / dyn_to_Pa)
    scalar = np.isscalar(S) and np.isscalar(P)
    S_arr = np.array(S, ndmin=1)
    logP_arr = np.array(logP_cgs, ndmin=1)
    if S_arr.shape != logP_arr.shape:
        S_arr, logP_arr = np.broadcast_arrays(S_arr, logP_arr)
    pts = np.stack((S_arr.ravel(), logP_arr.ravel()), axis=-1)
    vals = 10 ** logu_rgi_sp_Fel(pts).reshape(S_arr.shape) # erg/g
    return float(vals) if scalar else vals

def get_T_melt(P_GPa):
    Tm_fe = 1900 * (P_GPa / 31.3 + 1) ** (1/1.99) # Zhang et al. 2015
    return Tm_fe