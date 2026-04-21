import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
import os
from eos import ideal_eos, metals_eos, cms_eos, cd_eos, mls_eos, mh13_eos, scvh_eos
import pdb
from eos import smooth

raise Warning('This file has been depricated. Use eos_class.py instead.')
"""
    ROB: THIS HAS BEEN DEPRICATED. USE eos_class.py INSTEAD.
    This file provides access to H-He and H-He-Z mixtures.
    The mixtures are calculated via the volume addition law.
    For convenience, please use the _tab functions insteaad of the inversion functions.
    The _tab functions provide direct access to pre-calculated tables, and the inversion functions
    perform direct inversions from the P, T tables.

    Author: Roberto Tejada Arevalo

"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

pd.options.mode.chained_assignment = None

ideal_xy = ideal_eos.IdealHHeMix()
ideal_z = ideal_eos.IdealEOS(m=150)

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mh = 1
mhe = 4.0026

###### H-He mixing ######

def Y_to_n(_y):
    ''' Change between mass and number fraction OF HELIUM'''
    return ((_y/mhe)/(((1 - _y)/mh) + (_y/mhe)))
def n_to_Y(x):
    return (mhe * x)/(1 + 3.0026*x)

def x_H(_y, _z, mz):
    yeff = _y#/(1 - _z)
    Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
    return (1-yeff)*(1-_z)/mh/Ntot

def x_Z(_y, _z, mz):
    yeff = _y#/(1 - _z)
    Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
    return (_z/mz)/Ntot

def guarded_log(x):
    ''' Used to calculate ideal enetropy of mixing: xlogx'''
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('Number fraction went negative.')
        return x * np.log(x)
    return np.array([guarded_log(x_) for x_ in x])

def get_smix_id_y(Y):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    xhe = Y_to_n(Y)
    xh = 1 - xhe
    q = mh*xh + mhe*xhe
    return -1*(guarded_log(xh) + guarded_log(xhe)) / q

def get_smix_id_yz(Y, Z, mz):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    xh = x_H(Y, Z, mz)
    xz = x_Z(Y, Z, mz)
    xhe = 1 - xh - xz
    q = mh*xh + mhe*xhe + mz*xz
    return -1*(guarded_log(xh) + guarded_log(xhe) + guarded_log(xz)) / q

# def gauss_smooth(data, base_sigma=5, base_window=5):
#     "Returns smoothed data array"
#     smoothed_data = np.zeros_like(data)
#     differences = np.abs(np.diff(data, prepend=data[0]))  # prepend to match the length
#     median_difference = np.median(differences)

#     for i in range(len(data)):
#         # Adjust sigma based on local difference information
#         local_sigma = base_sigma * (1 + (median_difference - differences[i]) / (median_difference + 1e-6))

#         # Determine the window size dynamically based on the position within the array
#         start_index = max(0, i - base_window // 2)
#         end_index = min(len(data), i + base_window // 2 + 1)
#         actual_window_size = end_index - start_index

#         # Generate the Gaussian kernel for the actual window size
#         x = np.linspace(-(actual_window_size // 2), actual_window_size // 2, actual_window_size)
#         gauss_kernel = np.exp(-0.5 * (x / local_sigma) ** 2)
#         gauss_kernel /= gauss_kernel.sum()  # Normalize the kernel

#         # Apply the kernel to the data segment
#         smoothed_data[i] = np.dot(data[start_index:end_index], gauss_kernel)

#     return smoothed_data

#### P, _lgt mixtures ####

def get_s_pt(_lgp, _lgt, _y_prime, _z, hhe_eos, z_eos='aqua', hg=True):
    """
    This calculates the entropy for a metallicity mixture.
    The cms and mls EOSes already contain the HG23 non-ideal corrections
    to the entropy. These terms contain the ideal entropy of mixing, so
    for metal mixures, we subtract the H-He ideal entropy of mixing and
    add back the metal mixture entropy of mixing plus the non-ideal
    correction.

    The _y_prime parameter here is the Y in a pure H-He EOS. Therefore, it
    is Y/(1 - Z). So the y value that should be
    used to calculate the entropy of mixing should be Y*(1 - Z).
    """

    _y = _y_prime*(1 - _z)

    if (
        (np.isscalar(_y_prime) and _y_prime > 1.0)
        or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
        or (np.isscalar(_z) and _z > 1.0)
        or ((not np.isscalar(_z)) and np.any(_z > 1.0))
    ):
        raise Exception('Invalid mass fractions: X + Y + Z > 1.')

    if hhe_eos == 'cms' or hhe_eos == 'cms_ice':
        xy_eos = cms_eos

        smix_id = xy_eos.get_smix_id_y(_y_prime) / erg_to_kbbar
        if hg:
            s_nid_mix = xy_eos.smix_interp(_lgt, _lgp)*(1-_y_prime)*_y_prime - smix_id
        else:
            s_nid_mix = 0.0
        s_xy = xy_eos.get_s_pt(_lgp, _lgt, _y_prime, hg=False) - smix_id
    elif hhe_eos == 'cd' or hhe_eos == 'cd_ice':
        xy_eos = cd_eos
        smix_id = xy_eos.get_smix_id_y(_y_prime) / erg_to_kbbar
        s_xy = xy_eos.get_s_pt(_lgp, _lgt, _y_prime) - smix_id
        s_nid_mix = 0.0
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
        smix_id = xy_eos.get_smix_id_y(_y_prime) / erg_to_kbbar
        xy_eos = xy_eos.get_s_pt(_lgp, _lgt, _y_prime, hg=False) - smix_id
        if hg:
            s_nid_mix = xy_eos.smix_interp(_lgt, _lgp)*(1-_y_prime)*_y_prime - smix_id
        else:
            s_nid_mix = 0.0
    elif hhe_eos == 'mh13':
        xy_eos = mh13_eos
        smix_id = xy_eos.get_smix_id_y(_y_prime) / erg_to_kbbar
        s_xy = xy_eos.get_s_pt_tab(_lgp, _lgt, _y_prime)

    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos
        s_xy = xy_eos.get_s_pt_tab(_lgp, _lgt, _y_prime)
        s_nid_mix = 0.0
    elif hhe_eos == 'ideal':
        return ideal_xy.get_s_pt(_lgp, _lgt, _y_prime) / erg_to_kbbar

    else:
        raise Exception('Only cms, mls, mh13, scvh (CMS19+HG23, MLS22+HG23, MH13, and SCvH95) allowed for now')

    if z_eos == 'aqua':
        mz = 18.015
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='aqua')
    elif z_eos == 'ppv':
        mg = 24.305
        si = 28.085
        o3 = 48.000
        mz = mg+si+o3
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ppv')
    elif z_eos == 'serpentine':
        mz = 348.4
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='serpentine')
    elif z_eos == 'fo':
        mz = 140.69
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='serpentine')
    elif z_eos == 'iron':
        mz = 56
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='iron')
    elif z_eos == 'ideal':
        mz = 150 # olivine mean molecular weight
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ideal')
    elif z_eos == 'mixture':
        mz = 40
        xz = x_Z(_y, _z, mz)
        xh = x_H(_y, _z, mz)
        s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='mixture')
    elif z_eos is None:
        mz = 2.0 # doesn't matter because xz should be 0
        xz = 0.0
        s_z = 0.0
        xh = x_H(_y, _z, mz)
    else:
        raise Exception('Acceptable z_eos: aqua, ppv, serpentine, iron, ideal, mixture, None')

    # Calculating entropy of mixing terms
    xhe = 1 - xh - xz
    if np.any(xh + xhe + xz) != 1.0:
        raise Exception('X + Y + Z != 0')

    #qz = xh*mh + xhe*mh3 + xz*mz

    # Y that goes into three-component entropy of mixing is Y/(X + Y + X)
    s_id_zmix = get_smix_id_yz(_y, _z, mz) / erg_to_kbbar
    return s_xy*(1 - _z) + s_z * _z + s_id_zmix + s_nid_mix*(1 - _z)

def get_rho_pt(_lgp, _lgt, _y_prime, _z, hhe_eos, z_eos=None, hg=True):
    """
    This calculates the log10 of the density for a metallicity mixture.
    The cms and mls EOSes already contain the HG23 non-ideal corrections
    to the density.
    """

    _y = _y_prime*(1 - _z)

    if (
        (np.isscalar(_y_prime) and _y_prime > 1.0)
        or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
        or (np.isscalar(_z) and _z > 1.0)
        or ((not np.isscalar(_z)) and np.any(_z > 1.0))
    ):
        raise Exception('Invalid mass fractions: X + Y + Z > 1.')

    if hhe_eos == 'cms' or hhe_eos == 'cms_ice':
        xy_eos = cms_eos
        rho_hhe = 10**xy_eos.get_rho_pt(_lgp, _lgt, _y_prime, hg=hg)
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
        rho_hhe = 10**xy_eos.get_rho_pt(_lgp, _lgt, _y_prime, hg=hg)
    elif hhe_eos == 'mh13':
        xy_eos = mh13_eos
        rho_hhe = 10**xy_eos.get_rho_pt(_lgp, _lgt, _y_prime)
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos
        rho_hhe = 10**xy_eos.get_rho_pt_tab(_lgp, _lgt, _y_prime)
    elif hhe_eos == 'cd' or hhe_eos == 'cd_ice':
        xy_eos = cd_eos
        rho_hhe = 10**xy_eos.get_rho_pt(_lgp, _lgt, _y_prime)

    elif hhe_eos=='ideal':
        return ideal_xy.get_rho_pt(_lgp, _lgt, _y_prime)
    else:
        raise Exception('Only cms, cd, mls, mh13, scvh (CMS19+HG23, CD21, MLS22+HG23, MH13, and SCvH95) allowed for now')

    # Calculating volume of mixing terms
    if z_eos is not None:
        rho_z = 10**metals_eos.get_rho_pt_tab(_lgp, _lgt, z_eos)
    if z_eos is None:
        rho_z = 1.0 # doesn't matter because _z should be 0
        _z = 0.0
    return np.log10(1/((1 - _z)/rho_hhe + _z/rho_z))

def get_u_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos=None, hg=True):
    """
    This calculates the log10 of the specific internal energy.
    The volume addition law is used for all EOSes. There are
    no non-ideal correction terms for the internal energy yet.
    """
    if hhe_eos == 'cms' or hhe_eos == 'cms_ice':
        xy_eos = cms_eos
    elif hhe_eos == 'cd' or hhe_eos == 'cd_ice':
        xy_eos = cd_eos
    elif hhe_eos == 'mls':
        xy_eos = mls_eos
    elif hhe_eos == 'mh13':
        xy_eos = mh13_eos
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos
    elif hhe_eos == 'ideal':
        return ideal_xy.get_u_pt(_lgp, _lgt, _y)
    else:
        raise Exception('Only cms, cd, mls, mh13, scvh (CMS19+HG23, CD21, MLS22+HG23, MH13, and SCvH95) allowed for now')

    if z_eos is not None:
        u_z = 10**metals_eos.get_u_pt_tab(_lgp, _lgt, z_eos)
    if z_eos is None:
        u_z = 1.0 # doesn't matter because _z should be 0
        _z = 0.0
    # Calculating energy of mixtures via the volume addition law.
    # if hhe_eos == 'scvh':
    #     u_xy = 10**xy_eos.get_u_pt(_lgp, _lgt, _y)
    #     return np.log10((1 - _z)*u_xy + _z*u_z)
    if hhe_eos == 'mh13':
        u_xy = xy_eos.get_u_pt_tab(_lgp, _lgt, _y)
        return np.log10((1 - _z)*u_xy + _z*u_z)
    elif hhe_eos == 'cms' or hhe_eos=='cms_ice':
        u_xy = 10**xy_eos.get_u_pt(_lgp, _lgt, _y, hg=hg)
        return np.log10((1 - _z)*u_xy + _z*u_z)
    else:
        u_xy = 10**xy_eos.get_u_pt(_lgp, _lgt, _y)
        return np.log10((1 - _z)*u_xy + _z*u_z)

        #return np.log10((1 - _y)*(1 - _z) * u_h + _y * (1 - _z)* u_he + _z * u_z)

### error functions ###

def err_t_sp(_lgt, _s, _lgp, _y, _z, hhe_eos, z_eos, hg):
    s_test = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos=z_eos, hg=hg)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_rhot(_lgp, _lgrho, _lgt, _y, _z, hhe_eos, z_eos, hg):
    logrho_test = get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    return (logrho_test/(_lgrho+1e-30)) - 1

def err_t_srho(_lgt, _s, _lgrho, _y, _z, hhe_eos, z_eos, hg):
    s_test = get_s_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos, z_eos=z_eos, hg=hg)*erg_to_kbbar
    #s_test = get_s_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)*erg_to_kbbar
    #s_test = get_s_pt(logp, _lgt, _y, _z, hhe_eos, z_eos=z_eos)*erg_to_kbbar
    #s_test = get_s_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, hg=hg)*erg_to_kbbar
    #logp = get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, hg=hg)
    #s_test = get_s_pt(logp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)*erg_to_kbbar
    return (s_test/_s) - 1

def err_p_srho(_lgp, _s, _lgrho, _y, _z, hhe_eos, z_eos, hg):
    logt_test = get_t_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    logrho_test = get_rho_pt(_lgp, logt_test, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    return (logrho_test/(_lgrho+1e-20)) - 1

def err_pt_srho(pt_arg, _s, _lgrho, _y, _z, hhe_eos, z_eos, hg):
    _lgp, _lgt = pt_arg
    s_pt = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) * erg_to_kbbar
    rho_pt = get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    #rho_sp = get_rho_sp_tab(s_pt, _lgp, _y, _z, hhe_eos='cms')
    return [float((s_pt / _s) - 1), float((rho_pt/_lgrho) - 1)]

def err_t_rhop(_lgt, _lgrho, _lgp, _y, _z, hhe_eos, z_eos, hg):
    logrho_test = get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    return (logrho_test/_lgrho+1e-20) - 1

def err_grad(s_trial, _lgp, _y, _z, hhe_eos, hg, tab):
    grad_a = get_nabla_ad(s_trial, _lgp, _y, _z, hhe_eos=hhe_eos, hg=hg, tab=tab)
    if tab:
        logt = get_t_sp_tab(s_trial, _lgp, _y, _z, hhe_eos=hhe_eos, hg=hg)
    else:
        logt = get_t_sp(s_trial, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos='aqua', hg=hg)
    grad_prof = np.gradient(logt)/np.gradient(_lgp)
    return (grad_a/grad_prof) - 1

############################### inversion functions ###############################

TBOUNDS = [0, 17]
PBOUNDS = [0, 15]

XTOL = 1e-16

###### S, P ######

def get_t_sp(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', hg=True, multivariable=False, method='hybr'):

    if not multivariable:
        if np.isscalar(_s):
            _s, _lgp, _y, _z = np.array([_s]), np.array([_lgp]), np.array([_y]), np.array([_z])
            try:
                guess = ideal_xy.get_t_sp(_s, _lgp, _y)
            except:
                guess = get_t_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
            sol = root(err_t_sp, guess, tol=1e-8, method=method, args=(_s, _lgp, _y, _z, hhe_eos, z_eos, hg))
            return float(sol.x.item())

        # loops through arrays to find solulations, possibly an easier problem to solve than multivariable
        sol = np.array([get_t_sp(s, p, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for s, p, y, z in zip(_s, _lgp, _y, _z)])
        return sol

    else:
        if np.isscalar(_s):
            _s, _lgp, _y, _z = np.array([_s]), np.array([_lgp]), np.array([_y]), np.array([_z])
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
            sol = root(err_t_sp, guess, tol=1e-8, method=method, args=(_s, _lgp, _y, _z, hhe_eos, z_eos, hg))
            return float(sol.x.item())
        else:
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
            sol = root(err_t_sp, guess, tol=1e-8, method=method, args=(_s, _lgp, _y, _z, hhe_eos, z_eos, hg))
            return sol.x

# def get_rhot_sp(_s, _lgp, _y, _z, hhe_eos, z_eos=None):
#     logt = get_t_sp(_s, _lgp, _y, _z, hhe_eos, z_eos=z_eos)
#     return get_rho_pt(_lgp, logt, _y, _z, hhe_eos, z_eos=z_eos), logt

###### Rho, T ######

def get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    #if alg == 'root':
    if np.isscalar(_lgrho):
        _lgrho, _lgt, _y, _z = np.array([_lgrho]), np.array([_lgt]), np.array([_y]), np.array([_z])
        guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        sol = root(err_p_rhot, guess, tol=XTOL, method='hybr', args=(_lgrho, _lgt, _y, _z, hhe_eos, z_eos, hg))
        return float(sol.x.item())

    sol = np.array([get_p_rhot(rho, t, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for rho, t, y, z in zip(_lgrho, _lgt, _y, _z)])
    return sol

def get_s_rhot(_lgrho, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    logp = get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=True)
    return get_s_pt(logp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)


###### S, Rho ######

def get_t_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    #if alg == 'root':
    if np.isscalar(_s):
        _s, _lgrho, _y, _z = np.array([_s]), np.array([_lgrho]), np.array([_y]), np.array([_z])
        guess = ideal_xy.get_t_srho(_s, _lgrho, _y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, z_eos, hg))
        return float(sol.x.item())

    sol = np.array([get_t_srho(s, rho, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for s, rho, y, z in zip(_s, _lgrho, _y, _z)])
    return sol

def get_p_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    #if alg == 'root':
    if np.isscalar(_s):
        _s, _lgrho, _y, _z = np.array([_s]), np.array([_lgrho]), np.array([_y]), np.array([_z])
        guess = ideal_xy.get_p_srho(_s, _lgrho, _y)[0]
        sol = root(err_p_srho, guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, z_eos, hg))
        return float(sol.x.item())

    sol = np.array([get_p_srho(s, rho, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for s, rho, y, z in zip(_s, _lgrho, _y, _z)])
    return sol

def get_pt_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    if np.isscalar(_s):
        _s, _lgrho, _y, _z = np.array([_s]), np.array([_lgrho]), np.array([_y]), np.array([_z])
        guess = np.array(ideal_xy.get_pt_srho(_s, _lgrho, _y)[0])
        #print(np.shape(guess))
        sol = root(err_pt_srho, x0=guess, tol=1e-8, method='hybr', args=(_s, _lgrho, _y, _z, hhe_eos, z_eos, hg))
        return sol.x

    sol = np.array([get_pt_srho(s, rho, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for s, rho, y, z in zip(_s, _lgrho, _y, _z)])
    return sol

###### Rho, P ######

def get_t_rhop(_lgrho, _lgp, _y, _z, hhe_eos, z_eos='aqua', hg=True):
    #if alg == 'root':
    if np.isscalar(_lgrho):
        _lgrho, _lgp, _y, _z = np.array([_lgrho]), np.array([_lgp]), np.array([_y]), np.array([_z])
        guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y, _z, hhe_eos, z_eos, hg))
        return float(sol.x.item())

    sol = np.array([get_t_rhop(rho, p, y, z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg) for rho, p, y, z in zip(_lgrho, _lgp, _y, _z)])
    return sol

def get_s_rhop(_lgrho, _lgp, _y, _z, hhe_eos):
    logt_rhop = get_t_rhop(_lgrho, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos='aqua')
    return get_s_pt(_lgp, logt_rhop, _y, _z, hhe_eos=hhe_eos, z_eos='aqua')

############################### Tabulated EOS Functions ###############################

###### S, P ######


svals_sp_aqua = np.arange(3.0, 9.1, 0.1)
# svals_sp_aqua_cms = np.arange(4.0, 9.1, 0.1)
svals_sp_aqua_cms = np.arange(4.0, 9.05, 0.05)
#svals_sp_aqua_cms = np.arange(2.0, 9.1, 0.1)
svals_sp_aqua_cd = np.arange(4.0, 9.1, 0.1)
logpvals_sp_aqua = np.arange(6, 14.1, 0.1)

yvals_sp_cms = np.arange(0.05, 0.95, 0.05) # new CMS19+HG23 grid
#yvals_sp_cms = np.arange(0.05, 0.9, 0.05)
#zvals_sp_cms = np.arange(0, 0.92, 0.02)
zvals_sp_cms = np.arange(0, 0.95, 0.05)

yvals_sp = np.arange(0.05, 0.95, 0.1) # old grid, still good for the others
zvals_sp = np.arange(0, 1.0, 0.1)

yvals_sp_scvh = np.arange(0.15, 0.75, 0.05)
yvals_sp_mh13 = np.arange(0.246575, 0.95, 0.1)
#zvals_sp = np.arange(0, 1.0, 0.1)

# CMS sparse table for high-Z values meant for Uranus/Neptune sims
svals_sp_ice = np.arange(1.5, 8.1, 0.1)
logpvals_sp_ice = np.arange(6, 14.1, 0.1)
yvals_sp_ice = np.arange(0.05, 1.05, 0.1)
zvals_sp_ice = np.arange(0, 1.1, 0.1)

# CD sparse table for high-Z values meant for Uranus/Neptune sims
# svals_sp_cd_ice = np.arange(1.5, 8.6, 0.1)
# logpvals_sp_cd_ice = np.arange(6, 14.1, 0.1)
# yvals_sp_cd_ice = np.arange(0.05, 1.05, 0.1)
# zvals_sp_cd_ice = np.arange(0, 1.05, 0.05)

svals_sp_cd_ice = np.arange(0.5, 7.6, 0.1)
logpvals_sp_cd_ice = np.arange(6, 13.6, 0.1)
yvals_sp_cd_ice = np.arange(0.02, 0.85, 0.05)
zvals_sp_cd_ice = np.arange(0.0, 1.05, 0.05)

# CD & CMS19+HG23new extended table for low and high S
svals_sp_cd = np.arange(1.5, 10.1, 0.1)
logpvals_sp_cd = np.arange(6, 14.1, 0.1)
yvals_sp_cd = np.arange(0.02, 1.0, 0.05)
zvals_sp_cd = np.arange(0.0, 1.05, 0.05) # dense grid of Z

# logrho_res_sp_cms_aqua, logt_res_sp_cms_aqua = np.load('%s/cms/sp_base_z_aqua_cms_hg_updated.npy' % CURR_DIR)

logrho_res_sp_cms_aqua, logt_res_sp_cms_aqua = np.load('%s/cms/sp_base_z_aqua_cms_lows_highs_extended.npy' % CURR_DIR)

#logrho_res_sp_cd_ice, logt_res_sp_cd_ice = np.load('%s/cd/sp_base_z_aqua_cd_lows_ice_dense.npy' % CURR_DIR)
logrho_res_sp_cd_ice, logt_res_sp_cd_ice = np.load('%s/cd/sp_base_z_aqua_cd_lows_cold.npy' % CURR_DIR)

#logrho_res_sp_cms_aqua, logt_res_sp_cms_aqua = np.load('%s/cms/sp_base_z_aqua_cms_hg_updated_dense.npy' % CURR_DIR)

logrho_res_sp_cms_nohg_aqua, logt_res_sp_cms_nohg_aqua = np.load('%s/cms/sp_base_z_aqua_extended_nohg.npy' % CURR_DIR)

#logrho_res_sp_cd_aqua, logt_res_sp_cd_aqua = np.load('%s/cd/sp_base_z_aqua_extended.npy' % CURR_DIR)

logrho_res_sp_cd_aqua, logt_res_sp_cd_aqua = np.load('%s/cd/sp_base_z_aqua_cd_lows_highs_extended.npy' % CURR_DIR)

logrho_res_sp_scvh_aqua, logt_res_sp_scvh_aqua = np.load('%s/scvh/sp_base_z_aqua_extended_new.npy' % CURR_DIR)

logrho_res_sp_mls_aqua, logt_res_sp_mls_aqua = np.load('%s/mls/sp_base_z_aqua_extended.npy' % CURR_DIR)

logrho_res_sp_mh13_aqua, logt_res_sp_mh13_aqua = np.load('%s/mh13/sp_base_z_aqua_extended.npy' % CURR_DIR)

get_rho_rgi_sp_cms = RGI((svals_sp_cd, logpvals_sp_cd, yvals_sp_cd, zvals_sp_cd), logrho_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_cms = RGI((svals_sp_cd, logpvals_sp_cd, yvals_sp_cd, zvals_sp_cd), logt_res_sp_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

# get_rho_rgi_sp_ice = RGI((svals_sp_ice, logpvals_sp_ice, yvals_sp_ice, zvals_sp_ice), logrho_res_sp_cms_ice, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_sp_ice= RGI((svals_sp_ice, logpvals_sp_ice, yvals_sp_ice, zvals_sp_ice), logt_res_sp_cms_ice, method='linear', \
#             bounds_error=False, fill_value=None)

get_rho_rgi_sp_cd_ice = RGI((svals_sp_cd_ice, logpvals_sp_cd_ice, yvals_sp_cd_ice, zvals_sp_cd_ice), logrho_res_sp_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_cd_ice= RGI((svals_sp_cd_ice, logpvals_sp_cd_ice, yvals_sp_cd_ice, zvals_sp_cd_ice), logt_res_sp_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_cms_nohg = RGI((svals_sp_aqua_cd, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_cms_nohg = RGI((svals_sp_aqua_cd, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)

# get_rho_rgi_sp_cd = RGI((svals_sp_aqua_cd, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_cd_aqua, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_sp_cd = RGI((svals_sp_aqua_cd, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_cd_aqua, method='linear', \
#             bounds_error=False, fill_value=None)

get_rho_rgi_sp_cd = RGI((svals_sp_cd, logpvals_sp_cd, yvals_sp_cd, zvals_sp_cd), logrho_res_sp_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_cd = RGI((svals_sp_cd, logpvals_sp_cd, yvals_sp_cd, zvals_sp_cd), logt_res_sp_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_scvh = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp_scvh, zvals_sp), logrho_res_sp_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_scvh = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp_scvh, zvals_sp), logt_res_sp_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_mls = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logrho_res_sp_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_mls = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp, zvals_sp), logt_res_sp_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_mh13 = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp_mh13, zvals_sp), logrho_res_sp_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_mh13 = RGI((svals_sp_aqua, logpvals_sp_aqua, yvals_sp_mh13, zvals_sp), logt_res_sp_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_s):
                return float(get_rho_rgi_sp_cms(np.array([_s, _lgp, _y_call, _z]).T))
            else:
                return get_rho_rgi_sp_cms(np.array([_s, _lgp, _y_call, _z]).T)
        else:
            if np.isscalar(_s):
                return float(get_rho_rgi_sp_cms_nohg(np.array([_s, _lgp, _y_call, _z]).T))
            else:
                return get_rho_rgi_sp_cms_nohg(np.array([_s, _lgp, _y_call, _z]).T)

    # elif hhe_eos == 'cms_ice':
    #     if np.isscalar(_s):
    #         return float(get_rho_rgi_sp_ice(np.array([_s, _lgp, _y, _z]).T))
    #     else:
    #         return get_rho_rgi_sp_ice(np.array([_s, _lgp, _y, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_cd_ice(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_rho_rgi_sp_cd_ice(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_cd(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_rho_rgi_sp_cd(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_scvh(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_rho_rgi_sp_scvh(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_mls(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_rho_rgi_sp_mls(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_s):
            return float(get_rho_rgi_sp_mh13(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_rho_rgi_sp_mh13(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_rho_sp(_s, _lgp, _y_call))
        else:
            return ideal_xy.get_rho_sp(_s, _lgp, _y_call)

    else:
        raise Exception('Only cms, scvh, or mls available for now.')

def get_t_sp_tab(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_s):
                return float(get_t_rgi_sp_cms(np.array([_s, _lgp, _y_call, _z]).T))
            else:
                return get_t_rgi_sp_cms(np.array([_s, _lgp, _y_call, _z]).T)
        else:
            if np.isscalar(_s):
                return float(get_t_rgi_sp_cms_nohg(np.array([_s, _lgp, _y_call, _z]).T))
            else:
                return get_t_rgi_sp_cms_nohg(np.array([_s, _lgp, _y_call, _z]).T)

    # elif hhe_eos == 'cms_ice':
    #     if np.isscalar(_s):
    #         return float(get_t_rgi_sp_ice(np.array([_s, _lgp, _y, _z]).T))
    #     else:
    #         return get_t_rgi_sp_ice(np.array([_s, _lgp, _y, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_cd_ice(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_t_rgi_sp_cd_ice(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_cd(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_t_rgi_sp_cd(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_scvh(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_t_rgi_sp_scvh(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_mls(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_t_rgi_sp_mls(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_s):
            return float(get_t_rgi_sp_mh13(np.array([_s, _lgp, _y_call, _z]).T))
        else:
            return get_t_rgi_sp_mh13(np.array([_s, _lgp, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_t_sp(_s, _lgp, _y_call))
        else:
            return ideal_xy.get_t_sp(_s, _lgp, _y_call)
    else:
        raise Exception('Only cms, scvh, or mls available for now.')

def get_rhot_sp_tab(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', hg=True, tab=True, y_tot=True):
    if tab:
        rho, t = get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot), get_t_sp_tab(_s, _lgp , _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        if y_tot:
            _y_call = _y/(1 - _z)
        else:
            _y_call = _y
        t = get_t_sp(_s, _lgp, _y_call, _z, hhe_eos, z_eos=z_eos, hg=hg)
        rho = get_rho_pt(_lgp, t, _y_call, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    return rho, t

###### Rho, T ######

logrhovals_rhot = np.linspace(-5.0, 2.0, 100)
logtvals_rhot = np.arange(2, 5.05, 0.05)
yvals_rhot = np.arange(0.05, 0.95, 0.1)
zvals_rhot = np.arange(0, 1.0, 0.1)

yvals_rhot_cms = np.arange(0.05, 0.95, 0.05)
zvals_rhot_cms = np.arange(0, 0.95, 0.05)
#zvals_rhot_cms = np.arange(0, 0.91, 0.01) # dense grid

# CD ICE grid
logrhovals_rhot_cd_ice = np.linspace(-4.0, 2.0, 60)
logtvals_rhot_cd_ice = np.arange(2, 5.05, 0.05)
yvals_rhot_cd_ice = np.arange(0.05, 1.05, 0.1)
zvals_rhot_cd_ice = np.arange(0, 1.05, 0.05)

# CD & CMS19+HG23 low and high S
logrhovals_rhot_cd = np.linspace(-5.0, 2.0, 100)
logtvals_rhot_cd = np.arange(2.1, 5.05, 0.05)
yvals_rhot_cd = np.arange(0.02, 1.0, 0.05)
zvals_rhot_cd = np.arange(0.0, 1.05, 0.05) # dense grid of Z

#logp_res_rhot_cms_aqua, s_res_rhot_cms_aqua = np.load('%s/cms/rhot_base_z_aqua_extended_hg.npy' % CURR_DIR)
logp_res_rhot_cms_aqua, s_res_rhot_cms_aqua = np.load('%s/cms/rhot_base_z_aqua_cms_hg_updated.npy' % CURR_DIR)
# Dense grid cms+hg
logp_res_rhot_cms_aqua, s_res_rhot_cms_aqua = np.load('%s/cms/rhot_base_z_aqua_cms_lows_highs_extended.npy' % CURR_DIR)

logp_res_rhot_cms_nohg_aqua, s_res_rhot_cms_nohg_aqua = np.load('%s/cms/rhot_base_z_aqua_extended_nohg.npy' % CURR_DIR)

#logp_res_rhot_cd_aqua, s_res_rhot_cd_aqua = np.load('%s/cd/rhot_base_z_aqua_extended.npy' % CURR_DIR)

logp_res_rhot_cd_aqua, s_res_rhot_cd_aqua = np.load('%s/cd/rhot_base_z_aqua_cd_lows_highs_extended.npy' % CURR_DIR)

logp_res_rhot_cd_ice, s_res_rhot_cd_ice = np.load('%s/cd/rhot_base_z_aqua_cd_lows_ice_dense.npy' % CURR_DIR)

logp_res_rhot_scvh_aqua, s_res_rhot_scvh_aqua = np.load('%s/scvh/rhot_base_z_aqua_extended_new.npy' % CURR_DIR)

logp_res_rhot_mls_aqua, s_res_rhot_mls_aqua = np.load('%s/mls/rhot_base_z_aqua_extended.npy' % CURR_DIR)

logp_res_rhot_mh13_aqua, s_res_rhot_mh13_aqua = np.load('%s/mh13/rhot_base_z_aqua_extended.npy' % CURR_DIR)

get_p_rgi_rhot_cms = RGI((logrhovals_rhot_cd, logtvals_rhot_cd, yvals_rhot_cd, zvals_rhot_cd), logp_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_cms = RGI((logrhovals_rhot_cd, logtvals_rhot_cd, yvals_rhot_cd, zvals_rhot_cd), s_res_rhot_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_cms_nohg = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_cms_nohg = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_cd = RGI((logrhovals_rhot_cd, logtvals_rhot_cd, yvals_rhot_cd, zvals_rhot_cd), logp_res_rhot_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_cd = RGI((logrhovals_rhot_cd, logtvals_rhot_cd, yvals_rhot_cd, zvals_rhot_cd), s_res_rhot_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_cd_ice = RGI((logrhovals_rhot_cd_ice, logtvals_rhot_cd_ice, yvals_rhot_cd_ice, zvals_rhot_cd_ice), logp_res_rhot_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_cd_ice = RGI((logrhovals_rhot_cd_ice, logtvals_rhot_cd_ice, yvals_rhot_cd_ice, zvals_rhot_cd_ice), s_res_rhot_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_scvh = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_scvh = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_mls = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_mls = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_mh13 = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), logp_res_rhot_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_mh13 = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot, zvals_rhot), s_res_rhot_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_lgrho):
                return float(get_p_rgi_rhot_cms(np.array([_lgrho, _lgt, _y_call, _z]).T))
            else:
                return get_p_rgi_rhot_cms(np.array([_lgrho, _lgt, _y_call, _z]).T)
        else:
            if np.isscalar(_lgrho):
                return float(get_p_rgi_rhot_cms_nohg(np.array([_lgrho, _lgt, _y_call, _z]).T))
            else:
                return get_p_rgi_rhot_cms_nohg(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_cd(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_p_rgi_rhot_cd(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_cd_ice(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_p_rgi_rhot_cd_ice(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_p_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_mls(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_p_rgi_rhot_mls(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_lgrho):
            return float(get_p_rgi_rhot_mh13(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_p_rgi_rhot_mh13(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_p_rhot(_lgrho, _lgt, _y_call))
        else:
            return ideal_xy.get_p_rhot(_lgrho, _lgt, _y_call)

    else:
        raise Exception('Only cms, scvh, or mls available for now.')

def get_s_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_lgrho):
                return float(get_s_rgi_rhot_cms(np.array([_lgrho, _lgt, _y_call, _z]).T))
            else:
                return get_s_rgi_rhot_cms(np.array([_lgrho, _lgt, _y_call, _z]).T)
        else:
            if np.isscalar(_lgrho):
                return float(get_s_rgi_rhot_cms_nohg(np.array([_lgrho, _lgt, _y_call, _z]).T))
            else:
                return get_s_rgi_rhot_cms_nohg(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_cd(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_s_rgi_rhot_cd(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_cd_ice(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_s_rgi_rhot_cd_ice(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_s_rgi_rhot_scvh(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_mls(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_s_rgi_rhot_mls(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_lgrho):
            return float(get_s_rgi_rhot_mh13(np.array([_lgrho, _lgt, _y_call, _z]).T))
        else:
            return get_s_rgi_rhot_mh13(np.array([_lgrho, _lgt, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_s_rhot(_lgrho, _lgt, _y_call))
        else:
            return ideal_xy.get_s_rhot(_lgrho, _lgt, _y_call)

    else:
        raise Exception('Only cms, scvh, or mls available for now.')

def get_sp_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    logp_rhot = get_p_rhot_tab(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    return get_s_pt(logp_rhot, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot), logp_rhot

##### S, Rho #####

svals_srho = np.arange(3.0, 9.1, 0.05)
svals_srho_cms = np.arange(4.0, 9.05, 0.05)

logrhovals_srho = np.linspace(-5.0, 2.0, 100)
yvals_srho = np.arange(0.05, 0.95, 0.1)
zvals_srho = np.arange(0, 1.0, 0.1)

#svals_srho_cms = np.arange(3.0, 9.1, 0.1) # for dense Z table
yvals_srho_cms = np.arange(0.05, 0.95, 0.05)
zvals_srho_cms = np.arange(0, 0.95, 0.05)
#yvals_srho_cms = np.arange(0.05, 0.9, 0.05)
#zvals_srho_cms = np.arange(0, 0.92, 0.02)

# FOR ICE CD TABLES
# svals_srho_cd_ice = np.arange(1.5, 8.1, 0.1)
# logrhovals_srho_cd_ice = np.linspace(-3.0, 2.0, 60)
# yvals_srho_cd_ice = np.arange(0.05, 0.35, 0.01)
# zvals_srho_cd_ice = np.arange(0, 1.05, 0.05)

# svals_srho_cd_ice = np.arange(1.5, 8.1, 0.1)
# logrhovals_srho_cd_ice = np.linspace(-3.0, 2.0, 60)
# yvals_srho_cd_ice = np.arange(0.05, 1.05, 0.1)
# zvals_srho_cd_ice = np.arange(0, 1.05, 0.05)

svals_srho_cd_ice = np.arange(1.5, 8.6, 0.1)
logrhovals_srho_cd_ice = np.linspace(-3.0, 2.0, 70)
yvals_srho_cd_ice = np.arange(0.05, 1.05, 0.1)
zvals_srho_cd_ice = np.arange(0, 1.05, 0.05)

# EXTENDED HIGH S AND LOW S GRIDS
svals_srho_cd = np.arange(1.5, 10.1, 0.1)
logrhovals_srho_cd = np.linspace(-4.0, 2.0, 75)
yvals_srho_cd = np.arange(0.02, 1.0, 0.05)
zvals_srho_cd = np.arange(0, 1.05, 0.05)



# logp_res_srho_cms_aqua, logt_res_srho_cms_aqua = np.load('%s/cms/srho_base_z_aqua_cms_hg_updated.npy' % CURR_DIR)
#logp_res_srho_cms_aqua, logt_res_srho_cms_aqua = np.load('%s/cms/srho_base_z_aqua_cms_lows_highs_extended.npy' % CURR_DIR)
logp_res_srho_cms_aqua, logt_res_srho_cms_aqua = np.load('%s/cms/srho_base_z_aqua_cms_lows_highs_extended_pbased.npy' % CURR_DIR)

logp_res_srho_cms_nohg_aqua, logt_res_srho_cms_nohg_aqua = np.load('%s/cms/srho_base_z_aqua_extended_nohg.npy' % CURR_DIR)

#logp_res_srho_cd_aqua, logt_res_srho_cd_aqua = np.load('%s/cd/srho_base_z_aqua_extended.npy' % CURR_DIR)
#logp_res_srho_cd_aqua, logt_res_srho_cd_aqua = np.load('%s/cd/srho_base_z_aqua_cd_lows_highs_extended.npy' % CURR_DIR)

logp_res_srho_cd_aqua, logt_res_srho_cd_aqua = np.load('%s/cd/srho_base_z_aqua_cd_lows_highs_extended_pbased.npy' % CURR_DIR)

#logp_res_srho_cd_ice, logt_res_srho_cd_ice = np.load('%s/cd/srho_base_z_aqua_cd_lows_ice_dense.npy' % CURR_DIR)
logp_res_srho_cd_ice, logt_res_srho_cd_ice = np.load('%s/cd/srho_base_z_aqua_cd_lows_ice_dense_yextended.npy' % CURR_DIR)

logp_res_srho_scvh_aqua, logt_res_srho_scvh_aqua = np.load('%s/scvh/srho_base_z_aqua_extended_new.npy' % CURR_DIR)

logp_res_srho_mls_aqua, logt_res_srho_mls_aqua = np.load('%s/mls/srho_base_z_aqua_extended.npy' % CURR_DIR)

logp_res_srho_mh13_aqua, logt_res_srho_mh13_aqua = np.load('%s/mh13/srho_base_z_aqua_extended.npy' % CURR_DIR)

get_p_rgi_srho_cms = RGI((svals_srho_cd, logrhovals_srho_cd, yvals_srho_cd, zvals_srho_cd), logp_res_srho_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_cms = RGI((svals_srho_cd, logrhovals_srho_cd, yvals_srho_cd, zvals_srho_cd), logt_res_srho_cms_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_cms_nohg = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_cms_nohg = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_cms_nohg_aqua, method='linear', \
            bounds_error=False, fill_value=None)

# get_p_rgi_srho_cd = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_cd_aqua, method='linear', \
#             bounds_error=False, fill_value=None)
# get_t_rgi_srho_cd = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_cd_aqua, method='linear', \
#             bounds_error=False, fill_value=None)

get_p_rgi_srho_cd = RGI((svals_srho_cd, logrhovals_srho_cd, yvals_srho_cd, zvals_srho_cd), logp_res_srho_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_cd = RGI((svals_srho_cd, logrhovals_srho_cd, yvals_srho_cd, zvals_srho_cd), logt_res_srho_cd_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_cd_ice = RGI((svals_srho_cd_ice, logrhovals_srho_cd_ice, yvals_srho_cd_ice, zvals_srho_cd_ice), logp_res_srho_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_cd_ice = RGI((svals_srho_cd_ice, logrhovals_srho_cd_ice, yvals_srho_cd_ice, zvals_srho_cd_ice), logt_res_srho_cd_ice, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_scvh = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_scvh = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_scvh_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_mls = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_mls = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_mls_aqua, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_mh13 = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logp_res_srho_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_mh13 = RGI((svals_srho, logrhovals_srho, yvals_srho, zvals_srho), logt_res_srho_mh13_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_s):
                return float(get_p_rgi_srho_cms(np.array([_s, _lgrho, _y_call, _z]).T))
            else:
                return get_p_rgi_srho_cms(np.array([_s, _lgrho, _y_call, _z]).T)

        else:
            if np.isscalar(_s):
                return float(get_p_rgi_srho_cms_nohg(np.array([_s, _lgrho, _y_call, _z]).T))
            else:
                return get_p_rgi_srho_cms_nohg(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_cd(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_p_rgi_srho_cd(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_cd_ice(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_p_rgi_srho_cd_ice(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_scvh(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_p_rgi_srho_scvh(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_mls(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_p_rgi_srho_mls(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_s):
            return float(get_p_rgi_srho_mh13(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_p_rgi_srho_mh13(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_p_srho(_s, _lgrho, _y_call))
        else:
            return ideal_xy.get_p_srho(_s, _lgrho, _y_call)

def get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    if hhe_eos == 'cms':
        if hg:
            if np.isscalar(_s):
                return float(get_t_rgi_srho_cms(np.array([_s, _lgrho, _y_call, _z]).T))
            else:
                return get_t_rgi_srho_cms(np.array([_s, _lgrho, _y_call, _z]).T)
        else:
            if np.isscalar(_s):
                return float(get_t_rgi_srho_cms_nohg(np.array([_s, _lgrho, _y_call, _z]).T))
            else:
                return get_t_rgi_srho_cms_nohg(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'cd':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_cd(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_t_rgi_srho_cd(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'cd_ice':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_cd_ice(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_t_rgi_srho_cd_ice(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'scvh':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_scvh(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_t_rgi_srho_scvh(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'mls':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_mls(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_t_rgi_srho_mls(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'mh13':
        if np.isscalar(_s):
            return float(get_t_rgi_srho_mh13(np.array([_s, _lgrho, _y_call, _z]).T))
        else:
            return get_t_rgi_srho_mh13(np.array([_s, _lgrho, _y_call, _z]).T)

    elif hhe_eos == 'ideal':
        if np.isscalar(_s):
            return float(ideal_xy.get_t_srho(_s, _lgrho, _y_call))
        else:
            return ideal_xy.get_t_srho(_s, _lgrho, _y_call)

def get_pt_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True, y_tot=True):
    return get_p_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos=z_eos, hg=hg), get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos=z_eos, hg=hg)

def get_u_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', hg=True, tab=True, y_tot=True):
    if tab:
        _lgp, _lgt = get_p_srho_tab(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot), get_t_srho_tab(_s, _lgrho , _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        _lgt = get_t_srho(_s, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
        _lgp = get_p_rhot(_lgrho, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)
    return get_u_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos)

def get_s_ad(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=True, tab=True, y_tot=True):
    """This function returns the entropy value
    required for nabla - nabla_a = 0 at
    pressure and temperature profiles"""

    if y_tot:
        _y /= (1 - _z)

    guess = get_s_pt(_lgp, _lgt, _y, _z, hhe_eos=hhe_eos, hg=hg, z_eos=z_eos) * erg_to_kbbar

    sol = root(err_grad, guess, tol=1e-8, method='hybr', args=(_lgp, _y, _z, hhe_eos, hg, tab))
    return sol.x

############################### Derivatives ###############################

def get_dudy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, y_tot=True):

    u1 = 10**get_u_srho_tab(_s, _lgrho, _y - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    u2 = 10**get_u_srho_tab(_s, _lgrho, _y + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (u2 - u1)/(2 * dy)

def get_dudz_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, hg=True, y_tot=True):

    u1 = 10**get_u_srho_tab(_s, _lgrho, _y, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    u2 = 10**get_u_srho_tab(_s, _lgrho, _y, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (u2 - u1)/(2 * dz)

# du/ds_(rho, Y) = T test
def get_duds_rhoy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', order=1, ds=0.1, hg=True, tab=True, y_tot=True):

    S0 = _s/erg_to_kbbar
    S1 = S0*(1-ds)
    S2 = S0*(1+ds)

    U0 = 10**get_u_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)
    U1 = 10**get_u_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)
    U2 = 10**get_u_srho_tab(S2*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)

    if order == 2:
        return (U2 - U1)/(S2 - S1)
    elif order == 1:
        return (U2 - U0)/(S2 - S0)
    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

def get_dudrho_sy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', order=1, drho=0.1, hg=True, tab=True, y_tot=True):
    R0 = 10**_lgrho
    R1 = R0*(1-drho)
    R2 = R0*(1+drho)
    #rho1 = np.log10((10**rho)*(1+drho))
    U0 = 10**get_u_srho_tab(_s, np.log10(R0), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)
    U1 = 10**get_u_srho_tab(_s, np.log10(R1), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)
    U2 = 10**get_u_srho_tab(_s, np.log10(R2), _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, tab=tab, y_tot=y_tot)

    if order == 2:
        return (U2 - U1)/((1/R1) - (1/R2))
    elif order == 1:
        return (U2 - U0)/((1/R0) - (1/R2))
    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

# DS/DX|_rho, P - DERIVATIVES NECESSARY FOR THE LEDOUX CONDITION
def get_dpds_rhoy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', ds=1e-3, hg=True, y_tot=True, tab=True):

    if tab:
        p1 = 10**get_p_srho_tab(_s - ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        p2 = 10**get_p_srho_tab(_s + ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        if y_tot:
            _y_call = _y / (1 - _z)
        else:
            _y_call = _y
        p1 = 10**get_p_srho(_s - ds, _lgrho, _y_call, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        p2 = 10**get_p_srho(_s + ds, _lgrho, _y_call, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return (p2 - p1) / (2 * ds / erg_to_kbbar)

def get_dpdy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, y_tot=True, tab=True):

    if tab:
        p1 = 10**get_p_srho_tab(_s, _lgrho, _y - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        p2 = 10**get_p_srho_tab(_s, _lgrho, _y + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        if y_tot:
            _y_call = _y / (1 - _z)
        else:
            _y_call = _y
        p1 = 10**get_p_srho(_s, _lgrho, _y_call - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        p2 = 10**get_p_srho(_s, _lgrho, _y_call + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return (p2 - p1) / (2 * dy)


def get_dpdz_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, hg=True, y_tot=True, tab=True):


    if tab:
        p1 = 10**get_p_srho_tab(_s, _lgrho, _y, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        p2 = 10**get_p_srho_tab(_s, _lgrho, _y, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        if y_tot:
            _y_call = _y / (1 - _z)
        else:
            _y_call = _y
        p1 = 10**get_p_srho(_s, _lgrho, _y_call, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        p2 = 10**get_p_srho(_s, _lgrho, _y_call, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return (p2 - p1) / (2 * dz)

def get_dsdy_rhop_srho(_s, _lgrho, _y, _z, 
                        hhe_eos, z_eos='aqua', 
                        ds=1e-3, dy=1e-3,
                        hg=True, polyfit=False,
                         y_tot=True, tab=True):
    #dPdS|{rho, Y, Z}:
    dpds_rhoy_srho = get_dpds_rhoy_srho(_s, _lgrho, _y, _z, 
                                        hhe_eos=hhe_eos, ds=ds, hg=hg,
                                        y_tot=y_tot, tab=tab)
    #dPdZ|{S, rho, Y}:
    dpdy_srho = get_dpdy_srho(_s, _lgrho, _y, _z,
                            hhe_eos=hhe_eos, dy=dy, hg=hg,
                            y_tot=y_tot, tab=tab)

    #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
    dsdy_rhopy = -dpdy_srho/dpds_rhoy_srho # triple product rule
    if polyfit:
        return smooth.get_polyfit(dsdy_rhopy)
    else:
        return dsdy_rhopy


def get_dsdz_rhop_srho(_s, _lgrho, _y, _z, 
                        hhe_eos, z_eos='aqua', 
                        ds=1e-3, dz=1e-3,
                        hg=True, polyfit=False,
                         y_tot=True, tab=True):
    #dPdS|{rho, Y, Z}:
    dpds_rhoy_srho = get_dpds_rhoy_srho(_s, _lgrho, _y, _z, 
                                    hhe_eos=hhe_eos, ds=ds, hg=hg,
                                    y_tot=y_tot, tab=tab)
    #dPdZ|{S, rho, Y}:
    dpdz_srho = get_dpdz_srho(_s, _lgrho, _y, _z,
                            hhe_eos=hhe_eos, dz=dz, hg=hg,
                            y_tot=y_tot, tab=tab)

    #dSdZ|{rho, P, Y} = -dPdZ|{S, rho, Y} / dPdS|{rho, Y, Z}
    dsdz_rhopy = -dpdz_srho/dpds_rhoy_srho # triple product rule
    if polyfit:
        return smooth.get_polyfit(dsdz_rhopy)
    else:
        return dsdz_rhopy

# DS/DX|_P, T - DERIVATIVES NECESSARY FOR THE SCHWARZSCHILD CONDITION
def get_dsdy_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, polyfit=False, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    s1 = get_s_pt(_lgp, _lgt, _y_call - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    s2 = get_s_pt(_lgp, _lgt, _y_call + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    dsdy_pt = (s2 - s1)/(2 * dy)

    if polyfit:
        return smooth.get_polyfit(dsdy_pt)
    else:
        return dsdy_pt

def get_dsdz_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, hg=True, polyfit=False, y_tot=True):
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    s1 = get_s_pt(_lgp, _lgt, _y_call, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    s2 = get_s_pt(_lgp, _lgt, _y_call, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    dsdz_pt = (s2 - s1)/(2 * dz)

    if polyfit:
        return smooth.get_polyfit(dsdz_pt)
    else:
        return dsdz_pt


def get_c_s(_s, _lgp, _y, _z,hhe_eos, z_eos='aqua', order=1, dp=0.1, hg=True, y_tot=True):
    P0 = 10**_lgp
    P1 = P0*(1-dp)
    P2 = P0*(1+dp)
    R0 = get_rho_sp_tab(_s, np.log10(P0), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    R1 = get_rho_sp_tab(_s, np.log10(P1), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    R2 = get_rho_sp_tab(_s, np.log10(P2), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    if order == 2:
        return np.sqrt((P2 - P1)/(10**R2 - 10**R1))
    elif order == 1:
        return np.sqrt((P2 - P0)/(10**R2 - 10**R0))
    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

def get_dtdrho_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', order=1, drho=0.01, hg=True, y_tot=True):
    R0 = 10**_lgrho
    R1 = R0*(1-drho)
    R2 = R0*(1+drho)

    T0 = 10**get_t_srho_tab(_s, np.log10(R0), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T1 = 10**get_t_srho_tab(_s, np.log10(R1), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T2 = 10**get_t_srho_tab(_s, np.log10(R2), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    if order == 2:
        return (T2 - T1)/(R2 - R1)
    elif order == 1:
        return (T2 - T0)/(R2 - R0)
    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

def get_dtds_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', order=1, ds=0.01, hg=True, y_tot=True):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1-ds)
    S2 = S0*(1+ds)

    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T2 = 10**get_t_srho_tab(S2*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    if order == 2:
        return (T2 - T1)/(S2 - S1)
    elif order == 1:
        return (T2 - T0)/(S2 - S0)
    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

def get_dtds_sp(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', order=1, ds=0.01, hg=True, y_tot=True):
    S0 = _s/erg_to_kbbar
    S1 = S0*(1-ds)
    S2 = S0*(1+ds)

    T0 = 10**get_t_sp_tab(S0*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T1 = 10**get_t_sp_tab(S1*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T2 = 10**get_t_sp_tab(S2*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    if order == 2:
        return (T2 - T1)/(S2 - S1)
    elif order == 1:
        return (T2 - T0)/(S2 - S0)

    else:
        raise Exception('Only order = 1 or order = 2 allowed!')

def get_dtdy_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, y_tot=True):

    t1 = 10**get_t_srho_tab(_s, _lgrho, _y - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    t2 = 10**get_t_srho_tab(_s, _lgrho, _y + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (t2 - t1)/(2 * dy)

def get_dtdz_srho(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, hg=True, y_tot=True):

    t1 = 10**get_t_srho_tab(_s, _lgrho, _y, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    t2 = 10**get_t_srho_tab(_s, _lgrho, _y, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (t2 - t1)/(2 * dz)


def get_dlogp_dy_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, y_tot=True):
    # this is Chi_Y
    lgp1 = get_p_rhot_tab(_lgrho, _lgt, _y - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    lgp2 = get_p_rhot_tab(_lgrho, _lgt, _y + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return ((lgp2 - lgp1) * np.log(10))/(2 * dy)

def get_dlogp_dz_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, hg=True, y_tot=True):
    # this is Chi_Z
    lgp1 = get_p_rhot_tab(_lgrho, _lgt, _y, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    lgp2 = get_p_rhot_tab(_lgrho, _lgt, _y, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return ((lgp2 - lgp1) * np.log(10))/(2 * dz)

def get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos='aqua', dt=1e-2, hg=True, y_tot=True):
    # this is Chi_T
    lgp1 = get_p_rhot_tab(_lgrho, _lgt - dt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    lgp2 = get_p_rhot_tab(_lgrho, _lgt + dt, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (lgp2 - lgp1)/(2 * dt)

def get_dlogt_dy_rhop_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, dt=1e-2, hg=True, y_tot=True):
    # This is Chi_Y/Chi_T
    # To be used in the By term of Ledoux condition

    Chi_Y = get_dlogp_dy_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos=z_eos, dy=dy, hg=hg, y_tot=y_tot)
    Chi_T = get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos=z_eos, dt=dt, hg=hg, y_tot=y_tot)


    return Chi_Y/Chi_T

def get_dlogt_dz_rhop_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos='aqua', dz=1e-3, dt=1e-2, hg=True, y_tot=True):
    # This is Chi_Z/Chi_T
    # To be used in the Bz term of Ledoux condition

    Chi_Z = get_dlogp_dz_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos=z_eos, dz=dz, hg=hg, y_tot=y_tot)
    Chi_T = get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, hhe_eos, z_eos=z_eos, dt=dt, hg=hg, y_tot=y_tot)


    return Chi_Z/Chi_T


def get_drhodt_py(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dt=1e-3, hg=True, y_tot=True):
    # Chi_T/Chi_rho
    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    lgrho1 = get_rho_pt(_lgp, _lgt - dt, _y_call, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    lgrho2 = get_rho_pt(_lgp, _lgt + dt, _y_call, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return (lgrho2 - lgrho1)/(2 * dt)

def get_drhods_py(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', ds=1e-3, hg=True, y_tot=True):
    # (dlnrho / dS)_PY, where S is in cgs units (but input `_s` is in kbbar)
    lgrho2 = get_rho_sp_tab(_s + ds, _lgp, _y, _z,
        hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    lgrho1 = get_rho_sp_tab(_s - ds, _lgp, _y, _z,
        hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    return ((lgrho2 - lgrho1) * np.log(10)) / (2 * ds / erg_to_kbbar)

def get_drhody_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dy=1e-3, hg=True, y_tot=True):

    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    lgrho2 = get_rho_pt(_lgp, _lgt, _y_call + dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    lgrho1 = get_rho_pt(_lgp, _lgt, _y_call - dy, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return ((lgrho2 - lgrho1) * np.log(10))/(dy)

def get_drhodz_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dz=1e-4, hg=True, y_tot=True):

    if y_tot:
        _y_call = _y / (1 - _z)
    else:
        _y_call = _y

    lgrho2 = get_rho_pt(_lgp, _lgt, _y_call, _z + dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
    lgrho1 = get_rho_pt(_lgp, _lgt, _y_call, _z - dz, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return ((lgrho2 - lgrho1) * np.log(10))/(dz)

def get_c_v(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', ds=1e-3, tab=True, hg=True, smooth_gauss=False, base_sigma=5, base_window=10, y_tot=True):
    # ds/dlnT_{_lgrho, Y}

    if tab:
        #T0 = get_t_srho_tab(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt1 = get_t_srho_tab(_s - ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        lgt2 = get_t_srho_tab(_s + ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        #T0 = get_t_srho(S0*erg_to_kbbar, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt1 = get_t_srho(_s - ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt2 = get_t_srho(_s + ds, _lgrho, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    cv = (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))
    if smooth_gauss:
        return smooth.gauss_smooth(cv, base_sigma=base_sigma, base_window=base_window)
    else:
        return cv

def get_c_p(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', ds=1e-3, tab=True, hg=True, y_tot=True):
    # ds/dlnT_{P, Y}
    if tab:
        #T0 = get_t_sp_tab(S0*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt1 = get_t_sp_tab(_s - ds, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        lgt2 = get_t_sp_tab(_s + ds, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        #T0 = get_t_sp(S0*erg_to_kbbar, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt1 = get_t_sp(_s - ds, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)
        lgt2 = get_t_sp(_s + ds, _lgp, _y, _z, hhe_eos=hhe_eos, z_eos=z_eos, hg=hg)

    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))

def get_gamma1(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', dp = 0.01, hg=True, y_tot=True):
    # dlnP/dlnrho_S, Y, Z = dlogP/dlogrho_S, Y, Z
    #if tab:
    R0 = get_rho_sp_tab(_s, _lgp, _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    R1 = get_rho_sp_tab(_s, _lgp*(1-dp), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    R2 = get_rho_sp_tab(_s, _lgp*(1+dp), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)

    return (2*_lgp*dp)/(R2 - R1)

def get_nabla_ad(_s, _lgp, _y, _z, hhe_eos, z_eos='aqua', dp=1e-2, tab=True, hg=True, y_tot=True):
    if tab:
        lgt1 = get_t_sp_tab(_s, _lgp - dp, _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
        lgt2 = get_t_sp_tab(_s, _lgp + dp, _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    else:
        lgt1 = get_t_sp(_s, _lgp - dp, _y, _z, hhe_eos, z_eos=z_eos, hg=hg)
        lgt2 = get_t_sp(_s, _lgp + dp, _y, _z, hhe_eos, z_eos=z_eos, hg=hg)
    return (lgt2 - lgt1)/(2 * dp)

def get_gruneisen(_s, _lgrho, _y, _z, hhe_eos, z_eos='aqua', drho = 0.01, hg=True, y_tot=True):
    T0 = get_t_srho_tab(_s, _lgrho, _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T1 = get_t_srho_tab(_s, _lgrho*(1-drho), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    T2 = get_t_srho_tab(_s, _lgrho*(1+drho), _y, _z, hhe_eos, z_eos=z_eos, hg=hg, y_tot=y_tot)
    return (T2 - T1)/(2*_lgrho*drho)

def get_K(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dp = 0.01, hg=True, y_tot=True):

    if y_tot:
        _y /= (1 - _z)
    P0 = 10**_lgp
    P1 = P0*(1+dp)
    R0 = 10**get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=hg)
    R1 = 10**get_rho_pt(np.log10(P1), _lgt, _y, _z, hhe_eos, z_eos='aqua', hg=hg)

    return -R0*(P1 - P0)/(R1 - R0)

def get_alpha(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua', dt=0.1, hg=True, y_tot=True):
    '''
    Coefficient of thermal expansion
    '''

    if y_tot:
        _y /= (1 - _z)

    T0 = 10**_lgt
    T1 = T0*(1+dt)
    R0 = 10**get_rho_pt(_lgp, _lgt, _y, _z, hhe_eos, z_eos='aqua')
    R1 = 10**get_rho_pt(_lgp, np.log10(T1), _y, _z, hhe_eos, z_eos='aqua')
    return R0*((1/R1 - 1/R0)/(T1 - T0))

def hhe_thermo_consist(s, p, y, hhe_eos):
    if hhe_eos == 'cms':
        xy_eos = cms_eos
    elif hhe_eos == 'cd':
        xy_eos = cd_eos
    elif hhe_eos == 'scvh':
        xy_eos = scvh_eos

    logrho, logt = xy_eos.get_rhot_sp_tab(s, p, y)

    logt_test = np.log10(xy_eos.get_duds_rhoy_srho(s, logrho, y, ds=0.01))
    logp_test = np.log10(xy_eos.get_dudrho_sy_srho(s, logrho, y, drho=0.01))

    err_T = (logt - logt_test)/logt
    err_P = (p - logp_test)/p

    return err_T, err_P
