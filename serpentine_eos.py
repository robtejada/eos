import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos, aneos
import os

from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

"""
    This module borrows from the alice repository from Christopher 
    Mankovich (https://github.com/chkvch/alice) to provide access
    to the ANEOS Serpentine EOS in the same format as all other EOSes.

    Author: Roberto Tejada Arevalo

"""

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_fe = ideal_eos.IdealEOS(m=56)

aneos_serp = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='serpentine')

def get_s_pt_tab(p, t):
    if np.isscalar(p):
        return float((10**aneos_serp.get_logs(p, t)/11605.333))
    return 10**aneos_serp.get_logs(p, t)/11605.333

def get_rho_pt_tab(p, t):
    if np.isscalar(p):
        return float(aneos_serp.get_logrho(p, t))
    return aneos_serp.get_logrho(p, t)

def get_u_pt_tab(p, t):
    if np.isscalar(p):
        return float(aneos_serp.get_logu(p, t))
    return aneos_serp.get_logu(p, t)

#### S, P ####

sgrid = np.arange(0.0006, 1.1, 0.003)

logrho_res_sp, logt_res_sp = np.load('%s/aneos/aneos_serpentine_sp_base.npy' % CURR_DIR)

get_t_sp_rgi = RGI((sgrid, aneos_serp.logpvals), logt_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)
get_rho_sp_rgi = RGI((sgrid, aneos_serp.logpvals), logrho_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)

def get_t_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_t_sp_rgi(np.array([s, p]).T)) # in log GPa for now
    return get_t_sp_rgi(np.array([s, p]).T)

def get_rho_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_rho_sp_rgi(np.array([s, p]).T)) # in log GPa for now
    return get_rho_sp_rgi(np.array([s, p]).T)

#### S, rho ####

logrhovals_srho = np.arange(0.4, 1.5, 0.01)

logp_res_srho, logt_res_srho = np.load('%s/aneos/aneos_serpentine_srho_base.npy' % CURR_DIR)

get_p_rgi_srho = RGI((sgrid, logrhovals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((sgrid, logrhovals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, r]).T))
    else:
        return get_p_rgi_srho(np.array([s, r]).T)

def get_t_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, r]).T))
    else:
        return get_t_rgi_srho(np.array([s, r]).T)

#### error functions ####

def err_t_sp(logt, s_val, logp, alg):
    s_ = get_s_pt_tab(logp, logt)*erg_to_kbbar
   # s_val /= erg_to_kbbar # in cgs
    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval):
    #if np.any(zval) > 0.0:
        #sval = float(get_s_ptz(float(lgp), lgtval, yval, zval, z_eos = z_eos))*erg_to_kbbar
    logrho = get_rho_pt_tab(lgp, lgtval)
    #pdb.set_trace()
    return (logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval, alg):
    #sval = sval /erg_to_kbbar
    #lgp = get_p_rhot(rval, lgt, y)
    #if np.any(zval) > 0:
    lgp = get_p_rhot(rhoval, lgt, alg)
    #logrho = get_rho_sp(sval, lgp, alg)
    s_ = get_s_pt_tab(lgp, lgt)*erg_to_kbbar

    return (s_/sval) - 1

#### inversions ####

def get_t_sp(s, p, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, p = np.array([s]), np.array([p])
            guess = ideal_fe.get_t_sp(s, p, 0)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, alg))
            return float(sol.x.item())

        guess = ideal_fe.get_t_sp(s, p, 0)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            #try:
            sol = root_scalar(err_t_sp, bracket=[0, 8], xtol=1e-8, method='brenth', args=(s, p, alg)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
            return sol.root
            #except:
            #    raise
        sol = np.array([get_t_sp(s_, p_) for s_, p_ in zip(s, p)])
        return sol

def get_rho_sp(s, p, alg):
    t_ = get_t_sp(s, p, alg)
    return get_rho_pt_tab(p, t)

def get_p_rhot(rho, t, alg='brenth'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t = np.array([rho]), np.array([t])
            guess = ideal_fe.get_p_rhot(rho, t, 0)
            sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t))
            return float(sol.x.item())

        guess = ideal_fe.get_p_rhot(rho, t, 0)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t,))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=[6, 17], xtol=1e-8, method='brenth', args=(rho, t))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_) for rho_, t_ in zip(rho, t)])
        return sol

def get_t_srho(s, rho, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho = np.array([s]), np.array([rho])
            guess = ideal_fe.get_t_srho(s, rho, 0)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, alg))
            return float(sol.x.item())
        guess = ideal_fe.get_t_srho(s, rho, 0)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, alg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=[0, 8], xtol=1e-8, method='brenth', args=(s, rho, alg))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise
        sol = np.array([get_t_srho(s_, rho_) for s_, rho_ in zip(s, rho)])
        return sol

############## derivatives ##############

def get_dtdrho_srho(s, rho, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(s, np.log10(R0))
    T1 = 10**get_t_srho_tab(s, np.log10(R1))

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(s, rho, ds=0.01):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho)

    return (T1 - T0)/(S1 - S0)

def get_c_v(s, rho, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p)

    return (S1 - S0)/(T1 - T0)