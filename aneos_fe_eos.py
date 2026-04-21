import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos
import os

from astropy import units as u
#from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_fe = ideal_eos.IdealEOS(m=56)


# data gathered from Sarah Stewart's github: https://github.com/ststewart/aneos-iron-2020/

#### rho, T ####
logrhovals, logtvals = np.load('eos/aneos/aneos_fe_logrho.npy'), np.load('eos/aneos/aneos_fe_logt.npy')
logpvals, svals, loguvals = np.load('eos/aneos/aneos_fe_res.npy')

rgi_test_p = RGI((logtvals, logrhovals), logpvals, method='linear', bounds_error=False, fill_value=None)

rgi_test_s = RGI((logtvals, logrhovals), svals, method='linear', bounds_error=False, fill_value=None)

rgi_test_u = RGI((logtvals, logrhovals), loguvals, method='linear', bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(rgi_test_p(np.array([t, rho]).T).item()) # in log GPa for now
    return rgi_test_p(np.array([t, rho]).T)

def get_s_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(rgi_test_s(np.array([t, rho]).T).item())
    return rgi_test_s(np.array([t, rho]).T)

def get_u_rhot_tab(rho, t):
    if np.isscalar(rho):
        return float(rgi_test_u(np.array([t, rho]).T).item())
    return rgi_test_u(np.array([t, rho]).T)

#### P, T ####

logpgrid = np.arange(6, 15, 0.1)

logrho_res_pt, s_res_pt, logu_res_pt = np.load('%s/aneos/aneos_fe_pt_base.npy' % CURR_DIR)

get_rho_pt_rgi = RGI((logpgrid, logtvals), logrho_res_pt, \
                    method='linear', bounds_error=False, fill_value=None)
get_s_pt_rgi = RGI((logpgrid, logtvals), s_res_pt, \
                    method='linear', bounds_error=False, fill_value=None)

get_u_pt_rgi = RGI((logpgrid, logtvals), logu_res_pt, \
                    method='linear', bounds_error=False, fill_value=None)

def get_rho_pt_tab(p, t):
    if np.isscalar(p):
        return float(get_rho_pt_rgi(np.array([p, t]).T)) 
    return get_rho_pt_rgi(np.array([p, t]).T)

def get_s_pt_tab(p, t):
    if np.isscalar(p):
        return float(get_s_pt_rgi(np.array([p, t]).T))
    return get_s_pt_rgi(np.array([p, t]).T)

def get_u_pt_tab(p, t):
    if np.isscalar(p):
        return float(get_u_pt_rgi(np.array([p, t]).T)) 
    return get_u_pt_rgi(np.array([p, t]).T)

#### S, P ####

sgrid = np.arange(1000, 26000, 100)*erg_to_kbbar

logrho_res_sp, logt_res_sp = np.load('%s/aneos/aneos_fe_sp_base.npy' % CURR_DIR)

get_t_sp_rgi = RGI((sgrid, logpgrid), logt_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)
get_rho_sp_rgi = RGI((sgrid, logpgrid), logrho_res_sp, \
                    method='linear', bounds_error=False, fill_value=None)

def get_t_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_t_sp_rgi(np.array([s, p]).T)) 
    return get_t_sp_rgi(np.array([s, p]).T)

def get_rho_sp_tab(s, p):
    if np.isscalar(p):
        return float(get_rho_sp_rgi(np.array([s, p]).T)) 
    return get_rho_sp_rgi(np.array([s, p]).T)

def get_rhot_sp_tab(s, p):
    return get_rho_sp_tab(s, p), get_t_sp_tab(s, p)

#### S, rho ####

logp_res_srho, logt_res_srho = np.load('%s/aneos/aneos_fe_srho_base.npy' % CURR_DIR)

get_p_rgi_srho = RGI((sgrid, logrhovals), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((sgrid, logrhovals), logt_res_srho, method='linear', \
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

def err_rho_pt(lgrho, p_val, t_val):
    logp_ = get_p_rhot_tab(lgrho, t_val)
    return (logp_/p_val) - 1

def err_t_sp(logt, s_val, logp, alg):
    s_ = get_s_pt(logp, logt, alg)*erg_to_kbbar
   # s_val /= erg_to_kbbar # in cgs
    return (s_/s_val) - 1

def err_t_srho(lgt, sval, lgr):
    s_ = get_s_rhot_tab(lgr, lgt)*erg_to_kbbar
    return (s_/sval) - 1

#### inversions ####

def get_rho_pt(p, t, alg='brenth'):
    if alg == 'root':
        if np.isscalar(p):
            p, t = np.array([p]), np.array([t])
            guess = ideal_fe.get_rho_pt(p, t, 0)
            sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t))
            return float(sol.x.item())
        guess = ideal_fe.get_rho_pt(p, t, 0)
        sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(p):
            try:
                sol = root_scalar(err_rho_pt, bracket=[-20, 1.5], xtol=1e-8, method='brenth', args=(p, t))
                return sol.root
            except:
                #print('p={}, t={}, y={}'.format(p, t, y))
                raise

        sol = np.array([get_rho_pt(p_, t_) for p_, t_ in zip(p, t)])
        return sol

def get_s_pt(p, t, alg):
    rho = get_rho_pt(p, t, alg)
    #rho, T = get_rhot_sp()
    return get_s_rhot_tab(rho, t)

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

def get_t_srho(s, rho, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho = np.array([s]), np.array([rho])
            guess = ideal_fe.get_t_srho(s, rho, 0)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho))
            return float(sol.x.item())
        guess = ideal_fe.get_t_srho(s, rho, 0)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=[0, 8], xtol=1e-8, method='brenth', args=(s, rho))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise
        sol = np.array([get_t_srho(s_, rho_) for s_, rho_ in zip(s, rho)])
        return sol