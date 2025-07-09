import numpy as np
import pandas as pd
import os
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from eos import ideal_eos
import warnings
warnings.filterwarnings("ignore")

"""
    This file provides access to the AQUA table (Haldemann et al. 2020).
    The AQUA table accounts for different phases of water.

    As with the H-He tables, this file reads pre-computed inverted tables and
    includes the inversion functions used to produce the tables.

    The pre-computed table function names end with _tab.

    Authors: Roberto Tejada Arevalo

"""

mp = amu.to('g')
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
Pa_to_dyn = u.Pa.to('dyn/cm^2')
SI_to_cgs = (u.kg/u.meter**3).to('g/cm^3')
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_cgs = (u.J/u.kg).to('erg/g')

ideal_z = ideal_eos.IdealEOS(m=18)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def aqua_reader(basis):
    if basis == 'pt':
        cols = ['press', 'temp', 'rho', 'grada', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase']
        tab = np.loadtxt('%s/aqua/aqua_eos_%s_v1_0.dat' % (CURR_DIR, basis), skiprows=19)
    elif basis == 'rhot':
        cols = ['rho', 'temp', 'press', 'grada', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase']
        tab = np.loadtxt('%s/aqua/aqua_eos_%s_v1_0.dat' % (CURR_DIR, basis), skiprows=21)
    elif basis == 'rhou':
        cols = ['rho', 'u', 'press', 'grada', 's', 'temp', 'c', 'mmw', 'x_ion', 'x_d', 'phase']
        tab = np.loadtxt('%s/aqua/aqua_eos_%s_v1_0.dat' % (CURR_DIR, basis), skiprows=21)


    tab_df = pd.DataFrame(tab, columns=cols)

    tab_df['logp'] = np.log10(tab_df['press']*Pa_to_dyn) # in dyn/cm2
    tab_df['logrho'] = np.log10(tab_df['rho']*SI_to_cgs) # in g/cm3
    tab_df['logt'] = np.log10(tab_df['temp'])
    tab_df['s'] = tab_df['s']*J_to_erg # in erg/g/K
    tab_df['logu'] = np.log10(tab_df['u']*J_to_cgs)

    return tab_df


def grid_data(df, basis):
    # grids data for interpolation
    twoD = {}
    if basis == 'pt':
        shape = df['logp'].nunique(), -1
    elif basis == 'rhot' or basis == 'rhou':
        shape = df['logrho'].nunique(), -1

    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD


### P, T ####
aqua_data_pt = grid_data(aqua_reader('pt'), 'pt') # 2D grid

logpvals_pt = aqua_data_pt['logp'][:,0]
logtvals = aqua_data_pt['logt'][0]

svals_pt = aqua_data_pt['s']
logrhovals_pt = aqua_data_pt['logrho']
loguvals_pt = aqua_data_pt['logu']

s_rgi_pt = RGI((logpvals_pt, logtvals), svals_pt, method='linear', \
            bounds_error=False, fill_value=None)

rho_rgi = RGI((logpvals_pt, logtvals), logrhovals_pt, method='linear', \
            bounds_error=False, fill_value=None)

u_rgi_pt = RGI((logpvals_pt, logtvals), loguvals_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_pt_tab(lgp, lgt):
    if np.isscalar(lgp):
        return float(s_rgi_pt(np.array([lgp, lgt]).T))
    return s_rgi_pt(np.array([lgp, lgt]).T)

def get_logrho_pt_tab(lgp, lgt):
    if np.isscalar(lgp):
        return float(rho_rgi(np.array([lgp, lgt]).T))
    return rho_rgi(np.array([lgp, lgt]).T)

def get_logu_pt_tab(lgp, lgt):
    if np.isscalar(lgp):
        return float(u_rgi_pt(np.array([lgp, lgt]).T))
    return u_rgi_pt(np.array([lgp, lgt]).T)

### rho, T ###

aqua_data_rhot = grid_data(aqua_reader('rhot'), basis='rhot')
logrhovals_rhot = aqua_data_rhot['logrho'][:,0]
logtvals_rhot = aqua_data_rhot['logt'][0]

svals_rhot = aqua_data_rhot['s']
logpvals_rhot = aqua_data_rhot['logp']
loguvals_rhot = aqua_data_rhot['logu']

s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot), svals_rhot, method='linear', \
            bounds_error=False, fill_value=None)

p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot), logpvals_rhot, method='linear', \
            bounds_error=False, fill_value=None)

u_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot), loguvals_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_rhot_tab(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(s_rgi_rhot(np.array([lgrho, lgt]).T))
    return s_rgi_rhot(np.array([lgrho, lgt]).T)

def get_logp_rhot_tab(lgrho, lgt):
    if np.isscalar(lgrho):
        float(p_rgi_rhot(np.array([lgrho, lgt]).T))
    return p_rgi_rhot(np.array([lgrho, lgt]).T)

def get_logu_rhot_tab(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(u_rgi_rhot(np.array([lgrho, lgt]).T))
    return u_rgi_rhot(np.array([lgrho, lgt]).T)

# ### rho, U ###

aqua_data_rhou = grid_data(aqua_reader('rhou'), basis='rhou')
logrhovals_rhou = aqua_data_rhou['logrho'][:,0]
loguvals_rhou = aqua_data_rhou['logu'][0]

svals_rhou = aqua_data_rhou['s']
logpvals_rhou = aqua_data_rhou['logp']
logtvals_rhou = aqua_data_rhou['logt']

s_rgi_rhou = RGI((logrhovals_rhou, loguvals_rhou), svals_rhou, method='linear', \
            bounds_error=False, fill_value=None)

p_rgi_rhou = RGI((logrhovals_rhou, loguvals_rhou), logpvals_rhou, method='linear', \
            bounds_error=False, fill_value=None)

t_rgi_rhou = RGI((logrhovals_rhou, loguvals_rhou), logtvals_rhou, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_rhou_tab(lgrho, lgu):
    if np.isscalar(lgrho):
        return float(s_rgi_rhou(np.array([lgrho, lgu]).T))
    return s_rgi_rhou(np.array([lgrho, lgu]).T)

def get_p_rhou_tab(lgrho, lgu):
    if np.isscalar(lgrho):
        float(p_rgi_rhou(np.array([lgrho, lgu]).T))
    return p_rgi_rhou(np.array([lgrho, lgu]).T)

def get_t_rhou_tab(lgrho, lgu):
    if np.isscalar(lgrho):
        return float(t_rgi_rhou(np.array([lgrho, lgu]).T))
    return t_rgi_rhou(np.array([lgrho, lgu]).T)

####### Inverted Functions #######

### rho(s, p), t(s, p) ###

logrho_res_sp, logt_res_sp = np.load('%s/aqua/sp_base.npy' % CURR_DIR)

svals_sp = np.arange(0.1, 10.05, 0.05)
logpvals_sp = np.arange(5, 16, 0.1)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_logrho_sp_tab(s, p):
    if np.isscalar(s):
        return float(get_rho_rgi_sp(np.array([s, p]).T))
    else:
        return get_rho_rgi_sp(np.array([s, p]).T)

def get_logt_sp_tab(s, p):
    if np.isscalar(s):
        return float(get_t_rgi_sp(np.array([s, p]).T))
    else:
        return get_t_rgi_sp(np.array([s, p]).T)

def get_rhot_sp_tab(s, p):
    return get_logrho_sp_tab(s, p), get_logt_sp_tab(s, p)

### p(s, rho), T(s, rho) ###

logp_res_srho, logt_res_srho = np.load('%s/aqua/srho_base.npy' % CURR_DIR)

svals_srho = np.arange(1.2, 2.2, 0.01)
logrhovals_srho= np.arange(0.1, 2.11, 0.01)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_logp_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, r]).T))
    else:
        return get_p_rgi_srho(np.array([s, r]).T)

def get_logt_srho_tab(s, r):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, r]).T))
    else:
        return get_t_rgi_srho(np.array([s, r]).T)

####### inversion functions #######

TBOUNDS = [2, 5]
PBOUNDS = [0, 16]
XTOL = 1e-8

def err_t_sp(logt, s_val, logp):
    s_ = get_s_pt_tab(logp, logt)*erg_to_kbbar

    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval):
    #if zval > 0.0:
    logrho = get_logrho_pt_tab(lgp, lgtval)
    #pdb.set_trace()
    return logrho/rhoval - 1

def err_t_srho(lgt, sval, rhoval):
    lgp = get_p_rhot_tab(rhoval, lgt)
    s_ = get_s_pt_tab(lgp, lgt)*erg_to_kbbar
    return  s_/sval - 1

def get_p_rhot(rho, t ,alg='brenth'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t = np.array([rho]), np.array([t])
        guess = ideal_z.get_p_rhot(rho, t, 0)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_) for rho_, t_ in zip(rho, t)])
        return sol

def get_t_sp(s, p, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, p = np.array([s]), np.array([p])
        guess = ideal_z.get_t_sp(s, p, 0)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, p))
                return sol.root
            except:
                #print('s={}, p={}, y={}'.format(s, p, y))
                raise
        sol = np.array([get_t_sp(s_, p_) for s_, p_ in zip(s, p)])
        return sol

def get_t_srho(s, rho, alg='brenth'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho = np.array([s]), np.array([rho])
        guess = ideal_z.get_t_srho(s, rho, 0)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho))
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
