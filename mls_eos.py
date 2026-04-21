import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
#from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy.constants import m_p
#from numba import jit
import os
from eos import ideal_eos, aqua_eos
import pdb

"""
    This file provides thermodynamic quantities and derivatives for the Mazevet, Licari, and Soubiran (2022) H-He EOS.

    The functions rely on precomputed tables, and such functions end with _tab. 
    The functions used to invert the tables are also available.

    All independnet thermodynamic quantities are ordered in the following manner in the function arguments: s, rho, p, t, y, z
    Therefore, all functions will follow the same ordering convention.
    All functions have the dependent_independent naming convenction; e.g., get_rho_pt is \rho(P, T, Y), get_t_srho is T(s, rho, Y).

    Pressure is in log 10 in dyn/cm^2
    Temperature is in log 10 K
    density is in log 10 g/cm^3
    input entropy is in kb/baryon
    output entropy is in erg/g/K
    internal energy is in erg/g
    Y is the helium mass fraciton

    The entropy of mixing corrections by Howard & Guillot (2023) are added in get_s_pt. 

    Author: Roberto Tejada Arevalo

"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

pd.options.mode.chained_assignment = None
ideal_z = ideal_eos.IdealEOS(m=18)
ideal_xy = ideal_eos.IdealHHeMix()
ideal_x = ideal_eos.IdealEOS(m=2)
mz = 18
mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mh = 1 #* amu.value
mhe = 4.0026



###### reading tables ######

def cms_reader(tab_name):

    cols = ['logt','logp','logrho','logu','logs','dlrho/dlT_P','dlrho/dlP_T','dlS/dlT_P',
           'dlS/dlP_T','grad_ad']

    tab = np.loadtxt('%s/cms/DirEOS2019/%s' % (CURR_DIR, tab_name), comments='#')
    tab_df = pd.DataFrame(tab, columns=cols)
    data = tab_df[(tab_df['logt'] <= 5) & (tab_df['logt'] != 2.8)]# ROB: increased to 6.0 to test wider range for brenth
    #data = tab_df

    data['logp'] += 10 # 1 GPa = 1e10 cgs
    data['logu'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g
    data['logs'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g

    return data

def grid_data(df):
    # grids data for interpolation
    twoD = {}
    shape = df['logt'].nunique(), -1
    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD

cms_hdata = grid_data(cms_reader('TABLE_H_TP_v1'))
cms_hedata = grid_data(cms_reader('TABLE_HE_TP_v1'))

data_hc = pd.read_csv('%s/cms/HG23_Vmix_Smix.csv' % CURR_DIR, delimiter=',')
data_hc = data_hc[(data_hc['LOGT'] <= 5.0) & (data_hc['LOGT'] != 2.8)]
data_hc = data_hc.rename(columns={'LOGT':'logt', 'LOGP':'logp'}).sort_values(by=['logt', 'logp'])

grid_hc = grid_data(data_hc)
svals_hc = grid_hc['Smix']

logpvals_hc = grid_hc['logp'][0]
logtvals_hc = grid_hc['logt'][:,0]

# smix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Smix']) # Smix will be in cgs... not log cgs.
# vmix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Vmix'])

smix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Smix'], method='linear', bounds_error=False, fill_value=None) # Smix will be in cgs... not log cgs.
vmix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Vmix'], method='linear', bounds_error=False, fill_value=None)

def smix_interp(lgt, lgp):
    return smix_interp_rgi(np.array([lgt, lgp]).T)
def vmix_interp(lgt, lgp):
    return vmix_interp_rgi(np.array([lgt, lgp]).T)

# interpolations

logpvals = cms_hdata['logp'][0]
logtvals = cms_hdata['logt'][:,0]

logpvals_he = cms_hedata['logp'][0]
logtvals_he = cms_hedata['logt'][:,0]

#### H data ####

rhovals_h, svals_h, loguvals_h = np.load('%s/mls/mls_h_pt.npy' % CURR_DIR)

get_s_h_rgi = RGI((logtvals, logpvals), svals_h.T, method='linear', bounds_error=False, fill_value=None)
get_rho_h_rgi = RGI((logtvals, logpvals), rhovals_h.T, method='linear', bounds_error=False, fill_value=None)
get_logu_h_rgi = RGI((logtvals, logpvals), loguvals_h.T, method='linear', bounds_error=False, fill_value=None)

def get_s_h(lgt, lgp):
    return get_s_h_rgi(np.array([lgt, lgp]).T)
def get_rho_h(lgt, lgp):
    return get_rho_h_rgi(np.array([lgt, lgp]).T)
def get_logu_h(lgt, lgp):
    return get_logu_h_rgi(np.array([lgt, lgp]).T)

#### He data ####

svals_he = cms_hedata['logs']
rhovals_he = cms_hedata['logrho']
loguvals_he = cms_hedata['logu']

get_s_he_rgi = RGI((logtvals_he, logpvals_he), svals_he, method='linear', bounds_error=False, fill_value=None)
get_rho_he_rgi = RGI((logtvals_he, logpvals_he), rhovals_he, method='linear', bounds_error=False, fill_value=None)
get_logu_he_rgi = RGI((logtvals_he, logpvals_he), loguvals_he, method='linear', bounds_error=False, fill_value=None)

def get_s_he(lgt, lgp):
    return get_s_he_rgi(np.array([lgt, lgp]).T)
def get_rho_he(lgt, lgp):
    return get_rho_he_rgi(np.array([lgt, lgp]).T)
def get_logu_he(lgt, lgp):
    return get_logu_he_rgi(np.array([lgt, lgp]).T)

###### H-He mixing ######

def Y_to_n(Y):
    return ((Y/mhe)/(((1 - Y)/mh) + (Y/mhe)))
def n_to_Y(x):
    return (mhe * x)/(1 + 3.0026*x)

def x_H(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (1-Y)*(1-Z)/mh/Ntot

def x_Z(Y, Z, mz):
    Ntot = (1-Y)*(1-Z)/mh + (Y*(1-Z)/mhe) + Z/mz
    return (Z/mz)/Ntot

def guarded_log(x): # necessary for entropy of mixing calculations
    if np.isscalar(x):
        if x == 0:
            return 0
        elif x  < 0:
            raise ValueError('a')
        return x * np.log(x)

    return np.array([guarded_log(x_) for x_ in x])

### isolating the ideal and interacting entropy of mixing terms from HG23 ###

def get_smix_id_y(Y):
    #smix_hg23 = smix_interp.ev(lgt, lgp)*(1 - Y)*Y
    xhe = Y_to_n(Y)
    xh = 1 - xhe
    q = mh*xh + mhe*xhe
    return -1*(guarded_log(xh) + guarded_log(xhe)) / q

def get_smix_nd(Y, lgp, lgt): # non-ideal component of the entropy of mixing

    # the HG23 \Delta S_mix is the combination of the non-ideal and ideal entropy of mixing
    smix_hg23 = smix_interp(lgt, lgp)*(1 - Y)*Y
    smix_id = get_smix_id_y(Y) / erg_to_kbbar

    return smix_hg23 - smix_id

def get_s_pt(lgp, lgt, y, z=0.0, hg=True):
    s_h = get_s_h(lgt, lgp)
    s_he = 10 ** get_s_he(lgt, lgp)
    if hg:
        smix = smix_interp(lgt, lgp)*(1 - y)*y
    else:
        smix = get_smix_id_y(y) / erg_to_kbbar
    return (1 - y) * s_h + y * s_he + smix #

def get_rho_pt(lgp, lgt, y, z=0.0, hg=True):
    rho_h = 10 ** get_rho_h(lgt, lgp)
    rho_he = 10 ** get_rho_he(lgt, lgp)
    if hg:
        vmix = vmix_interp(lgt, lgp)
    else:
        vmix = 0.0
    return np.log10(1/(((1 - y) / rho_h) + (y / rho_he) + vmix*(1 - y)*y))

def get_u_pt(lgp, lgt, y, z=0.0):
    u_h = 10**get_logu_h(lgt, lgp) # MJ/kg to erg/g
    u_he = 10**get_logu_he(lgt, lgp)
    return np.log10((1 - y)*u_h + y*u_he)

###### inverted tables ######

logrho_res_sp, logt_res_sp = np.load('%s/mls/sp_base_comb_hg_corr.npy' % CURR_DIR)

# svals_sp = np.arange(5.25, 10.1, 0.05) # old mls22 grid
# logpvals_sp = np.arange(5.5, 14, 0.05)
# yvals_sp = np.arange(0.05, 1.05, 0.05)

svals_sp = np.arange(5.5, 10.05, 0.05)
logpvals_sp = np.arange(5.5, 14.05, 0.05)
yvals_sp = np.arange(0.05, 1.0, 0.05)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, y, z=0.0):
    if np.isscalar(s):
        return float(get_rho_rgi_sp(np.array([s, p, y]).T))
    else:
        return get_rho_rgi_sp(np.array([s, p, y]).T)

def get_t_sp_tab(s, p, y, z=0.0):
    if np.isscalar(s):
        return float(get_t_rgi_sp(np.array([s, p, y]).T))
    else:
        return get_t_rgi_sp(np.array([s, p, y]).T)

def get_rhot_sp_tab(s, p, y, z=0.0):
    return get_rho_sp_tab(s, p, y), get_t_sp_tab(s, p, y)

### P(rho, T, Y), s(rho, T, Y) tables ###

logp_res_rhot, s_res_rhot = np.load('%s/mls/rhot_base_comb_hg_corr.npy' % CURR_DIR)

logrhovals_rhot = np.arange(-5, 1.5, 0.05)
logtvals_rhot = np.arange(2.1, 5.1, 0.05)
yvals_rhot = np.arange(0.05, 1.0, 0.05)

get_p_rhot_rgi = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rhot_rgi = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, y, z=0.0):
    if np.isscalar(rho):
        return float(get_p_rhot_rgi(np.array([rho, t, y]).T))
    else:
        return get_p_rhot_rgi(np.array([rho, t, y]).T)

def get_s_rhot_tab(rho, t, y, z=0.0):
    if np.isscalar(rho):
        return float(get_s_rhot_rgi(np.array([rho, t, y]).T))
    else:
        return get_s_rhot_rgi(np.array([rho, t, y]).T)



### P(s, rho, Y), T(s, rho, Y) tables ###
logp_res_srho, logt_res_srho = np.load('%s/mls/srho_base_comb_hg_corr.npy' % CURR_DIR)

svals_srho = np.arange(5.5, 9.05, 0.05)
logrhovals_srho = np.linspace(-4.5, 2.0, 100)
yvals_srho = np.arange(0.05, 1.0, 0.05)

get_p_rgi = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, r, y, z=0.0):
    if np.isscalar(s):
        return float(get_p_rgi(np.array([s, r, y]).T))
    else:
        return get_p_rgi(np.array([s, r, y]).T)

def get_t_srho_tab(s, r, y, z=0.0):
    if np.isscalar(s):
        return float(get_t_rgi(np.array([s, r, y]).T))
    else:
        return get_t_rgi(np.array([s, r, y]).T)


### error functions ###

def err_p_srho(lgp, lgr, s_val, y):
    t = get_t_srho(s_val, lgr, y)
    s_ = get_s_pt(lgp, t, y)
    s_val /= erg_to_kbbar
    return (s_/s_val) - 1

def err_t_sp(logt, s_val, logp, y):
    s_ = get_s_pt(logp, logt, y)*erg_to_kbbar
    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval, yval):
    logrho = get_rho_pt(lgp, lgtval, yval)
    return (logrho/rhoval) - 1

def err_p_srho(lgp, sval, rhoval, yval):
    logrho = get_rho_sp_tab(sval, lgp, yval) # circumvents temperature
    return (logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval, yval):
    #logp = get_p_rhot(rhoval, lgt, yval, alg='root')
    s_test = get_s_rhot_tab(rhoval, lgt, yval)*erg_to_kbbar
    return (s_test/sval) - 1

def err_t_rhop(_lgt, _lgrho, _lgp, _y):
    logrho_test = get_rho_pt(_lgp, _lgt, _y)
    return (logrho_test/_lgrho) - 1

### inversion functions ###

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-8
    
###### Temperature ######
def get_t_sp(s, p, y, hg=True, alg='root', z_eos=None):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
            guess = ideal_xy.get_t_sp(s, p, y)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y))
            return float(sol.x.item())
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, p, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
                return sol.root
            except:
                raise
        sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
        return sol

def get_t_srho(s, rho, y, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho, y = np.array([s]), np.array([rho]), np.array([y])
            guess = ideal_xy.get_t_srho(s, rho, y)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
            return float(sol.x.item())
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise
        sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
        return sol

def get_t_rhop(_lgrho, _lgp, _y, alg='root'):
    if alg == 'root':
        if np.isscalar(_lgrho):
            _lgrho, _lgp, _y = np.array([_lgrho]), np.array([_lgp]), np.array([_y])
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
            sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y))
            return float(sol.x.item())

        guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(_lgrho, _lgp, _y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(_lgrho):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_rhop, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(_lgrho, _lgp, _y))
                return sol.root
            except:
                raise
        sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(_lgrho, _lgp, _y)])
        return sol

###### Density ######
def get_rhot_sp(s, p, y, z=0.0):
    rho, t = get_rhot_sp_tab(s, p, y)
    return rho, t
    

###### Pressure ######
def get_p_rhot(rho, t, y, alg='root'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t, y = np.array([rho]), np.array([t]), np.array([y])
            guess = ideal_x.get_p_rhot(rho, t, y)
            sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y))
            return float(sol.x.item())
        guess = ideal_x.get_p_rhot(rho, t, y)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t, y))
                return sol.root
            except:
                #print('rho={}, t={}, y={}'.format(rho, t, y))
                raise
        sol = np.array([get_p_rhot(rho_, t_, y_) for rho_, t_, y_ in zip(rho, t, y)])
        return sol

def get_p_srho(s, rho, y, alg='root'):
    logt = get_t_srho(s, rho, y, alg)
    return get_p_rhot_tab(rho, logt, y)


def get_sp_rhot(rho, t, y):
    logp = get_p_rhot_tab(rho, t, y)
    s = get_s_pt(logp, t, y)
    return s, logp

def get_pt_srho(s, rho, y):
    return get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)

def get_u_sp(s, p, y):
    t = get_t_sp(s, p, y)
    return get_u_pt(p, t, y)

def get_u_rhot(rho, t, y):
    p = get_p_rhot_tab(rho, t, y) 
    return get_u_pt(p, t, y)

def get_u_srho(s, rho, y, z=0.0):
    p, t = get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)
    return get_u_pt(p, t, y)

def get_s_rhop(rho, p, y):
    t = get_t_rhop(rho, p, y)
    #y = cms.n_to_Y(y)
    return get_s_pt(p, t, y)

############## derivatives ##############


### entropy gradients ###

def get_dsdy_rhop(rho, p, y, dy=0.1):
    S0 = get_s_rhop(rho, p, y)
    S1 = get_s_rhop(rho, p, y*(1+dy))

    return (S1 - S0)/(y*dy)

def get_dsdy_rhop_srho(s, rho, y, z=0.0, ds=0.1, dy=0.1, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    if not tab:
        P0 = 10**get_p_srho(S0*erg_to_kbbar, rho, y)
        P1 = 10**get_p_srho(S1*erg_to_kbbar, rho, y)
        P2 = 10**get_p_srho(S0*erg_to_kbbar, rho, y*(1+dy))
    else: 
        P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y)
        P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y)
        P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y*(1+dy))      
    
    dpds_rhoy = (P1 - P0)/(S1 - S0)
    dpdy_srho = (P2 - P0)/(y*dy)

    return -dpdy_srho/dpds_rhoy


def get_dsdy_rhot(rho, t, y, z=0.0, dy=0.01):
    S0 = get_s_rhot_tab(rho, t, y)
    S1 = get_s_rhot_tab(rho, t, y*(1+dy))

    dsdy = (S1 - S0)/(y*dy)
    return dsdy

def get_dsdy_pt(p, t, y, z=0.0, dy=0.01):
    S0 = get_s_pt(p, t, y)
    S1 = get_s_pt(p, t, y*(1+dy))

    return (S1 - S0)/(y*dy)

def get_dsdt_ry_rhot(rho, t, y, z=0.0, dt=0.1):
    T0 = 10**t
    T1 = T0*(1+dt)
    S0 = get_s_rhot_tab(rho, np.log10(T0), y)
    S1 = get_s_rhot_tab(rho, np.log10(T1), y)

    return (S1 - S0)/(T1 - T0)

def get_c_s(s, p, y, z=0.0, dp=0.1):
    P0 = 10**p
    P1 = P0*(1+dp)
    R0 = get_rho_sp_tab(s, np.log10(P0), y)
    R1 = get_rho_sp_tab(s, np.log10(P1), y)

    return np.sqrt((P1 - P0)/(10**R1 - 10**R0))

def get_c_v(s, rho, y, z=0.0, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho, y)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho, y)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, y, z=0.0, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p, y)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p, y)

    return (S1 - S0)/(T1 - T0)

### pressure gradients ###

def get_dpdt_rhot(rho, t, y, z=0.0, dT=0.01):
    T0 = 10**t
    T1 = T0*(1+dT)
    P0 = get_p_rhot_tab(rho, np.log10(T0), y)
    P1 = get_p_rhot_tab(rho, np.log10(T1), y)
    return (P1 - P0)/(T1 - T0)

def get_gamma1(s, p, y, z=0.0, dp = 0.01):
    R0 = get_rho_sp_tab(s, p, y)
    R1 = get_rho_sp_tab(s, p*(1+dp), y)
    return (p*dp)/(R1 - R0)


### energy gradients ###

# to get chemical potential:
def get_dudy_srho(s, rho, y, z=0.0, dy=0.1):
    U0 = 10**get_u_srho(s, rho, y)
    U1 = 10**get_u_srho(s, rho, y*(1+dy))
    return (U1 - U0)/(y*dy)

# du/ds_(rho, Y) = T test
def get_duds_rhoy_srho(s, rho, y, z=0.0, ds=0.1):
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)
    U0 = 10**get_u_srho(S1*erg_to_kbbar, rho, y)
    U1 = 10**get_u_srho(S2*erg_to_kbbar, rho, y)
    return (U1 - U0)/(S1*ds)

def get_dudrho_sy_srho(s, rho, y, z=0.0, drho=0.1):
    R1 = 10**rho
    R2 = R1*(1+drho)
    #rho1 = np.log10((10**rho)*(1+drho))
    U0 = 10**get_u_srho(s, np.log10(R1), y)
    U1 = 10**get_u_srho(s, np.log10(R2), y)
    #return (U1 - U0)/(R1*drho)
    return (U1 - U0)/((1/R1) - (1/R2))

def get_dudrho_rhot(rho, t, y, z=0.0, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    U0 = 10**get_u_rhot(rho, t, y)
    U1 = 10**get_u_rhot(np.log10(R1), t, y)
    return (U1 - U0)/(R1 - R0)

### density gradients ###

def get_drhods_py(s, p, y, z=0.0, ds=0.01):
    
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)

    rho0 = 10**get_rhot_sp(S1*erg_to_kbbar, p, y)[0]
    rho1 = 10**get_rhot_sp(S2*erg_to_kbbar, p, y)[0]

    drhods = (rho1 - rho0)/(S2 - S1)

    return drhods


def get_drhodt_py(p, t, y, z=0.0, dt=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt(p, t, y)
    rho1 = get_rho_pt(p, t*(1+dt), y)

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

### temperature gradients ###

def get_dtdy_sp(s, p, y, z=0.0, dy=0.01):
    # t0 = get_t_sp(s, p, y)
    # t1 = get_t_sp(s, p, y*(1+dy))
    T0 = 10**get_t_sp_tab(s, p, y) # this was returning dlogT/dY before
    T1 = 10**get_t_sp_tab(s, p, y*(1+dy))

    dtdy = (T1 - T0)/(y*dy)
    return dtdy

def get_dtdy_srho(s, rho, y, z=0.0, dy=0.1, tab=True):
    if not tab:
        T0 = 10**get_t_srho(s, rho, y)
        T1 = 10**get_t_srho(s, rho, y*(1+dy))
    else:
        T0 = 10**get_t_srho_tab(s, rho, y)
        T1 = 10**get_t_srho_tab(s, rho, y*(1+dy)) 

    return (T1 - T0)/(y*dy)

def get_dtdy_rhop(rho, p, y, z=0.0, dy=0.01):
    t0 = 10**get_t_rhop(rho, p, y)
    t1 = 10**get_t_rhop(rho, p, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

def get_dtdrho_sy_srho(s, rho, y, z=0.0, drho = 0.01, tab=True): # dlogT/dlogrho_{s, Y}
    R0 = 10**rho
    R1 = R0*(1+drho)
    if not tab:
        T0 = 10**get_t_srho(s, np.log10(R0), y)
        T1 = 10**get_t_srho(s, np.log10(R1), y)
    else:
        T0 = 10**get_t_srho_tab(s, np.log10(R0), y)
        T1 = 10**get_t_srho_tab(s, np.log10(R1), y)
    return (T1 - T0)/(R1 - R0)

def get_dtds_rhoy_srho(s, rho, y, z=0.0, ds=0.01, tab=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    if not tab:
        T0 = 10**get_t_srho(S0*erg_to_kbbar, rho, y)
        T1 = 10**get_t_srho(S1*erg_to_kbbar, rho, y)
    else:
        T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho, y)
        T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho, y)
    return (T1 - T0)/(S1 - S0)

def get_nabla_ad(s, p, y, z=0.0, dp=0.01):
    T0 = get_t_sp_tab(s, p, y)
    T1 = get_t_sp_tab(s, p*(1+dp), y)
    return (T1 - T0)/(p*dp)

def get_gruneisen(s, rho, y, z=0.0, drho = 0.01):
    T0 = get_t_srho_tab(s, rho, y)
    T1 = get_t_srho_tab(s, rho*(1+drho), y)
    return (T1 - T0)/(rho*drho)