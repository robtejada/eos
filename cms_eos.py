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
from eos import ideal_eos
import pdb

"""
    This file provides thermodynamic quantities and derivatives for the Chabrier, Mazevet, and Soubiran (2019) H-He EOS.

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

ideal_xy = ideal_eos.IdealHHeMix()
ideal_x = ideal_eos.IdealEOS(m=2)

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

data_hc = pd.read_csv('%s/cms/HG23_Vmix_Smix_Umix.csv' % CURR_DIR, delimiter=',')
data_hc = data_hc[(data_hc['LOGT'] <= 5.0) & (data_hc['LOGT'] != 2.8)]
data_hc = data_hc.rename(columns={'LOGT':'logt', 'LOGP':'logp'}).sort_values(by=['logt', 'logp'])

grid_hc = grid_data(data_hc)
svals_hc = grid_hc['Smix']
uvals_hc = grid_hc['Umix']

logpvals_hc = grid_hc['logp'][0]
logtvals_hc = grid_hc['logt'][:,0]

# smix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Smix']) # Smix will be in cgs... not log cgs.
# vmix_interp = RBS(logtvals_hc, logpvals_hc, grid_hc['Vmix'])

smix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Smix'], method='linear', bounds_error=False, fill_value=None) # Smix will be in cgs... not log cgs.
vmix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Vmix'], method='linear', bounds_error=False, fill_value=None)
umix_interp_rgi = RGI((logtvals_hc, logpvals_hc), grid_hc['Umix'], method='linear', bounds_error=False, fill_value=None)

def smix_interp(lgt, lgp):
    return smix_interp_rgi(np.array([lgt, lgp]).T)
def vmix_interp(lgt, lgp):
    return vmix_interp_rgi(np.array([lgt, lgp]).T)
def umix_interp(lgt, lgp):
    return umix_interp_rgi(np.array([lgt, lgp]).T)

# interpolations

logpvals = cms_hdata['logp'][0]
logtvals = cms_hdata['logt'][:,0]

logpvals = cms_hedata['logp'][0]
logtvals = cms_hedata['logt'][:,0]

#### H data ####

svals_h = cms_hdata['logs']
rhovals_h = cms_hdata['logrho']
loguvals_h = cms_hdata['logu']

# get_s_h = RBS(logtvals, logpvals, svals_h) # x or y are not changing so can leave out to speed things up
# get_rho_h = RBS(logtvals, logpvals, rhovals_h)
# get_logu_h = RBS(logtvals, logpvals, loguvals_h, s=0)

get_s_h_rgi = RGI((logtvals, logpvals), svals_h, method='linear', bounds_error=False, fill_value=None)
get_rho_h_rgi = RGI((logtvals, logpvals), rhovals_h, method='linear', bounds_error=False, fill_value=None)
get_logu_h_rgi = RGI((logtvals, logpvals), loguvals_h, method='linear', bounds_error=False, fill_value=None)

def get_s_h(lgt, lgp):
    return get_s_h_rgi(np.array([lgt, lgp]).T)
def get_rho_h(lgt, lgp):
    return get_rho_h_rgi(np.array([lgt, lgp]).T)
def get_logu_h(lgt, lgp):
    return get_logu_h_rgi(np.array([lgt, lgp]).T)

# derivatives

# get_rhot_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlT_P'])
# get_rhop_h = RBS(logtvals, logpvals, cms_hdata['dlrho/dlP_T'])
# get_sp_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlP_T'])
# get_st_h = RBS(logtvals, logpvals, cms_hdata['dlS/dlT_P'])

#### He data ####

svals_he = cms_hedata['logs']
rhovals_he = cms_hedata['logrho']
loguvals_he = cms_hedata['logu']

get_s_he_rgi = RGI((logtvals, logpvals), svals_he, method='linear', bounds_error=False, fill_value=None)
get_rho_he_rgi = RGI((logtvals, logpvals), rhovals_he, method='linear', bounds_error=False, fill_value=None)
get_logu_he_rgi = RGI((logtvals, logpvals), loguvals_he, method='linear', bounds_error=False, fill_value=None)

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

def guarded_log(x):
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

def get_smix_nd(Y, lgp, lgt):

    # the HG23 \Delta S_mix is the combination of the non-ideal and ideal entropy of mixing
    smix_hg23 = smix_interp(lgt, lgp)*(1 - Y)*Y
    smix_id = get_smix_id_y(Y) / erg_to_kbbar

    return smix_hg23 - smix_id

def get_s_pt(lgp, lgt, y, z=0.0, hg=True):
    s_h = 10 ** get_s_h(lgt, lgp)
    s_he = 10 ** get_s_he(lgt, lgp)
    if hg:
        smix = smix_interp(lgt, lgp)*(1 - y)*y
    else:
        smix = get_smix_id_y(y) / erg_to_kbbar

    return (1 - y) * s_h + y * s_he + smix

def get_rho_pt(lgp, lgt, y, z=0.0, hg=True):
    rho_h = 10 ** get_rho_h(lgt, lgp)
    rho_he = 10 ** get_rho_he(lgt, lgp)
    if hg:
        vmix = vmix_interp(lgt, lgp)
    else:
        vmix = 0.0
    return np.log10(1/(((1 - y) / rho_h) + (y / rho_he) + vmix*(1 - y)*y))

def get_u_pt(lgp, lgt, y, z=0.0, hg=True):
    u_h = 10**get_logu_h(lgt, lgp) # MJ/kg to erg/g
    u_he = 10**get_logu_he(lgt, lgp)
    if hg:
        umix = umix_interp(lgt, lgp)*(1 - y)*y
    else:
        umix = 0.0
    return np.log10((1 - y)*u_h + y*u_he + umix)

###### inverted tables ######

## t, rho (s, p, y) ##
"""To revert to the old version with the HG corrections, uncomment the first line.
All functions should be the same for ease of use."""


""" The commented values below are the old S, P basis. While the values are okay, the derivatives are suspect."""
# s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/cms/cms_hg_thermo.npy' % CURR_DIR)
# #s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, grada_arr = np.load('%s/cms/cms_thermo.npy' % CURR_DIR)

# get_rho_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)
# get_t_ = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)

# get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
# get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

# get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
# get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

# get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
# get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

# def get_rho_t(s, p, y):
#     return get_rho_(np.array([y, s, p]).T), get_t_(np.array([y, s, p]).T)

# def get_c_p(s, p, y):
#     cp_res = get_cp(np.array([y, s, p]).T)
#     return cp_res

# def get_c_v(s, p, y):
#     cv_res = get_cv(np.array([y, s, p]).T)
#     return cv_res

# def get_chi_rho(s, p, y):
#     chirho_res = get_chirho(np.array([y, s, p]).T)
#     return chirho_res

# def get_chi_t(s, p, y):
#     chit_res = get_chit(np.array([y, s, p]).T)
#     return chit_res

# def get_grad_ad(s, p, y):
#     grada = get_grada(np.array([y, s, p]).T)
#     return grada

#### S, P ####

#logrho_res_sp, logt_res_sp = np.load('%s/cms/sp_base_comb.npy' % CURR_DIR)
logrho_res_sp, logt_res_sp = np.load('%s/cms/sp_base_comb_hg_corr.npy' % CURR_DIR) #base_comb was last table
logrho_res_sp_nohg, logt_res_sp_nohg = np.load('%s/cms/sp_base_comb_nohg.npy' % CURR_DIR)

svals_sp = np.arange(5.5, 10.05, 0.05)
logpvals_sp = np.arange(5.5, 14.05, 0.05)
yvals_sp = np.arange(0.05, 1.0, 0.05)

get_rho_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logrho_res_sp, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp = RGI((svals_sp, logpvals_sp, yvals_sp), logt_res_sp, method='linear', \
            bounds_error=False, fill_value=None)

get_rho_rgi_sp_nohg = RGI((svals_sp, logpvals_sp, yvals_sp), logrho_res_sp_nohg, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_sp_nohg = RGI((svals_sp, logpvals_sp, yvals_sp), logt_res_sp_nohg, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, y, z=0.0, hg=True):
    if np.isscalar(s):
        if hg:
            return float(get_rho_rgi_sp(np.array([s, p, y]).T))
        else:
            return float(get_rho_rgi_sp_nohg(np.array([s, p, y]).T))
    else:
        if hg:
            return get_rho_rgi_sp(np.array([s, p, y]).T)
        else:
           return get_rho_rgi_sp_nohg(np.array([s, p, y]).T) 

def get_t_sp_tab(s, p, y, z=0.0, hg=True):
    if np.isscalar(s):
        if hg:
            return float(get_t_rgi_sp(np.array([s, p, y]).T))
        else:
            return float(get_t_rgi_sp_nohg(np.array([s, p, y]).T))
    else:
        if hg:
            return get_t_rgi_sp(np.array([s, p, y]).T)
        else:
           return get_t_rgi_sp_nohg(np.array([s, p, y]).T) 

def get_rhot_sp_tab(s, p, y, z=0.0, hg=True):
    return get_rho_sp_tab(s, p, y, hg=hg), get_t_sp_tab(s, p, y, hg=hg)


#### Rho, T ####

#logp_res_rhot, s_res_rhot = np.load('%s/cms/rhot_base_comb.npy' % CURR_DIR)
logp_res_rhot, s_res_rhot = np.load('%s/cms/rhot_base_comb_hg_corr.npy' % CURR_DIR)
logp_res_rhot_nohg, s_res_rhot_nohg = np.load('%s/cms/rhot_base_comb_nohg.npy' % CURR_DIR)

logrhovals_rhot = np.arange(-5, 1.5, 0.05)
logtvals_rhot = np.arange(2.1, 5.1, 0.05)
yvals_rhot = np.arange(0.05, 1.05, 0.05)

get_p_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), logp_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), s_res_rhot, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_rhot_nohg = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), logp_res_rhot_nohg, method='linear', \
            bounds_error=False, fill_value=None)
get_s_rgi_rhot_nohg = RGI((logrhovals_rhot, logtvals_rhot, yvals_rhot), s_res_rhot_nohg, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, y, z=0.0, hg=True):
    if np.isscalar(rho):
        if hg:
            return float(get_p_rgi_rhot(np.array([rho, t, y]).T))
        else:
            return float(get_p_rgi_rhot_nohg(np.array([rho, t, y]).T))
    else:
        if hg:
            return get_p_rgi_rhot(np.array([rho, t, y]).T)
        else:
            return get_p_rgi_rhot_nohg(np.array([rho, t, y]).T)

def get_s_rhot_tab(rho, t, y, z=0.0, hg=True):
    if np.isscalar(rho):
        if hg:
            return float(get_s_rgi_rhot(np.array([rho, t, y]).T))
        else:
            return float(get_s_rgi_rhot_nohg(np.array([rho, t, y]).T))
    else:
        if hg:
            return get_s_rgi_rhot(np.array([rho, t, y]).T)
        else:
            return get_s_rgi_rhot_nohg(np.array([rho, t, y]).T)


#### S, Rho ####
logp_res_srho, logt_res_srho = np.load('%s/cms/srho_base_comb_hg_corr.npy' % CURR_DIR)
logp_res_srho_nohg, logt_res_srho_nohg = np.load('%s/cms/srho_base_comb_nohg.npy' % CURR_DIR)

svals_srho = np.arange(5.5, 9.05, 0.05) # new grid
logrhovals_srho = np.linspace(-4.5, 2.0, 100)
yvals_srho = np.arange(0.05, 1.0, 0.05)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

get_p_rgi_srho_nohg = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho_nohg, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho_nohg = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho_nohg, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, rho, y, z=0.0, hg=True):
    if np.isscalar(s):
        if hg:
            return float(get_p_rgi_srho(np.array([s, rho, y]).T))
        else:
            return float(get_p_rgi_srho_nohg(np.array([s, rho, y]).T))
    else:
        if hg:
            return get_p_rgi_srho(np.array([s, rho, y]).T)
        else:
            return get_p_rgi_srho_nohg(np.array([s, rho, y]).T)

def get_t_srho_tab(s, rho, y, z=0.0, hg=True):
    if np.isscalar(s):
        if hg:
            return float(get_t_rgi_srho(np.array([s, rho, y]).T))
        else:
            return float(get_t_rgi_srho_nohg(np.array([s, rho, y]).T))
    else:
        if hg:
            return get_t_rgi_srho(np.array([s, rho, y]).T)
        else:
            return get_t_rgi_srho_nohg(np.array([s, rho, y]).T)

### error functions ###

def err_p_srho(lgp, lgr, s_val, y):
    t = get_t_srho(s_val, lgr, y)
    s_ = get_s_pt(lgp, t, y)
    s_val /= erg_to_kbbar
    return (s_/s_val) - 1

def err_t_sp(logt, s_val, logp, y, hg):
    s_ = get_s_pt(logp, logt, y, hg=hg)*erg_to_kbbar
    return (s_/s_val) - 1

def err_p_rhot(lgp, rhoval, lgtval, yval, hg):
    logrho = get_rho_pt(lgp, lgtval, yval, hg=hg)
    return (logrho/rhoval) - 1

def err_p_srho(lgp, sval, rhoval, yval):
    logrho = get_rho_sp_tab(sval, lgp, yval) # circumvents temperature
    return (logrho/rhoval) - 1

def err_t_srho(lgt, sval, rhoval, yval, hg):
    s_test = get_s_rhot_tab(rhoval, lgt, yval, hg=hg)*erg_to_kbbar
    return (s_test/sval) - 1

def err_t_rhop(_lgt, _lgrho, _lgp, _y):
    logrho_test = get_rho_pt(_lgp, _lgt, _y)
    return (logrho_test/_lgrho) - 1

### inversion functions ###

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-8
    
###### Temperature ######
def get_t_sp(s, p, y, hg=True, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
            guess = ideal_xy.get_t_sp(s, p, y)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y, hg))
            return float(sol.x.item())
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s, p, y, hg))
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

def get_t_srho(s, rho, y, alg='root', hg=True):
    if alg == 'root':
        if np.isscalar(s):
            s, rho, y = np.array([s]), np.array([rho]), np.array([y])
            guess = ideal_xy.get_t_srho(s, rho, y)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y, hg))
            return float(sol.x.item())
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y, hg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
        #guess = 2.5
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y, hg))
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
def get_p_rhot(rho, t, y, alg='root', hg=True):
    if alg == 'root':
        if np.isscalar(rho):
            rho, t, y = np.array([rho]), np.array([t]), np.array([y])
            guess = ideal_x.get_p_rhot(rho, t, y)
            sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y, hg))
            return float(sol.x.item())
        guess = ideal_x.get_p_rhot(rho, t, y)
        sol = root(err_p_rhot, guess, tol=1e-8, method='hybr', args=(rho, t, y, hg))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_p_rhot, bracket=PBOUNDS, xtol=XTOL, method='brenth', args=(rho, t, y, hg))
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

def get_u_srho(s, rho, y, z=0.0, hg=True):
    p, t = get_p_srho_tab(s, rho, y, hg=hg), get_t_srho_tab(s, rho, y, hg=hg)
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

def get_dsdy_rhop_srho(s, rho, y, z=0.0, ds=0.1, dy=0.1, tab=True, hg=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    if not tab:
        P0 = 10**get_p_srho(S0*erg_to_kbbar, rho, y, hg)
        P1 = 10**get_p_srho(S1*erg_to_kbbar, rho, y, hg)
        P2 = 10**get_p_srho(S0*erg_to_kbbar, rho, y*(1+dy), hg)
    else: 
        P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y, hg)
        P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y, hg)
        P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y*(1+dy), hg)      
    
    dpds_rhoy = (P1 - P0)/(S1 - S0)
    dpdy_srho = (P2 - P0)/(y*dy)

    return -dpdy_srho/dpds_rhoy


def get_dsdy_rhot(rho, t, y, z=0.0, dy=0.01):
    S0 = get_s_rhot_tab(rho, t, y)
    S1 = get_s_rhot_tab(rho, t, y*(1+dy))

    dsdy = (S1 - S0)/(y*dy)
    return dsdy

def get_dsdy_pt(p, t, y, z=0.0, dy=0.01, hg=True):
    S0 = get_s_pt(p, t, y, hg)
    S1 = get_s_pt(p, t, y*(1+dy), hg)

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

def get_c_v(s, rho, y, z=0.0, ds=0.1, hg=True):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho, y, hg)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho, y, hg)
 
    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, y, z=0.0, ds=0.1, hg=True):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_sp_tab(S0*erg_to_kbbar, p, y, hg)
    T1 = get_t_sp_tab(S1*erg_to_kbbar, p, y, hg)

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
def get_dudy_srho(s, rho, y, z=0.0, dy=0.1, hg=True):
    U0 = 10**get_u_srho(s, rho, y, hg=hg)
    U1 = 10**get_u_srho(s, rho, y*(1+dy), hg=hg)
    return (U1 - U0)/(y*dy)

# du/ds_(rho, Y) = T test
def get_duds_rhoy_srho(s, rho, y, z=0.0, ds=0.1, hg=True):
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)
    U0 = 10**get_u_srho(S1*erg_to_kbbar, rho, y, hg=hg)
    U1 = 10**get_u_srho(S2*erg_to_kbbar, rho, y, hg=hg)
    return (U1 - U0)/(S1*ds)

def get_dudrho_sy_srho(s, rho, y, z=0.0, drho=0.1, hg=True):
    R1 = 10**rho
    R2 = R1*(1+drho)
    #rho1 = np.log10((10**rho)*(1+drho))
    U0 = 10**get_u_srho(s, np.log10(R1), y, hg=hg)
    U1 = 10**get_u_srho(s, np.log10(R2), y, hg=hg)
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

def get_dtdy_srho(s, rho, y, z=0.0, dy=0.1, tab=True, hg=True):
    if not tab:
        T0 = 10**get_t_srho(s, rho, y, hg)
        T1 = 10**get_t_srho(s, rho, y*(1+dy), hg)
    else:
        T0 = 10**get_t_srho_tab(s, rho, y, hg)
        T1 = 10**get_t_srho_tab(s, rho, y*(1+dy), hg) 

    return (T1 - T0)/(y*dy)

def get_dtdy_rhop(rho, p, y, z=0.0, dy=0.01):
    t0 = 10**get_t_rhop(rho, p, y)
    t1 = 10**get_t_rhop(rho, p, y*(1+dy))

    dtdy = (t1 - t0)/(y*dy)
    return dtdy

def get_dtdrho_sy_srho(s, rho, y, z=0.0, drho = 0.01, tab=True, hg=True): # dlogT/dlogrho_{s, Y}
    R0 = 10**rho
    R1 = R0*(1+drho)
    if not tab:
        T0 = 10**get_t_srho(s, np.log10(R0), y, hg)
        T1 = 10**get_t_srho(s, np.log10(R1), y, hg)
    else:
        T0 = 10**get_t_srho_tab(s, np.log10(R0), y, hg)
        T1 = 10**get_t_srho_tab(s, np.log10(R1), y, hg)
    return (T1 - T0)/(R1 - R0)

def get_dtds_rhoy_srho(s, rho, y, z=0.0, ds=0.01, tab=True, hg=True):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    if not tab:
        T0 = 10**get_t_srho(S0*erg_to_kbbar, rho, y, hg)
        T1 = 10**get_t_srho(S1*erg_to_kbbar, rho, y, hg)
    else:
        T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho, y, hg)
        T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho, y, hg)
    return (T1 - T0)/(S1 - S0)

def get_nabla_ad(s, p, y, z=0.0, dp=0.01, hg=True):
    T0 = get_t_sp_tab(s, p, y, hg)
    T1 = get_t_sp_tab(s, p*(1+dp), y, hg)
    return (T1 - T0)/(p*dp)

def get_gruneisen(s, rho, y, z=0.0, drho = 0.01, hg=True):
    T0 = get_t_srho_tab(s, rho, y, hg)
    T1 = get_t_srho_tab(s, rho*(1+drho), y, hg)
    return (T1 - T0)/(rho*drho)


### Ledoux terms ###
# def get_B1(s, p, y, dy=0.1): # neesd to be multiplied by dY/dP
#     #p = get_p_srho(s, rho, y)
#     c_p = get_c_p(s, p, y)
#     rho, T = get_rhot_sp(s, p, y)
#     dsdy_rhop = get_dsdy_rhop(rho, p, y, dy=dy)
#     return ((10**p)/c_p)*dsdy_rhop

# def get_B2(s, p, y, dy=0.1): # neesd to be multiplied by dY/dP
#     t = get_t_sp(s, p, y)
#     c_p = get_c_p(s, p, y)
#     dsdy_pt = get_dsdy_pt(p, t, y, dy=dy)
#     return -((10**p)/c_p)*dsdy_pt


### second derivatives ###

# def get_d2sdy2_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
#     # second derivative of dsdy_rhop
#     A0 = get_dsdy_rhop_srho(s, rho, y, ds, dy, tab)
#     A1 = get_dsdy_rhop_srho(s, rho, y*(1+dy), ds, dy, tab)
#     return (A1 - A0)/(y*dy)

# def get_d2sds2_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
#     # second derivative of dsdy_rhop
#     S0 = s/erg_to_kbbar
#     S1 = S0*(1+ds)
#     A0 = get_dsdy_rhop_srho(S0*erg_to_kbbar, rho, y, ds, dy, tab)
#     A1 = get_dsdy_rhop_srho(S1*erg_to_kbbar, rho, y, ds, dy, tab)
#     return (A1 - A0)/(S1 - S0)

# def get_dcv_ds_srho(s, rho, y, ds=0.1, tab=True):
#     S0 = s/erg_to_kbbar
#     S1 = S0*(1+ds)
#     CV0 = get_c_v(S0*erg_to_kbbar, rho, y, ds, tab)
#     CV1 = get_c_v(S1*erg_to_kbbar, rho, y, ds, tab)

#     return (CV1 - CV0)/(S1 - S0)

# def get_dcv_dy_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
#     # S0 = s/erg_to_kbbar
#     # S1 = S0*(1+ds)
#     CV0 = get_c_v(s, rho, y, ds, tab)
#     CV1 = get_c_v(s, rho, y*(1+dy), ds, tab)

#     return (CV1 - CV0)/(y*dy)

# def get_dcp_ds_srho(s, rho, y, ds=0.1, tab=True):
#     S0 = s/erg_to_kbbar
#     S1 = S0*(1+ds)
#     CP0 = get_c_p(S0*erg_to_kbbar, rho, y, ds, tab)
#     CP1 = get_c_p(S1*erg_to_kbbar, rho, y, ds, tab)

#     return (CP1 - CP0)/(S1 - S0)

# def get_dcp_dy_srho(s, rho, y, ds=0.1, dy=0.1, tab=True):
#     # S0 = s/erg_to_kbbar
#     # S1 = S0*(1+ds)
#     CP0 = get_c_p(s, rho, y, ds, tab)
#     CP1 = get_c_p(s, rho, y*(1+dy), ds, tab)

#     return (CP1 - CP0)/(y*dy)

##### speed tests #####


#def test():
# import time
# from joblib import Parallel, delayed

# stest = np.zeros(240)+6.2
# rhotest =  np.zeros(240)-4
# ytest = np.zeros(240)+0.25

# start = time.time()
# p, t = get_pt_srho(stest, rhotest, ytest)
# end = time.time()
# print('get_pt_srho test:', end - start)

# start = time.time()
# t = get_t_srho(stest, rhotest, ytest)
# end = time.time()
# print('get_t_srho test:', end - start)

# start = time.time()
# p, t = get_p_srho_tab(stest, rhotest, ytest), get_t_srho_tab(stest, rhotest, ytest)
# end = time.time()
# print('tables:', end - start)

# start = time.time()
# Parallel(n_jobs=2, prefer="threads" )(delayed(get_t_srho)(s_, rho_, y_) for s_, rho_, y_ in zip(stest, rhotest, ytest))
# end = time.time()
# print('parallel code:', end - start)

# from multiprocessing import Pool

# def get_t_srho_par(iter):
#     print(iter)
#     logt = get_t_srho(stest[iter], rhotest[iter], ytest[iter])
#     return logt

# if __name__ == '__main__':
#     pool = Pool(2)
#     start = time.time()
#     results = pool.map(get_t_srho_par, range(len(stest)))
#     end = time.time()
#     print('parallel:', end - start)
#     np.save('%s/cms/parallel_test_tsrho.npy' % CURR_DIR, results)

    



