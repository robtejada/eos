import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import RectBivariateSpline as RBS
from astropy import units as u
from scipy.optimize import root, root_scalar
from astropy.constants import k_B, m_e
from astropy.constants import u as amu
from eos import ideal_eos

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
pd.options.mode.chained_assignment = None

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
evel_to_erg = -(u.eV/m_e).to('erg/g').value
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

mp = amu.to('g') # grams

kb = k_B.to('erg/K') # ergs/K

ideal_xy = ideal_eos.IdealHHeMix()

def kbel_to_kbbar(X):
    # will be useful for MH13 data down the road; multiply their S values by this, given that X = 1-Y
    # ne/nb
    return (1+X)/2

def mh13_reader(tab):
    mh13_ad = np.genfromtxt(tab, dtype=None, encoding=None)

    out_list = []
    for i, row in enumerate(mh13_ad):
        clean_row = []
        for j, val in enumerate(row):
            if j % 2:
                clean_row.append(val)
        out_list.append(clean_row)

    eos_mh13 = pd.DataFrame(np.array(out_list), 
                            columns = ['rs','rho','T','E','P','F','S(kb/el)'])#.sort_values(['T', 'rho'])
    
    #eos_mh13 = eos_mh13[eos_mh13['S(kb/el)'] > 0.0]
    eos_mh13['S(kb/bar)'] = eos_mh13['S(kb/el)']*kbel_to_kbbar((1-0.246575))
    eos_mh13['logrho'] = np.log10(eos_mh13['rho'])
    eos_mh13['logp'] = np.log10(eos_mh13['P'])+10 # in cgs
    eos_mh13['logt'] = np.log10(eos_mh13['T'])
    eos_mh13['u'] = -eos_mh13['E']*evel_to_erg
    #eos_mh13['logs'] = np.log10(eos_mh13['S(kb/bar)']*erg_to_kbbar)#log cgs for interpolation
    return eos_mh13

def grid_data_mh13(tab):
    
    df = mh13_reader(tab)
    twoD = {}
    shape = df['logrho'].nunique(), -1
    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    
    #s_table = twoD['S(kb/bar)'] # 2-D array with entropies
    #logtvals = np.log10(twoD['T'][:, 0]) # 1-D array with unique temps
    #logrhovals = np.log10(twoD['rho'][0, :]) # 1-D array with unique densities
    
    return twoD

### Y = 0.245 mixture ###
data_mix = grid_data_mh13('%s/mh13/mh13_eos.out' % CURR_DIR)
logtvals_mix = data_mix['logt'][0]
logrhovals_mix = data_mix['logrho'][:,0]

stab_mix = data_mix['S(kb/bar)']
ptab_mix = data_mix['logp']
utab_mix = data_mix['u']

get_s_mix_rgi = RGI((logrhovals_mix, logtvals_mix), stab_mix, method='linear', bounds_error=False, fill_value=None)
get_p_mix_rgi = RGI((logrhovals_mix, logtvals_mix), ptab_mix, method='linear', bounds_error=False, fill_value=None)
get_u_mix_rgi = RGI((logrhovals_mix, logtvals_mix), utab_mix, method='linear', bounds_error=False, fill_value=None)

def get_s_rhot_mix(lgrho, lgt): # in cgs
    if np.isscalar(lgrho):
        return float(get_s_mix_rgi(np.array([lgrho, lgt]).T)/erg_to_kbbar)
    return get_s_mix_rgi(np.array([lgrho, lgt]).T)/erg_to_kbbar

def get_p_rhot_mix(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(get_p_mix_rgi(np.array([lgrho, lgt]).T))
    return get_p_mix_rgi(np.array([lgrho, lgt]).T)

def get_u_rhot_mix(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(get_u_mix_rgi(np.array([lgrho, lgt]).T))
    return get_u_mix_rgi(np.array([lgrho, lgt]).T)

### helium tables ###
data_he = grid_data_mh13('%s/mh13/mh13_he_eos.out' % CURR_DIR)
logtvals_he = data_he['logt'][0]
logrhovals_he = data_he['logrho'][:,0]

stab_he = data_he['S(kb/bar)']
ptab_he = data_he['logp']
utab_he = data_he['u']

get_s_he_rgi = RGI((logrhovals_he, logtvals_he), stab_he, method='linear', bounds_error=False, fill_value=None)
get_p_he_rgi = RGI((logrhovals_he, logtvals_he), ptab_he, method='linear', bounds_error=False, fill_value=None)
get_u_he_rgi = RGI((logrhovals_he, logtvals_he), utab_he, method='linear', bounds_error=False, fill_value=None)

def get_s_rhot_he(lgrho, lgt): # in cgs
    if np.isscalar(lgrho):
        return float(get_s_he_rgi(np.array([lgrho, lgt]).T)/erg_to_kbbar)
    return get_s_he_rgi(np.array([lgrho, lgt]).T)/erg_to_kbbar

def get_p_rhot_he(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(get_p_he_rgi(np.array([lgrho, lgt]).T))
    return get_p_he_rgi(np.array([lgrho, lgt]).T)

def get_u_rhot_he(lgrho, lgt):
    if np.isscalar(lgrho):
        return float(get_u_he_rgi(np.array([lgrho, lgt]).T))
    return get_u_he_rgi(np.array([lgrho, lgt]).T)


##### MH13 + CMS19+HG23 combined tables for better coverage  #####

logrhovals_rhot = np.linspace(-5, 1.5, 100)
logtvals_rhot = np.arange(2.1, 5.1, 0.05)

logp_res_rhot_mix, s_res_rhot_mix, u_res_rhot_mix = np.load('%s/mh13/mh13_cmshg_mix.npy' % CURR_DIR)
logp_res_rhot_he, s_res_rhot_he, u_res_rhot_he = np.load('%s/mh13/mh13_cmshg_he.npy' % CURR_DIR)

stabs = [s_res_rhot_mix, s_res_rhot_he]
ptabs = [logp_res_rhot_mix, logp_res_rhot_he]
utabs = [u_res_rhot_mix, u_res_rhot_he]
ys = [0.246575, 1.0]

get_p_rhot_rgi = RGI((ys, logrhovals_rhot, logtvals_rhot), ptabs, method='linear', bounds_error=False, fill_value=None)
get_s_rhot_rgi = RGI((ys, logrhovals_rhot, logtvals_rhot), stabs, method='linear', bounds_error=False, fill_value=None)
get_u_rhot_rgi = RGI((ys, logrhovals_rhot, logtvals_rhot), utabs, method='linear', bounds_error=False, fill_value=None)

def get_p_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_p_rhot_rgi(np.array([y, rho, t]).T))
    else:
        return get_p_rhot_rgi(np.array([y, rho, t]).T)

def get_s_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_s_rhot_rgi(np.array([y, rho, t]).T))
    else:
        return get_s_rhot_rgi(np.array([y, rho, t]).T)

def get_u_rhot_tab(rho, t, y):
    if np.isscalar(rho):
        return float(get_u_rhot_rgi(np.array([y, rho, t]).T))
    else:
        return get_u_rhot_rgi(np.array([y, rho, t]).T)

### mixtures ###
# y0 = 0.246575 
# def get_p_rhot(rho, t, y, z=0.0):
#     pmix = 10**get_p_rhot_mix(rho, t)
#     phe = 10**get_p_rhot_he(rho, t)
#     #print(pmix, phe)
#     return np.log10(((1 - y)/(1 - y0))*pmix + ((y - y0)/(1 - y0))*phe)

# def get_s_rhot(rho, t, y, z=0.0):
#     smix = get_s_rhot_mix(rho, t)
#     she = get_s_rhot_he(rho, t)
#     #smix_id = cms_eos.get_smix_id_y(y)
#     return ((1 - y)/(1 - y0))*smix + ((y - y0)/(1 - y0))*she #+ smix_id/erg_to_kbbar

# def get_u_rhot(rho, t, y, z=0.0):
#     umix = 10**get_u_rhot_mix(rho, t)
#     uhe = 10**get_u_rhot_he(rho, t)
#     #print(pmix, phe)
#     return np.log10(((1 - y)/(1 - y0))*umix + ((y - y0)/(1 - y0))*uhe)

#### error functions ####

def err_rho_pt(lgrhoval, lgpval, lgtval, yval):
    logp_test = get_p_rhot_tab(lgrhoval, lgtval, yval)
    return (lgpval/logp_test) - 1

def err_t_sp(logt, s_val, logp, y):
    s_ = get_s_pt_tab(logp, logt, y)*erg_to_kbbar
    return (s_/s_val) - 1

def err_t_srho(lgt, sval, rhoval, yval):
    s_test = get_s_rhot_tab(rhoval, lgt, yval)*erg_to_kbbar
    return (s_test/sval) - 1

### inversion functions ####

TBOUNDS = [2, 7]
PBOUNDS = [0, 15]

XTOL = 1e-8

### Temperature ###

def get_t_sp(s, p, y, alg='root'):
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

### density ###

def get_rho_pt(p, t, y, z=0.0):
    if np.isscalar(p):
        p, t, y = np.array([p]), np.array([t]), np.array([y])
        guess = ideal_xy.get_rho_pt(p, t, y)
        sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t, y))
        return float(sol.x.item())

    guess = ideal_xy.get_rho_pt(p, t, y)
    sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t, y))
    return sol.x


### P, T tables ###

logpvals = np.linspace(5, 14, 100)
logtvals = np.linspace(2.1, 5.05, 100)

s_mix, logrho_mix = np.load('%s/mh13/pt_basis_mix.npy' % CURR_DIR)
s_he, logrho_he = np.load('%s/mh13/pt_basis_he.npy' % CURR_DIR)

stabs = [s_mix, s_he]
rhotabs = [logrho_mix, logrho_he]
ys = [0.246575, 1.0]

get_rho_pt_rgi = RGI((ys, logpvals, logtvals), rhotabs, method='linear', bounds_error=False, fill_value=None)
get_s_pt_rgi = RGI((ys, logpvals, logtvals), stabs, method='linear', bounds_error=False, fill_value=None)

def get_rho_pt_tab(p, t, y):
    if np.isscalar(p):
        return float(get_rho_pt_rgi(np.array([y, p, t]).T))
    else:
        return get_rho_pt_rgi(np.array([y, p, t]).T)

def get_s_pt_tab(p, t, y):
    if np.isscalar(p):
        return float(get_s_pt_rgi(np.array([y, p, t]).T))
    else:
        return get_s_pt_rgi(np.array([y, p, t]).T)

def get_u_pt_tab(p, t, y):
    rho = get_rho_pt_tab(p, t, y)
    return np.abs(get_u_rhot_tab(rho, t, y))

### S, P tables ###

logrho_res_sp, logt_res_sp = np.load('%s/mh13/sp_basis.npy' % CURR_DIR)

svals_sp = np.arange(5.0, 10.1, 0.1)
logpvals_sp = np.linspace(5, 14, 100)
yvals_sp = np.arange(0.246575, 1.0, 0.01)

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

### S, rho tables ###

logp_res_srho, logt_res_srho = np.load('%s/mh13/srho_basis.npy' % CURR_DIR)

# svals_srho = np.arange(5.0, 10.1, 0.05) # new grid
# logrhovals_srho = np.arange(-5, 1.5, 0.05)
# yvals_srho = np.arange(0.05, 1.05, 0.05)

svals_srho = np.arange(5.5, 10.1, 0.1) # new grid
logrhovals_srho = np.linspace(-5, 1.5, 100)
yvals_srho = np.arange(0.246575, 1.0, 0.01)

get_p_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_rgi_srho = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, rho, y, z=0.0):
    if np.isscalar(s):
        return float(get_p_rgi_srho(np.array([s, rho, y]).T))
    else:
        return get_p_rgi_srho(np.array([s, rho, y]).T)

def get_t_srho_tab(s, rho, y, z=0.0):
    if np.isscalar(s):
        return float(get_t_rgi_srho(np.array([s, rho, y]).T))
    else:
        return get_t_rgi_srho(np.array([s, rho, y]).T)

def get_u_srho(s, rho, y, z=0.0):
    p, t = get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)
    return get_u_pt_tab(p, t, y)

############## derivatives ##############


### entropy gradients ###

# def get_dsdy_rhop(rho, p, y, dy=0.1):
#     S0 = get_s_rhop(rho, p, y)
#     S1 = get_s_rhop(rho, p, y*(1+dy))

#     return (S1 - S0)/(y*dy)

def get_dsdy_rhop_srho(s, rho, y, z=0.0, ds=0.1, dy=0.1):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    # if not tab:
    #     P0 = 10**get_p_srho(S0*erg_to_kbbar, rho, y)
    #     P1 = 10**get_p_srho(S1*erg_to_kbbar, rho, y)
    #     P2 = 10**get_p_srho(S0*erg_to_kbbar, rho, y*(1+dy))
    #else: 
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
    S0 = get_s_pt_tab(p, t, y)
    S1 = get_s_pt_tab(p, t, y*(1+dy))

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
    U0 = 10**get_u_rhot_tab(rho, t, y)
    U1 = 10**get_u_rhot_tab(np.log10(R1), t, y)
    return (U1 - U0)/(R1 - R0)

### density gradients ###

def get_drhods_py(s, p, y, z=0.0, ds=0.01):
    
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)

    rho0 = 10**get_rho_sp_tab(S1*erg_to_kbbar, p, y)
    rho1 = 10**get_rho_sp_tab(S2*erg_to_kbbar, p, y)

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

# def get_dtdy_rhop(rho, p, y, z=0.0, dy=0.01):
#     t0 = 10**get_t_rhop(rho, p, y)
#     t1 = 10**get_t_rhop(rho, p, y*(1+dy))

#     dtdy = (t1 - t0)/(y*dy)
#     return dtdy

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