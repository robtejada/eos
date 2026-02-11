import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
import pandas as pd

from eos import serpentine_eos

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg/u.K).to('erg/(K * g)')

GPa_to_cgs = u.GPa.to('dyn/cm^2')
GPa_to_bar = u.GPa.to('bar')

logrhovals, logtvals = np.load('eos/aneos/forsterite_eos_arrays.npy', allow_pickle=True)

# S units: MJ/K/kg
# P units: GPa
# U units: MJ/kg
s_grid, p_grid, u_grid = np.load('eos/aneos/forsterite_eos_grids.npy', allow_pickle=True)

p_grid = p_grid#*GPa_to_cgs # in dyn/cm^2
s_grid = s_grid#/erg_to_MJ # in erg/K/g
u_grid = u_grid#*MJ_to_erg # in erg/g

rgi_p = RGI((logtvals, logrhovals), p_grid, method='linear', bounds_error=False, fill_value=None)
rgi_s = RGI((logtvals, logrhovals), s_grid, method='linear', bounds_error=False, fill_value=None)
rgi_u = RGI((logtvals, logrhovals), u_grid, method='linear', bounds_error=False, fill_value=None)

def get_p_rhot_tab(logrho, logt):
    #print('WARNING-- PRESSURE IS IN GPA FOR INVERSION PURPOSES')
    if np.isscalar(logrho):
        # taking log10 here because original data has 0 GPa in the beginning
        return float(rgi_p(np.array([logt, logrho]).T))
    return rgi_p(np.array([logt, logrho]).T) # returns in GPa

def get_s_rhot_tab(logrho, logt):
    if np.isscalar(logrho):
        return float(rgi_s(np.array([logt, logrho]).T))
    return rgi_s(np.array([logt, logrho]).T) # returns in MJ/kg/K units

def get_u_rhot_tab(logrho, logt):
    if np.isscalar(logrho):
        return float(rgi_u(np.array([logt, logrho]).T))
    return rgi_u(np.array([logt, logrho]).T) # returns in MJ/kg units

def err_rho_pt(logrho, p_val, logtval):
    logp_ = get_p_rhot_tab(logrho, logtval)
    return (logp_/p_val) - 1

def get_rho_pt(p_GPa, logt):
    
    #p_GPa = 10**(logp-10) 
    
    if np.isscalar(p_GPa):
        p_GPa, logt = np.array([p_GPa]), np.array([logt])
        guess = serpentine_eos.get_rho_pt_tab(np.log10(p_GPa*1e10), logt)
        #guess = 1
        sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p_GPa, logt))
        return float(sol.x)
    
    sol = np.array([get_rho_pt(p, t) for p, t in zip(p_GPa, logt)])
    return sol

### INVERSIONS ###

# RHO(P, T)

logrho_grid_res = np.load('eos/aneos/aneos_forsterite_pt_base.npy')

p_grid_inv = np.logspace(np.log10(1.2), 4, 100) # grid is in GPa-- equiv to 10.07--14 in log dyn/cm^2
logt_grid_inv = np.linspace(2, 6, 150)

rgi_rho = RGI((logt_grid_inv, p_grid_inv), logrho_grid_res, method='linear', 
              bounds_error=False, fill_value=None)

def get_rho_pt_tab(p_GPa, logt):
    #p_GPa = 10**(logp-10) 
    if np.isscalar(p_GPa):
        # needs to be in p_GPa... seems to work best for inversion
        return float(rgi_rho(np.array([logt, p_GPa]).T))
    return rgi_rho(np.array([logt, p_GPa]).T) # returns logrho

def get_s_pt_tab(p_GPa, logt, tab=True):
    #p_GPa = 10**(logp-10) 
    if tab:
        return get_s_rhot_tab(get_rho_pt_tab(p_GPa, logt), logt)
    else:
        return get_s_rhot_tab(get_rho_pt(p_GPa, logt), logt) # RETURNS IN MJ/kg/K

def get_u_pt_tab(p_GPa, logt, tab=True):
    #p_GPa = 10**(logp-10) 
    if tab:
        return get_u_rhot_tab(get_rho_pt_tab(p_GPa, logt), logt)
    else:
        return get_u_rhot_tab(get_rho_pt(p_GPa, logt), logt) # RETURNS IN MJ/kg/K
    
def err_t_sp(logt, s_val, logp_val):
    s_ = get_s_pt_tab(logp_val, logt)
    return (s_/s_val) - 1

logt_test = 2 # WHY DOES THIS WORK???

def get_t_sp(s_MJ, p_GPa):
    #p_GPa = 10**(logp-10) 
    #s = _s/MJ_to_kbbar 
    if np.isscalar(p_GPa):
        s_MJ, p_GPa = np.array([s_MJ]), np.array([p_GPa])
        # THIS IS THE WRONG GUESS FUNCTION BUT IT WORKS???
        # IT WORKS MUCH BETTER THAN THE GET_T_SP FUNC!
        guess = serpentine_eos.get_s_pt_tab(np.log10(p_GPa*1e10), logt_test)#/MJ_to_erg
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(s_MJ, p_GPa))
        return float(sol.x)

    sol = np.array([get_t_sp(s_, p_) for s_, p_ in zip(s_MJ, p_GPa)])
    return sol

# T(S, P)

p_low = 1e-4 # GPa
t_low = 2
# logT
s_low = get_s_pt_tab(p_low, t_low, tab=True)

p_high = 1e4 # GPa
t_high = 6
# logT
s_high = get_s_pt_tab(p_high, t_high, tab=True)

# print(s_test_low, s_test_high)

s_grid_inv = np.linspace(s_low, s_high, 1000) # this is in MJ/kg/K
p_grid_inv_sp = np.logspace(-4, 4, 500) # grid is in GPa-- equiv to 1--100 bar

logt_grid_res = np.load('eos/aneos/aneos_forsterite_sp_base.npy')

rgi_t = RGI((s_grid_inv, p_grid_inv_sp), logt_grid_res, method='linear', 
              bounds_error=False, fill_value=None) # p_grid is in GPa... plan accordingly

def get_t_sp_tab(s_MJ, p_GPa):
    #p_GPa = 10**(logp-10) # converting log(dyn/cm^2) to GPa
    #s_MJ = s/MJ_to_kbbar # converting MJ/kg/K to kb/baryon...
    if np.isscalar(p_GPa):
        # needs to be in p_GPa... seems to work best for inversion
        return float(rgi_t(np.array([s_MJ, p_GPa]).T))
    return rgi_t(np.array([s_MJ, p_GPa]).T) # returns logrho

def get_rho_sp_tab(s_MJ, p_GPa, tab=True):
    #p_GPa = 10**(logp-10) 
    #s_MJ = s/MJ_to_kbbar
    if tab:
        return get_rho_pt_tab(p_GPa, get_t_sp_tab(s_MJ, p_GPa))
    else:
        return get_rho_pt(p_GPa, get_t_sp(s_MJ, p_GPa))
