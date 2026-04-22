import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
# from scipy.optimize import root, root_scalar
from eos import ideal_eos, aqua_eos, ppv_eos, ppv2_eos, fe_eos, iron2_eos, zmix_eos, mgsio3_liquid_eos_formix
import os

from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu

"""
    This file puts together all of the available metal EOSes of this repo.
    Instead of calling each module individually, one can import this file
    and select the metal EOS to use with the EOS argument in each function.

    This file further provides access to metal mixtures. For now, the 'mixture'
    option provides a mixture of 50% water, 33.3% post-perovskite, and 16.6%
    iron. A dependency on water fraction will be added in the near future.

    Author: Roberto Tejada Arevalo

"""

mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

# For Forsterite EOS units
MJ_to_erg_S = (u.MJ/u.kg/u.K).to('erg/(K * g)')
MJ_to_erg_U = (u.MJ/u.kg).to('erg/g')
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)

ideal_water = ideal_eos.IdealEOS(m=18) # default for ideal eos is water for now

eos_l = mgsio3_liquid_eos_formix.MGSIO3_LIQUID_EOS()

#### P, T ####

def get_rho_pt_tab(p, t, eos, f_ppv=0.333, f_fe=0.166, z_eos1='aqua', z_eos2='mgsio3_l', z_eos3='iron'):
    if eos == 'aqua':
        return aqua_eos.get_logrho_pt_tab(p, t)
    elif eos == 'mgsio3_l':
        return eos_l.get_rho_pt(p, t)
    elif eos == 'aqua_mlcp':
        return aqua_mlcp_eos.get_logrho_pt_tab(p, t)
    elif eos == 'ice_aneos':
        return ice_aneos_eos.get_logrho_pt_tab(p, t)
    elif eos == 'ppv':
        return ppv_eos.get_rho_pt_tab(p, t)
    elif eos == 'ppv2':
        return ppv2_eos.get_logrho_pt_tab(p, t)
    elif eos == 'serpentine':
        return serpentine_eos.get_rho_pt_tab(p, t)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_rho_pt_tab((10**p)*1e-10, t)
    elif eos == 'iron':
        return fe_eos.get_rho_pt_tab(p, t)
    elif eos == 'iron2':
        return iron2_eos.get_logrho_pt_tab(p, t)
    elif eos == 'ideal':
        return ideal_water.get_rho_pt(p, t, 0)
    elif eos == 'mixture':
        return zmix_eos.get_rho_pt_tab(p, t, f_ppv, f_fe, z_eos1=z_eos1, z_eos2=z_eos2, z_eos3=z_eos3) # the standard was 0.333 and 0.166 for ppv and iron fractions

    # elif eos == 'mgsio3_s':
    #     return eos_s.get_rho_pt_tab(p, t)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_s_pt_tab(p, t, eos, f_ppv=0.333, f_fe=0.166, z_eos1='aqua', z_eos2='mgsio3_l', z_eos3='iron'):
    if eos == 'aqua':
        return aqua_eos.get_s_pt_tab(p, t)
    elif eos == 'mgsio3_l':
        return eos_l.get_s_pt(p, t)
    elif eos == 'aqua_mlcp':
        return aqua_mlcp_eos.get_s_pt_tab(p, t)
    elif eos == 'ice_aneos':
        return ice_aneos_eos.get_s_pt_tab(p, t)
    elif eos == 'ppv':
        return ppv_eos.get_s_pt_tab(p, t)
    elif eos == 'ppv2':
        return ppv2_eos.get_s_pt_tab(p, t)
    elif eos == 'serpentine':
        return serpentine_eos.get_s_pt_tab(p, t)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_s_pt_tab((10**p)*1e-10, t) * MJ_to_erg_S
    elif eos == 'iron':
        return fe_eos.get_s_pt_tab(p, t)
    elif eos == 'iron2':
        return iron2_eos.get_s_pt_tab(p, t)
    elif eos == 'ideal':
        return ideal_water.get_s_pt(p, t, 0)/erg_to_kbbar
    elif eos == 'mixture':
        return zmix_eos.get_s_pt_tab(p, t, f_ppv, f_fe, z_eos1=z_eos1, z_eos2=z_eos2, z_eos3=z_eos3)

    # elif eos == 'mgsio3_s':
    #     return eos_s.get_s_pt_tab(p, t)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_u_pt_tab(p, t, eos, f_ppv=0.333, f_fe=0.166, z_eos1='aqua', z_eos2='mgsio3_l', z_eos3='iron'):
    if eos == 'aqua':
        return aqua_eos.get_logu_pt_tab(p, t)
    elif eos == 'mgsio3_l':
        return eos_l.get_u_pt(p, t) # not log
    elif eos == 'aqua_mlcp':
        return aqua_mlcp_eos.get_logu_pt_tab(p, t)
    elif eos == 'ice_aneos':
        return ice_aneos_eos.get_logu_pt_tab(p, t)
    elif eos == 'ppv':
        return ppv_eos.get_u_pt_tab(p, t)
    elif eos == 'ppv2':
        return ppv2_eos.get_logu_pt_tab(p, t)
    elif eos == 'serpentine':
        return serpentine_eos.get_u_pt_tab(p, t)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_u_pt_tab((10**p)*1e-10, t) * MJ_to_erg_U
    elif eos == 'iron':
        return fe_eos.get_u_pt_tab(p, t)
    elif eos == 'iron2':
        return iron2_eos.get_logu_pt_tab(p, t)
    elif eos == 'ideal':
        return ideal_water.get_u_pt(p, t)
    elif eos == 'mixture':
        return zmix_eos.get_u_pt_tab(p, t, f_ppv, f_fe, z_eos1=z_eos1, z_eos2=z_eos2, z_eos3=z_eos3)

    # elif eos == 'mgsio3_s':
    #     return eos_s.get_u_pt_tab(p, t) # not log
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

#### rho, T ####

def get_p_rhot_tab(rho, t, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_p_rhot_tab(rho, t)
    elif eos == 'mlcp':
        return np.log10(mazevet_eos.get_p_rhot_tab(rho, t))
    elif eos == 'ppv':
        return ppv_eos.get_p_rhot_tab(rho, t)
    # elif eos == 'serpentine':
    #     #return serpentine_eos.get_p_rhot_tab(rho, t)
    elif eos == 'fo':
        return np.log10(aneos_forsterite_eos.get_p_rhot_tab(rho, t)*1e10)
    elif eos == 'iron':
        return fe_eos.get_p_rhot_tab(rho, t)
    elif eos == 'ideal':
        return ideal_water.get_p_rhot(rho, t, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(rho))-f_ppv-f_fe
        #f_ice = np.zeros(len(rho))+1-0.333-0.166 # constant for now
        return zmix_eos.get_p_rhot_tab(rho, t, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_s_rhot_tab(rho, t, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_s_rhot_tab(rho, t)
    elif eos == 'mlcp':
        return mazevet_eos.get_s_rhot_tab(rho, t)
    elif eos == 'ppv':
        return ppv_eos.get_s_rhot_tab(rho, t)
    # elif eos == 'serpentine':
    #     #return serpentine_eos.get_p_rhot_tab(rho, t)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_s_rhot_tab(rho, t) * MJ_to_erg_S
    elif eos == 'iron':
        return fe_eos.get_s_rhot_tab(rho, t)
    elif eos == 'ideal':
        return ideal_water.get_s_rhot(rho, t, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(rho))-f_ppv-f_fe
        return zmix_eos.get_s_rhot_tab(rho, t, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

#### S, P ####

def get_t_sp_tab(s, p, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_t_sp_tab(s, p)
    elif eos == 'ppv':
        return ppv_eos.get_logt_sp_tab(s, p)
    elif eos == 'ppv2':
        return ppv2_eos.get_logt_sp_tab(s, p)
    elif eos == 'serpentine':
        return serpentine_eos.get_t_sp_tab(s, p)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_t_sp_tab(s/MJ_to_kbbar, (10**p)*1e-10)
    elif eos == 'iron':
        return fe_eos.get_logt_sp_tab(s, p)
    elif eos == 'ideal':
        return ideal_water.get_t_sp(s, p, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(s))-f_ppv-f_fe
        return zmix_eos.get_t_sp_tab(s, p, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_rho_sp_tab(s, p, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_rho_sp_tab(s, p)
    elif eos == 'ppv':
        return ppv_eos.get_logrho_sp_tab(s, p)
    elif eos == 'ppv2':
        return ppv2_eos.get_logrho_sp_tab(s, p)
    elif eos == 'serpentine':
        return serpentine_eos.get_rho_sp_tab(s, p)
    elif eos == 'fo':
        return aneos_forsterite_eos.get_rho_sp_tab(s/MJ_to_kbbar, (10**p)*1e-10)
    elif eos == 'iron':
        return fe_eos.get_logrho_sp_tab(s, p)
    elif eos == 'ideal':
        return ideal_water.get_rho_sp(s, p, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(s))-f_ppv-f_fe
        return zmix_eos.get_rho_sp_tab(s, p, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, ideal, or mixture')

def get_rhot_sp_tab(s, p, eos):
    return get_rho_sp_tab(s, p, eos), get_t_sp_tab(s, p, eos)

#### S, rho ####

def get_t_srho_tab(s, rho, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_t_srho_tab(s, rho)
    elif eos == 'ppv':
        return ppv_eos.get_t_srho_tab(s, rho)
    elif eos == 'serpentine':
        return serpentine_eos.get_t_srho_tab(s, rho)
    elif eos == 'iron':
        return fe_eos.get_t_srho_tab(s, rho)
    elif eos == 'ideal':
        return ideal_water.get_t_srho(s, rho, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(s))-f_ppv-f_fe
        return zmix_eos.get_t_srho_tab(s, rho, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

def get_p_srho_tab(s, rho, eos, f_ppv=0.333, f_fe=0.166):
    if eos == 'aqua':
        return aqua_eos.get_p_srho_tab(s, rho)
    elif eos == 'ppv':
        return ppv_eos.get_p_srho_tab(s, rho)
    elif eos == 'serpentine':
        return serpentine_eos.get_p_srho_tab(s, rho)
    elif eos == 'iron':
        return fe_eos.get_p_srho_tab(s, rho)
    elif eos == 'ideal':
        return ideal_water.get_p_srho(s, rho, 0)
    elif eos == 'mixture':
        f_ice = np.ones(len(s))-f_ppv-f_fe
        return zmix_eos.get_p_srho_tab(s, rho, f_ice)
    else:
        raise Exception('EOS must be aqua, ppv, serpentine, iron, or ideal')

############## derivatives ##############

# def get_dsdy_rhop_srho(s, rho, y, ds=0.1, dy=0.1):
#     S0 = s/erg_to_kbbar
#     S1 = S0*(1+ds)
#     P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y)
#     P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y)
#     P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y*(1+dy))

#     dpds_rhoy = (P1 - P0)/(S1 - S0)
#     dpdy_srho = (P2 - P0)/(y*dy)

#     return -dpdy_srho/dpds_rhoy

# def get_dsdy_pt(p, t, y, dy=0.01):
#     S0 = get_s_pt_tab(p, t, y)
#     S1 = get_s_pt_tab(p, t, y*(1+dy))

#     return (S1 - S0)/(y*dy)

def get_dtdrho_srho(s, rho, eos, drho=0.01):
    R0 = 10**rho
    R1 = R0*(1+drho)
    T0 = 10**get_t_srho_tab(s, np.log10(R0), eos)
    T1 = 10**get_t_srho_tab(s, np.log10(R1), eos)

    return (T1 - T0)/(R1 - R0)

def get_dtds_srho(s, rho, eos, ds=0.01):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    T0 = 10**get_t_srho_tab(S0*erg_to_kbbar, rho, eos)
    T1 = 10**get_t_srho_tab(S1*erg_to_kbbar, rho, eos)

    return (T1 - T0)/(S1 - S0)

def get_c_v(s, rho, eos, f_ppv, f_fe, ds=1e-3):
    # ds/dlogT_{rho, Y}

    lgt2 = get_t_srho_tab(s + ds, rho, eos, f_ppv=f_ppv, f_fe=f_fe)
    lgt1 = get_t_srho_tab(s - ds, rho, eos, f_ppv=f_ppv, f_fe=f_fe)

    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))

def get_c_p(s, p, eos, f_ppv, f_fe, ds=1e-3):
    # ds/dlogT_{P, Y}

    lgt2 = get_t_sp_tab(s + ds, p, eos, f_ppv=f_ppv, f_fe=f_fe)
    lgt1 = get_t_sp_tab(s - ds, p, eos, f_ppv=f_ppv, f_fe=f_fe)

    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))