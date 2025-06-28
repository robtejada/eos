import numpy as np
#import scvh_nr
from scipy.optimize import root, root_scalar
from scipy.interpolate import RegularGridInterpolator as RGI
from eos import aneos, scvh_man, ideal_eos
import os

"""
    This file provides thermodynamic quantities and derivatives for the Saumon, Chabrier, and van Horn (1995) H-He EOS.

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

    Authors: Roberto Tejada Arevalo, Adam Burrows

"""

erg_to_kbbar = 1.202723550011625e-08
mh = 1
mhe = 4.0026

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

eos_aneos = aneos.eos(path_to_data='%s/aneos' % CURR_DIR, material='serpentine')
eos_scvh = scvh_man.eos(path_to_data='%s/scvh_mesa' % CURR_DIR)

ideal_x = ideal_eos.IdealEOS(m=2)
ideal_xy = ideal_eos.IdealHHeMix()

# def scvh_reader(tab_name):
#     tab = []
#     head = []
#     with open('%s/scvh/eos/%s' % (CURR_DIR, tab_name)) as file:
#         for j, line in enumerate(file):
#             line = line.rstrip('\n')
#             if line.startswith("#"):
#                 line = line.lstrip('#')
#                 head.append(line)

#             else:# j > head: # skipping the header
#                 tab.append(list(float(line[i:i+8]) for i in range(0, len(line), 8)))

#     header = list(filter(None, ' '.join(head).split(' ')))

#     tab = np.array(tab)
#     tab_ = np.reshape(tab, (300, 100))

#     return [float(val) for val in header], tab_

# bounds_s, stab = scvh_reader('stabnew_adam.dat')
# _, INDEX, R1, R2, T1, T2, T11, T12 = bounds_s # same bounds

# stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
# ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']

# stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
# ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

# bounds_s, stab = scvh_reader('stabnew_adam.dat')
# R1, R2, T1, T2, T11, T12 = bounds_s[2:8] # same bounds

# logrho_arr = np.linspace(R1, R2, 300)
# stab_names = ['s22scz.dat', 'stabnew_adam.dat', 's28scz.dat', 's30.dat']
# stabs = np.array([scvh_reader(name)[-1] for name in stab_names])
# ptab_names = ['p22scz.dat', 'ptabnew_adam.dat', 'p28scz.dat', 'p30.dat']
# ptabs = np.array([scvh_reader(name)[-1] for name in ptab_names])

# def x(R):
#     return (R - R1)*(300)/(R2 - R1)

# def T1p(R):
#     m1 = (T11 - T1)/(R2 - R1)
#     return m1*(R - R1) + T1

# def T2p(R):
#     m2 = (T12 - T2)/(R2 - R1)
#     return m2*(R - R1) + T2

# def get_logtarr(R):
#     return np.linspace(T1p(R), T2p(R), 100)

# def y(R, T):
#     return ((T - T1p(R))/(T2p(R) - T1p(R)))*(100)

# yhe_arr = np.array([0.22, 0.25, 0.28, 0.30])

# #x_arr = np.arange(0, 300, 1)
# x_arr = x(logrho_arr)
# y_arr = np.arange(0, 100, 1)

# interp_s = RGI((yhe_arr, x_arr, y_arr), stabs, method='linear', bounds_error=False, fill_value=None)
# interp_p = RGI((yhe_arr, x_arr, y_arr), ptabs, method='linear', bounds_error=False, fill_value=None)


###### derivatives ######

s_arr, p_arr, t_arr, r_arr, y_arr, cp_arr, cv_arr, chirho_arr, chit_arr, gamma1_arr, grada_arr = np.load('%s/scvh/scvh_thermo.npy' % CURR_DIR)

get_rho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), r_arr, method='linear', bounds_error=False, fill_value=None)
get_t= RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), t_arr, method='linear', bounds_error=False, fill_value=None)

get_cp = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cp_arr, method='linear', bounds_error=False, fill_value=None)
get_cv = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), cv_arr, method='linear', bounds_error=False, fill_value=None)

get_chirho = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chirho_arr, method='linear', bounds_error=False, fill_value=None)
get_chit = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), chit_arr, method='linear', bounds_error=False, fill_value=None)

get_grada = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), grada_arr, method='linear', bounds_error=False, fill_value=None)
get_gamma1 = RGI((y_arr[:,0][:,0], s_arr[0][:,0], p_arr[0,:][0]), gamma1_arr, method='linear', bounds_error=False, fill_value=None)

def get_rho_sp_tab(s, p, _y, z = 0.0):
    if np.isscalar(s):
        return float(get_rho(np.array([_y, s, p]).T))
    return get_rho(np.array([_y, s, p]).T)

def get_t_sp_tab(s, p, _y, z = 0.0):
    if np.isscalar(s):
        return float(get_t(np.array([_y, s, p]).T))
    return get_t(np.array([_y, s, p]).T)

def get_rhot_sp_tab(s, p, _y, z = 0.0):
    return get_rho_sp_tab(s, p, _y), get_t_sp_tab(s, p, _y)

# def get_s_rhot_tab(r, t, _y, z = 0.0):

#     if np.isscalar(r):
#         return (10**float(interp_s((_y, x(r), y(r, t)))))/erg_to_kbbar
#     else:
#         return (10**interp_s(np.array([_y, x(r), y(r, t)]).T))/erg_to_kbbar

# def get_p_rhot_tab(r, t, _y, z = 0.0):
#     if np.isscalar(r):
#         return float(interp_p((_y, x(r), y(r, t))))
#     else:
#         return interp_p(np.array([_y, x(r), y(r, t)]).T)

# def get_sp_rhot(r, t, _y, z = 0.0):
#     if np.isscalar(r):
#         return (10**float(interp_s((_y, x(r), y(r, t)))))/erg_to_kbbar, float(interp_p((y, x(r), y(r, t))))
#     else:
#         return (10**interp_s(np.array([_y, x(r), y(r, t)]).T))/erg_to_kbbar, interp_p(np.array([y, x(r), y(r, t)]).T)

### rho(P, T, Y), s(P, T, Y) tables ###

#logrho_res_pt, s_res_pt = np.load('%s/scvh/pt_base_comb.npy' % CURR_DIR)
logrho_res_pt, s_res_pt = np.load('%s/scvh/pt_base_new.npy' % CURR_DIR)

# logpvals_pt = np.arange(6, 14.1, 0.1) # new grid
# logtvals_pt = np.arange(2.1, 5.1, 0.05)
# yvals_pt = np.arange(0.15, 0.75, 0.05)

logpvals_pt = p_arr[0][0]
logtvals_pt = np.arange(2.1, 5.05, 0.05)
yvals_pt = y_arr[:,0][:,0]

get_rho_pt_rgi = RGI((logpvals_pt, logtvals_pt, yvals_pt), logrho_res_pt, method='linear', \
            bounds_error=False, fill_value=None)
get_s_pt_rgi = RGI((logpvals_pt, logtvals_pt, yvals_pt), s_res_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_rho_pt_tab(p, t, y, z = 0.0):
    if np.isscalar(p):
        return float(get_rho_pt_rgi(np.array([p, t, y]).T))
    else:
        return get_rho_pt_rgi(np.array([p, t, y]).T)

def get_s_pt_tab(p, t, y, z = 0.0):
    if np.isscalar(p):
        return float(get_s_pt_rgi(np.array([p, t, y]).T))
    else:
        return get_s_pt_rgi(np.array([p, t, y]).T)

### Rho, t basis ###

logp_res_rhot, s_res_rhot = np.load('%s/scvh/rhot_base_new.npy' % CURR_DIR)

logrhovals_rhot = np.linspace(-5, 1.5, 100)
logtvals_rhot = logtvals_pt.copy()
yvals_rhot = yvals_pt.copy()

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
        return float(get_s_rhot_rgi(np.array([rho, t, y]).T))/erg_to_kbbar
    else:
        return get_s_rhot_rgi(np.array([rho, t, y]).T)/erg_to_kbbar

def get_sp_rhot_tab(rho, t, y, z=0.0):
    return get_s_rhot_tab(rho, t, y)/erg_to_kbbar, get_p_rhot_tab(rho, t, y)

### P(s, rho, Y), T(s, rho, Y) tables ###
#p_srho, t_srho = np.load('%s/cms/p_sry.npy' % CURR_DIR), np.load('%s/cms/t_sry.npy' % CURR_DIR)
#logp_res_srho, logt_res_srho = np.load('%s/scvh/srho_base_comb.npy' % CURR_DIR)
logp_res_srho, logt_res_srho = np.load('%s/scvh/srho_base_updated.npy' % CURR_DIR)

# svals_srho = np.arange(5.0, 10.1, 0.05) # new grid
# logrhovals_srho = np.arange(-6, 1.5, 0.05)
# yvals_srho = np.arange(0.15, 0.75, 0.05)

svals_srho = s_arr[0][:,0]
logrhovals_srho = np.linspace(-5, 1.5, 100)
yvals_srho = y_arr[:,0][:,0]

get_p_srho_rgi = RGI((svals_srho, logrhovals_srho, yvals_srho), logp_res_srho, method='linear', \
            bounds_error=False, fill_value=None)
get_t_srho_rgi = RGI((svals_srho, logrhovals_srho, yvals_srho), logt_res_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_p_srho_tab(s, rho, y, z = 0.0):
    if np.isscalar(s):
        return float(get_p_srho_rgi(np.array([s, rho, y]).T))
    else:
        return get_p_srho_rgi(np.array([s, rho, y]).T)

def get_t_srho_tab(s, rho, y, z = 0.0):
    if np.isscalar(s):
        return float(get_t_srho_rgi(np.array([s, rho, y]).T))
    else:
        return get_t_srho_rgi(np.array([s, rho, y]).T)


### error functions ###
def err_rhot(rt_pair, sval, pval, y, z = 0.0):
    rho, t = rt_pair
    s, p = get_s_rhot_tab(rho, t, y), get_p_rhot_tab(rho, t, y)
    sval /= erg_to_kbbar
    return  s/sval - 1, p/pval -1

def err_rho_pt(lgrho, pval, t, y, z = 0.0):
    logp_ = get_p_rhot_tab(lgrho, t, y)
    return logp_/pval - 1

def err_t_rhop(lgt, lgp, rhoval, y, z = 0.0):
    #lgp, lgt = pt_pair
    logrho_ = get_rho_pt(lgp, lgt, y)
    #s *= erg_to_kbbar
    return  logrho_/rhoval - 1

def err_t_sp(logt, logp, s_val, y, z = 0.0):
    s_ = get_s_pt_tab(logp, logt, y)
    s_val /= erg_to_kbbar # in cgs

    return (s_/s_val) - 1

def err_t_srho(lgt, sval, rhoval, yval):
    #logp = get_p_rhot(rhoval, lgt, yval, alg='root')
    s_test = get_s_rhot_tab(rhoval, lgt, yval)*erg_to_kbbar
    return (s_test/sval) - 1

### inversions ###

TBOUNDS = [0, 7] # logrho and logt_sp test passes with [0, 6]
PBOUNDS = [0, 15] # works with 1, 15
RHOBOUNDS = [-7, 1.5] # works with +1 and -2

XTOL = 1e-8

def get_rho_pt(p, t, y, alg='root'):
    if alg == 'root':
        if np.isscalar(p):
            p, t, y = np.array([p]), np.array([t]), np.array([y])
            guess = ideal_xy.get_rho_pt(p, t, y)
            sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t, y))
            return float(sol.x)
        guess = ideal_xy.get_rho_pt(p, t, y)
        sol = root(err_rho_pt, guess, tol=1e-8, method='hybr', args=(p, t, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(p):
            try:
                sol = root_scalar(err_rho_pt, bracket=RHOBOUNDS, xtol=XTOL, method='brenth', args=(p, t, y))
                return sol.root
            except:
                #print('p={}, t={}, y={}'.format(p, t, y))
                raise

        sol = np.array([get_rho_pt(p_, t_, y_) for p_, t_, y_ in zip(p, t, y)])
        return sol

def get_s_pt(p, t, y, z = 0.0):
    rho = get_rho_pt(p, t, y)
    return get_s_rhot_tab(rho, t, y)

def get_t_rhop(rho, p, y, alg='root'):
    if alg == 'root':
        if np.isscalar(rho):
            rho, p, y = np.array([rho]), np.array([p]), np.array([y])
        guess = ideal_xy.get_t_rhop(rho, p, y)
        sol = root(err_t_rhop, guess, tol=1e-8, method='hybr', args=(p, rho, y))
        return sol.x
    elif alg == 'brenth':
        TBOUNDS2 = [0, 4.1]
        if np.isscalar(rho):
            try:
                sol = root_scalar(err_t_rhop, bracket=TBOUNDS2, xtol=XTOL, method='brenth', args=(p, rho, y))
                return sol.root
            except:
                raise

        sol = np.array([get_t_rhop(rho_, p_, y_) for rho_, p_, y_ in zip(rho, p, y)])
        return sol

def get_t_sp(s, p, y, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, p, y = np.array([s]), np.array([p]), np.array([y])
            guess = ideal_xy.get_t_sp(s, p, y)
            sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(p, s, y))
            return float(sol.x)
        guess = ideal_xy.get_t_sp(s, p, y)
        sol = root(err_t_sp, guess, tol=1e-8, method='hybr', args=(p, s, y))
        return sol.x
    elif alg == 'brenth':
        if np.isscalar(s):
            #try:
            sol = root_scalar(err_t_sp, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(p, s, y)) # range should be 2, 5 but doesn't converge for higher z unless it's lower
            return sol.root
            #except:
            #    raise
        sol = np.array([get_t_sp(s_, p_, y_) for s_, p_, y_ in zip(s, p, y)])
        return sol

def get_t_srho(s, rho, y, alg='root'):
    if alg == 'root':
        if np.isscalar(s):
            s, rho, y = np.array([s]), np.array([rho]), np.array([y])
            guess = ideal_xy.get_t_srho(s, rho, y)
            sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
            return float(sol.x)
        guess = ideal_xy.get_t_srho(s, rho, y)
        sol = root(err_t_srho, guess, tol=1e-8, method='hybr', args=(s, rho, y))
        return sol.x

    elif alg == 'brenth':
        if np.isscalar(s):
            try:
                sol = root_scalar(err_t_srho, bracket=TBOUNDS, xtol=XTOL, method='brenth', args=(s, rho, y))
                return sol.root
            except:
                #print('s={}, rho={}, y={}'.format(s, rho, y))
                raise

        sol = np.array([get_t_srho(s_, rho_, y_) for s_, rho_, y_ in zip(s, rho, y)])
        return sol

def get_pt_srho(s, rho, y):
    return get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)

#def get_pt_srho(s, rho, y)

def get_s_rhop(rho, p, y, z = 0.0):
    t = get_t_rhop(rho, p, y)
    #y = cms.n_to_Y(y)
    s = get_s_rhot_tab(rho, t, y)
    return s # in cgs

def get_u_pt(p, t, y, z = 0.0):
    u = eos_scvh.get_logu(p, t, y) # volume law
    return u

def get_u_srho(s, rho, y, z = 0.0):
    # if not tab:
    #     p, t = get_pt_srho(s, rho, y)
    # else:
    p, t = get_p_srho_tab(s, rho, y), get_t_srho_tab(s, rho, y)
    return get_u_pt(p, t, y)

############## derivatives ##############

### pressure gradients ###

def get_dpdy_srho(s, rho, y, z = 0.0, dy=0.01):
    P0 = 10**get_p_srho_tab(s, rho, y)
    P1 = 10**get_p_srho_tab(s, rho, y*(1+dy))

    return (P1 - P0)/(y*dy) #dlogP/dY

def get_dpds_rhoy_srho(s, rho, y, z = 0.0, ds=0.1):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y)
    P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y)

    return (P1 - P0)/(S1 - S0)

### entropy gradients ###

def get_dsdy_rhop_srho(s, rho, y, z = 0.0, ds=0.1, dy=0.1):
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)
    P0 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y)
    P1 = 10**get_p_srho_tab(S1*erg_to_kbbar, rho, y)
    P2 = 10**get_p_srho_tab(S0*erg_to_kbbar, rho, y*(1+dy))

    dpds_rhoy = (P1 - P0)/(S1 - S0)
    dpdy_srho = (P2 - P0)/(y*dy)

    return -dpdy_srho/dpds_rhoy

def get_dsdy_rhot(rho, t, y, z = 0.0, dy=0.01):
    S0 = get_s_rhot_tab(rho, t, y)
    S1 = get_s_rhot_tab(rho, t, y*(1+dy))

    dsdy = (S1 - S0)/(y*dy)
    return dsdy

def get_dsdy_pt(p, t, y, z = 0.0, dy=0.01, tab=True):
    if not tab:
        S1 = get_s_pt(p, t, y)
        S2 = get_s_pt(p, t, y*(1+dy))
    else:
        S1 = get_s_pt_tab(p, t, y)
        S2 = get_s_pt_tab(p, t, y*(1+dy))

    return (S2 - S1)/(y*dy)

def get_dsdt_ry_rhot(rho, t, y, z = 0.0, dt=0.1):
    T0 = 10**t
    T1 = T0*(1+dt)

    S0 = get_s_rhot_tab(rho, np.log10(T0), y)
    S1 = get_s_rhot_tab(rho, np.log10(T1), y)

    return (S1 - S0)/(T1 - T2)

def get_c_v(s, rho, y, z = 0.0, ds=0.1):
    # ds/dlogT_{rho, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_t_srho_tab(S0*erg_to_kbbar, rho, y)
    T1 = get_t_srho_tab(S1*erg_to_kbbar, rho, y)

    return (S1 - S0)/(T1 - T0)

def get_c_p(s, p, y, z = 0.0, ds=0.1):
    # ds/dlogT_{P, Y}
    S0 = s/erg_to_kbbar
    S1 = S0*(1+ds)

    T0 = get_rhot_sp_tab(S0*erg_to_kbbar, p, y)[-1]
    T1 = get_rhot_sp_tab(S1*erg_to_kbbar, p, y)[-1]

    return (S1 - S0)/(T1 - T0)

### energy gradients ###

def get_dudy_srho(s, rho, y, z = 0.0, dy=0.1, tab=True):
    U0 = 10**get_u_srho(s, rho, y, tab)
    U1 = 10**get_u_srho(s, rho, y*(1+dy), tab)
    return (U1 - U0)/(y*dy)


# du/ds_(rho, Y) = T test
def get_duds_rhoy_srho(s, rho, y, z = 0.0, ds=0.1):
    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)
    U0 = 10**get_u_srho(S1*erg_to_kbbar, rho, y)
    U1 = 10**get_u_srho(S2*erg_to_kbbar, rho, y)
    return (U1 - U0)/(S1*ds)

def get_dudrho_sy_srho(s, rho, y, z = 0.0, drho=0.1):
    R1 = 10**rho
    R2 = R1*(1+drho)
    #rho1 = np.log10((10**rho)*(1+drho))
    U0 = 10**get_u_srho(s, np.log10(R1), y)
    U1 = 10**get_u_srho(s, np.log10(R2), y)
    #return (U1 - U0)/(R1*drho)
    return (U1 - U0)/((1/R1) - (1/R2))

### temperature gradients ###

def get_dtdy_srho(s, rho, y, z = 0.0, dy=0.01, tab=True):
    if not tab:
        T0 = 10**get_t_srho(s, rho, y)
        T1 = 10**get_t_srho(s, rho, y*(1+dy))
    else:
        T0 = 10**get_t_srho_tab(s, rho, y)
        T1 = 10**get_t_srho_tab(s, rho, y*(1+dy))

    return (T1 - T0)/(y*dy)

### density gradients ###

def get_drhods_py(s, p, y, z = 0.0, ds=0.01):

    S1 = s/erg_to_kbbar
    S2 = S1*(1+ds)

    rho0 = 10**get_rhot_sp_tab(S1*erg_to_kbbar, p, y)[0]
    rho1 = 10**get_rhot_sp_tab(S2*erg_to_kbbar, p, y)[0]

    drhods = (rho1 - rho0)/(S2 - S1)

    return drhods


def get_drhodt_py(p, t, y, z = 0.0, dt=0.1):
    #y = cms.n_to_Y(x)
    rho0 = get_rho_pt_tab(p, t, y)
    rho1 = get_rho_pt_tab(p, t*(1+dt), y)

    drhodt = (rho1 - rho0)/(t*dt)

    return drhodt

def get_gamma1(s, p, y, z=0.0, dp = 0.01):
    R0 = get_rho_sp_tab(s, p, y)
    R1 = get_rho_sp_tab(s, p*(1+dp), y)
    return (p*dp)/(R1 - R0)