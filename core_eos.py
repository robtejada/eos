from eos import aqua_eos as aqua
from eos import aqua_mlcp_eos as aqua_mlcp
from eos import ppv2_eos as ppv2
from eos import iron2_eos as iron2
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

def get_s_pt_val(_lgp, _lgt, _frock, _firon=0.0):
    s_water = aqua_mlcp.get_s_pt_tab(_lgp, _lgt)
    s_rock = ppv2.get_s_pt_tab(_lgp, _lgt)
    s_iron = iron2.get_s_pt_tab(_lgp, _lgt)

    return s_water*(1 - _frock)*(1 - _firon) + s_rock*_frock*(1 - _firon) + s_iron*_firon

def get_logrho_pt_val(_lgp, _lgt, _frock, _firon=0.0):
    rho_water = 10**aqua_mlcp.get_logrho_pt_tab(_lgp, _lgt)
    rho_rock = 10**ppv2.get_logrho_pt_tab(_lgp, _lgt)
    rho_iron = 10**iron2.get_logrho_pt_tab(_lgp, _lgt)

    rho_mix_inv = (1 - _frock)*(1 - _firon)/rho_water + _frock*(1 - _firon)/rho_rock + _firon/rho_iron

    return np.log10(1/rho_mix_inv)


def get_logu_pt_val(_lgp, _lgt, _frock, _firon=0.0):

    u_water = 10**aqua_mlcp.get_logu_pt_tab(_lgp, _lgt)
    u_rock = 10**ppv2.get_logu_pt_tab(_lgp, _lgt)
    u_iron = 10**iron2.get_logu_pt_tab(_lgp, _lgt)

    return np.log10(u_water*(1 - _frock)*(1 - _firon) + u_rock*_frock*(1 - _firon) + u_iron*_firon)

##### INVERSION FUNCTIONS #####

def get_logt_sp_inv(_s, _lgp, _frock, _firon, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    _s = np.atleast_1d(_s)
    _lgp = np.atleast_1d(_lgp)
    _frock = np.atleast_1d(_frock)
    _firon = np.atleast_1d(_firon)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgp, _frock, _firon = np.broadcast_arrays(_s, _lgp, _frock, _firon)

    if ideal_guess:
        guess = aqua.get_t_sp_tab(_s, _lgp)*(1 - _frock)*(1 - _firon) + ppv2.get_logt_sp_tab(_s, _lgp)*_frock*(1 - _firon) + \
                            _firon*iron2.get_logt_sp_tab(_s, _lgp)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgp_i, _frock_i, _firon_i, guess_i):
        def err(_lgt):
            # Error function for logt(S, logp)

            s_test = get_s_pt_val(lgp_i, _lgt, _frock_i, _firon_i)*erg_to_kbbar
            return (s_test/s_i) - 1

        if method == 'root':
            sol = root(err, guess_i, tol=1e-8)
            if sol.success:
                return sol.x[0], True
            else:
                return np.nan, False  # Assign np.nan to non-converged elements

        elif method == 'newton':
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                #Convergence failed
                return np.nan, False
            except Exception as e:
                #Handle other exceptions
                return np.nan, False

        elif method == 'brentq':
            # Define an initial interval around the guess
            delta = 0.1  # Initial interval half-width
            a = guess_i - delta
            b = guess_i + delta

            # Try to find a valid interval where the function changes sign
            max_attempts = 5
            factor = 2.0  # Factor to expand the interval if needed

            for attempt in range(max_attempts):
                try:
                    fa = err(a)
                    fb = err(b)
                    if np.isnan(fa) or np.isnan(fb):
                        raise ValueError("Function returned NaN.")

                    if fa * fb < 0:
                        # Valid interval found
                        sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                        return sol_root, True
                    else:
                        # Expand the interval and try again
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor  # Increase delta for next iteration
                except ValueError:
                    # If err() cannot be evaluated, expand the interval
                    a -= delta * factor
                    b += delta * factor
                    delta *= factor

        elif method == 'newton_brentq':
            # Try the Newton method first
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                # Fall back to the Brentq method if Newton fails
                delta = 0.1
                a = guess_i - delta
                b = guess_i + delta
                max_attempts = 5
                factor = 2.0

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")
                        if fa * fb < 0:
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    except ValueError:
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor
                return np.nan, False
            # If no valid interval is found after max_attempts
            return np.nan, False
        else:
            raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
    # Vectorize the root_func
    vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

    # Apply the vectorized function
    temperature, converged = vectorized_root_func(_s, _lgp, _frock, _firon, guess)

    return temperature

def get_logrho_sp_inv(_s, _lgp, _frock, _firon):
    logt = get_logt_sp_inv(_s, _lgp, _frock, _firon)
    return get_logrho_pt_val(_lgp, logt, _frock, _firon)

def get_logp_srho_inv(_s, _lgrho, _frock, _firon, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    _s = np.atleast_1d(_s)
    _lgrho = np.atleast_1d(_lgrho)
    _frock = np.atleast_1d(_frock)
    _firon = np.atleast_1d(_firon)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgrho, _frock, _firon = np.broadcast_arrays(_s, _lgrho, _frock, _firon)

    if ideal_guess:
        guess = aqua.get_p_srho_tab(_s, _lgrho)# + ppv2.get_logt_sp_tab(_s, _lgp)*_frock*(1 - _firon) + \
        #                     _firon*iron2.get_logt_sp_tab(_s, _lgp)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgrho_i, _frock_i, _firon_i, guess_i):
        def err(_lgp):
            # Error function for logt(S, logp)

            logrho_test = get_logrho_sp_inv(s_i, _lgp, _frock_i, _firon_i)
            return (logrho_test/lgrho_i) - 1

        if method == 'root':
            sol = root(err, guess_i, tol=1e-8)
            if sol.success:
                return sol.x[0], True
            else:
                return np.nan, False  # Assign np.nan to non-converged elements

        elif method == 'newton':
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                #Convergence failed
                return np.nan, False
            except Exception as e:
                #Handle other exceptions
                return np.nan, False

        elif method == 'brentq':
            # Define an initial interval around the guess
            delta = 0.1  # Initial interval half-width
            a = guess_i - delta
            b = guess_i + delta

            # Try to find a valid interval where the function changes sign
            max_attempts = 5
            factor = 2.0  # Factor to expand the interval if needed

            for attempt in range(max_attempts):
                try:
                    fa = err(a)
                    fb = err(b)
                    if np.isnan(fa) or np.isnan(fb):
                        raise ValueError("Function returned NaN.")

                    if fa * fb < 0:
                        # Valid interval found
                        sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                        return sol_root, True
                    else:
                        # Expand the interval and try again
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor  # Increase delta for next iteration
                except ValueError:
                    # If err() cannot be evaluated, expand the interval
                    a -= delta * factor
                    b += delta * factor
                    delta *= factor

        elif method == 'newton_brentq':
            # Try the Newton method first
            try:
                sol_root = newton(err, x0=guess_i, tol=1e-5, maxiter=100)
                return sol_root, True
            except RuntimeError:
                # Fall back to the Brentq method if Newton fails
                delta = 0.1
                a = guess_i - delta
                b = guess_i + delta
                max_attempts = 5
                factor = 2.0

                for attempt in range(max_attempts):
                    try:
                        fa = err(a)
                        fb = err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")
                        if fa * fb < 0:
                            sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                            return sol_root, True
                        else:
                            a -= delta * factor
                            b += delta * factor
                            delta *= factor
                    except ValueError:
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor
                return np.nan, False
            # If no valid interval is found after max_attempts
            return np.nan, False
        else:
            raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
    # Vectorize the root_func
    vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

    # Apply the vectorized function
    pressure, converged = vectorized_root_func(_s, _lgrho, _frock, _firon, guess)

    return pressure

def get_logt_srho_inv(_s, _lgrho, _frock, _firon):
    logp = get_logp_srho_inv(_s, _lgrho, _frock, _firon)
    return get_logt_sp_inv(_s, logp, _frock, _firon)


#### INVERSION TABLES ####

# sp_data = np.load('eos/metal_mixtures/water_ppv2_iron_sp.npz')

# # S, P basis
# svals_sp = sp_data['s_vals'] # erg/g/K
# logpvals_sp = sp_data['logpvals'] # log K
# frockvals_sp = sp_data['f_rock_vals']
# fironvals_sp = sp_data['f_iron_vals']

# logrho_grid_sp = sp_data['logrho_sp'] # in g/cm^3
# logt_grid_sp = sp_data['logt_sp'] # in K
# logu_grid_sp = sp_data['logu_sp'] # in erg/g

# logt_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp, fironvals_sp), logt_grid_sp, method='linear', \
#             bounds_error=False, fill_value=None)
# logrho_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp, fironvals_sp), logrho_grid_sp, method='linear', \
#             bounds_error=False, fill_value=None)
# logu_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp, fironvals_sp), logu_grid_sp, method='linear', \
#             bounds_error=False, fill_value=None)

# def get_logt_sp_tab(_s, _lgp, _frock, _firon):
#     args = (_s, _lgp, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logt_rgi_sp(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result

# def get_logrho_sp_tab(_s, _lgp, _frock, _firon): # returns in erg/g/K
#     args = (_s, _lgp, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logrho_rgi_sp(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result

# def get_logu_sp_tab(_s, _lgp, _frock, _firon): # returns in erg/g
#     args = (_s, _lgp, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logu_rgi_sp(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result

sp_data = np.load('eos/metal_mixtures/water_ppv2_sp.npz')

# S, P basis
svals_sp = sp_data['s_vals'] # erg/g/K
logpvals_sp = sp_data['logpvals'] # log K
frockvals_sp = sp_data['f_rock_vals']
# fironvals_sp = sp_data['f_iron_vals']

logrho_grid_sp = sp_data['logrho_sp'] # in g/cm^3
logt_grid_sp = sp_data['logt_sp'] # in K
logu_grid_sp = sp_data['logu_sp'] # in erg/g

logt_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp), logt_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logrho_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp), logrho_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_sp = RGI((svals_sp, logpvals_sp, frockvals_sp), logu_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)

def get_logt_sp_tab(_s, _lgp, _frock, _firon=0.0):
    args = (_s, _lgp, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logt_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_tab(_s, _lgp, _frock, _firon=0.0): # returns in erg/g/K
    args = (_s, _lgp, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_sp_tab(_s, _lgp, _frock, _firon=0.0): # returns in erg/g
    args = (_s, _lgp, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result


srho_data = np.load('eos/metal_mixtures/water_ppv2_srho.npz')

# S, Rho basis
svals_srho = srho_data['s_vals'] # erg/g/K
logrhovals_srho = srho_data['logrhovals'] # log K
frockvals_srho = srho_data['f_rock_vals']
# fironvals_srho = srho_data['f_iron_vals']

logp_grid_srho = srho_data['logp_srho'] # in g/cm^3
logt_grid_srho = srho_data['logt_srho'] # in K
logu_grid_srho = srho_data['logu_srho'] # in erg/g

logp_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho), logp_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)
logt_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho), logt_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho), logu_grid_srho, method='linear', \
            bounds_error=False, fill_value=None)

def get_logp_srho_tab(_s, _lgrho, _frock, _firon=0.0):
    args = (_s, _lgrho, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logp_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logt_srho_tab(_s, _lgrho, _frock, _firon=0.0): # returns in erg/g/K
    args = (_s, _lgrho, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logt_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_srho_tab(_s, _lgrho, _frock, _firon=0.0): # returns in erg/g
    args = (_s, _lgrho, _frock)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_srho(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

# srho_data = np.load('eos/metal_mixtures/water_ppv2_iron_srho.npz')

# # S, Rho basis
# svals_srho = srho_data['s_vals'] # erg/g/K
# logrhovals_srho = srho_data['logrhovals'] # log K
# frockvals_srho = srho_data['f_rock_vals']
# fironvals_srho = srho_data['f_iron_vals']

# logp_grid_srho = srho_data['logp_srho'] # in g/cm^3
# logt_grid_srho = srho_data['logt_srho'] # in K
# logu_grid_srho = srho_data['logu_srho'] # in erg/g

# logp_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho, fironvals_srho), logp_grid_srho, method='linear', \
#             bounds_error=False, fill_value=None)
# logt_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho, fironvals_srho), logt_grid_srho, method='linear', \
#             bounds_error=False, fill_value=None)
# logu_rgi_srho = RGI((svals_srho, logrhovals_srho, frockvals_srho, fironvals_srho), logu_grid_srho, method='linear', \
#             bounds_error=False, fill_value=None)

# def get_logp_srho_tab(_s, _lgrho, _frock, _firon):
#     args = (_s, _lgrho, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logp_rgi_srho(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result

# def get_logt_srho_tab(_s, _lgrho, _frock, _firon): # returns in erg/g/K
#     args = (_s, _lgrho, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logt_rgi_srho(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result

# def get_logu_srho_tab(_s, _lgrho, _frock, _firon): # returns in erg/g
#     args = (_s, _lgrho, _frock, _firon)
#     v_args = [np.atleast_1d(arg) for arg in args]
#     pts = np.column_stack(v_args)
#     result = logu_rgi_srho(pts)
#     if all(np.isscalar(arg) for arg in args):
#         return result.item()
#     else:
#         return result


#### RELEVANT DERIVATIVES #####

def get_c_v(_s, _lgrho, _frock, _firon, ds=1e-3):
    # ds/dlogT_{rho, Y}

    lgt2 = get_logt_srho_tab(_s + ds, _lgrho, _frock, _firon)
    lgt1 = get_logt_srho_tab(_s - ds, _lgrho, _frock, _firon)

    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))

def get_c_p(_s, _lgp, _frock, _firon, ds=1e-3):
    # ds/dlogT_{P, Y}

    lgt2 = get_logt_sp_tab(_s + ds, _lgp, _frock, _firon)
    lgt1 = get_logt_sp_tab(_s - ds, _lgp, _frock, _firon)

    return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))