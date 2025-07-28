from eos import fe_eos as iron
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u


mp = amu.to('g') # grams
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
J_K_kg_to_erg_K_g = (u.J / (u.kg * u.K)).to('erg/(K*g)') # specific entropy conversion
J_kg_to_erg_g = (u.J / u.kg).to('erg/g') # specific internal energy conversion
J_to_kbbar = (u.J / (u.kg * u.K)).to(k_B/mp) # specific entropy conversion
kg_to_g = (u.kg/u.m**3).to('g/cm^3')
Pa_to_cgs = u.Pa.to('dyn/cm^2')

from eos import ideal_eos

ideal_water = ideal_eos.IdealEOS(m=56)

### P, T ###
pvals_pt = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Pgrid_Fel.txt')
logpvals_pt = np.log10(pvals_pt[pvals_pt > 3.5e9]*Pa_to_cgs) # avoids negative density values
logtvals_pt = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Tgrid_Fel.txt'))

s_grid_pt = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/s_Fel.txt')[pvals_pt > 3.5e9]*J_K_kg_to_erg_K_g
logrho_grid_pt = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/rho_Fel.txt')[pvals_pt > 3.5e9]*kg_to_g)
logu_grid_pt = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/Eth_Fel.txt')[pvals_pt > 3.5e9]*J_kg_to_erg_g)
cp_grid_pt = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cP_Fel.txt')[pvals_pt > 3.5e9]*J_K_kg_to_erg_K_g
cv_grid_pt = np.loadtxt('eos/zhang_eos/zhang_multiphase/iron2/cV_Fel.txt')[pvals_pt > 3.5e9]*J_K_kg_to_erg_K_g

logrho_rgi_pt = RGI((logpvals_pt, logtvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logpvals_pt, logtvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_pt = RGI((logpvals_pt, logtvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
cp_rgi_pt = RGI((logpvals_pt, logtvals_pt), cp_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
cv_rgi_pt = RGI((logpvals_pt, logtvals_pt), cv_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

### S, P ###

sp_data = np.load('eos/zhang_eos/zhang_multiphase/iron2_sp.npz')

# S, P basis
svals_sp = sp_data['s_vals'] # erg/g/K
logpvals_sp = sp_data['logpvals'] # log K

logrho_grid_sp = sp_data['logrho_sp'] # in g/cm^3
logt_grid_sp = sp_data['logt_sp'] # in K
logu_grid_sp = sp_data['logu_sp'] # in erg/g

logt_rgi_sp = RGI((svals_sp, logpvals_sp), logt_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logrho_rgi_sp = RGI((svals_sp, logpvals_sp), logrho_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_sp = RGI((svals_sp, logpvals_sp), logu_grid_sp, method='linear', \
            bounds_error=False, fill_value=None)

# TABLE FUNCTIONS (P, T)

def get_logrho_pt_tab(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_tab(_lgp, _lgt): # returns in erg/g/K
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_tab(_lgp, _lgt): # returns in erg/g
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_c_p_pt(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = cp_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_c_v_pt(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = cv_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logt_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logt_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_tab(_s, _lgp): # returns in erg/g/K
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_sp_tab(_s, _lgp): # returns in erg/g
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_sp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

##### S, P inversion function #####

def get_logt_sp_inv(_s, _lgp, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    _s = np.atleast_1d(_s)
    _lgp = np.atleast_1d(_lgp)

    _s, _lgp = np.broadcast_arrays(_s, _lgp)

    if ideal_guess:
        guess = iron.get_logt_sp_tab(_s, _lgp)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgp_i, guess_i):
        def err(_lgt):
            # Error function for logt(S, logp)

            s_test = get_s_pt_tab(lgp_i, _lgt)*erg_to_kbbar
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
    temperature, converged = vectorized_root_func(_s, _lgp, guess)

    # s, logrho, logu = get_s_logrho_logu_pt_mixture(_lgp, temperature, _frock, _firon)

    return temperature

def get_logp_srho_inv(_s, _lgrho, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    """
    Compute the pressure given entropy and density.

    Parameters:
        _s (array_like): entropy values in k_B/baryon.
        _lgrho (array_like): Log10 density values.
        ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
        logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

    Returns:
        ndarray: Computed temperature values.
    """

    _s = np.atleast_1d(_s)
    _lgrho = np.atleast_1d(_lgrho)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgrho = np.broadcast_arrays(_s, _lgrho)

    if ideal_guess:
        guess = iron.get_p_srho_tab(_s, _lgrho)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgrho_i, guess_i):
        def err(_lgp):
            # Error function for logt(S, logp)
            logrho_test = get_logrho_sp_tab(s_i, _lgp)
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
    pressure, converged = vectorized_root_func(_s, _lgrho, guess)

    return pressure

def get_logt_srho(_s, _lgrho, ideal_guess=True, arr_guess=None, method='newton_brentq'):
    logp_srho = get_logp_srho_inv(_s, _lgrho, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
    logt_srho = get_logt_sp_tab(_s, logp_srho)
    return logt_srho


### DERIVATIVES ###
def get_c_p(_s, _lgp):
    logt = get_logt_sp_tab(_s, _lgp)
    return get_c_p_pt(_lgp, logt)

def get_c_v(_s, _lgp):
    logt = get_logt_sp_tab(_s, _lgp)
    return get_c_v_pt(_lgp, logt)

# Adiabatic temperature gradient
def get_nabla_ad(_s, _lgp, dp=0.1):

    # Use the tabulated functions
    lgt1 = get_logt_sp_tab(_s, _lgp - dp)
    lgt2 = get_logt_sp_tab(_s, _lgp + dp)

    return (lgt2 - lgt1)/(2 * dp)

def get_gruneisen(_s, _lgrho, drho=0.1):

    # Use the tabulated functions
    lgt1 = get_logt_srho(_s, _lgrho - drho)
    lgt2 = get_logt_srho(_s, _lgrho + drho)

    return (lgt2 - lgt1)/(2 * drho)