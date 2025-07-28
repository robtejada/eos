from eos import ideal_eos
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
from tqdm import tqdm

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_kbbar = (u.J / (u.kg * u.K)).to(k_B/mp)

mg = 24.305
si = 28.085
o3 = 48.000

mgsio3 = mg+si+o3 # molecular weight of post-perovskite

# for guesses
ideal_z = ideal_eos.IdealEOS(m=mgsio3)

### S, P ###
s_grid = np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/s_grid.txt')*J_to_kbbar
logpgrid = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/P_grid.txt')*10)

logtvals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/T_table_ppv.txt'))
logrhovals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/rho_table_ppv.txt')*1e-3)
loguvals = np.log10(np.loadtxt('eos/zhang_eos/zhang_multiphase/ppv2/E_table_ppv.txt')*J_to_erg)

rho_rgi = RGI((s_grid, logpgrid), logrhovals, method='linear', bounds_error=False, fill_value=None)
t_rgi = RGI((s_grid, logpgrid), logtvals, method='linear', bounds_error=False, fill_value=None)
u_rgi = RGI((s_grid, logpgrid), loguvals, method='linear', bounds_error=False, fill_value=None)

### P, T ###

pt_data = np.load('eos/zhang_eos/zhang_multiphase/ppv2/zhang_ppv_2024_pt.npz')

logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrhovals'] # in dyn/cm2
s_grid_pt = pt_data['svals'] # in erg/g/K
logu_grid_pt = pt_data['loguvals'] # in erg/g

rho_rgi_pt = RGI((logpvals_pt, logtvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logpvals_pt, logtvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
u_rgi_pt = RGI((logpvals_pt, logtvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

def get_logrho_pt_tab(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi_pt(pts)
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

def get_logu_pt_tab(_lgp, _lgt): # returns in log10 erg/g
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = rho_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
def get_logt_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = t_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_sp_tab(_s, _lgp):
    args = (_s, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = u_rgi(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result


##### P, T inversion function #####

def get_s_pt_inv(_lgp, _lgt, ideal_guess=True, arr_guess=None, method='newton_brentq'):

    """
    Compute the pressure given density, temperature, helium abundance, and metallicity.

    Parameters:
        _lgrho (array_like): Log10 density values.
        _lgt (array_like): Log10 temperature values.
        ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
        logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

    Returns:
        ndarray: Computed temperature values.
    """

    _lgp = np.atleast_1d(_lgp)
    _lgt = np.atleast_1d(_lgt)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _lgp, _lgt = np.broadcast_arrays(_lgp, _lgt)

    if ideal_guess:
        guess = ideal_z.get_s_pt(_lgp, _lgt, 0)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(lgp_i, lgt_i, guess_i):
        def err(_s):
            # Error function for logt(S, logp)
            logt_test = get_logt_sp_tab(_s, lgp_i)
            return (logt_test/lgt_i) - 1

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
    entropy, converged = vectorized_root_func(_lgp, _lgt, guess)

    return entropy

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
        guess = ideal_z.get_p_srho(_s, _lgrho, 0)
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
def get_c_p(_s, _lgp, ds=0.01):
    logt1 = get_logt_sp_tab(_s - ds, _lgp)
    logt2 = get_logt_sp_tab(_s + ds, _lgp)
    return (2 * ds / erg_to_kbbar) / ((logt2 - logt1) * np.log(10))

def get_c_v(_s, _lgrho, ds=0.01):
    logt1 = get_logt_srho(_s - ds, _lgrho)
    logt2 = get_logt_srho(_s + ds, _lgrho)
    return (2 * ds / erg_to_kbbar) / ((logt2 - logt1) * np.log(10))

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