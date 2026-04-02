import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
import pandas as pd
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from eos import ideal_eos
# from eos import ice_aneos_eos as ice
# for initial guesses in s, p table
from eos import aqua_eos as aqua

mp = amu.to('g')
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
Pa_to_dyn = u.Pa.to('dyn/cm^2')
SI_to_cgs = (u.kg/u.meter**3).to('g/cm^3')
J_to_erg = (u.J / (u.kg * u.K)).to('erg/(K*g)')
J_to_cgs = (u.J/u.kg).to('erg/g')

ideal_water = ideal_eos.IdealEOS(m=18) # ideal EOS for water

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

######### ORIGINAL MAZEVET EOS #########

def read_water_table(file_path):
    # Read the file and skip initial lines without data
    try:
        data = pd.read_csv(
            file_path,
            sep=r'\s+',  # Handle whitespace-separated values
            skiprows=1,  # Skip the header line
            names=[
                "rho", "T", "P", "P_ni", "F_Ni", "U_Ni", "U", "CV_Ni", "S_Ni"
            ]
        )
        # print("File successfully read.")
        # data = data[data['P'] < 0]
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Path to your water_table.txt file
file_path = "eos/mazevet/mlcp_water_table_new.txt"

# Read the data table
data = read_water_table(file_path)

# data
# Display the DataFrame if successfully loaded
# if data is not None:
#     print(data.head())


def grid_data(df, basis):
    # grids data for interpolation
    twoD = {}
    if basis == 'pt':
        shape = df['logp'].nunique(), -1
    elif basis == 'rhot':
        shape = df['T'].nunique(), -1
         # shape = 150, 300

    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD


water_data_rhot = grid_data(data, basis='rhot')
logrhovals_rhot_mlcp = np.log10(water_data_rhot['rho'][0])
logtvals_rhot_mlcp = np.log10(water_data_rhot['T'][:,0])

s_rgi_rhot_mlcp = RGI((logtvals_rhot_mlcp, logrhovals_rhot_mlcp), water_data_rhot['S_Ni']/6/erg_to_kbbar, method='linear', \
            bounds_error=False, fill_value=None)

logp_rgi_rhot_mlcp = RGI((logtvals_rhot_mlcp, logrhovals_rhot_mlcp), np.log10(water_data_rhot['P']*1e12), method='linear', \
            bounds_error=False, fill_value=None)

logu_rgi_rhot_mlcp = RGI((logtvals_rhot_mlcp, logrhovals_rhot_mlcp), np.log10(water_data_rhot['U']), method='linear', \
            bounds_error=False, fill_value=None)

def get_logp_rhot_mlcp(_lgrho, _lgt): # returns in dyn/cm^2
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logp_rgi_rhot_mlcp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_rhot_mlcp(_lgrho, _lgt): # returns in erg/g/K
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_rhot_mlcp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result
def get_logu_pt_mlcp(_lgrho, _lgt): # returns in erg/g
    args = (_lgt, _lgrho)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_rhot_mlcp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

######### ORIGINAL AQUA EOS #########

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def aqua_reader(basis):
    if basis == 'pt':
        cols = ['press', 'temp', 'rho', 'grada', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase']
    elif basis == 'rhot':
        cols = ['rho', 'temp', 'press', 'grada', 's', 'u', 'c', 'mmw', 'x_ion', 'x_d', 'phase']

    tab = np.loadtxt('%s/aqua/aqua_eos_%s_v1_0.dat' % (CURR_DIR, basis))
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
    elif basis == 'rhot':
        shape = df['logrho'].nunique(), -1

    for i in df.keys():
        twoD[i] = np.reshape(np.array(df[i]), shape)
    return twoD

aqua_data_rhot = grid_data(aqua_reader('rhot'), basis='rhot')
logrhovals_rhot_aqua = aqua_data_rhot['logrho'][:,0]
logtvals_rhot_aqua = aqua_data_rhot['logt'][0]

s_rgi_rhot_aqua = RGI((logrhovals_rhot_aqua, logtvals_rhot_aqua), aqua_data_rhot['s'], method='linear', \
            bounds_error=False, fill_value=None)

logp_rgi_rhot_aqua = RGI((logrhovals_rhot_aqua, logtvals_rhot_aqua), aqua_data_rhot['logp'], method='linear', \
            bounds_error=False, fill_value=None)

logu_rgi_rhot_aqua = RGI((logrhovals_rhot_aqua, logtvals_rhot_aqua), aqua_data_rhot['logu'], method='linear', \
            bounds_error=False, fill_value=None)

def get_s_rhot_aqua(_lgrho, _lgt):
    args = (_lgrho, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_rhot_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logp_rhot_aqua(_lgrho, _lgt):
    args = (_lgrho, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logp_rgi_rhot_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_rhot_aqua(_lgrho, _lgt):
    args = (_lgrho, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_rhot_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

aqua_data_pt = grid_data(aqua_reader('pt'), 'pt') # 2D grid

logpvals_pt_aqua = aqua_data_pt['logp'][:,0]
logtvals_pt_aqua = aqua_data_pt['logt'][0]

svals_pt_aqua = aqua_data_pt['s']
logrhovals_pt_aqua = aqua_data_pt['logrho']
loguvals_pt_aqua = aqua_data_pt['logu']

s_rgi_pt_aqua = RGI((logpvals_pt_aqua, logtvals_pt_aqua), svals_pt_aqua, method='linear', \
            bounds_error=False, fill_value=None)

logrho_rgi_pt_aqua = RGI((logpvals_pt_aqua, logtvals_pt_aqua), logrhovals_pt_aqua, method='linear', \
            bounds_error=False, fill_value=None)

logu_rgi_pt_aqua = RGI((logpvals_pt_aqua, logtvals_pt_aqua), loguvals_pt_aqua, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_pt_aqua(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logrho_pt_aqua(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_pt_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_aqua(_lgp, _lgt):
    args = (_lgp, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_rhot_aqua(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

######### COMBINED AQUA+MLCP data #########

s_aqua_mlcp = np.load('eos/mazevet/aqua_mlcp_entropy_correction.npy')
logu_aqua_mlcp = np.load('eos/mazevet/aqua_mlcp_energy_correction.npy')

s_rgi_rhot_aqua_mlcp = RGI((logrhovals_rhot_aqua, logtvals_rhot_aqua), s_aqua_mlcp, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_rhot_aqua_mlcp = RGI((logrhovals_rhot_aqua, logtvals_rhot_aqua), logu_aqua_mlcp, method='linear', \
            bounds_error=False, fill_value=None)

def get_s_rhot_tab(_lgrho, _lgt):
    args = (_lgrho, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_rhot_aqua_mlcp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_rhot_tab(_lgrho, _lgt):
    args = (_lgrho, _lgt)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_rhot_aqua_mlcp(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logp_rhot_tab(_lgrho, _lgt):
    return get_logp_rhot_aqua(_lgrho, _lgt)


######### COMBINED AQUA+MLCP PT data #########

pt_data = np.load('eos/mazevet/aqua_mlcp_pt.npz')

# P, T basis
logpvals_pt = pt_data['logpvals'] # log g/cm^3
logtvals_pt = pt_data['logtvals'] # log K
logrho_grid_pt = pt_data['logrho_pt'] # in dyn/cm2
s_grid_pt = pt_data['s_pt'] # in erg/g/K
logu_grid_pt = pt_data['logu_pt'] # in erg/g

logrho_rgi_pt = RGI((logtvals_pt, logpvals_pt), logrho_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
s_rgi_pt = RGI((logtvals_pt, logpvals_pt), s_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)
logu_rgi_pt = RGI((logtvals_pt, logpvals_pt), logu_grid_pt, method='linear', \
            bounds_error=False, fill_value=None)

# TABLE FUNCTIONS

def get_logrho_pt_tab(_lgp, _lgt): 
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logrho_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_s_pt_tab(_lgp, _lgt): # returns in erg/g/K
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = s_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result

def get_logu_pt_tab(_lgp, _lgt): # returns in erg/g
    args = (_lgt, _lgp)
    v_args = [np.atleast_1d(arg) for arg in args]
    pts = np.column_stack(v_args)
    result = logu_rgi_pt(pts)
    if all(np.isscalar(arg) for arg in args):
        return result.item()
    else:
        return result


######### WRAPPER FUNCTIONS TO BE USED IN INVERSIONS AND METALS_EOS AND Z_MIX EOS #########



# INVERSION FUNCTION

def get_logrho_pt_inv(_lgp, _lgt, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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
        guess = get_logrho_pt_aqua(_lgp, _lgt)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(lgp_i, lgt_i, guess_i):
        def err(_lgrho):
            # Error function for logt(S, logp)
            logp_test = get_logp_rhot_tab(_lgrho, lgt_i)
            return (logp_test/lgp_i) - 1

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
    density, converged = vectorized_root_func(_lgp, _lgt, guess)

    return density, converged

def get_s_pt(_lgp, _lgt):
    logrho = get_logrho_pt_inv(_lgp, _lgt)[0]
    return get_s_rhot_tab(logrho, _lgt)

def get_logu_pt(_lgp, _lgt):
    logrho = get_logrho_pt_inv(_lgp, _lgt)[0]
    return get_logu_rhot_tab(logrho, _lgt)

def get_logt_sp_inv(_s, _lgp, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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

    _s = np.atleast_1d(_s)
    _lgp = np.atleast_1d(_lgp)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgrho = np.broadcast_arrays(_s, _lgp)

    if ideal_guess:
        guess = aqua.get_t_sp_tab(_s, _lgp)
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

    return temperature, converged

def get_logt_srho_inv(_s, _lgrho, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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

    _s = np.atleast_1d(_s)
    _lgrho = np.atleast_1d(_lgrho)

    #_y = _y if self.y_prime else _y / (1 - _z)
    # Ensure inputs are numpy arrays and broadcasted to the same shape
    _s, _lgrho = np.broadcast_arrays(_s, _lgrho)

    if ideal_guess:
        guess = ideal_water.get_t_srho(_s, _lgrho, 0)
    else:
        if arr_guess is None:
            raise ValueError("logt_guess must be provided when ideal_guess is False.")
        guess = arr_guess
   # Define a function to compute root and capture convergence
    def root_func(s_i, lgrho_i, guess_i):
        def err(_lgt):
            # Error function for logt(S, logp)
            
            s_test = get_s_rhot_tab(lgrho_i, _lgt)*erg_to_kbbar
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
    density, converged = vectorized_root_func(_s, _lgrho, guess)

    return density, converged