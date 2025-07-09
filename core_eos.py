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
import pdb

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)

class rock_eos:
    """
    Class to handle rock EOS calculations.
    """
    def __init__(self, sp_inv = False, srho_inv = False):

        self.water = aqua # water EOS
        self.rock = ppv2 # rock EOS
        self.iron = iron2 # iron EOS

        self.sp_inv = sp_inv
        self.srho_inv = srho_inv

        rgi_args = {'method': 'linear', 'bounds_error': False, 'fill_value': None}

        if not self.sp_inv:

            # self.sp_data = np.load('eos/metal_mixtures/water_ppv2_sp.npz')
            self.sp_data = np.load('eos/metal_mixtures/ppv2_iron2_sp.npz')
            self.rhot_data = np.load('eos/metal_mixtures/ppv2_iron2_rhot.npz')

            # S, P basis
            self.svals_sp = self.sp_data['s_vals'] # erg/g/K
            self.logpvals_sp = self.sp_data['logpvals'] # log K
            #self.frockvals_sp = self.sp_data['f_rock_vals']
            self.fironvals_sp = self.sp_data['f_iron_vals']

            self.logrho_grid_sp = self.sp_data['logrho_sp'] # in g/cm^3
            self.logt_grid_sp = self.sp_data['logt_sp'] # in K
            self.logu_grid_sp = self.sp_data['logu_sp'] # in erg/g

            # # 1-D independent grids (rho, T)
            self.logrhovals_rhot = self.rhot_data['logrhovals'] # log10 g/cc
            self.logtvals_rhot = self.rhot_data['logtvals']
            self.fironvals_rhot = self.rhot_data['f_iron_vals']
            # 4-D dependent grids (rho T)
            self.s_rhot_tab = self.rhot_data['s_rhot'] # erg/g/K
            self.logp_rhot_tab = self.rhot_data['logp_rhot']

            self.logt_rgi_sp = RGI((self.svals_sp, self.logpvals_sp,
                                    #self.frockvals_sp
                                    self.fironvals_sp
                                    ), self.logt_grid_sp, **rgi_args)
            self.logrho_rgi_sp = RGI((self.svals_sp, self.logpvals_sp,
                                       self.fironvals_sp
                                       ), self.logrho_grid_sp, **rgi_args)
            self.logu_rgi_sp = RGI((self.svals_sp, self.logpvals_sp,
                                    self.fironvals_sp
                                    ), self.logu_grid_sp, **rgi_args)

            self.s_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.fironvals_rhot),
                                    self.s_rhot_tab, **rgi_args)
            self.logp_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.fironvals_rhot),
                                    self.logp_rhot_tab, **rgi_args)

            if not self.srho_inv:

                self.srho_data = np.load('eos/metal_mixtures/ppv2_iron2_srho.npz')
                self.svals_srho = self.srho_data['s_vals'] # erg/g/K
                self.logrhovals_srho = self.srho_data['logrhovals'] # log K
                #self.frockvals_srho = self.srho_data['f_rock_vals']
                self.fironvals_srho = self.srho_data['f_iron_vals']

                self.logp_grid_srho = self.srho_data['logp_srho'] # in g/cm^3
                self.logt_grid_srho = self.srho_data['logt_srho'] # in K
                self.logu_grid_srho = self.srho_data['logu_srho'] # in erg/g

                self.logp_rgi_srho = RGI((self.svals_srho, self.logrhovals_srho,
                                          self.fironvals_srho
                                          ), self.logp_grid_srho, method='linear', \
                            bounds_error=False, fill_value=None)
                self.logt_rgi_srho = RGI((self.svals_srho, self.logrhovals_srho,
                                          self.fironvals_srho
                                          ), self.logt_grid_srho, method='linear', \
                            bounds_error=False, fill_value=None)
                self.logu_rgi_srho = RGI((self.svals_srho, self.logrhovals_srho,
                                          self.fironvals_srho
                                          ), self.logu_grid_srho, method='linear', \
                            bounds_error=False, fill_value=None)

    def get_s_pt_val(self, _lgp, _lgt, _frock, _firon=0.0):
        s_water = self.water.get_s_pt_tab(_lgp, _lgt)
        s_rock = self.rock.get_s_pt_tab(_lgp, _lgt)
        s_iron = self.iron.get_s_pt_tab(_lgp, _lgt)

        # output in erg/g/K
        return s_water * (1 - _frock) * (1 - _firon) + s_rock * _frock * (1 - _firon) + s_iron * _firon

    def get_logrho_pt_val(self, _lgp, _lgt, _frock, _firon=0.0):
        rho_water = 10 ** self.water.get_logrho_pt_tab(_lgp, _lgt)
        rho_rock = 10 ** self.rock.get_logrho_pt_tab(_lgp, _lgt)
        rho_iron = 10 ** self.iron.get_logrho_pt_tab(_lgp, _lgt)

        v_mix_inv = (1 - _frock) * (1 - _firon) / rho_water + _frock * (1 - _firon) / rho_rock + _firon / rho_iron

        # output in log10 g/cm^3
        return np.log10(1/v_mix_inv)


    def get_logu_pt_val(self, _lgp, _lgt, _frock, _firon=0.0):
        u_water = 10 ** self.water.get_logu_pt_tab(_lgp, _lgt)
        u_rock = 10 ** self.rock.get_logu_pt_tab(_lgp, _lgt)
        u_iron = 10 ** self.iron.get_logu_pt_tab(_lgp, _lgt)

        # output in erg/g
        return np.log10(u_water * (1 - _frock) * (1 - _firon) + u_rock * _frock * (1 - _firon) + u_iron * _firon)

    #### INVERSION TABLES ####

    def get_logt_sp_tab(self, _s, _lgp, _frock=0.0, _firon=0.0):
        args = (_s, _lgp, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logt_rgi_sp(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_sp_tab(self, _s, _lgp, _frock=0.0, _firon=0.0): # returns in erg/g/K
        args = (_s, _lgp, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logrho_rgi_sp(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

   # logrho, logt tables
    def get_s_rhot_tab(self, _lgrho, _lgt, _frock=0.0, _firon=0.0):
        args = (_lgrho, _lgt, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logp_rhot_tab(self, _lgrho, _lgt, _frock=0.0, _firon=0.0):
        args = (_lgrho, _lgt, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logu_sp_tab(self, _s, _lgp, _frock=0.0, _firon=0.0): # returns in erg/g
        args = (_s, _lgp, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logu_rgi_sp(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result


    def get_logp_srho_tab(self, _s, _lgrho, _frock=0.0, _firon=0.0):
        args = (_s, _lgrho, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logp_rgi_srho(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logt_srho_tab(self, _s, _lgrho, _frock=0.0, _firon=0.0): # returns in erg/g/K
        args = (_s, _lgrho, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logt_rgi_srho(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logu_srho_tab(self, _s, _lgrho, _frock=0.0, _firon=0.0): # returns in erg/g
        args = (_s, _lgrho, _firon)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logu_rgi_srho(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    ##### INVERSION FUNCTIONS #####

    def get_logt_sp_inv(self, _s, _lgp, _frock, _firon, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        _s = np.atleast_1d(_s)
        _lgp = np.atleast_1d(_lgp)
        _frock = np.atleast_1d(_frock)
        _firon = np.atleast_1d(_firon)

        #_y = _y if self.y_prime else _y / (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgp, _frock, _firon = np.broadcast_arrays(_s, _lgp, _frock, _firon)

        if ideal_guess:
            guess = aqua.get_t_sp_tab(_s, _lgp)
            # *(1 - _frock)*(1 - _firon) + ppv2.get_logt_sp_tab(_s, _lgp)*_frock*(1 - _firon) + \
            #                     _firon*iron2.get_logt_sp_tab(_s, _lgp)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
        # Define a function to compute root and capture convergence
        def root_func(s_i, lgp_i, _frock_i, _firon_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)

                s_test = self.get_s_pt_val(lgp_i, _lgt, _frock_i, _firon_i) * erg_to_kbbar
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

    def get_logrho_sp_inv(self, _s, _lgp, _frock, _firon):
        logt = self.get_logt_sp_inv(_s, _lgp, _frock, _firon)
        # if len(logt) == 1:
        #     return get_logrho_pt_val(_lgp, logt[0], _frock, _firon)
        return self.get_logrho_pt_val(_lgp, logt, _frock, _firon)

    def get_logp_rhot_inv(self, _lgrho, _lgt, _frock, _firon, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the pressure given density, temperature, helium abundance, and metallicity.

        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgt (array_like): Log10 temperature values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _lgrho = np.atleast_1d(_lgrho)
        _lgt = np.atleast_1d(_lgt)
        _frock = np.atleast_1d(_frock)
        _firon = np.atleast_1d(_firon)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgt, _frock, _firon = np.broadcast_arrays(_lgrho, _lgt, _frock, _firon)

        if ideal_guess:
            # guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
            guess = self.water.get_logp_rhot_tab(_lgrho, _lgt)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
       # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgt_i, frock_i, firon_i, guess_i):
            def err(_lgp):
                #pdb.set_trace()
                logrho_test = self.get_logrho_pt_val(_lgp, lgt_i, frock_i, firon_i)
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
        pressure, converged = vectorized_root_func(_lgrho, _lgt, _frock, _firon, guess)

        return pressure

    def get_logp_srho_inv(self, _s, _lgrho, _frock, _firon, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _frock = np.atleast_1d(_frock)
        _firon = np.atleast_1d(_firon)

        #_y = _y if self.y_prime else _y / (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _frock, _firon = np.broadcast_arrays(_s, _lgrho, _frock, _firon)

        if ideal_guess:
            guess = self.water.get_logp_srho_tab(_s, _lgrho)# + ppv2.get_logt_sp_tab(_s, _lgp)*_frock*(1 - _firon) + \
            #                     _firon*iron2.get_logt_sp_tab(_s, _lgp)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, _frock_i, _firon_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                #logt_test = self.get_logt_sp_inv(s_i, _lgp, _frock_i, _firon_i)

                logrho_test = self.get_logrho_sp_tab(s_i, _lgp, _frock_i, _firon_i)
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

    def get_logt_srho_inv(self, _s, _lgrho, _frock, _firon):
        logp = self.get_logp_srho_inv(_s, _lgrho, _frock, _firon)
        return self.get_logt_sp_inv(_s, logp, _frock, _firon)


    #### RELEVANT DERIVATIVES #####

    def get_c_v_srho(self, _s, _lgrho, _firon, ds=1e-3):
        # ds/dlogT_{rho, Y}

        lgt2 = self.get_logt_srho_tab(_s + ds, _lgrho, _firon=_firon)
        lgt1 = self.get_logt_srho_tab(_s - ds, _lgrho, _firon=_firon)

        return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))

    def get_c_v_rhot(self, _lgrho, _lgt, _firon, dt=1e-3):

        s1 = self.get_s_rhot_tab(_lgrho, _lgt - dt, _firon=_firon)
        s2 = self.get_s_rhot_tab(_lgrho, _lgt + dt, _firon=_firon)

        return (s2 - s1) / (2 * dt * np.log(10))

    def get_c_p(self, _s, _lgp, _firon, ds=1e-3):
        # ds/dlogT_{P, Y}

        lgt2 = self.get_logt_sp_tab(_s + ds, _lgp, _firon=_firon)
        lgt1 = self.get_logt_sp_tab(_s - ds, _lgp, _firon=_firon)

        return (2 * ds / erg_to_kbbar)/((lgt2 - lgt1) * np.log(10))