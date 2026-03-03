import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
from scipy.optimize import root, root_scalar, newton, brentq
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u
import importlib
from tqdm import tqdm
import os
from eos import aqua_eos, ppv2_eos, iron2_eos, aqua_mlcp_eos

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
erg_to_kJ = (u.erg/u.gram).to('kJ/g')

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

class ice_eos:
    """
    Class for calculating the equation of state of ices using the ICE EOS model.
    """

    def __init__(self, path_to_data=None, use_mlcp=False):
        """
        Initialize the class with the path to the data files.

        Parameters
        ----------
        path_to_data : str
            Path to the directory containing the data files.
        """

        self.methane = np.load('%s/methane_ammonia/methane_eos_pt_extended.npz' % CURR_DIR)
        self.ammonia = np.load('%s/methane_ammonia/ammonia_eos_pt_extended.npz' % CURR_DIR)

        if use_mlcp:
            self.water = aqua_mlcp_eos
        else:
            self.water = aqua_eos

        self.rho_pt_methane_rgi = RGI((self.methane["logT"], self.methane["logP"]),
                                self.methane["logrho"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

        self.u_pt_methane_rgi = RGI((self.methane["logT"], self.methane["logP"]),
                                self.methane["u"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

        self.s_pt_methane_rgi = RGI((self.methane["logT"], self.methane["logP"]),
                                self.methane["s"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

        # For ammonia:

        self.rho_pt_ammonia_rgi = RGI((self.ammonia["logT"], self.ammonia["logP"]),
                                self.ammonia["logrho"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

        self.u_pt_ammonia_rgi = RGI((self.ammonia["logT"], self.ammonia["logP"]),
                                self.ammonia["u"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

        self.s_pt_ammonia_rgi = RGI((self.ammonia["logT"], self.ammonia["logP"]),
                                self.ammonia["s"],
                                method='linear',
                                bounds_error=False,
                                fill_value=None)

    def guarded_log(self, x):
        ''' Used to calculate ideal enetropy of mixing: xlogx'''
        if np.isscalar(x):
            if x == 0:
                return 0
            elif x  < 0:
                raise ValueError('Number fraction went negative.')
            return x * np.log(x)
        return np.array([self.guarded_log(x_) for x_ in x])

    # --------------------- Volume Addition Law ---------------------
    def get_logrho_pt_val(self, _lgp, _lgt, _zm, _za):
        """
        Interpolates the density for a mixture using the inverted EOS tables
        (in log-space) for methane, ammonia, and water. The independent variables
        are log10(P) and log10(T).

        Parameters
        ----------
        _lgp : float or array_like
            log10(Pressure) value(s) (using the same convention as in your tables).
        _lgt : float or array_like
            log10(Temperature) value(s) (in the table units).
        _zm  : float
            Mass fraction of methane.
        _za  : float
            Mass fraction of ammonia.
            (Water mass fraction is taken as (1 - _zm) * (1 - _za).)

        Returns
        -------
        mixture_logrho : float or array
            The mixture density (log10 [g/cm³]) computed via the linear mixing rule.
            In the low-(T,P) region (logt < 3.0 or logp < 10.0), the result is that
            of water only.
        """
        # Ensure inputs are arrays and broadcast to a common shape:
        logt_arr = np.atleast_1d(_lgt)
        logp_arr = np.atleast_1d(_lgp)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        logp_arr, logt_arr, _zm, _za = np.broadcast_arrays(logp_arr, logt_arr, _zm, _za)

        # Build the interpolation points from logT and logP:
        pts = np.column_stack((logt_arr, logp_arr))

        # Interpolate log10 density from methane and ammonia tables.
        # They are stored as log10(rho); convert them back to linear.
        logrho_methane = self.rho_pt_methane_rgi(pts)
        logrho_ammonia = self.rho_pt_ammonia_rgi(pts)
        rho_methane = 10 ** logrho_methane
        rho_ammonia = 10 ** logrho_ammonia

        # Water EOS: assume water.get_rho_pt_tab returns log10(rho) on the same grid.
        logrho_water = self.water.get_logrho_pt_tab(logp_arr, logt_arr)
        rho_water = 10 ** logrho_water

        # Compute the specific volumes for each component:
        v_water    = 1.0 / rho_water
        v_methane  = 1.0 / rho_methane
        v_ammonia  = 1.0 / rho_ammonia

        f_water   = (1 - _zm) * (1 - _za)
        f_methane = _zm * (1 - _za)
        f_ammonia = _za

        v_mix = f_water * v_water + f_methane * v_methane  + f_ammonia * v_ammonia
        # Convert back to density (log10 scale):
        rho_mix = 1.0 / v_mix
        mixture_logrho = np.log10(rho_mix)

        # Return scalar if inputs were scalars.
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return mixture_logrho.item()
        return mixture_logrho

    # ----------------------------------------------------------
    def get_u_pt_val(self, _lgp, _lgt, _zm, _za):
        """
        Returns the mixture specific internal energy (u_mix, in erg/g)
        by weight-averaging the pure-component internal energies.

        For logt < 3.0 or logp < 10.0 the value of water is returned,
        regardless of the methane and ammonia mass fractions.

        Parameters
        ----------
        _lgp : float or array_like
            log10(Pressure) values.
        _lgt : float or array_like
            log10(Temperature) values.
        _zm : float
            Mass fraction of methane.
        _za : float
            Mass fraction of ammonia.
            (Water mass fraction is (1-_zm)*(1-_za).)

        Returns
        -------
        u_mix : float or array
            Mixture specific internal energy (erg/g).
        """
        # Ensure inputs are arrays and broadcast to a common shape:
        logt_arr = np.atleast_1d(_lgt)
        logp_arr = np.atleast_1d(_lgp)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        logp_arr, logt_arr, _zm, _za = np.broadcast_arrays(logp_arr, logt_arr, _zm, _za)
        pts = np.column_stack((logt_arr, logp_arr))

        # Get pure-component internal energies (assumed in erg/g)
        u_methane = self.u_pt_methane_rgi(pts)
        u_ammonia = self.u_pt_ammonia_rgi(pts)
        u_water = 10 ** self.water.get_logu_pt_tab(logp_arr, logt_arr)

        # Mass fractions:
        f_water   = (1 - _zm) * (1 - _za)
        f_methane = _zm * (1 - _za)
        f_ammonia = _za

        u_mix = f_water * u_water + f_methane * u_methane + f_ammonia * u_ammonia

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return u_mix.item() if hasattr(u_mix, 'item') else u_mix
        return u_mix

    # ----------------------------------------------------------
    def get_s_pt_val(self, _lgp, _lgt, _zm, _za):
        """
        Returns the mixture specific entropy (s_mix in erg/(g·K))
        by combining the pure-component entropies and adding
        the ideal entropy of mixing.

        For logt < 3.0 or logp < 10.0 the value is that of water only,
        regardless of the user-specified _zm and _za.

        The mixing rule is:
        s_mix = f_water*s_water + f_methane*s_methane + f_ammonia*s_ammonia + s_mix^id,
        where s_mix^id is computed from number fractions.

        Parameters
        ----------
        _lgp : float or array_like
            log10(Pressure) values.
        _lgt : float or array_like
            log10(Temperature) values.
        _zm : float
            Mass fraction of methane.
        _za : float
            Mass fraction of ammonia.
            (Water mass fraction is (1-_zm)*(1-_za).)

        Returns
        -------
        s_intrinsic : float or array
            Mixture specific entropy (erg/(g·K)) (NOTE: Without ideal entropy of mixing).
        """
        # Ensure inputs are arrays and broadcast to a common shape:
        logt_arr = np.atleast_1d(_lgt)
        logp_arr = np.atleast_1d(_lgp)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        logp_arr, logt_arr, _zm, _za = np.broadcast_arrays(logp_arr, logt_arr, _zm, _za)
        pts = np.column_stack((logt_arr, logp_arr))

        # Interpolate pure-component entropies (in erg/(g·K)).
        s_methane = self.s_pt_methane_rgi(pts)
        s_ammonia = self.s_pt_ammonia_rgi(pts)
        s_water   = self.water.get_s_pt_tab(logp_arr, logt_arr)

        # Mass fractions:
        f_water   = (1 - _zm) * (1 - _za)
        f_methane = _zm * (1 - _za)
        f_ammonia = _za

        s_intrinsic = f_water * s_water + f_methane * s_methane + f_ammonia * s_ammonia


        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return s_intrinsic.item() if hasattr(s_intrinsic, 'item') else s_intrinsic
        return s_intrinsic

