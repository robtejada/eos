import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import eos.const as const
import pdb
import pandas as pd
from tqdm import tqdm
from numba import njit
from eos import ideal_eos, metals_eos, ice_eos
from eos import ideal_eos, metals_eos, scvh_eos
from eos.smooth import smooth_eos_table, hampel_filter_1d
from scipy.optimize import root, newton, brentq, brenth, minimize, least_squares
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_xy = ideal_eos.IdealHHeMix()

mh = 1
mhe = 4.0026

##### useful unit conversions #####

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

log10_to_loge = np.log(10)

class hhe_eos:
    def __init__(self, hhe_eos, smooth_hhe=False):
        self.hhe_eos = hhe_eos

        if self.hhe_eos == 'cms':
            self.hdata = self.grid_data(self.table_reader('TABLE_H_TP_v1'))
        elif self.hhe_eos == 'cd':
            self.hdata = self.grid_data(self.table_reader('TABLE_H_TP_effective'))

        self.hedata = self.grid_data(self.table_reader('TABLE_HE_TP_v1'))

        self.logpvals = self.hdata['logp'][0]
        self.logtvals = self.hdata['logt'][:,0]

        self.svals_h = self.hdata['logs']
        self.logrhovals_h = self.hdata['logrho']
        self.loguvals_h = self.hdata['logu']

        self.svals_he = self.hedata['logs']
        self.logrhovals_he = self.hedata['logrho']
        self.loguvals_he = self.hedata['logu']

        if smooth_hhe:
            # --- Smooth H tables before RGI creation ---
            h_grids = {'logrho': self.logrhovals_h, 'logs': self.svals_h, 'logu': self.loguvals_h}
            h_smooth = smooth_eos_table(h_grids, self.logtvals, self.logpvals)
            self.svals_h = h_smooth['logs']
            self.logrhovals_h = h_smooth['logrho']
            self.loguvals_h = h_smooth['logu']

            # --- Smooth He tables before RGI creation ---
            he_grids = {'logrho': self.logrhovals_he, 'logs': self.svals_he, 'logu': self.loguvals_he}
            he_smooth = smooth_eos_table(he_grids, self.logtvals, self.logpvals)
            self.svals_he = he_smooth['logs']
            self.logrhovals_he = he_smooth['logrho']
            self.loguvals_he = he_smooth['logu']

        self.data_hc = pd.read_csv(f'{CURR_DIR}/cms/HG23_Vmix_Smix_Umix.csv', delimiter=',')
        self.data_hc = self.data_hc[(self.data_hc['LOGT'] <= 6.0) & (self.data_hc['LOGT'] != 2.8)].copy()
        self.data_hc = self.data_hc.rename(columns={'LOGT': 'logt', 'LOGP': 'logp'}).sort_values(by=['logt', 'logp'])

        self.grid_hc = self.grid_data(self.data_hc)
        self.svals_hc = self.grid_hc['Smix']
        self.uvals_hc = self.grid_hc['Umix']

        self.logpvals_hc = self.grid_hc['logp'][0]
        self.logtvals_hc = self.grid_hc['logt'][:,0]

        #### H data ####

        self.get_s_h_rgi = RGI((self.logtvals, self.logpvals), self.svals_h, method='linear', bounds_error=False, fill_value=None)
        self.get_logrho_h_rgi = RGI((self.logtvals, self.logpvals), self.logrhovals_h, method='linear', bounds_error=False, fill_value=None)
        self.get_logu_h_rgi = RGI((self.logtvals, self.logpvals), self.loguvals_h, method='linear', bounds_error=False, fill_value=None)

        #### He data ####

        self.get_s_he_rgi = RGI((self.logtvals, self.logpvals), self.svals_he, method='linear', bounds_error=False, fill_value=None)
        self.get_logrho_he_rgi = RGI((self.logtvals, self.logpvals), self.logrhovals_he, method='linear', bounds_error=False, fill_value=None)
        self.get_logu_he_rgi = RGI((self.logtvals, self.logpvals), self.loguvals_he, method='linear', bounds_error=False, fill_value=None)


        #### Non-ideal mixing terms for VAL ####
        self.smix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Smix'], method='linear', bounds_error=False, fill_value=None) # Smix will be in cgs... not log cgs.
        self.vmix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Vmix'], method='linear', bounds_error=False, fill_value=None)
        self.umix_interp_rgi = RGI((self.logtvals_hc, self.logpvals_hc), self.grid_hc['Umix'], method='linear', bounds_error=False, fill_value=None)


    def table_reader(self, tab_name):

        cols = ['logt', 'logp', 'logrho', 'logu', 'logs', 'dlrho/dlT_P', 'dlrho/dlP_T',
                'dlS/dlT_P', 'dlS/dlP_T', 'grad_ad']

        if self.hhe_eos == 'cms':
            tab = np.loadtxt(f'{CURR_DIR}/cms/DirEOS2019/{tab_name}', comments='#')
        elif self.hhe_eos == 'cd':
            tab = np.loadtxt(f'{CURR_DIR}/cms/DirEOS2021/{tab_name}', comments='#')
        else:
            raise ValueError(f"Invalid hhe_eos value '{self.hhe_eos}'. Only 'cms' and 'cd' are supported.")

        tab_df = pd.DataFrame(tab, columns=cols)
        # Explicitly make a copy of the slice
        data = tab_df[(tab_df['logt'] <= 6) & (tab_df['logt'] != 2.8)].copy()

        data['logp'] += 10  # 1 GPa = 1e10 cgs
        data['logu'] += 10  # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g
        data['logs'] += 10  # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g

        return data

    def grid_data(self, df):
    # grids data for interpolation
        twoD = {}
        shape = df['logt'].nunique(), -1
        for i in df.keys():
            twoD[i] = np.reshape(np.array(df[i]), shape)
        return twoD


    def _interpolate(self, interpolator, _lgp, _lgt):
        args = (_lgt, _lgp)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = interpolator(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def smix_interp(self, _lgp, _lgt):
        return self._interpolate(self.smix_interp_rgi, _lgp, _lgt)

    def vmix_interp(self, _lgp, _lgt):
        return self._interpolate(self.vmix_interp_rgi, _lgp, _lgt)

    def umix_interp(self, _lgp, _lgt):
        return self._interpolate(self.umix_interp_rgi, _lgp, _lgt)

    def get_s_h(self, _lgp, _lgt):
        return self._interpolate(self.get_s_h_rgi, _lgp, _lgt)

    def get_logrho_h(self, _lgp, _lgt):
        return self._interpolate(self.get_logrho_h_rgi, _lgp, _lgt)

    def get_logu_h(self, _lgp, _lgt):
        return self._interpolate(self.get_logu_h_rgi, _lgp, _lgt)

    def get_s_he(self, _lgp, _lgt):
        return self._interpolate(self.get_s_he_rgi, _lgp, _lgt)

    def get_logrho_he(self, _lgp, _lgt):
        return self._interpolate(self.get_logrho_he_rgi, _lgp, _lgt)

    def get_logu_he(self, _lgp, _lgt):
        return self._interpolate(self.get_logu_he_rgi, _lgp, _lgt)


class z_eos:
    """Reads heavy-element EOS tables directly from raw files.

    Supported species (via `species` parameter):
      - 'water'    : AQUA water EOS (Haldemann et al. 2020), P-T and rho-T bases
      - 'methane'  : CH4 EOS (Setzmann + DFT blend), P-T basis only
      - 'ammonia'  : NH3 EOS (Gao + DFT blend), P-T basis only
      - 'mg2sio4'  : Mg2SiO4 forsterite ANEOS, P-T basis only

    The ``aqua_version`` parameter selects which AQUA table to load:
      - 'revised'  : Cano Amoros et al. — revised entropies (default), P-T only
      - 'original' : Haldemann et al. 2020 — original table, P-T and rho-T

    In ``val_mixtures`` / ``hhe_z_mixtures``, use 'water_revised' or
    'water' in the species_list to select the table version.

    Units are consistent with hhe_eos:
      - logp : log10(P) in dyn/cm^2
      - logt : log10(T) in K
      - logrho : log10(rho) in g/cm^3
      - logs : log10(S) in erg/g/K  (NaN where S <= 0)
      - logu : log10(U) in erg/g
    """

    # Unit conversions: AQUA raw tables are in SI
    _Pa_to_dyn = 10.0             # 1 Pa = 10 dyn/cm^2
    _kgm3_to_gcm3 = 1e-3          # 1 kg/m^3 = 1e-3 g/cm^3
    _J_kgK_to_erg_gK = 1e4        # 1 J/(kg*K) = 1e4 erg/(g*K)
    _J_kg_to_erg_g = 1e4          # 1 J/kg = 1e4 erg/g
    _GPa_to_dyncm2 = 1e10         # 1 GPa = 1e10 dyn/cm^2

    # Mean molecular weights for ideal-gas initial guesses
    _SPECIES_MU = {
        'water':   18.015,   # H2O
        'methane': 16.043,   # CH4
        'ammonia': 17.031,   # NH3
        'mg2sio4': 140.69,   # Mg2SiO4 (forsterite)
    }

    def __init__(self, species='water', smooth_z=False, aqua_version='revised'):
        self.species = species
        self.aqua_version = aqua_version

        # Ideal EOS for initial guesses in inversions
        mu = self._SPECIES_MU.get(species, 18.0)
        self._ideal = ideal_eos.IdealEOS(mu)

        if species == 'water':
            if aqua_version == 'revised':
                self._load_aqua_revised(smooth_z)
            elif aqua_version == 'original':
                self._load_aqua(smooth_z)
            else:
                raise ValueError(f"Unknown aqua_version '{aqua_version}'. "
                                 f"Use 'revised' or 'original'.")
        elif species == 'methane':
            self._load_ch4_nh3('methane', smooth_z)
        elif species == 'ammonia':
            self._load_ch4_nh3('ammonia', smooth_z)
        elif species == 'mg2sio4':
            self._load_mg2sio4(smooth_z)
        else:
            raise ValueError(f"Unknown z_eos species '{species}'. "
                             f"Use 'water', 'methane', 'ammonia', or 'mg2sio4'.")

    # -----------------------------------------------------------------
    # AQUA loader
    # -----------------------------------------------------------------
    def _load_aqua(self, smooth_z):
        # --- P-T basis ---
        pt_cols = ['press', 'temp', 'rho', 'grada', 's', 'u',
                   'c', 'mmw', 'x_ion', 'x_d', 'phase']
        pt_raw = np.loadtxt(f'{CURR_DIR}/aqua/aqua_eos_pt_v1_0.dat', skiprows=19)
        pt_df = pd.DataFrame(pt_raw, columns=pt_cols)

        pt_df['logp'] = np.log10(pt_df['press'] * self._Pa_to_dyn)
        pt_df['logt'] = np.log10(pt_df['temp'])
        pt_df['logrho'] = np.log10(pt_df['rho'] * self._kgm3_to_gcm3)
        pt_df['logu'] = np.log10(pt_df['u'] * self._J_kg_to_erg_g)

        s_cgs = pt_df['s'].values * self._J_kgK_to_erg_gK
        with np.errstate(invalid='ignore', divide='ignore'):
            pt_df['logs'] = np.where(s_cgs > 0, np.log10(s_cgs), np.nan)

        n_p_pt = pt_df['logp'].nunique()
        shape_pt = (n_p_pt, -1)

        self.logpvals_pt = np.reshape(pt_df['logp'].values, shape_pt)[:, 0]
        self.logtvals_pt = np.reshape(pt_df['logt'].values, shape_pt)[0, :]

        self.logrho_pt = np.reshape(pt_df['logrho'].values, shape_pt)
        self.logs_pt = np.reshape(pt_df['logs'].values, shape_pt)
        self.logu_pt = np.reshape(pt_df['logu'].values, shape_pt)
        self.phase_pt = np.reshape(pt_df['phase'].values, shape_pt)

        # --- rho-T basis ---
        rhot_cols = ['rho', 'temp', 'press', 'grada', 's', 'u',
                     'c', 'mmw', 'x_ion', 'x_d', 'phase']
        rhot_raw = np.loadtxt(f'{CURR_DIR}/aqua/aqua_eos_rhot_v1_0.dat', skiprows=21)
        rhot_df = pd.DataFrame(rhot_raw, columns=rhot_cols)

        rhot_df['logp'] = np.log10(rhot_df['press'] * self._Pa_to_dyn)
        rhot_df['logt'] = np.log10(rhot_df['temp'])
        rhot_df['logrho'] = np.log10(rhot_df['rho'] * self._kgm3_to_gcm3)
        rhot_df['logu'] = np.log10(rhot_df['u'] * self._J_kg_to_erg_g)

        s_cgs_rhot = rhot_df['s'].values * self._J_kgK_to_erg_gK
        with np.errstate(invalid='ignore', divide='ignore'):
            rhot_df['logs'] = np.where(s_cgs_rhot > 0, np.log10(s_cgs_rhot), np.nan)

        n_rho_rhot = rhot_df['logrho'].nunique()
        shape_rhot = (n_rho_rhot, -1)

        self.logrhovals_rhot = np.reshape(rhot_df['logrho'].values, shape_rhot)[:, 0]
        self.logtvals_rhot = np.reshape(rhot_df['logt'].values, shape_rhot)[0, :]

        self.logp_rhot = np.reshape(rhot_df['logp'].values, shape_rhot)
        self.logs_rhot = np.reshape(rhot_df['logs'].values, shape_rhot)
        self.logu_rhot = np.reshape(rhot_df['logu'].values, shape_rhot)
        self.phase_rhot = np.reshape(rhot_df['phase'].values, shape_rhot)

        if smooth_z:
            self._smooth_aqua_lowp_lowt()

        # --- Build RGI interpolators (P-T basis) ---
        self.logrho_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                  self.logrho_pt, method='linear',
                                  bounds_error=False, fill_value=None)
        self.logs_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                self.logs_pt, method='linear',
                                bounds_error=False, fill_value=None)
        self.logu_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                self.logu_pt, method='linear',
                                bounds_error=False, fill_value=None)

        # --- Build RGI interpolators (rho-T basis) ---
        self.logp_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot),
                                  self.logp_rhot, method='linear',
                                  bounds_error=False, fill_value=None)
        self.logs_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot),
                                  self.logs_rhot, method='linear',
                                  bounds_error=False, fill_value=None)
        self.logu_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot),
                                  self.logu_rhot, method='linear',
                                  bounds_error=False, fill_value=None)

    # -----------------------------------------------------------------
    # Revised AQUA loader (Cano Amoros et al.)
    # -----------------------------------------------------------------
    def _load_aqua_revised(self, smooth_z):
        """Load the revised AQUA EOS (P-T basis only) from CSV.

        Revised entropies from Mazevet et al. (2021) and new superionic
        data from French et al. (2016).  Sentinel entropy = -1 is
        replaced with NaN.
        """
        csv_path = f'{CURR_DIR}/aqua/aqua_revised_amoros/AQUA_revised_eos_pt.csv'
        pt_df = pd.read_csv(csv_path)

        # Rename columns to match internal convention
        pt_df.rename(columns={
            'pressure_Pa': 'press',
            'temperature_K': 'temp',
            'density_kg_m3': 'rho',
            'entropy_J_kgK': 's',
            'internal_energy_J_kg': 'u',
        }, inplace=True)

        # Replace sentinel entropy values
        pt_df.loc[pt_df['s'] == -1, 's'] = np.nan

        # Unit conversions (SI → CGS)
        pt_df['logp'] = np.log10(pt_df['press'] * self._Pa_to_dyn)
        pt_df['logt'] = np.log10(pt_df['temp'])
        pt_df['logrho'] = np.log10(pt_df['rho'] * self._kgm3_to_gcm3)
        pt_df['logu'] = np.log10(pt_df['u'] * self._J_kg_to_erg_g)

        s_cgs = pt_df['s'].values * self._J_kgK_to_erg_gK
        with np.errstate(invalid='ignore', divide='ignore'):
            pt_df['logs'] = np.where(s_cgs > 0, np.log10(s_cgs), np.nan)

        # Reshape to 2D grid
        n_p_pt = pt_df['logp'].nunique()
        shape_pt = (n_p_pt, -1)

        self.logpvals_pt = np.reshape(pt_df['logp'].values, shape_pt)[:, 0]
        self.logtvals_pt = np.reshape(pt_df['logt'].values, shape_pt)[0, :]

        self.logrho_pt = np.reshape(pt_df['logrho'].values, shape_pt)
        self.logs_pt = np.reshape(pt_df['logs'].values, shape_pt)
        self.logu_pt = np.reshape(pt_df['logu'].values, shape_pt)
        self.phase_pt = np.reshape(pt_df['phase'].values, shape_pt)

        # Additional fields from revised table
        self.flag_pt = np.reshape(pt_df['flag'].values, shape_pt)
        self.ad_grad_pt = np.reshape(pt_df['ad_grad'].values, shape_pt)

        # No rho-T basis available for revised table
        self.has_rhot = False

        if smooth_z:
            self._smooth_aqua_lowp_lowt()

        # Build P-T RGI interpolators
        self.logrho_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                  self.logrho_pt, method='linear',
                                  bounds_error=False, fill_value=None)
        self.logs_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                self.logs_pt, method='linear',
                                bounds_error=False, fill_value=None)
        self.logu_pt_rgi = RGI((self.logpvals_pt, self.logtvals_pt),
                                self.logu_pt, method='linear',
                                bounds_error=False, fill_value=None)

    # -----------------------------------------------------------------
    # CH4 / NH3 loader
    # -----------------------------------------------------------------
    def _load_ch4_nh3(self, molecule, smooth_z):
        if molecule == 'methane':
            data = np.load(f'{CURR_DIR}/methane_ammonia/methane_eos_pt_extended.npz')
            from eos.ch4 import smooth_pt_tables
        else:
            data = np.load(f'{CURR_DIR}/methane_ammonia/ammonia_eos_pt_extended.npz')
            from eos.nh3 import smooth_pt_tables

        # NPZ keys: logT, logP, logrho, s, u — already in CGS
        # Grid is (n_T, n_P); logrho is log10(g/cm³),
        # s is linear erg/(g·K), u is linear erg/g
        logt_1d = data['logT']
        logp_1d = data['logP']
        logrho_raw = data['logrho']
        s_raw = data['s']
        u_raw = data['u']

        if smooth_z:
            logrho_raw, s_raw, u_raw = smooth_pt_tables(
                logrho_raw, s_raw, u_raw, logt_1d, logp_1d)

        # Convert to log form consistent with hhe_eos
        with np.errstate(invalid='ignore', divide='ignore'):
            logs_raw = np.where(s_raw > 0, np.log10(s_raw), np.nan)
            logu_raw = np.where(u_raw > 0, np.log10(u_raw), np.nan)

        # Store with same attribute names as AQUA, but axes are (logT, logP)
        self.logtvals_pt = logt_1d
        self.logpvals_pt = logp_1d
        self.logrho_pt = logrho_raw     # (n_T, n_P)
        self.logs_pt = logs_raw
        self.logu_pt = logu_raw

        # Build RGI interpolators — axes are (logT, logP) for CH4/NH3
        rgi_kw = dict(method='linear', bounds_error=False, fill_value=None)
        self.logrho_pt_rgi = RGI((logt_1d, logp_1d), self.logrho_pt, **rgi_kw)
        self.logs_pt_rgi = RGI((logt_1d, logp_1d), self.logs_pt, **rgi_kw)
        self.logu_pt_rgi = RGI((logt_1d, logp_1d), self.logu_pt, **rgi_kw)

    # -----------------------------------------------------------------
    # Mg2SiO4 loader
    # -----------------------------------------------------------------
    def _load_mg2sio4(self, smooth_z):
        from eos.mg2sio4_aneos_eos import smooth_mg2sio4_pt

        data = np.load(f'{CURR_DIR}/rock_eos/mg2sio4_aneos_PT.npz')

        # PT table: P in GPa, T in K, rho in g/cm³,
        # s in erg/g/K (already CGS), u in erg/g (already CGS)
        P_GPa = np.asarray(data['pvals_pt'], dtype=float)
        T_K = np.asarray(data['tvals_pt'], dtype=float)
        n_P, n_T = P_GPa.size, T_K.size

        rho_raw = np.asarray(data['rho_grid_pt'], dtype=float)
        s_raw = np.asarray(data['s_grid_pt'], dtype=float)
        u_raw = np.asarray(data['u_grid_pt'], dtype=float)

        # Ensure shape is (n_P, n_T)
        if rho_raw.shape == (n_T, n_P):
            rho_raw, s_raw, u_raw = rho_raw.T, s_raw.T, u_raw.T

        # Convert axes to log CGS
        logp_cgs = np.log10(P_GPa * self._GPa_to_dyncm2)
        logt_1d = np.log10(T_K)

        if smooth_z:
            rho_raw, s_raw, u_raw = smooth_mg2sio4_pt(
                rho_raw, s_raw, u_raw, logp_cgs, logt_1d)

        # Convert to log form
        with np.errstate(invalid='ignore', divide='ignore'):
            logrho_raw = np.where(rho_raw > 0, np.log10(rho_raw), np.nan)
            logs_raw = np.where(s_raw > 0, np.log10(s_raw), np.nan)
            logu_raw = np.where(u_raw > 0, np.log10(u_raw), np.nan)

        # Store — grid is (n_P, n_T), axes (logP, logT), same as AQUA
        self.logpvals_pt = logp_cgs
        self.logtvals_pt = logt_1d
        self.logrho_pt = logrho_raw
        self.logs_pt = logs_raw
        self.logu_pt = logu_raw

        # RGI axes: (logP, logT) like AQUA
        rgi_kw = dict(method='linear', bounds_error=False, fill_value=None)
        self.logrho_pt_rgi = RGI((logp_cgs, logt_1d), self.logrho_pt, **rgi_kw)
        self.logs_pt_rgi = RGI((logp_cgs, logt_1d), self.logs_pt, **rgi_kw)
        self.logu_pt_rgi = RGI((logp_cgs, logt_1d), self.logu_pt, **rgi_kw)

    # -----------------------------------------------------------------
    # AQUA smoothing
    # -----------------------------------------------------------------
    def _smooth_aqua_lowp_lowt(self):
        """Smooth discontinuities at low-P / low-T phase boundaries
        in the AQUA P-T table using a localized 2D Gaussian filter.
        """
        logp = self.logpvals_pt
        logt = self.logtvals_pt

        for attr in ['logrho_pt', 'logs_pt', 'logu_pt']:
            grid = getattr(self, attr)

            filled = grid.copy()
            nan_mask = np.isnan(filled)
            if nan_mask.any():
                for i in range(filled.shape[0]):
                    row = filled[i]
                    nans = np.isnan(row)
                    if nans.all():
                        continue
                    valid = ~nans
                    filled[i] = np.interp(np.arange(len(row)),
                                          np.where(valid)[0], row[valid])

            smoothed_full = gaussian_filter(filled, sigma=[3.0, 3.0],
                                            mode='nearest')

            logp_2d = logp[:, np.newaxis]
            logt_2d = logt[np.newaxis, :]

            mask_p = 0.5 * (1.0 - np.tanh((logp_2d - 8.0) / 1.5))
            mask_t = 0.5 * (1.0 - np.tanh((logt_2d - 3.0) / 0.3))
            mask = mask_p * mask_t

            blended = (1.0 - mask) * filled + mask * smoothed_full
            blended[nan_mask] = np.nan

            setattr(self, attr, blended)

    def _interpolate_pt(self, interpolator, _lgp, _lgt):
        # AQUA/Mg2SiO4 RGI axes: (logP, logT); CH4/NH3 RGI axes: (logT, logP)
        if self.species in ('water', 'mg2sio4'):
            args = (_lgp, _lgt)
        else:
            args = (_lgt, _lgp)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = interpolator(pts)
        if all(np.isscalar(a) for a in (_lgp, _lgt)):
            return result.item()
        return result

    def _interpolate_rhot(self, interpolator, _lgrho, _lgt):
        args = (_lgrho, _lgt)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = interpolator(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        return result

    def get_logrho_pt(self, lgp, lgt):
        return self._interpolate_pt(self.logrho_pt_rgi, lgp, lgt)

    def get_logs_pt(self, lgp, lgt):
        return self._interpolate_pt(self.logs_pt_rgi, lgp, lgt)

    def get_logu_pt(self, lgp, lgt):
        return self._interpolate_pt(self.logu_pt_rgi, lgp, lgt)

    def get_logp_rhot(self, lgrho, lgt):
        if not hasattr(self, 'logp_rhot_rgi'):
            raise AttributeError(
                "rho-T basis not available for revised AQUA table. "
                "Use aqua_version='original' or the P-T basis methods.")
        return self._interpolate_rhot(self.logp_rhot_rgi, lgrho, lgt)

    def get_logs_rhot(self, lgrho, lgt):
        if not hasattr(self, 'logs_rhot_rgi'):
            raise AttributeError(
                "rho-T basis not available for revised AQUA table. "
                "Use aqua_version='original' or the P-T basis methods.")
        return self._interpolate_rhot(self.logs_rhot_rgi, lgrho, lgt)

    def get_logu_rhot(self, lgrho, lgt):
        if not hasattr(self, 'logu_rhot_rgi'):
            raise AttributeError(
                "rho-T basis not available for revised AQUA table. "
                "Use aqua_version='original' or the P-T basis methods.")
        return self._interpolate_rhot(self.logu_rhot_rgi, lgrho, lgt)

    # =================================================================
    # Inversion methods
    # =================================================================

    def _newton_1d_z(self, err_func, guess, lo_abs, hi_abs,
                     max_iter=30, tol=1e-8, h=1e-4):
        """Newton-Raphson with adaptive brentq fallback (z_eos version).

        Identical algorithm to ``hhe_z_mixtures._newton_1d`` but
        self-contained so that z_eos has no dependency on the mixture
        class.

        Parameters
        ----------
        err_func : callable
            f(x) = 0 at the root.
        guess : float
            Initial estimate.
        lo_abs, hi_abs : float
            Hard bounds for brentq bracket expansion.

        Returns
        -------
        solution, converged : float, bool
        """
        x = float(np.clip(guess, lo_abs, hi_abs))

        for _ in range(max_iter):
            f_val = err_func(x)
            if not np.isfinite(f_val):
                break
            if abs(f_val) < tol:
                return x, True
            f_plus = err_func(x + h)
            f_minus = err_func(x - h)
            if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                break
            fp = (f_plus - f_minus) / (2.0 * h)
            if abs(fp) < 1e-30:
                break
            step = f_val / fp
            if abs(step) > 1.0:
                step = np.sign(step) * 1.0
            x_new = np.clip(x - step, lo_abs, hi_abs)
            if abs(x_new - x) < 1e-12:
                f_new = err_func(x_new)
                return x_new, (np.isfinite(f_new)
                               and abs(f_new) < tol * 100)
            x = x_new

        # Adaptive brentq fallback
        delta = 0.5
        factor = 2.0
        a = np.clip(x - delta, lo_abs, hi_abs)
        b = np.clip(x + delta, lo_abs, hi_abs)
        for _ in range(8):
            fa = err_func(a)
            fb = err_func(b)
            if (np.isfinite(fa) and np.isfinite(fb)
                    and fa * fb < 0):
                try:
                    sol = brentq(err_func, a, b,
                                 xtol=1e-6, maxiter=100)
                    return sol, True
                except (ValueError, RuntimeError):
                    pass
            a = np.clip(a - delta * factor, lo_abs, hi_abs)
            b = np.clip(b + delta * factor, lo_abs, hi_abs)
            delta *= factor
            if a == lo_abs and b == hi_abs:
                fa = err_func(a)
                fb = err_func(b)
                if (np.isfinite(fa) and np.isfinite(fb)
                        and fa * fb < 0):
                    try:
                        return brentq(err_func, a, b,
                                      xtol=1e-6, maxiter=100), True
                    except (ValueError, RuntimeError):
                        pass
                return np.nan, False
        return np.nan, False

    def get_logt_sp(self, _s_kb, lgp):
        """Temperature from (S, P) via Newton-Raphson.

        Inverts S(P, T) = _s_kb to find logT.  Consistent with
        ``hhe_z_mixtures.get_logt_sp`` — the input entropy is in
        kb/baryon, converted internally to log10(erg/g/K) for
        comparison with the ``logs_pt`` forward model.

        Parameters
        ----------
        _s_kb : float or array
            Target entropy in kb/baryon.
        lgp : float or array
            log10 P [dyn/cm²].

        Returns
        -------
        logt : float or array
            log10 T [K].  NaN where no solution found.
        """
        scalar = np.isscalar(_s_kb) and np.isscalar(lgp)
        _s_kb = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        lgp = np.atleast_1d(np.asarray(lgp, dtype=float))
        _s_kb, lgp = np.broadcast_arrays(_s_kb, lgp)
        out = np.full_like(_s_kb, np.nan, dtype=float)

        # Allow extrapolation beyond table T range (RGI with
        # fill_value=None does nearest-neighbor extrapolation).
        lo_t = 2.0
        hi_t = 7.0

        # Convert kb/baryon → log10(erg/g/K)
        # S [erg/g/K] = S [kb/baryon] / erg_to_kbbar
        logs_target = np.log10(_s_kb / erg_to_kbbar)

        prev_sol = None
        for idx in np.ndindex(logs_target.shape):
            s_i = float(logs_target[idx])
            s_kb_i = float(_s_kb[idx])
            p_i = float(lgp[idx])

            def err(lgt, _s=s_i, _p=p_i):
                return float(self.get_logs_pt(_p, lgt) - _s)

            if prev_sol is not None:
                guess = prev_sol
            else:
                # Ideal-gas guess using species molecular weight
                guess = float(np.clip(
                    self._ideal.get_t_sp(s_kb_i, p_i, 0.0),
                    lo_t, hi_t))
            sol, ok = self._newton_1d_z(err, guess, lo_t, hi_t)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        return out.item() if scalar else out

    def get_logt_rhop(self, lgrho_target, lgp):
        """Temperature from (rho, P) via Newton-Raphson.

        Inverts logrho(P, T) = lgrho_target to find logT.

        Parameters
        ----------
        lgrho_target : float or array
            Target log10(rho) in g/cm³.
        lgp : float or array
            log10 P [dyn/cm²].

        Returns
        -------
        logt : float or array
            log10 T [K].  NaN where no solution found.
        """
        scalar = np.isscalar(lgrho_target) and np.isscalar(lgp)
        lgrho_target = np.atleast_1d(np.asarray(lgrho_target, dtype=float))
        lgp = np.atleast_1d(np.asarray(lgp, dtype=float))
        lgrho_target, lgp = np.broadcast_arrays(lgrho_target, lgp)
        out = np.full_like(lgrho_target, np.nan, dtype=float)

        # Allow extrapolation beyond table T range
        lo_t = 2.0
        hi_t = 7.0

        prev_sol = None
        for idx in np.ndindex(lgrho_target.shape):
            rho_i = float(lgrho_target[idx])
            p_i = float(lgp[idx])

            def err(lgt, _rho=rho_i, _p=p_i):
                return float(self.get_logrho_pt(_p, lgt) - _rho)

            if prev_sol is not None:
                guess = prev_sol
            else:
                guess = float(np.clip(
                    self._ideal.get_t_rhop(rho_i, p_i, 0.0),
                    lo_t, hi_t))
            sol, ok = self._newton_1d_z(err, guess, lo_t, hi_t)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        return out.item() if scalar else out

    def get_logp_rhot_inv(self, lgrho_target, lgt):
        """Pressure from (rho, T) via Newton-Raphson on the P-T table.

        Inverts logrho(P, T) = lgrho_target to find logP.
        Unlike ``get_logp_rhot`` (which requires the rho-T basis table),
        this works for all species by root-finding on the P-T forward model.

        Parameters
        ----------
        lgrho_target : float or array
            Target log10(rho) in g/cm³.
        lgt : float or array
            log10 T [K].

        Returns
        -------
        logp : float or array
            log10 P [dyn/cm²].  NaN where no solution found.
        """
        scalar = np.isscalar(lgrho_target) and np.isscalar(lgt)
        lgrho_target = np.atleast_1d(np.asarray(lgrho_target, dtype=float))
        lgt = np.atleast_1d(np.asarray(lgt, dtype=float))
        lgrho_target, lgt = np.broadcast_arrays(lgrho_target, lgt)
        out = np.full_like(lgrho_target, np.nan, dtype=float)

        lo_p = float(self.logpvals_pt[0])
        hi_p = float(self.logpvals_pt[-1])

        prev_sol = None
        for idx in np.ndindex(lgrho_target.shape):
            rho_i = float(lgrho_target[idx])
            t_i = float(lgt[idx])

            def err(lgp, _rho=rho_i, _t=t_i):
                return float(self.get_logrho_pt(lgp, _t) - _rho)

            if prev_sol is not None:
                guess = prev_sol
            else:
                guess = float(np.clip(
                    self._ideal.get_p_rhot(rho_i, t_i, 0.0),
                    lo_p, hi_p))
            sol, ok = self._newton_1d_z(err, guess, lo_p, hi_p)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        return out.item() if scalar else out


class val_mixtures:
    """Volume Addition Law (VAL) mixer for H-He-Z compositions.

    Combines hhe_eos (hydrogen + helium) with up to four z_eos species
    (water, methane, ammonia, mg2sio4) using the linear mixing /
    volume addition law.

    Metal sub-fractions follow the nested convention used throughout
    ORCHARD:
        _zm  : methane fraction within the metal budget
        _za  : ammonia fraction in the remainder after methane
        _zr  : rock fraction in the remainder after ammonia
        Physical mass fractions (within Z):
            f_water   = (1 - _zm) * (1 - _za) * (1 - _zr)
            f_methane = _zm * (1 - _za) * (1 - _zr)
            f_ammonia = _za * (1 - _zr)
            f_rock    = _zr

    Y_prime = Y / (1 - Z) so that it ranges from 0 to 1.

    HG23 non-ideal corrections (Smix, Vmix, Umix) are only applied
    when ``hhe_eos_name == 'cms'`` and ``hg=True``.
    Ideal entropy of mixing is always included.

    All quantities are in CGS:
        logP   : log10(dyn/cm²)
        logT   : log10(K)
        logrho : log10(g/cm³)
        S      : erg/(g·K)  (linear, not log)
        U      : erg/g      (linear, not log)
    """

    # Molecular weights for ideal entropy of mixing
    _m_h_atomic    = 1.0       # atomic / ionized hydrogen
    _m_h_molecular = 2.0       # molecular H2
    _m_he      = 4.0026
    _m_water   = 18.015
    _m_methane = 16.04
    _m_ammonia = 17.031
    _m_rock    = 140.6935   # Mg2SiO4 (forsterite)

    # H2 dissociation boundary from CMS19 C_P peak analysis:
    #   logT_dissoc = _DISSOC_A + _DISSOC_B * logP_cgs
    # Below this line: molecular H2 (mu=2). Above: atomic/ionized H (mu=1).
    _DISSOC_A = 2.9656
    _DISSOC_B = 0.0974

    def __init__(self, hhe_eos_name='cd', hg=True,
                 smooth_hhe=False, smooth_z=False,
                 species_list=None, mu_h_vary=True):
        """
        Parameters
        ----------
        hhe_eos_name : str
            Which H-He EOS to use ('cms' or 'cd').
        hg : bool
            Include HG23 non-ideal mixing corrections (CMS only).
        smooth_hhe : bool
            Smooth H-He tables before RGI creation.
        smooth_z : bool
            Smooth Z tables before RGI creation.
        species_list : list of str or None
            Which Z species to load. Default: all four
            ['water_revised', 'methane', 'ammonia', 'mg2sio4'].
            Use 'water_revised' for the revised AQUA table (Cano Amoros
            et al.) or 'water' for the original (Haldemann et al. 2020).
        mu_h_vary : bool
            If True (default), use P-T dependent molecular weight for
            hydrogen: mu_H = 2 below the H2 dissociation boundary and
            mu_H = 1 above it.  If False, use the legacy value mu_H = 1
            everywhere.
        """
        if species_list is None:
            species_list = ['water_revised', 'methane', 'ammonia', 'mg2sio4']

        self.hhe_eos_name = hhe_eos_name
        self.hg = hg
        self.mu_h_vary = mu_h_vary

        # H-He EOS
        self.hhe = hhe_eos(hhe_eos_name, smooth_hhe=smooth_hhe)

        # Z EOS instances — one per species
        # 'water_revised' and 'water' both map to the 'water' key in self.z
        # but load different AQUA table versions.
        self.z = {}
        for sp in species_list:
            if sp == 'water_revised':
                self.z['water'] = z_eos(species='water', smooth_z=smooth_z,
                                        aqua_version='revised')
            elif sp == 'water':
                self.z['water'] = z_eos(species='water', smooth_z=smooth_z,
                                        aqua_version='original')
            else:
                self.z[sp] = z_eos(species=sp, smooth_z=smooth_z)

    # =================================================================
    # helpers
    # =================================================================

    @staticmethod
    def _guarded_xlogx(x):
        """x * ln(x), returning 0 when x == 0."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos = x > 0.0
        out[pos] = x[pos] * np.log(x[pos])
        return out

    def _metal_fractions(self, _zm, _za, _zr):
        """Physical mass fractions **within** the metal budget Z."""
        f_water   = (1.0 - _zm) * (1.0 - _za) * (1.0 - _zr)
        f_methane = _zm * (1.0 - _za) * (1.0 - _zr)
        f_ammonia = _za * (1.0 - _zr)
        f_rock    = _zr
        return f_water, f_methane, f_ammonia, f_rock

    # -----------------------------------------------------------------
    # P-T dependent hydrogen molecular weight
    # -----------------------------------------------------------------
    def _get_mu_h(self, _lgp, _lgt):
        """Return the effective mean molecular weight of hydrogen.

        Uses the H₂ dissociation boundary derived from the CMS19
        C_P peak analysis:
            logT_dissoc = 2.9656 + 0.0974 * logP_cgs

        Below this line hydrogen is predominantly molecular (μ_H = 2).
        Above it hydrogen is atomic or ionized (μ_H = 1).

        When ``self.mu_h_vary`` is False, always returns 1.0 (legacy).

        Parameters
        ----------
        _lgp, _lgt : float or array
            log10(P [dyn/cm²]), log10(T [K]).

        Returns
        -------
        mu_h : float or array
            Effective molecular weight of hydrogen (1.0 or 2.0).
        """
        if not self.mu_h_vary:
            if np.isscalar(_lgp) and np.isscalar(_lgt):
                return self._m_h_atomic
            return np.full_like(np.atleast_1d(_lgp), self._m_h_atomic,
                                dtype=float)

        _lgp_arr = np.atleast_1d(_lgp)
        _lgt_arr = np.atleast_1d(_lgt)

        logt_dissoc = self._DISSOC_A + self._DISSOC_B * _lgp_arr

        # Smooth tanh transition over ~0.3 dex in logT to avoid
        # a step discontinuity in the ideal entropy of mixing
        # that would cause kinks in T(S,P) inversions.
        # f = 0 (molecular, μ=2) at low T, f = 1 (atomic, μ=1) at high T
        width = 0.15  # half-width of transition in dex
        f = 0.5 * (1.0 + np.tanh((_lgt_arr - logt_dissoc) / width))
        mu_h = self._m_h_molecular * (1.0 - f) + self._m_h_atomic * f

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return mu_h.item()
        return mu_h

    # -----------------------------------------------------------------
    # ideal entropy of mixing
    # -----------------------------------------------------------------
    def _smix_ideal(self, f_h, f_he, f_water=0.0, f_methane=0.0,
                    f_ammonia=0.0, f_rock=0.0, m_h=None):
        """Ideal entropy of mixing  -Σ(x_i ln x_i) / q   [kb/baryon].

        Accepts *physical* mass fractions (must sum to 1).  When the
        metal fractions are all zero this reduces to the pure H-He
        ideal entropy of mixing, so a single function handles both
        the X-Y and the X-Y-Z cases.

        Parameters
        ----------
        f_h, f_he, ... : float
            Physical mass fractions of each species.
        m_h : float or None
            Molecular weight to use for hydrogen.  If None, defaults
            to ``_m_h_atomic`` (1.0) for backward compatibility.

        Returns value in units of kb/baryon.  Caller must divide by
        ``erg_to_kbbar`` to convert to erg/(g·K).
        """
        if m_h is None:
            m_h = self._m_h_atomic

        m = self
        n_h       = f_h       / m_h
        n_he      = f_he      / m._m_he
        n_water   = np.where(np.asarray(f_water)   > 0, np.asarray(f_water)   / m._m_water,   0.0)
        n_methane = np.where(np.asarray(f_methane) > 0, np.asarray(f_methane) / m._m_methane, 0.0)
        n_ammonia = np.where(np.asarray(f_ammonia) > 0, np.asarray(f_ammonia) / m._m_ammonia, 0.0)
        n_rock    = np.where(np.asarray(f_rock)    > 0, np.asarray(f_rock)    / m._m_rock,    0.0)

        Ntot = n_h + n_he + n_water + n_methane + n_ammonia + n_rock

        x_h       = n_h       / Ntot
        x_he      = n_he      / Ntot
        x_water   = n_water   / Ntot
        x_methane = n_methane / Ntot
        x_ammonia = n_ammonia / Ntot
        x_rock    = n_rock    / Ntot

        q = (m_h * x_h + m._m_he * x_he + m._m_water * x_water
             + m._m_methane * x_methane + m._m_ammonia * x_ammonia
             + m._m_rock * x_rock)

        s_id = -(self._guarded_xlogx(x_h) + self._guarded_xlogx(x_he)
                 + self._guarded_xlogx(x_water) + self._guarded_xlogx(x_methane)
                 + self._guarded_xlogx(x_ammonia) + self._guarded_xlogx(x_rock)) / q

        return s_id

    # -----------------------------------------------------------------
    # HG23 non-ideal corrections (CMS only)
    # -----------------------------------------------------------------
    def _vmix_nonideal(self, _lgp, _lgt, _y_prime):
        """HG23 volume of mixing (cm³/g). Zero for non-CMS."""
        if self.hg and self.hhe_eos_name == 'cms':
            return self.hhe.vmix_interp(_lgp, _lgt) * (1.0 - _y_prime) * _y_prime
        return 0.0

    def _umix_nonideal(self, _lgp, _lgt, _y_prime):
        """HG23 internal energy of mixing (erg/g). Zero for non-CMS."""
        if self.hg and self.hhe_eos_name == 'cms':
            return self.hhe.umix_interp(_lgp, _lgt) * (1.0 - _y_prime) * _y_prime
        return 0.0

    # =================================================================
    # Metal-only mixing (water + methane + ammonia + rock)
    # =================================================================

    def get_logrho_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture density via VAL (log10 g/cm³)."""
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)

        v_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)

        if f_w > 0 and 'water' in self.z:
            rho_w = 10.0 ** self.z['water'].get_logrho_pt(_lgp, _lgt)
            v_mix = v_mix + f_w / rho_w
        if f_m > 0 and 'methane' in self.z:
            rho_m = 10.0 ** self.z['methane'].get_logrho_pt(_lgp, _lgt)
            v_mix = v_mix + f_m / rho_m
        if f_a > 0 and 'ammonia' in self.z:
            rho_a = 10.0 ** self.z['ammonia'].get_logrho_pt(_lgp, _lgt)
            v_mix = v_mix + f_a / rho_a
        if f_r > 0 and 'mg2sio4' in self.z:
            rho_r = 10.0 ** self.z['mg2sio4'].get_logrho_pt(_lgp, _lgt)
            v_mix = v_mix + f_r / rho_r

        result = np.log10(1.0 / v_mix)
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return result.item()
        return result

    def get_s_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture entropy, mass-weighted (erg/(g·K)).

        NOTE: does NOT include ideal entropy of mixing — that is added
        at the full H-He-Z level in get_s_pt_val.
        """
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)

        s_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)

        if f_w > 0 and 'water' in self.z:
            s_mix = s_mix + f_w * 10.0 ** self.z['water'].get_logs_pt(_lgp, _lgt)
        if f_m > 0 and 'methane' in self.z:
            s_mix = s_mix + f_m * 10.0 ** self.z['methane'].get_logs_pt(_lgp, _lgt)
        if f_a > 0 and 'ammonia' in self.z:
            s_mix = s_mix + f_a * 10.0 ** self.z['ammonia'].get_logs_pt(_lgp, _lgt)
        if f_r > 0 and 'mg2sio4' in self.z:
            s_mix = s_mix + f_r * 10.0 ** self.z['mg2sio4'].get_logs_pt(_lgp, _lgt)

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return s_mix.item()
        return s_mix

    def get_u_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture internal energy, mass-weighted (erg/g)."""
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)

        u_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)

        if f_w > 0 and 'water' in self.z:
            u_mix = u_mix + f_w * 10.0 ** self.z['water'].get_logu_pt(_lgp, _lgt)
        if f_m > 0 and 'methane' in self.z:
            u_mix = u_mix + f_m * 10.0 ** self.z['methane'].get_logu_pt(_lgp, _lgt)
        if f_a > 0 and 'ammonia' in self.z:
            u_mix = u_mix + f_a * 10.0 ** self.z['ammonia'].get_logu_pt(_lgp, _lgt)
        if f_r > 0 and 'mg2sio4' in self.z:
            u_mix = u_mix + f_r * 10.0 ** self.z['mg2sio4'].get_logu_pt(_lgp, _lgt)

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return u_mix.item()
        return u_mix

    # =================================================================
    # Full H-He-Z mixing
    # =================================================================

    def get_logrho_pt_val(self, _lgp, _lgt, _y_prime, _z=0.0,
                          _zm=0.0, _za=0.0, _zr=0.0):
        """Density of H-He-Z mixture via VAL (returns log10 g/cm³).

        Parameters
        ----------
        _lgp, _lgt : float or array
            log10(P [dyn/cm²]), log10(T [K]).
        _y_prime : float or array
            Y' = Y/(1-Z), ranges 0..1.
        _z : float or array
            Total metal mass fraction.
        _zm, _za, _zr : float
            Nested sub-fractions within Z.
        """
        # H-He specific volume
        rho_h  = 10.0 ** self.hhe.get_logrho_h(_lgp, _lgt)
        rho_he = 10.0 ** self.hhe.get_logrho_he(_lgp, _lgt)
        vmix   = self._vmix_nonideal(_lgp, _lgt, _y_prime)
        v_xy   = (1.0 - _y_prime) / rho_h + _y_prime / rho_he + vmix

        # Metal density
        rho_z = 10.0 ** self.get_logrho_z(_lgp, _lgt, _zm, _za, _zr)

        # VAL
        v_total = v_xy * (1.0 - _z) + _z / rho_z
        result = np.log10(1.0 / v_total)

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return np.atleast_1d(result).item()
        return result

    def get_s_pt_val(self, _lgp, _lgt, _y_prime, _z=0.0,
                     _zm=0.0, _za=0.0, _zr=0.0):
        """Entropy of H-He-Z mixture via VAL (returns erg/(g·K)).

        Mixing logic
        ------------
        1. Start with mass-weighted pure-species entropies:
               s_xy*(1-Z) + s_z*Z
        2. For **H-He only** (Z=0):
               + smix_xy_ideal                             (always)
               + smix_xy_nonideal                          (CMS + hg only)
        3. For **H-He-Z** (Z>0):
               - smix_xy_ideal*(1-Z)   (subtract X-Y ideal that we're replacing)
               + smix_xyz_ideal        (re-add full X-Y-Z ideal for all species)
               + smix_xy_nonideal*(1-Z)(HG23 non-ideal for CMS; 0 for CD)
           where smix_xy_nonideal = HG23_full - smix_xy_ideal.

        When Z=0, smix_xyz_ideal == smix_xy_ideal, so the subtract/add
        cancels and we recover case 2 identically.
        """
        # --- H-He component entropies (stored as log10) ---
        s_h  = 10.0 ** self.hhe.get_s_h(_lgp, _lgt)
        s_he = 10.0 ** self.hhe.get_s_he(_lgp, _lgt)
        s_xy = s_h * (1.0 - _y_prime) + s_he * _y_prime

        # --- Metal entropy (mass-weighted, no mixing terms) ---
        s_z = self.get_s_z(_lgp, _lgt, _zm, _za, _zr)

        # --- Physical mass fractions ---
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)
        f_h  = (1.0 - _y_prime) * (1.0 - _z)
        f_he = _y_prime * (1.0 - _z)

        # --- P-T dependent hydrogen molecular weight ---
        mu_h = self._get_mu_h(_lgp, _lgt)

        # --- Ideal entropy of mixing ---
        # X-Y only (pure H-He)
        smix_xy_ideal = self._smix_ideal(
            1.0 - _y_prime, _y_prime, m_h=mu_h
        ) / erg_to_kbbar

        # X-Y-Z (all species present) — reduces to smix_xy_ideal when Z=0
        smix_xyz_ideal = self._smix_ideal(
            f_h, f_he,
            f_w * _z, f_m * _z, f_a * _z, f_r * _z,
            m_h=mu_h
        ) / erg_to_kbbar

        # --- HG23 non-ideal H-He correction (CMS only) ---
        # smix_xy_nonideal = HG23_total − smix_xy_ideal
        # For CD (or hg=False): smix_xy_nonideal = 0
        smix_xy_nonideal = 0.0
        if self.hg and self.hhe_eos_name == 'cms':
            smix_hg23 = (self.hhe.smix_interp(_lgp, _lgt)
                         * (1.0 - _y_prime) * _y_prime)
            smix_xy_nonideal = smix_hg23 - smix_xy_ideal

        # --- Assemble ---
        # Subtract X-Y ideal (scaled by 1-Z) and re-add X-Y-Z ideal.
        # When Z=0 this is a no-op because smix_xyz_ideal == smix_xy_ideal.
        result = (
            s_xy * (1.0 - _z)
            + s_z * _z
            + smix_xyz_ideal
            + smix_xy_nonideal * (1.0 - _z)
        )

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return np.atleast_1d(result).item()
        return result

    def get_u_pt_val(self, _lgp, _lgt, _y_prime, _z=0.0,
                     _zm=0.0, _za=0.0, _zr=0.0):
        """Internal energy of H-He-Z mixture via VAL (returns erg/g)."""
        # H-He component energies (stored as log10)
        u_h  = 10.0 ** self.hhe.get_logu_h(_lgp, _lgt)
        u_he = 10.0 ** self.hhe.get_logu_he(_lgp, _lgt)
        umix = self._umix_nonideal(_lgp, _lgt, _y_prime)
        u_xy = u_h * (1.0 - _y_prime) + u_he * _y_prime + umix

        # Metal energy
        u_z = self.get_u_z(_lgp, _lgt, _zm, _za, _zr)

        result = u_xy * (1.0 - _z) + u_z * _z

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return np.atleast_1d(result).item()
        return result


class hhe_z_mixtures():
    """H-He-Z EOS with rhomboid S-P inversion support.

    Wraps ``val_mixtures`` (smoothed H-He + Z species via VAL) and
    provides infrastructure to map the non-rectangular physical
    entropy domain onto a rectangular grid using a normalised entropy
    coordinate xi in [0, 1].

    Pipeline overview
    -----------------
    1. **Raw tables** — ``hhe_eos`` and ``z_eos`` load raw EOS data
       (CD21 or CMS19 for H-He; AQUA, CH4, etc. for Z species).

    2. **Per-component smoothing** — If ``smooth_hhe=True`` /
       ``smooth_z=True``, per-component tables are smoothed
       *before* combining (see ``eos.smooth.smooth_eos_table``).
       This is important because the raw CD21/CMS19 and AQUA tables
       contain non-physical kinks from table construction artifacts.

    3. **Volume Addition Law** — ``val_mixtures`` combines the H-He
       and Z EOSes via ``1/v = (1-Z)/v_HHe + Z/v_Z`` to produce
       thermodynamic quantities (S, rho, U) at arbitrary (P, T, Y', Z).

    4. **P-T table** — A 4-D S(P, T, Y', Z) table is pre-computed
       from the forward model.  This avoids repeated calls to
       ``val_mixtures`` during inversions.

    5. **S bounds** — At each (P, Y', Z), evaluate the forward model
       at T_min and T_max to get S_lo(P, Y', Z) and S_hi(P, Y', Z).
       These bounds define the physical entropy range.  They are
       smoothed along the P axis (Hampel + Gaussian) to suppress
       spikes from AQUA phase transitions (ionization/dissociation
       at high T) that would otherwise cause severe kinks in the
       xi mapping.  A 2% margin is added after smoothing so that
       brentq never fails at the boundaries.

    6. **xi mapping** — The normalised coordinate

           xi = (S - S_lo) / (S_hi - S_lo)

       maps the trapezoidal S(P) domain to a unit rectangle.  This
       allows ``RegularGridInterpolator`` to operate on a regular
       4-D grid (xi, P, Y', Z).

    7. **S-P table** — ``build_sp_table`` uses brentq to solve
       S(P, T) = S_target at each (xi, P, Y', Z) grid point, storing
       logT(xi, P, Y', Z).  NaN cells are repaired via interpolation.

    8. **Query** — ``get_logt_sp`` first tries the RGI table.  To
       avoid cross-Z xi mismatch (where the same xi maps to different
       physical S at different Z grid points), the lookup brackets
       the query Z between the two nearest grid points, computes xi
       at each grid Z separately, and linearly interpolates the
       resulting logT.  Points where xi is out of bounds (> 1.05 or
       < -0.05) are returned as NaN, triggering a per-point brentq
       fallback in ``get_logt_sp``.

    Parameters
    ----------
    hhe_eos_name : str
        'cms' or 'cd'.
    hg : bool
        Include HG23 non-ideal mixing (CMS only).
    smooth_hhe, smooth_z : bool
        Smooth H-He / Z tables before RGI creation.
    mu_h_vary : bool
        P-T dependent hydrogen molecular weight.
    species_list : list of str or None
        Z species for val_mixtures. Use 'water_revised' for the revised
        AQUA table (Cano Amoros et al.) or 'water' for the original
        (Haldemann et al. 2020). Default includes 'water_revised'.
    logp_range : tuple (lo, hi)
        log10 P [dyn/cm²] bounds for the S-P grid.
    logp_step : float
        Step in logP.
    logt_range : tuple (lo, hi)
        log10 T [K] bounds of the underlying P-T table.
    n_xi : int
        Number of normalised entropy grid points.
    """

    # Table naming convention: {hhe_eos}/{hhe_eos}_{z_eos}_sp_adaptive.npz
    # Auto-discovered relative to CURR_DIR (eos/ directory).
    _TABLE_BASES = {
        'pt':   '{hhe}_{z}_pt_adaptive.npz',
        'sp':   '{hhe}_{z}_sp_adaptive.npz',
        'rhot': '{hhe}_{z}_rhot_adaptive.npz',
        'rhop': '{hhe}_{z}_rhop_adaptive.npz',
        'srho': '{hhe}_{z}_srho_adaptive.npz',
    }

    def __init__(self, hhe_eos_name='cd', hg=True,
                 smooth_hhe=True, smooth_z=True,
                 mu_h_vary=True,
                 species_list=None,
                 z_eos='aqua_revised',
                 pt_tab=True,
                 inv_tab=True,
                 y_prime=True,
                 logp_range=(5.0, 15.0), logp_step=0.05,
                 logt_range=(1.3, 6.0),
                 logrho_range=(-8.0, 2.0), logrho_step=0.05,
                 n_xi=150):
        """
        Parameters
        ----------
        hhe_eos_name : str
            'cms' or 'cd'.
        hg : bool
            Include HG23 non-ideal mixing corrections (CMS only).
        smooth_hhe, smooth_z : bool
            Smooth H-He / Z tables before RGI creation.
        mu_h_vary : bool
            P-T dependent hydrogen molecular weight.
        species_list : list of str or None
            Z species for val_mixtures. Use 'water_revised' for revised
            AQUA (Cano Amoros et al.) or 'water' for original (Haldemann
            et al. 2020).
        z_eos : str
            Label used in table filenames (e.g. 'water', 'ice_mixture').
        pt_tab : bool
            If True (default), auto-load the pre-computed P-T basis
            table for S(P,T), ρ(P,T), U(P,T).  Uses smooth RGI
            interpolation instead of raw VAL evaluation.
        inv_tab : bool
            If True (default), auto-load pre-computed inverted tables
            (S-P, ρ-T, ρ-P, S-ρ) for fast inversions and derivatives.
            Set False to use on-the-fly root-finding.
        logp_range : tuple (lo, hi)
            log10 P [dyn/cm²] bounds for the S-P grid.
        logp_step : float
            Step in logP.
        logt_range : tuple (lo, hi)
            log10 T [K] bounds of the underlying P-T table.
        logrho_range : tuple (lo, hi)
            log10 ρ [g/cm³] bounds for the ρ-T grid.
        logrho_step : float
            Step in logrho.
        n_xi : int
            Number of normalised entropy grid points.
        """

        self.hhe_eos_name = hhe_eos_name
        self.z_eos_label = z_eos
        self.pt_tab = pt_tab
        self.inv_tab = inv_tab
        self.y_prime = y_prime

        # --- Forward-model mixer ---
        self.val = val_mixtures(
            hhe_eos_name=hhe_eos_name, hg=hg,
            smooth_hhe=smooth_hhe, smooth_z=smooth_z,
            mu_h_vary=mu_h_vary,
            species_list=species_list)

        # --- Grid parameters ---
        self.logp_vals = np.arange(logp_range[0],
                                   logp_range[1] + logp_step * 0.1,
                                   logp_step)
        self.logt_min = logt_range[0]
        self.logt_max = logt_range[1]
        self.n_xi = n_xi
        # Default ξ grid (overwritten when a table is loaded)
        _alpha = 2.0
        _t = np.linspace(0.0, 1.0, n_xi)
        self.xi_vals = np.sinh(_alpha * _t) / np.sinh(_alpha)

        self.logrho_vals = np.arange(logrho_range[0],
                                      logrho_range[1] + logrho_step * 0.1,
                                      logrho_step)
        self.logt_vals = np.arange(logt_range[0], logt_range[1] + 0.01,
                                    logp_step)  # same step as logP

        # Placeholders — populated by compute_s_bounds()
        self._s_lo = None      # (nP,) or (nP, nY, nZ)
        self._s_hi = None
        self._s_lo_rgi = None  # RGI for interpolating bounds
        self._s_hi_rgi = None

        # Pre-computed tables (None until loaded)
        self._s_pt_rgi = None
        self._logrho_pt_rgi = None
        self._logu_pt_rgi = None
        self._logt_sp_rgi = None
        self._logp_rhot_rgi = None
        self._logt_rhop_rgi = None
        self._rho_lo_rhop_rgi = None
        self._rho_hi_rhop_rgi = None
        self._srho_rgi_p = None
        self._srho_rgi_t = None
        self._s_lo_srho = None
        self._s_hi_srho = None
        self._s_lo_srho_rgi = None
        self._s_hi_srho_rgi = None

        # --- Auto-load tables based on pt_tab and inv_tab flags ---
        self._auto_load_tables()

    # =================================================================
    # Auto-loading
    # =================================================================

    def _table_path(self, table_type):
        """Return the expected file path for a given table type."""
        fname = self._TABLE_BASES[table_type].format(
            hhe=self.hhe_eos_name, z=self.z_eos_label)
        return os.path.join(CURR_DIR, self.hhe_eos_name, fname)

    def _auto_load_tables(self):
        """Try to load pre-computed tables from disk.

        Controlled by ``self.pt_tab`` (P-T basis table) and
        ``self.inv_tab`` (inverted S-P, ρ-T, ρ-P, S-ρ tables).
        """
        # P-T basis table (controlled by pt_tab)
        if self.pt_tab:
            pt_path = self._table_path('pt')
            if os.path.isfile(pt_path):
                self.load_pt_table(pt_path)

        # Inverted tables (controlled by inv_tab)
        if self.inv_tab:
            for basis in ('sp', 'rhot', 'rhop', 'srho'):
                path = self._table_path(basis)
                if os.path.isfile(path):
                    getattr(self, f'load_{basis}_table')(path)

    # =================================================================
    # Y → Y' conversion
    # =================================================================

    def _to_yprime(self, _y, _z):
        """Convert absolute Y to Y' = Y/(1-Z) if y_prime=False.

        When self.y_prime=True, _y is already Y' and is returned as-is.
        When self.y_prime=False, _y is absolute Y and is divided by (1-Z).
        Works for scalar and array inputs.
        """
        if self.y_prime:
            return _y
        _z_arr = np.asarray(_z, dtype=float)
        return np.asarray(_y, dtype=float) / (1.0 - _z_arr + 1e-6)

    # =================================================================
    # P-T basis table
    # =================================================================

    def build_pt_table(self, yvals, zvals,
                       _zm=0.0, _za=0.0, _zr=0.0,
                       verbose=True):
        """Build the P-T basis table: S, logrho, logU on a regular
        (logP, logT, Y', Z) grid from the VAL forward model.

        No inversion or xi-mapping needed — just forward evaluation.

        Parameters
        ----------
        yvals, zvals : array_like
            1-D grids of Y' and Z values.
        verbose : bool
            Print progress.

        Returns
        -------
        result : dict with keys logpvals, logtvals, yvals, zvals,
                 s_pt, logrho_pt, logu_pt (all float32).
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        logp = self.logp_vals
        logt = self.logt_vals

        nP, nT, nY, nZ = len(logp), len(logt), len(yvals), len(zvals)

        if verbose:
            print(f"Building P-T table: "
                  f"logP=[{logp[0]:.2f}, {logp[-1]:.2f}] ({nP} pts), "
                  f"logT=[{logt[0]:.2f}, {logt[-1]:.2f}] ({nT} pts)")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {nP}×{nT}×{nY}×{nZ} = "
                  f"{nP*nT*nY*nZ:,}")
            est_mb = nP * nT * nY * nZ * 3 * 4 / 1e6
            print(f"  Est. size: ~{est_mb:.0f} MB (float32, 3 arrays)")

        s_pt = np.empty((nP, nT, nY, nZ), dtype=float)
        logrho_pt = np.empty((nP, nT, nY, nZ), dtype=float)
        logu_pt = np.empty((nP, nT, nY, nZ), dtype=float)

        total = nY * nZ
        pbar = tqdm(total=total,
                     desc="Building P-T table",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        # Vectorized: evaluate all (P, T) pairs at once for each (Y', Z)
        lgp_2d, lgt_2d = np.meshgrid(logp, logt, indexing='ij')
        lgp_flat = lgp_2d.ravel()
        lgt_flat = lgt_2d.ravel()

        for iy, yp in enumerate(yvals):
            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                try:
                    s_flat = self.val.get_s_pt_val(
                        lgp_flat, lgt_flat, yp, zv, _zm, _za, _zr)
                    rho_flat = self.val.get_logrho_pt_val(
                        lgp_flat, lgt_flat, yp, zv, _zm, _za, _zr)
                    u_flat = self.val.get_u_pt_val(
                        lgp_flat, lgt_flat, yp, zv, _zm, _za, _zr)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        logu_flat = np.where(u_flat > 0,
                                              np.log10(u_flat), np.nan)
                except (ZeroDivisionError, FloatingPointError):
                    s_flat = np.full(nP * nT, np.nan)
                    rho_flat = np.full(nP * nT, np.nan)
                    logu_flat = np.full(nP * nT, np.nan)

                s_pt[:, :, iy, iz] = s_flat.reshape(nP, nT)
                logrho_pt[:, :, iy, iz] = rho_flat.reshape(nP, nT)
                logu_pt[:, :, iy, iz] = logu_flat.reshape(nP, nT)

        pbar.close()

        # Fill NaN in all arrays first (so Hampel has no gaps).
        # Uses PT-axis ordering: (P, T, Y', Z).
        for arr, label in [(s_pt, 'S'), (logrho_pt, 'logrho'),
                           (logu_pt, 'logU')]:
            n_nan = np.isnan(arr).sum()
            if n_nan > 0:
                if verbose:
                    print(f"Filling {n_nan} NaN cells in {label} ...")
                self._fill_nans_2axis(arr, axis0=0, axis1=1)

        # Detect and replace finite outlier spikes along the P axis.
        # The CD21 H-He forward model can produce isolated entropy
        # spikes (e.g. 260x at logP=12.75, low T) from numerical
        # artifacts in the underlying tables.  Only S is filtered;
        # rho and U at the same grid points are typically smooth.
        if verbose:
            print("Running Hampel outlier filter on S along P axis ...")
        flat = s_pt.reshape(nP, -1)  # (nP, nT*nY*nZ)
        n_cols = flat.shape[1]
        n_outliers = 0
        for j in range(n_cols):
            col = flat[:, j]
            cleaned, n_rep = hampel_filter_1d(
                col, window=7, n_sigma=3.0)
            if n_rep > 0:
                changed = (cleaned != col) & np.isfinite(col)
                flat[changed, j] = cleaned[changed]
                n_outliers += n_rep
        if n_outliers > 0 and verbose:
            print(f"Replaced {n_outliers} finite outlier cells "
                  f"in S (Hampel filter along P axis)")

        s_f32 = s_pt.astype(np.float32)
        logrho_f32 = logrho_pt.astype(np.float32)
        logu_f32 = logu_pt.astype(np.float32)

        result = {
            'logpvals':  logp,
            'logtvals':  logt,
            'yvals':     yvals,
            'zvals':     zvals,
            's_pt':      s_f32,
            'logrho_pt': logrho_f32,
            'logu_pt':   logu_f32,
        }

        # Load into this instance
        self._load_pt_from_arrays(logp, logt, yvals, zvals,
                                   s_f32, logrho_f32, logu_f32)

        if verbose:
            n_total = s_pt.size
            n_good = np.isfinite(s_pt).sum()
            mem_mb = (s_f32.nbytes + logrho_f32.nbytes + logu_f32.nbytes) / 1e6
            print(f"Done. {n_good}/{n_total} cells finite, "
                  f"table size: {mem_mb:.0f} MB (float32)")

        return result

    def _load_pt_from_arrays(self, logp, logt, yvals, zvals,
                              s_pt, logrho_pt, logu_pt):
        """Build P-T RGI interpolators from arrays."""
        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._s_pt_rgi = RGI((logp, logt, yvals, zvals),
                              s_pt, **rgi_kw)
        self._logrho_pt_rgi = RGI((logp, logt, yvals, zvals),
                                   logrho_pt, **rgi_kw)
        self._logu_pt_rgi = RGI((logp, logt, yvals, zvals),
                                 logu_pt, **rgi_kw)

    def load_pt_table(self, path):
        """Load a pre-computed P-T table from NPZ."""
        data = np.load(path)
        self._load_pt_from_arrays(
            data['logpvals'], data['logtvals'],
            data['yvals'], data['zvals'],
            data['s_pt'], data['logrho_pt'], data['logu_pt'])

    def save_pt_table(self, result, path=None):
        """Save a P-T table to NPZ."""
        if path is None:
            path = self._table_path('pt')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # Internal P-T queries (Y' already converted, no _frock)
    # =================================================================

    def _query_pt_rgi(self, rgi, lgp, lgt, yp, z):
        """Query a P-T RGI with linear extrapolation in logT.

        scipy's RGI with fill_value=None does nearest-neighbor
        extrapolation, which produces isothermal plateaus outside
        the grid.  This method linearly extrapolates using the
        gradient at the grid boundary when logT is out of bounds.
        """
        scalar = np.isscalar(lgp) and np.isscalar(lgt)
        lgp_a = np.atleast_1d(lgp)
        lgt_a = np.atleast_1d(lgt)
        yp_a = np.atleast_1d(yp)
        z_a = np.atleast_1d(z)
        lgp_a, lgt_a, yp_a, z_a = np.broadcast_arrays(
            lgp_a, lgt_a, yp_a, z_a)

        # Standard RGI query (clamps out-of-bounds to boundary)
        pts = np.column_stack((lgp_a.ravel(), lgt_a.ravel(),
                                yp_a.ravel(), z_a.ravel()))
        result = rgi(pts).reshape(lgp_a.shape)

        # Linear extrapolation for logT below the grid minimum
        logt_grid = rgi.grid[1]  # axis 1 = logT
        t_lo = logt_grid[0]
        t_lo1 = logt_grid[1]
        dt = t_lo1 - t_lo

        below = lgt_a < t_lo
        if np.any(below):
            # Value and gradient at the lower boundary
            pts_lo = pts.copy()
            pts_lo[:, 1] = t_lo
            f_lo = rgi(pts_lo).reshape(lgp_a.shape)

            pts_lo1 = pts.copy()
            pts_lo1[:, 1] = t_lo1
            f_lo1 = rgi(pts_lo1).reshape(lgp_a.shape)

            grad = (f_lo1 - f_lo) / dt
            result = np.where(below,
                               f_lo + grad * (lgt_a - t_lo),
                               result)

        # Linear extrapolation for logT above the grid maximum
        t_hi = logt_grid[-1]
        t_hi1 = logt_grid[-2]
        above = lgt_a > t_hi
        if np.any(above):
            pts_hi = pts.copy()
            pts_hi[:, 1] = t_hi
            f_hi = rgi(pts_hi).reshape(lgp_a.shape)

            pts_hi1 = pts.copy()
            pts_hi1[:, 1] = t_hi1
            f_hi1 = rgi(pts_hi1).reshape(lgp_a.shape)

            grad = (f_hi - f_hi1) / (t_hi - t_hi1)
            result = np.where(above,
                               f_hi + grad * (lgt_a - t_hi),
                               result)

        if scalar:
            return result.item()
        return result

    def _s_pt(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0):
        """S(P, T, Y', Z) — uses table RGI if loaded, else VAL."""
        if self._s_pt_rgi is not None:
            return self._query_pt_rgi(self._s_pt_rgi, lgp, lgt, yp, z)
        return self.val.get_s_pt_val(lgp, lgt, yp, z, _zm, _za, _zr)

    def _logrho_pt(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0):
        """logrho(P, T, Y', Z) — uses table RGI if loaded, else VAL."""
        if self._logrho_pt_rgi is not None:
            return self._query_pt_rgi(self._logrho_pt_rgi, lgp, lgt, yp, z)
        return self.val.get_logrho_pt_val(lgp, lgt, yp, z, _zm, _za, _zr)

    def _logu_pt(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0):
        """logU(P, T, Y', Z) — uses table RGI if loaded, else VAL."""
        if self._logu_pt_rgi is not None:
            return self._query_pt_rgi(self._logu_pt_rgi, lgp, lgt, yp, z)
        return np.log10(self.val.get_u_pt_val(lgp, lgt, yp, z, _zm, _za, _zr))

    # =================================================================
    # P-T table query wrappers (public, with Y' conversion)
    # =================================================================

    def get_s_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0,
                     val=False, **kw):
        """S(P, T, Y, Z) in erg/(g·K).

        Uses the pre-computed P-T table RGI by default.
        Set val=True to call val.get_s_pt_val() directly.
        """
        _y = self._to_yprime(_y, _z)
        if val or self._s_pt_rgi is None:
            return self.val.get_s_pt_val(_lgp, _lgt, _y, _z)
        # Table path: 4-D RGI query
        _lgp_a = np.atleast_1d(_lgp)
        _lgt_a = np.atleast_1d(_lgt)
        _y_a = np.atleast_1d(_y)
        _z_a = np.atleast_1d(_z)
        _lgp_a, _lgt_a, _y_a, _z_a = np.broadcast_arrays(
            _lgp_a, _lgt_a, _y_a, _z_a)
        pts = np.column_stack((_lgp_a.ravel(), _lgt_a.ravel(),
                                _y_a.ravel(), _z_a.ravel()))
        result = self._s_pt_rgi(pts).reshape(_lgp_a.shape)
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return result.item()
        return result

    def get_logrho_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0,
                           val=False, **kw):
        """log10 ρ(P, T, Y, Z) in g/cm³.

        Uses the pre-computed P-T table RGI by default.
        Set val=True to call val.get_logrho_pt_val() directly.
        """
        _y = self._to_yprime(_y, _z)
        if val or self._logrho_pt_rgi is None:
            return self.val.get_logrho_pt_val(_lgp, _lgt, _y, _z)
        _lgp_a = np.atleast_1d(_lgp)
        _lgt_a = np.atleast_1d(_lgt)
        _y_a = np.atleast_1d(_y)
        _z_a = np.atleast_1d(_z)
        _lgp_a, _lgt_a, _y_a, _z_a = np.broadcast_arrays(
            _lgp_a, _lgt_a, _y_a, _z_a)
        pts = np.column_stack((_lgp_a.ravel(), _lgt_a.ravel(),
                                _y_a.ravel(), _z_a.ravel()))
        result = self._logrho_pt_rgi(pts).reshape(_lgp_a.shape)
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return result.item()
        return result

    def get_logu_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0,
                         val=False, **kw):
        """log10 U(P, T, Y, Z) in erg/g.

        Uses the pre-computed P-T table RGI by default.
        Set val=True to call val.get_u_pt_val() directly.
        """
        _y = self._to_yprime(_y, _z)
        if val or self._logu_pt_rgi is None:
            return np.log10(self.val.get_u_pt_val(_lgp, _lgt, _y, _z))
        _lgp_a = np.atleast_1d(_lgp)
        _lgt_a = np.atleast_1d(_lgt)
        _y_a = np.atleast_1d(_y)
        _z_a = np.atleast_1d(_z)
        _lgp_a, _lgt_a, _y_a, _z_a = np.broadcast_arrays(
            _lgp_a, _lgt_a, _y_a, _z_a)
        pts = np.column_stack((_lgp_a.ravel(), _lgt_a.ravel(),
                                _y_a.ravel(), _z_a.ravel()))
        result = self._logu_pt_rgi(pts).reshape(_lgp_a.shape)
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return result.item()
        return result

    # =================================================================
    # S-bound computation
    # =================================================================

    def compute_s_bounds(self, _y_prime, _z,
                         _zm=0.0, _za=0.0, _zr=0.0):
        """Compute the physical entropy bounds at each pressure.

        Evaluates S(P, T_min) and S(P, T_max) via val_mixtures for
        a single composition (Y', Z, metal fractions) and stores
        the min/max as 1-D arrays over pressure.

        Parameters
        ----------
        _y_prime : float
            Helium fraction Y' = Y/(1-Z).
        _z : float
            Total metal mass fraction.
        _zm, _za, _zr : float
            Nested metal sub-fractions.

        Sets
        ----
        self._s_lo, self._s_hi : 1-D arrays of shape (nP,) in kb/baryon
        self._s_lo_rgi, self._s_hi_rgi : interp1d callables
        """
        _y_prime = self._to_yprime(_y_prime, _z)
        nP = len(self.logp_vals)

        s_at_tmin = np.empty(nP)
        s_at_tmax = np.empty(nP)

        for ip, lgp in enumerate(self.logp_vals):
            s_at_tmin[ip] = self._s_pt(
                lgp, self.logt_min, _y_prime, _z, _zm, _za, _zr)
            s_at_tmax[ip] = self._s_pt(
                lgp, self.logt_max, _y_prime, _z, _zm, _za, _zr)

        # Convert erg/(g·K) → kb/baryon
        s_at_tmin *= erg_to_kbbar
        s_at_tmax *= erg_to_kbbar

        # S_lo = min, S_hi = max (ordering can flip with P)
        self._s_lo = np.minimum(s_at_tmin, s_at_tmax)
        self._s_hi = np.maximum(s_at_tmin, s_at_tmax)

        # 1-D interpolators for the bounds
        self._s_lo_rgi = interp1d(self.logp_vals, self._s_lo,
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
        self._s_hi_rgi = interp1d(self.logp_vals, self._s_hi,
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')

    def compute_s_bounds_grid(self, yvals, zvals,
                              _zm=0.0, _za=0.0, _zr=0.0):
        """Compute S bounds over a grid of (P, Y', Z).

        Like ``compute_s_bounds`` but for arrays of Y' and Z values.
        Stores 3-D arrays and builds RGI interpolators over
        (logP, Y', Z).

        After computing raw S_lo and S_hi from the forward model at
        T_min and T_max, the bounds are smoothed along the P axis
        at each (Y', Z) slice.  This is critical because the AQUA
        water EOS has ionization/dissociation entropy spikes at high
        T that can cause S_hi(P) to swing by >10x over a narrow P
        range.  Without smoothing, the xi mapping produces kinked
        isentropes (confirmed: up to 0.9 dex logT jumps at Z=0.9).

        A 2% margin is added to the smoothed bounds so that brentq
        does not fail at ξ near 0 or 1 where smoothing may have
        shifted the bound past the true T_min/T_max entropy.

        Parameters
        ----------
        yvals : 1-D array of Y' values
        zvals : 1-D array of Z values
        _zm, _za, _zr : float  (fixed for all Y', Z)
        """
        nP = len(self.logp_vals)
        nY = len(yvals)
        nZ = len(zvals)

        self._yvals = np.asarray(yvals, dtype=float)
        self._zvals = np.asarray(zvals, dtype=float)

        s_lo_3d = np.empty((nP, nY, nZ))
        s_hi_3d = np.empty((nP, nY, nZ))

        pbar = tqdm(total=nY * nZ,
                     desc="Computing S bounds",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')
        for iy, yp in enumerate(yvals):
            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)
                for ip, lgp in enumerate(self.logp_vals):
                    s_cold = self._s_pt(
                        lgp, self.logt_min, yp, zv, _zm, _za, _zr)
                    s_hot = self._s_pt(
                        lgp, self.logt_max, yp, zv, _zm, _za, _zr)
                    s_cold *= erg_to_kbbar
                    s_hot  *= erg_to_kbbar
                    s_lo_3d[ip, iy, iz] = min(s_cold, s_hot)
                    s_hi_3d[ip, iy, iz] = max(s_cold, s_hot)
        pbar.close()

        # Cap S_hi to prevent the ξ range from being dominated by
        # extreme values at low P / high T.  At low pressures,
        # logt_max=7 gives S > 600 kb/baryon, while physical
        # planetary entropies are ≲ 50.  Without capping, physical
        # isentropes map to ξ < 0.03 at low P, destroying table
        # resolution.
        #
        # Strategy: use the 75th percentile of S_hi across the
        # full P grid as the cap (with a 50% margin), so that
        # the bulk of the ξ range covers physically relevant S.
        # Combined with sinh-clustered ξ, this ensures good
        # resolution at both low and high pressures.
        for iy_i in range(nY):
            for iz_i in range(nZ):
                col = s_hi_3d[:, iy_i, iz_i]
                cap = np.nanpercentile(col, 75) * 1.5
                # Don't cap below the minimum s_hi (safety)
                cap = max(cap, np.nanmin(col))
                s_hi_3d[:, iy_i, iz_i] = np.minimum(col, cap)

        # Smooth S bounds along the P axis to remove spikes from AQUA
        # phase transitions (ionization/dissociation at high T).
        # Without this, S_hi(P) can swing by 16x over 0.5 dex in logP,
        # causing the xi mapping to produce kinked isentropes.
        # Hampel filter removes outlier spikes; Gaussian smooths residual
        # roughness.  Must be done BEFORE building the SP table so that
        # brentq solves at the smoothed physical S values.
        for iy_i in range(nY):
            for iz_i in range(nZ):
                s_hi_3d[:, iy_i, iz_i], _ = hampel_filter_1d(
                    s_hi_3d[:, iy_i, iz_i], window=10, n_sigma=3.0)
                s_hi_3d[:, iy_i, iz_i] = gaussian_filter1d(
                    s_hi_3d[:, iy_i, iz_i], sigma=5.0)
                s_lo_3d[:, iy_i, iz_i], _ = hampel_filter_1d(
                    s_lo_3d[:, iy_i, iz_i], window=10, n_sigma=3.0)
                s_lo_3d[:, iy_i, iz_i] = gaussian_filter1d(
                    s_lo_3d[:, iy_i, iz_i], sigma=5.0)

        # Pad bounds by 2% so brentq doesn't fail at boundaries where
        # smoothing may have pushed S_lo above the true minimum or
        # S_hi below the true maximum.
        margin = 0.02 * (s_hi_3d - s_lo_3d)
        s_lo_3d -= margin
        s_hi_3d += margin

        self._s_lo = s_lo_3d
        self._s_hi = s_hi_3d

        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._s_lo_rgi = RGI((self.logp_vals, self._yvals, self._zvals),
                              s_lo_3d, **rgi_kw)
        self._s_hi_rgi = RGI((self.logp_vals, self._yvals, self._zvals),
                              s_hi_3d, **rgi_kw)

    # =================================================================
    # Coordinate transforms
    # =================================================================

    def s_to_xi(self, _s_kb, _lgp, _yp=None, _z=None):
        """Convert physical entropy (kb/baryon) → normalised ξ ∈ [0,1].

        Uses the stored (and smoothed) S bounds.  Works for scalar or
        array inputs.  ξ < 0 or ξ > 1 means S is outside the range
        covered by the table at that (P, Y', Z); this triggers brentq
        fallback at query time.
        """
        s_lo, s_hi = self._get_bounds(_lgp, _yp, _z)
        denom = s_hi - s_lo
        # Guard against zero-width range
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        return (_s_kb - s_lo) / denom

    def xi_to_s(self, _xi, _lgp, _yp=None, _z=None):
        """Convert normalised ξ → physical entropy (kb/baryon)."""
        s_lo, s_hi = self._get_bounds(_lgp, _yp, _z)
        return s_lo + _xi * (s_hi - s_lo)

    def _get_bounds(self, _lgp, _yp=None, _z=None):
        """Interpolate S_lo and S_hi at the requested (P, [Y', Z])."""
        if self._s_lo_rgi is None:
            raise RuntimeError("Call compute_s_bounds() or "
                               "compute_s_bounds_grid() first.")

        _lgp_arr = np.atleast_1d(_lgp)

        if isinstance(self._s_lo_rgi, RGI):
            # 3-D bounds (P, Y', Z)
            _yp_arr = np.atleast_1d(_yp)
            _z_arr  = np.atleast_1d(_z)
            _lgp_arr, _yp_arr, _z_arr = np.broadcast_arrays(
                _lgp_arr, _yp_arr, _z_arr)
            pts = np.column_stack((_lgp_arr.ravel(),
                                   _yp_arr.ravel(),
                                   _z_arr.ravel()))
            s_lo = self._s_lo_rgi(pts).reshape(_lgp_arr.shape)
            s_hi = self._s_hi_rgi(pts).reshape(_lgp_arr.shape)
        else:
            # 1-D bounds (P only) — interp1d callable
            s_lo = self._s_lo_rgi(_lgp_arr)
            s_hi = self._s_hi_rgi(_lgp_arr)

        if np.isscalar(_lgp):
            return float(s_lo), float(s_hi)
        return s_lo, s_hi

    # =================================================================
    # On-the-fly S-P inversion (root-finding)
    # =================================================================

    def _s_bounds_at_point(self, lgp_i, _yp, _z, _zm, _za, _zr):
        """Compute S_lo, S_hi (kb/baryon) at a single P for given composition.

        Uses the forward model directly — no pre-computed bounds needed.
        """
        _yp = self._to_yprime(_yp, _z)
        s_cold = (self._s_pt(lgp_i, self.logt_min,
                                         _yp, _z, _zm, _za, _zr)
                  * erg_to_kbbar)
        s_hot  = (self._s_pt(lgp_i, self.logt_max,
                                         _yp, _z, _zm, _za, _zr)
                  * erg_to_kbbar)
        return min(s_cold, s_hot), max(s_cold, s_hot)

    # =================================================================
    # Unified Newton-Raphson solvers (1-D and 2-D)
    # =================================================================

    def _newton_1d(self, err_func, guess, lo_abs, hi_abs,
                   max_iter=30, tol=1e-8, h=1e-4):
        """Newton-Raphson with adaptive brentq fallback for any 1-D inversion.

        The solver tries Newton-Raphson first.  If Newton fails to
        converge, the brentq fallback places a bracket around Newton's
        last iterate and expands it recursively until a sign change is
        found, up to the hard bounds ``[lo_abs, hi_abs]``.

        Parameters
        ----------
        err_func : callable
            f(x) = 0 at the desired root.  Must return float or NaN.
        guess : float
            Initial estimate (from ideal_eos or previous solution).
        lo_abs, hi_abs : float
            Hard bounds for the brentq bracket expansion
            (e.g. logT=[1.5, 7.0] or logP bounds).
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on |f(x)|.
        h : float
            Step for central-difference derivative.

        Returns
        -------
        solution : float
            Converged x, or NaN on failure.
        converged : bool
        """
        x = float(np.clip(guess, lo_abs, hi_abs))

        # --- Phase 1: Newton-Raphson ---
        for _ in range(max_iter):
            f_val = err_func(x)
            if not np.isfinite(f_val):
                break
            if abs(f_val) < tol:
                return x, True

            f_plus = err_func(x + h)
            f_minus = err_func(x - h)
            if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                break
            fp = (f_plus - f_minus) / (2.0 * h)
            if abs(fp) < 1e-30:
                break

            step = f_val / fp
            if abs(step) > 1.0:
                step = np.sign(step) * 1.0
            x_new = np.clip(x - step, lo_abs, hi_abs)
            if abs(x_new - x) < 1e-12:
                f_new = err_func(x_new)
                return x_new, (np.isfinite(f_new) and abs(f_new) < tol * 100)
            x = x_new

        # --- Phase 2: Adaptive brentq fallback ---
        # Bracket starts around Newton's last iterate and expands
        # recursively until a sign change is found.
        delta = 0.5
        factor = 2.0
        max_attempts = 8
        center = x  # Newton's last iterate

        a = np.clip(center - delta, lo_abs, hi_abs)
        b = np.clip(center + delta, lo_abs, hi_abs)

        for _ in range(max_attempts):
            fa = err_func(a)
            fb = err_func(b)

            if (np.isfinite(fa) and np.isfinite(fb)
                    and fa * fb < 0):
                try:
                    sol = brentq(err_func, a, b,
                                 xtol=1e-6, maxiter=100)
                    return sol, True
                except (ValueError, RuntimeError,
                        ZeroDivisionError):
                    pass

            # Expand bracket
            a = np.clip(a - delta * factor, lo_abs, hi_abs)
            b = np.clip(b + delta * factor, lo_abs, hi_abs)
            delta *= factor

            # If we've hit both hard bounds without sign change → give up
            if a == lo_abs and b == hi_abs:
                # One last try with full range
                fa = err_func(a)
                fb = err_func(b)
                if (np.isfinite(fa) and np.isfinite(fb)
                        and fa * fb < 0):
                    try:
                        sol = brentq(err_func, a, b,
                                     xtol=1e-6, maxiter=100)
                        return sol, True
                    except (ValueError, RuntimeError,
                            ZeroDivisionError):
                        pass
                return np.nan, False

        return np.nan, False

    def _newton_2d(self, residuals_func, guess, lb, ub,
                   max_iter=15, tol=1e-8, h=1e-4):
        """2-D Newton-Raphson for (S, rho) → (P, T) with least_squares fallback.

        Parameters
        ----------
        residuals_func : callable
            f(x) → array of length 2, where x = [logP, logT].
        guess : array_like, shape (2,)
            Initial [logP, logT] estimate.
        lb, ub : array_like, shape (2,)
            Lower/upper bounds for [logP, logT].
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on max(|residual|).
        h : float
            Step for numerical Jacobian.

        Returns
        -------
        solution : ndarray, shape (2,)
            Converged [logP, logT], or [NaN, NaN].
        converged : bool
        """
        x = np.clip(np.asarray(guess, dtype=float), lb, ub)

        for _ in range(max_iter):
            r = residuals_func(x)
            if not np.all(np.isfinite(r)):
                break
            if np.max(np.abs(r)) < tol:
                return x, True

            # Numerical Jacobian (2x2)
            J = np.empty((2, 2))
            for j in range(2):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[j] += h
                x_minus[j] -= h
                r_plus = residuals_func(x_plus)
                r_minus = residuals_func(x_minus)
                if not (np.all(np.isfinite(r_plus))
                        and np.all(np.isfinite(r_minus))):
                    J[:, j] = 0.0
                else:
                    J[:, j] = (r_plus - r_minus) / (2.0 * h)

            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-30:
                break

            # Solve J @ dx = r  →  dx = J^{-1} @ r
            dx = np.array([
                ( J[1, 1] * r[0] - J[0, 1] * r[1]) / det,
                (-J[1, 0] * r[0] + J[0, 0] * r[1]) / det])

            # Limit step size
            for j in range(2):
                if abs(dx[j]) > 1.0:
                    dx[j] = np.sign(dx[j]) * 1.0

            x_new = np.clip(x - dx, lb, ub)
            if np.max(np.abs(x_new - x)) < 1e-12:
                r_new = residuals_func(x_new)
                return x_new, (np.all(np.isfinite(r_new))
                               and np.max(np.abs(r_new)) < tol * 100)
            x = x_new

        # --- Fallback: least_squares (limited budget) ---
        try:
            sol = least_squares(
                residuals_func, x,
                bounds=(lb, ub),
                method='trf',
                xtol=1e-8, ftol=1e-8,
                gtol=1e-8, max_nfev=50)
            if sol.success and np.all(np.isfinite(sol.x)):
                return sol.x, True
        except Exception:
            pass

        return np.array([np.nan, np.nan]), False

    def get_logt_sp(self, _s_kb, _lgp, _yp, _z=0.0,
                    _zm=0.0, _za=0.0, _zr=0.0, **kw):
        """Temperature from (S, P) via Newton-Raphson on the forward model.

        Uses ``ideal_xy`` for initial guesses and the previous
        converged solution as the starting point when sweeping arrays
        (so isentropes naturally track the correct high-T branch).
        Falls back to adaptive brentq bracketed around Newton's last
        iterate, bounded by [1.5, 7.0] in logT.

        If a pre-computed S-P table has been loaded, the RGI lookup
        is used first; Newton/brentq only fills NaN entries.

        Parameters
        ----------
        _s_kb : float or array
            Entropy in kb/baryon.
        _lgp : float or array
            log10 P [dyn/cm²].
        _yp : float
            Y' = Y/(1-Z).
        _z : float
            Total metal mass fraction (default 0).
        _zm, _za, _zr : float
            Nested metal sub-fractions.

        Returns
        -------
        logt : float or array
            log10 T [K].  NaN where S is outside the physical domain.
        """
        _yp = self._to_yprime(_yp, _z)

        def _make_err(s_i, p_i, yp_i, z_i, zm_i, za_i, zr_i):
            def err(lgt):
                try:
                    s_test = self._s_pt(
                        p_i, lgt, yp_i, z_i, zm_i, za_i, zr_i)
                    return float(s_test * erg_to_kbbar - s_i)
                except (ZeroDivisionError, FloatingPointError):
                    return np.nan
            return err

        # --- Fast path: pre-computed table ---
        if self._logt_sp_rgi is not None:
            result = self._lookup_sp_table(
                _s_kb, _lgp, _yp, _z, self._logt_sp_rgi)
            result_arr = np.atleast_1d(result)
            if np.all(np.isfinite(result_arr)):
                return result
            # Fill NaN entries via _newton_1d
            if np.isscalar(_s_kb) and np.isscalar(_lgp):
                pass  # single NaN → fall through to slow path
            else:
                _s_arr = np.atleast_1d(np.asarray(_s_kb, dtype=float))
                _p_arr = np.atleast_1d(np.asarray(_lgp, dtype=float))
                _yp_arr = np.atleast_1d(np.asarray(_yp, dtype=float))
                _z_arr = np.atleast_1d(np.asarray(_z, dtype=float))
                _s_arr, _p_arr, _yp_arr, _z_arr = np.broadcast_arrays(
                    _s_arr, _p_arr, _yp_arr, _z_arr)
                out = result_arr.copy()
                bad = ~np.isfinite(out)
                _zm_s = float(np.atleast_1d(_zm).ravel()[0])
                _za_s = float(np.atleast_1d(_za).ravel()[0])
                _zr_s = float(np.atleast_1d(_zr).ravel()[0])
                if bad.any():
                    prev_sol = None
                    for idx in np.where(bad.ravel())[0]:
                        s_i = float(_s_arr.ravel()[idx])
                        p_i = float(_p_arr.ravel()[idx])
                        yp_i = float(_yp_arr.ravel()[idx])
                        z_i = float(_z_arr.ravel()[idx])
                        err_f = _make_err(s_i, p_i, yp_i, z_i,
                                          _zm_s, _za_s, _zr_s)
                        guess = (prev_sol if prev_sol is not None
                                 else ideal_xy.get_t_sp(s_i, p_i, yp_i))
                        sol, ok = self._newton_1d(
                            err_f, guess, 1.5, 7.0)
                        if np.isfinite(sol):
                            out.ravel()[idx] = sol
                            prev_sol = sol
                return out.reshape(result_arr.shape)

        # --- Slow path: per-point Newton-Raphson ---
        scalar_input = np.isscalar(_s_kb) and np.isscalar(_lgp)
        _s_kb = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgp  = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _s_kb, _lgp = np.broadcast_arrays(_s_kb, _lgp)
        out = np.full_like(_s_kb, np.nan, dtype=float)

        prev_sol = None
        for idx in np.ndindex(_s_kb.shape):
            s_i   = float(_s_kb[idx])
            lgp_i = float(_lgp[idx])

            err_f = _make_err(s_i, lgp_i, _yp, _z, _zm, _za, _zr)
            guess = (prev_sol if prev_sol is not None
                     else ideal_xy.get_t_sp(s_i, lgp_i, _yp))
            sol, ok = self._newton_1d(err_f, guess, 1.5, 7.0)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        if scalar_input:
            return out.item()
        return out

    def get_logrho_sp(self, _s_kb, _lgp, _yp, _z=0.0,
                      _zm=0.0, _za=0.0, _zr=0.0, **kw):
        """Density from (S, P) — calls get_logt_sp then forward model."""
        _yp = self._to_yprime(_yp, _z)
        logt = self.get_logt_sp(_s_kb, _lgp, _yp, _z, _zm, _za, _zr)
        logt_arr = np.atleast_1d(logt)
        _lgp_arr = np.atleast_1d(_lgp)
        logt_arr, _lgp_arr = np.broadcast_arrays(logt_arr, _lgp_arr)

        out = np.full_like(logt_arr, np.nan, dtype=float)
        good = np.isfinite(logt_arr)
        if good.any():
            out[good] = self._logrho_pt(
                _lgp_arr[good], logt_arr[good], _yp, _z, _zm, _za, _zr)

        if out.size == 1:
            return out.item()
        return out

    # =================================================================
    # Pre-computed table lookup
    # =================================================================

    def _lookup_sp_table(self, _s_kb, _lgp, _yp, _z, rgi):
        """Query a pre-computed (ξ, logP, Y', Z) RGI table.

        To avoid cross-Z ξ mismatch, this brackets the query Z between
        the two nearest grid points, computes the correct ξ at each
        grid Z (so that both correspond to the same physical S), looks
        up logT at each, and linearly interpolates in Z.
        """
        _s_kb = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgp  = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _yp_a = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a  = np.atleast_1d(np.asarray(_z, dtype=float))
        _s_kb, _lgp, _yp_a, _z_a = np.broadcast_arrays(
            _s_kb, _lgp, _yp_a, _z_a)

        zg = self._zvals  # Z grid points
        flat = _z_a.ravel()

        # Bracket: find k such that zg[k] <= Z < zg[k+1]
        k = np.searchsorted(zg, flat, side='right') - 1
        k = np.clip(k, 0, len(zg) - 2)
        z_lo = zg[k]
        z_hi = zg[k + 1]
        dz = z_hi - z_lo
        dz[dz == 0] = 1.0  # guard (shouldn't happen)
        w = ((flat - z_lo) / dz).reshape(_z_a.shape)

        # Compute ξ at each bracket Z using the S bounds at that Z
        z_lo_arr = z_lo.reshape(_z_a.shape)
        z_hi_arr = z_hi.reshape(_z_a.shape)

        xi_lo = self.s_to_xi(_s_kb, _lgp, _yp_a, z_lo_arr)
        xi_hi = self.s_to_xi(_s_kb, _lgp, _yp_a, z_hi_arr)

        # Out-of-bounds check on BOTH bracket ξ values
        oob_lo = (xi_lo < -0.05) | (xi_lo > 1.05) | ~np.isfinite(xi_lo)
        oob_hi = (xi_hi < -0.05) | (xi_hi > 1.05) | ~np.isfinite(xi_hi)
        out_of_bounds = oob_lo | oob_hi

        xi_lo = np.where(np.isfinite(xi_lo), np.clip(xi_lo, 0.0, 1.0), 0.5)
        xi_hi = np.where(np.isfinite(xi_hi), np.clip(xi_hi, 0.0, 1.0), 0.5)

        # Query RGI at each bracket Z (ξ, P, Y', Z_grid)
        pts_lo = np.column_stack((xi_lo.ravel(), _lgp.ravel(),
                                  _yp_a.ravel(), z_lo_arr.ravel()))
        pts_hi = np.column_stack((xi_hi.ravel(), _lgp.ravel(),
                                  _yp_a.ravel(), z_hi_arr.ravel()))
        val_lo = rgi(pts_lo).reshape(_z_a.shape)
        val_hi = rgi(pts_hi).reshape(_z_a.shape)

        # Linear interpolation in Z
        out = (1.0 - w) * val_lo + w * val_hi

        # Force NaN for out-of-bounds points → triggers brentq fallback
        out[out_of_bounds] = np.nan

        if out.size == 1:
            return out.item()
        return out

    def _load_from_arrays(self, xi_vals, logp, yvals, zvals,
                          logt_sp, s_lo, s_hi):
        """Build RGI interpolators from arrays (shared by load and build)."""
        self.xi_vals = xi_vals
        self.logp_vals = logp
        self._yvals = yvals
        self._zvals = zvals
        self._s_lo = s_lo
        self._s_hi = s_hi

        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._logt_sp_rgi = RGI((xi_vals, logp, yvals, zvals),
                                 logt_sp, **rgi_kw)
        self._s_lo_rgi = RGI((logp, yvals, zvals), s_lo, **rgi_kw)
        self._s_hi_rgi = RGI((logp, yvals, zvals), s_hi, **rgi_kw)

    def load_sp_table(self, path):
        """Load a pre-computed adaptive S-P table from NPZ.

        Expected keys: xi_vals, logpvals, yvals, zvals,
                        logt_sp, s_lo, s_hi, logt_min, logt_max.

        logrho is not stored — it is computed on-the-fly from the
        forward model via ``get_logrho_sp``.
        """
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        self._load_from_arrays(
            data['xi_vals'], data['logpvals'],
            data['yvals'], data['zvals'],
            data['logt_sp'], data['s_lo'], data['s_hi'])

    # =================================================================
    # NaN repair for tables
    # =================================================================

    @staticmethod
    def _fill_nans_2axis(arr, axis0=0, axis1=1):
        """Interpolate NaN cells in a 4-D array along two axes (in-place).

        Pass 1: interpolate along ``axis0`` at each slice of the other dims.
        Pass 2: interpolate along ``axis1`` for any remaining NaN.
        """
        shape = arr.shape
        n0, n1 = shape[axis0], shape[axis1]
        other = [i for i in range(4) if i not in (axis0, axis1)]
        n2, n3 = shape[other[0]], shape[other[1]]

        for i2 in range(n2):
            for i3 in range(n3):
                # Build index template
                idx = [None] * 4
                idx[other[0]] = i2
                idx[other[1]] = i3

                # Pass 1: along axis0
                for j1 in range(n1):
                    idx[axis1] = j1
                    idx[axis0] = slice(None)
                    col = arr[tuple(idx)]
                    bad = np.isnan(col)
                    if not bad.any():
                        continue
                    good = ~bad
                    if good.sum() < 2:
                        continue
                    col[bad] = np.interp(
                        np.where(bad)[0], np.where(good)[0], col[good])
                    arr[tuple(idx)] = col

                # Pass 2: along axis1
                for j0 in range(n0):
                    idx[axis0] = j0
                    idx[axis1] = slice(None)
                    row = arr[tuple(idx)]
                    bad = np.isnan(row)
                    if not bad.any():
                        continue
                    good = ~bad
                    if good.sum() < 2:
                        continue
                    row[bad] = np.interp(
                        np.where(bad)[0], np.where(good)[0], row[good])
                    arr[tuple(idx)] = row

    @staticmethod
    def _fill_table_nans(table):
        """Interpolate over NaN cells in a 4-D (ξ, P, Y', Z) table.

        Strategy (applied in order of priority):
        1. Along the ξ axis (axis 0) — interpolate from valid
           neighbours at the same (P, Y', Z).  This is the most
           physically meaningful direction since ξ maps linearly
           to entropy.
        2. Along the P axis (axis 1) — for entire ξ-columns that
           are NaN (e.g. at extreme P), interpolate from
           neighbouring pressures.
        3. Any cells still NaN after both passes are filled by
           nearest-neighbour extrapolation along ξ.

        Returns a copy; the original is not modified.
        """
        out = table.copy()
        n_xi, nP, nY, nZ = out.shape

        # --- Pass 1: interpolate along ξ (axis 0) ---
        for ip in range(nP):
            for iy in range(nY):
                for iz in range(nZ):
                    col = out[:, ip, iy, iz]
                    bad = np.isnan(col)
                    if not bad.any():
                        continue
                    good = ~bad
                    if good.sum() < 2:
                        continue  # not enough data — leave for pass 2
                    col[bad] = np.interp(
                        np.where(bad)[0],
                        np.where(good)[0],
                        col[good])
                    out[:, ip, iy, iz] = col

        # --- Pass 2: interpolate along P (axis 1) ---
        for ixi in range(n_xi):
            for iy in range(nY):
                for iz in range(nZ):
                    row = out[ixi, :, iy, iz]
                    bad = np.isnan(row)
                    if not bad.any():
                        continue
                    good = ~bad
                    if good.sum() < 2:
                        continue
                    row[bad] = np.interp(
                        np.where(bad)[0],
                        np.where(good)[0],
                        row[good])
                    out[ixi, :, iy, iz] = row

        # --- Pass 3: nearest-neighbour extrapolation along ξ ---
        for ip in range(nP):
            for iy in range(nY):
                for iz in range(nZ):
                    col = out[:, ip, iy, iz]
                    bad = np.isnan(col)
                    if not bad.any():
                        continue
                    good = ~bad
                    if not good.any():
                        continue  # entire column NaN — nothing to do
                    # Forward-fill then back-fill
                    good_idx = np.where(good)[0]
                    for bi in np.where(bad)[0]:
                        nearest = good_idx[np.argmin(np.abs(good_idx - bi))]
                        col[bi] = col[nearest]
                    out[:, ip, iy, iz] = col

        return out

    # =================================================================
    # Table generation
    # =================================================================

    def build_sp_table(self, yvals, zvals,
                       _zm=0.0, _za=0.0, _zr=0.0,
                       n_xi=None, verbose=True):
        """Build the full logT(ξ, P, Y', Z) and logrho(ξ, P, Y', Z) tables.

        For each (P, Y', Z) grid point the physical entropy range
        [S_lo, S_hi] is computed from the T boundaries of the P-T
        table.  ``n_xi`` evenly spaced ξ values in [0, 1] are then
        mapped to physical S values within that range and inverted
        via ``brentq`` to obtain logT.  logrho is computed from the
        forward model at (P, T).

        Parameters
        ----------
        yvals : array_like
            1-D array of Y' values.
        zvals : array_like
            1-D array of Z values.
        _zm, _za, _zr : float
            Fixed nested metal sub-fractions.
        n_xi : int or None
            Number of ξ grid points (default: ``self.n_xi``).
        verbose : bool
            Print progress.

        Returns
        -------
        result : dict
            Keys: xi_vals, logpvals, yvals, zvals,
                  logt_sp  (n_xi, nP, nY, nZ),
                  logrho_sp (n_xi, nP, nY, nZ),
                  s_lo (nP, nY, nZ),
                  s_hi (nP, nY, nZ),
                  logt_min, logt_max.

        Also loads the table into this instance (sets the RGI
        interpolators so ``get_logt_sp`` / ``get_logrho_sp`` use
        the fast table path).
        """
        if n_xi is None:
            n_xi = self.n_xi

        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)

        # Non-uniform ξ spacing: cluster points near ξ=0 where
        # physical planetary entropies (S ≈ 4–15 kb/baryon) live.
        # At low P and high Z, the S range is huge (s_hi > 100) but
        # physical S maps to ξ < 0.1, so uniform spacing wastes
        # resolution.  A sinh-based mapping puts ~half the points
        # in the lower 20% of ξ.
        #
        #   ξ = sinh(α * t) / sinh(α)   where t ∈ [0, 1]
        #
        # α controls the clustering strength (α=0 → uniform,
        # larger α → more clustered near 0).
        alpha = 2.0
        t_uniform = np.linspace(0.0, 1.0, n_xi)
        xi_vals = np.sinh(alpha * t_uniform) / np.sinh(alpha)

        logp = self.logp_vals

        nP, nY, nZ = len(logp), len(yvals), len(zvals)

        # --- Step 1: compute S bounds ---
        if verbose:
            frac_below_20 = np.sum(xi_vals < 0.2) / n_xi
            print(f"Building S-P table: n_xi={n_xi} "
                  f"(sinh-clustered, α={alpha}, "
                  f"{frac_below_20:.0%} of pts below ξ=0.2), "
                  f"logP=[{logp[0]:.2f}, {logp[-1]:.2f}] "
                  f"(dlogP={logp[1]-logp[0]:.2f}, {nP} pts), "
                  f"logT=[{self.logt_min:.1f}, {self.logt_max:.1f}]")
            print(f"  Y' grid: {nY} pts [{yvals[0]:.3f} .. {yvals[-1]:.3f}], "
                  f"Z grid: {nZ} pts [{zvals[0]:.3f} .. {zvals[-1]:.3f}]")
            print(f"  Total cells: {n_xi}×{nP}×{nY}×{nZ} = "
                  f"{n_xi*nP*nY*nZ:,}")
        self.compute_s_bounds_grid(yvals, zvals, _zm, _za, _zr)
        s_lo = self._s_lo   # (nP, nY, nZ)
        s_hi = self._s_hi

        # --- Step 2: invert at each (ξ, P, Y', Z) ---
        # Only store logT; logrho is computed on-the-fly from
        # val.get_logrho_pt_val(P, T, Y', Z) to halve memory.
        logt_sp = np.full((n_xi, nP, nY, nZ), np.nan, dtype=float)

        total = nY * nZ
        pbar = tqdm(total=total,
                     desc="Inverting P,T → S,P",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        # Solver strategy:
        #   - At the START of each (Y', Z) composition (iz==0):
        #     use brentq to get reliable initial solutions.
        #   - For subsequent Z steps: use newton warm-started from
        #     the previous Z solution (small composition step → fast
        #     convergence).  Fall back to brentq on newton failure.
        #   - prev_logt[ixi, ip] caches the last converged logT at
        #     each (ξ, P) cell across Z steps within the same Y'.
        prev_logt = np.full((n_xi, nP), np.nan)
        lmin, lmax = self.logt_min, self.logt_max

        for iy, yp in enumerate(yvals):
            # Reset cache when Y' changes (Z wraps from 1→0)
            prev_logt[:] = np.nan

            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                use_newton = iz > 0  # first Z uses brentq only

                for ip in range(nP):
                    lgp_i = logp[ip]
                    slo_i = s_lo[ip, iy, iz]
                    shi_i = s_hi[ip, iy, iz]

                    if (not np.isfinite(slo_i)) or (not np.isfinite(shi_i)):
                        continue
                    if shi_i <= slo_i:
                        continue

                    for ixi in range(n_xi):
                        s_phys = slo_i + xi_vals[ixi] * (shi_i - slo_i)

                        def err(lgt, _s=s_phys, _p=lgp_i):
                            try:
                                s_test = self._s_pt(
                                    _p, lgt, yp, zv, _zm, _za, _zr)
                                return float(
                                    s_test * erg_to_kbbar - _s)
                            except (ZeroDivisionError,
                                    FloatingPointError):
                                return np.nan

                        # Use previous solution or mid-range as guess
                        guess = prev_logt[ixi, ip]
                        if not np.isfinite(guess):
                            guess = 0.5 * (lmin + lmax)
                        lgt_sol, ok = self._newton_1d(
                            err, guess, 1.5, 7.0)
                        if np.isfinite(lgt_sol):
                            logt_sp[ixi, ip, iy, iz] = lgt_sol
                            prev_logt[ixi, ip] = lgt_sol

        pbar.close()

        # --- Step 3: fill NaNs first (so Hampel has no gaps) ---
        n_nan_before = np.isnan(logt_sp).sum()
        if n_nan_before > 0:
            if verbose:
                print(f"Filling {n_nan_before} NaN cells by "
                      f"interpolation ...")
            logt_sp = self._fill_table_nans(logt_sp)
            n_nan_after = np.isnan(logt_sp).sum()
            if verbose and n_nan_after > 0:
                print(f"  WARNING: {n_nan_after} NaNs remain after "
                      f"interpolation")

        # --- Step 4: Hampel outlier filter on logT along ξ axis ---
        if verbose:
            print("Running Hampel outlier filter on logT along ξ axis ...")
        flat = logt_sp.reshape(n_xi, -1)
        n_outliers = 0
        for j in range(flat.shape[1]):
            col = flat[:, j]
            cleaned, n_rep = hampel_filter_1d(
                col, window=5, n_sigma=3.0)
            if n_rep > 0:
                changed = (cleaned != col) & np.isfinite(col)
                flat[changed, j] = cleaned[changed]
                n_outliers += n_rep
        if n_outliers > 0 and verbose:
            print(f"  Replaced {n_outliers} outlier cells in logT")

        # --- Step 5: cast to float32 to halve memory ---
        logt_sp_f32 = logt_sp.astype(np.float32)
        s_lo_f32    = s_lo.astype(np.float32)
        s_hi_f32    = s_hi.astype(np.float32)

        if verbose:
            mem_mb = logt_sp_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        # --- Step 5: package result ---
        result = {
            'xi_vals':   xi_vals,
            'logpvals':  logp,
            'yvals':     yvals,
            'zvals':     zvals,
            'logt_sp':   logt_sp_f32,
            's_lo':      s_lo_f32,
            's_hi':      s_hi_f32,
            'logt_min':  self.logt_min,
            'logt_max':  self.logt_max,
        }

        # --- Step 6: load into this instance ---
        self._load_from_arrays(xi_vals, logp, yvals, zvals,
                               logt_sp_f32, s_lo_f32, s_hi_f32)

        if verbose:
            n_total = logt_sp.size
            n_good = np.isfinite(logt_sp).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan_before} were interpolated")

        return result

    def save_sp_table(self, result, path=None):
        """Save a table dict (from ``build_sp_table``) to NPZ.

        Parameters
        ----------
        result : dict
            Output of ``build_sp_table``.
        path : str or None
            File path.  If None, uses the default auto-load path:
            ``eos/{hhe_eos}/{hhe_eos}_{z_eos}_sp_adaptive.npz``.
        """
        if path is None:
            path = self._table_path('sp')
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # rho-T inversion: P(rho, T, Y', Z)
    # =================================================================

    def get_logp_rhot(self, _lgrho, _lgt, _yp, _z=0.0,
                      _zm=0.0, _za=0.0, _zr=0.0, **kw):
        """Pressure from (rho, T) via root-finding or pre-computed table.

        Inverts rho(P, T, Y', Z) = 10^_lgrho to find logP.
        """
        # --- Fast path: ξ-mapped table ---
        if self._logp_rhot_rgi is not None:
            return self._lookup_rhot_table(
                _lgrho, _lgt, _yp, _z)

        _yp = self._to_yprime(_yp, _z)
        # --- Slow path: Newton-Raphson per-point ---
        scalar_input = np.isscalar(_lgrho) and np.isscalar(_lgt)
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgt   = np.atleast_1d(np.asarray(_lgt, dtype=float))
        _lgrho, _lgt = np.broadcast_arrays(_lgrho, _lgt)
        out = np.full_like(_lgrho, np.nan, dtype=float)

        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]

        prev_sol = None
        for idx in np.ndindex(_lgrho.shape):
            rho_i = float(_lgrho[idx])
            lgt_i = float(_lgt[idx])

            def err(lgp, _r=rho_i, _t=lgt_i):
                try:
                    rho_test = self._logrho_pt(
                        lgp, _t, _yp, _z, _zm, _za, _zr)
                    return float(rho_test - _r)
                except (ZeroDivisionError, FloatingPointError):
                    return np.nan

            guess = (prev_sol if prev_sol is not None
                     else ideal_xy.get_p_rhot(rho_i, lgt_i, _yp))
            sol, ok = self._newton_1d(err, guess, lgp_lo, lgp_hi)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        if scalar_input:
            return out.item()
        return out

    def compute_rho_bounds_rhot(self, yvals, zvals,
                                _zm=0.0, _za=0.0, _zr=0.0):
        """Compute physical logrho bounds at each (logT, Y', Z).

        At each temperature, the accessible density range is
        [rho(P_min, T), rho(P_max, T)].

        Stores ``self._rho_lo_rhot`` and ``self._rho_hi_rhot``
        as 3-D arrays (nT, nY, nZ).
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        logt = self.logt_vals
        nT, nY, nZ = len(logt), len(yvals), len(zvals)
        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]

        rho_lo = np.full((nT, nY, nZ), np.nan)
        rho_hi = np.full((nT, nY, nZ), np.nan)

        for iy, yp in enumerate(yvals):
            for iz, zv in enumerate(zvals):
                for it, lgt in enumerate(logt):
                    try:
                        r_plo = self._logrho_pt(
                            lgp_lo, lgt, yp, zv, _zm, _za, _zr)
                        r_phi = self._logrho_pt(
                            lgp_hi, lgt, yp, zv, _zm, _za, _zr)
                    except (ZeroDivisionError, FloatingPointError):
                        continue
                    if np.isfinite(r_plo) and np.isfinite(r_phi):
                        rho_lo[it, iy, iz] = min(r_plo, r_phi)
                        rho_hi[it, iy, iz] = max(r_plo, r_phi)

        self._rho_lo_rhot = rho_lo.astype(np.float32)
        self._rho_hi_rhot = rho_hi.astype(np.float32)
        self._yvals_rhot = yvals
        self._zvals_rhot = zvals

        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._rho_lo_rhot_rgi = RGI(
            (logt, yvals, zvals), self._rho_lo_rhot, **rgi_kw)
        self._rho_hi_rhot_rgi = RGI(
            (logt, yvals, zvals), self._rho_hi_rhot, **rgi_kw)

    def rho_to_xi_rhot(self, _lgrho, _lgt, _yp, _z):
        """Convert logrho → ξ in the ρ-T rhomboid."""
        pts = np.column_stack((
            np.atleast_1d(_lgt).ravel(),
            np.atleast_1d(_yp).ravel(),
            np.atleast_1d(_z).ravel()))
        rlo = self._rho_lo_rhot_rgi(pts).reshape(np.atleast_1d(_lgt).shape)
        rhi = self._rho_hi_rhot_rgi(pts).reshape(np.atleast_1d(_lgt).shape)
        denom = rhi - rlo
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        xi = (np.atleast_1d(_lgrho) - rlo) / denom
        if np.isscalar(_lgrho) and np.isscalar(_lgt):
            return float(xi.ravel()[0])
        return xi

    def _lookup_rhot_table(self, _lgrho, _lgt, _yp, _z):
        """Query the pre-computed ξ-mapped (ξ, logT, Y', Z) RGI."""
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgt   = np.atleast_1d(np.asarray(_lgt, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _lgrho, _lgt, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho, _lgt, _yp_a, _z_a)

        xi = self.rho_to_xi_rhot(_lgrho, _lgt, _yp_a, _z_a)
        out = np.full_like(xi, np.nan, dtype=float)
        good = (xi >= 0.0) & (xi <= 1.0) & np.isfinite(xi)
        if good.any():
            pts = np.column_stack((xi[good], _lgt[good],
                                   _yp_a[good], _z_a[good]))
            out[good] = self._logp_rhot_rgi(pts)

        if out.size == 1:
            return out.item()
        return out

    def load_rhot_table(self, path):
        """Load a pre-computed ξ-mapped ρ-T → P table from NPZ."""
        data = np.load(path)
        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        xi = data['xi_vals']
        logt = data['logtvals']
        yv = data['yvals']
        zv = data['zvals']

        self._logp_rhot_rgi = RGI(
            (xi, logt, yv, zv), data['logp_rhot'], **rgi_kw)
        self.logt_vals = logt
        self._rho_lo_rhot = data['rho_lo_rhot']
        self._rho_hi_rhot = data['rho_hi_rhot']
        self._yvals_rhot = yv
        self._zvals_rhot = zv
        self._rho_lo_rhot_rgi = RGI(
            (logt, yv, zv), self._rho_lo_rhot, **rgi_kw)
        self._rho_hi_rhot_rgi = RGI(
            (logt, yv, zv), self._rho_hi_rhot, **rgi_kw)

    def build_rhot_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         n_xi=None, verbose=True):
        """Build logP(ξ_ρ, logT, Y', Z) table with ξ-mapping on ρ.

        At each (logT, Y', Z), the density range [ρ_lo, ρ_hi] comes
        from evaluating ρ(P_min, T) and ρ(P_max, T).  ``n_xi`` evenly
        spaced ξ values map this range to [0, 1].
        """
        if n_xi is None:
            n_xi = self.n_xi

        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        alpha_xi = 2.0
        t_uniform = np.linspace(0.0, 1.0, n_xi)
        xi_vals = np.sinh(alpha_xi * t_uniform) / np.sinh(alpha_xi)
        logt = self.logt_vals
        nT, nY, nZ = len(logt), len(yvals), len(zvals)
        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]

        if verbose:
            print(f"Building ρ-T table (ξ-mapped): n_xi={n_xi}, "
                  f"logT=[{logt[0]:.2f}, {logt[-1]:.2f}] ({nT} pts), "
                  f"logP=[{lgp_lo:.1f}, {lgp_hi:.1f}]")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {n_xi}×{nT}×{nY}×{nZ} = "
                  f"{n_xi*nT*nY*nZ:,}")

        # Step 1: compute ρ bounds
        if verbose:
            print("Computing ρ bounds at each (T, Y', Z) ...")
        self.compute_rho_bounds_rhot(yvals, zvals, _zm, _za, _zr)
        rho_lo = self._rho_lo_rhot  # (nT, nY, nZ)
        rho_hi = self._rho_hi_rhot

        # Step 2: invert
        logp_tab = np.full((n_xi, nT, nY, nZ), np.nan, dtype=float)

        total = nY * nZ
        pbar = tqdm(total=total,
                     desc="Inverting P,T → ρ,T",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        prev_logp = np.full((n_xi, nT), np.nan)

        for iy, yp in enumerate(yvals):
            prev_logp[:] = np.nan

            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                use_newton = iz > 0

                for it in range(nT):
                    lgt_i = logt[it]
                    rlo_i = rho_lo[it, iy, iz]
                    rhi_i = rho_hi[it, iy, iz]

                    if (not np.isfinite(rlo_i)) or (not np.isfinite(rhi_i)):
                        continue
                    if rhi_i <= rlo_i:
                        continue

                    for ixi in range(n_xi):
                        rho_phys = rlo_i + xi_vals[ixi] * (rhi_i - rlo_i)

                        def err(lgp, _rho=rho_phys, _t=lgt_i):
                            try:
                                rho_test = self._logrho_pt(
                                    lgp, _t, yp, zv, _zm, _za, _zr)
                                return float(rho_test - _rho)
                            except (ZeroDivisionError,
                                    FloatingPointError):
                                return np.nan

                        guess = prev_logp[ixi, it]
                        if not np.isfinite(guess):
                            guess = 0.5 * (lgp_lo + lgp_hi)
                        lgp_sol, ok = self._newton_1d(
                            err, guess, lgp_lo, lgp_hi)
                        if np.isfinite(lgp_sol):
                            logp_tab[ixi, it, iy, iz] = lgp_sol
                            prev_logp[ixi, it] = lgp_sol

        pbar.close()

        # Step 3: fill NaNs first (so Hampel has no gaps)
        n_nan = np.isnan(logp_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logp_tab = self._fill_table_nans(logp_tab)

        # Step 4: Hampel outlier filter on logP along ξ axis
        if verbose:
            print("Running Hampel outlier filter on logP along ξ axis ...")
        flat = logp_tab.reshape(n_xi, -1)
        n_outliers = 0
        for j in range(flat.shape[1]):
            col = flat[:, j]
            cleaned, n_rep = hampel_filter_1d(
                col, window=5, n_sigma=3.0)
            if n_rep > 0:
                changed = (cleaned != col) & np.isfinite(col)
                flat[changed, j] = cleaned[changed]
                n_outliers += n_rep
        if n_outliers > 0 and verbose:
            print(f"  Replaced {n_outliers} outlier cells in logP")

        logp_f32 = logp_tab.astype(np.float32)

        if verbose:
            mem_mb = logp_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        result = {
            'xi_vals':      xi_vals,
            'logtvals':     logt,
            'yvals':        yvals,
            'zvals':        zvals,
            'logp_rhot':    logp_f32,
            'rho_lo_rhot':  rho_lo.astype(np.float32),
            'rho_hi_rhot':  rho_hi.astype(np.float32),
            'logt_min':     self.logt_min,
            'logt_max':     self.logt_max,
        }

        # Load into this instance
        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._logp_rhot_rgi = RGI(
            (xi_vals, logt, yvals, zvals), logp_f32, **rgi_kw)

        if verbose:
            n_total = logp_tab.size
            n_good = np.isfinite(logp_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def save_rhot_table(self, result, path=None):
        """Save a ρ-T table dict to NPZ."""
        if path is None:
            path = self._table_path('rhot')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # ρ-P inversion: T(ρ, P, Y', Z) — 1-D with ξ-mapping on ρ axis
    # =================================================================

    def compute_rho_bounds_rhop(self, yvals, zvals,
                                _zm=0.0, _za=0.0, _zr=0.0):
        """Compute the physical logrho bounds at each (logP, Y', Z).

        At each pressure, rho ranges from rho(P, T_max) (hot, low ρ)
        to rho(P, T_min) (cold, high ρ).

        Stores ``self._rho_lo_rhop`` and ``self._rho_hi_rhop`` as
        3-D arrays (nP, nY, nZ).
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        nP, nY, nZ = len(self.logp_vals), len(yvals), len(zvals)

        rho_lo = np.full((nP, nY, nZ), np.nan)
        rho_hi = np.full((nP, nY, nZ), np.nan)

        for iy, yp in enumerate(yvals):
            for iz, zv in enumerate(zvals):
                for ip, lgp in enumerate(self.logp_vals):
                    r_hot = self._logrho_pt(
                        lgp, self.logt_max, yp, zv, _zm, _za, _zr)
                    r_cold = self._logrho_pt(
                        lgp, self.logt_min, yp, zv, _zm, _za, _zr)
                    if np.isfinite(r_hot) and np.isfinite(r_cold):
                        rho_lo[ip, iy, iz] = min(r_hot, r_cold)
                        rho_hi[ip, iy, iz] = max(r_hot, r_cold)

        self._rho_lo_rhop = rho_lo.astype(np.float32)
        self._rho_hi_rhop = rho_hi.astype(np.float32)
        self._yvals_rhop = yvals
        self._zvals_rhop = zvals

        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._rho_lo_rhop_rgi = RGI(
            (self.logp_vals, yvals, zvals), self._rho_lo_rhop, **rgi_kw)
        self._rho_hi_rhop_rgi = RGI(
            (self.logp_vals, yvals, zvals), self._rho_hi_rhop, **rgi_kw)

    def rho_to_xi_rhop(self, _lgrho, _lgp, _yp, _z):
        """Convert logrho → ξ in the ρ-P rhomboid."""
        pts = np.column_stack((
            np.atleast_1d(_lgp).ravel(),
            np.atleast_1d(_yp).ravel(),
            np.atleast_1d(_z).ravel()))
        rlo = self._rho_lo_rhop_rgi(pts).reshape(np.atleast_1d(_lgp).shape)
        rhi = self._rho_hi_rhop_rgi(pts).reshape(np.atleast_1d(_lgp).shape)
        denom = rhi - rlo
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        xi = (np.atleast_1d(_lgrho) - rlo) / denom
        if np.isscalar(_lgrho) and np.isscalar(_lgp):
            return float(xi.ravel()[0])
        return xi

    def xi_to_rho_rhop(self, _xi, _lgp, _yp, _z):
        """Convert ξ → logrho in the ρ-P rhomboid."""
        pts = np.column_stack((
            np.atleast_1d(_lgp).ravel(),
            np.atleast_1d(_yp).ravel(),
            np.atleast_1d(_z).ravel()))
        rlo = self._rho_lo_rhop_rgi(pts).reshape(np.atleast_1d(_lgp).shape)
        rhi = self._rho_hi_rhop_rgi(pts).reshape(np.atleast_1d(_lgp).shape)
        return rlo + np.atleast_1d(_xi) * (rhi - rlo)

    def get_logt_rhop(self, _lgrho, _lgp, _yp, _z=0.0,
                      _zm=0.0, _za=0.0, _zr=0.0, **kw):
        """Temperature from (ρ, P) via 1-D root-finding or table.

        Inverts ρ(P, T, Y', Z) = 10^logrho to find logT.

        Parameters
        ----------
        _lgrho : float or array
            log10 ρ [g/cm³].
        _lgp : float or array
            log10 P [dyn/cm²].
        _yp : float
            Y' = Y/(1-Z).
        _z : float
            Total metal mass fraction.

        Returns
        -------
        logt : float or array
            log10 T [K].  NaN where no solution.
        """
        _yp = self._to_yprime(_yp, _z)
        # Fast path: pre-computed table
        if self._logt_rhop_rgi is not None:
            return self._lookup_rhop_table(_lgrho, _lgp, _yp, _z)

        # Slow path: Newton-Raphson per point
        scalar = np.isscalar(_lgrho) and np.isscalar(_lgp)
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgp   = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _lgrho, _lgp = np.broadcast_arrays(_lgrho, _lgp)
        out = np.full_like(_lgrho, np.nan, dtype=float)

        prev_sol = None
        for idx in np.ndindex(_lgrho.shape):
            rho_i = float(_lgrho[idx])
            lgp_i = float(_lgp[idx])

            def err(lgt, _r=rho_i, _p=lgp_i):
                try:
                    return float(self._logrho_pt(
                        _p, lgt, _yp, _z, _zm, _za, _zr) - _r)
                except (ZeroDivisionError, FloatingPointError):
                    return np.nan

            guess = (prev_sol if prev_sol is not None
                     else ideal_xy.get_t_rhop(rho_i, lgp_i, _yp))
            sol, ok = self._newton_1d(err, guess, 1.5, 7.0)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        if scalar:
            return out.item()
        return out

    def _lookup_rhop_table(self, _lgrho, _lgp, _yp, _z):
        """Query the pre-computed (ξ, logP, Y', Z) ρ-P RGI."""
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgp   = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _lgrho, _lgp, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho, _lgp, _yp_a, _z_a)

        xi = self.rho_to_xi_rhop(_lgrho, _lgp, _yp_a, _z_a)
        out = np.full_like(xi, np.nan, dtype=float)
        good = (xi >= 0.0) & (xi <= 1.0) & np.isfinite(xi)
        if good.any():
            pts = np.column_stack((xi[good], _lgp[good],
                                   _yp_a[good], _z_a[good]))
            out[good] = self._logt_rhop_rgi(pts)

        if out.size == 1:
            return out.item()
        return out

    def build_rhop_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         n_xi=None, verbose=True):
        """Build logT(ξ_ρ, logP, Y', Z) table with ξ-mapping on ρ.

        At each (logP, Y', Z), the density range [ρ_lo, ρ_hi] is
        computed from the T boundaries.  ``n_xi`` evenly spaced ξ
        values map this range to [0, 1].
        """
        if n_xi is None:
            n_xi = self.n_xi

        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        alpha_xi = 2.0
        t_uniform = np.linspace(0.0, 1.0, n_xi)
        xi_vals = np.sinh(alpha_xi * t_uniform) / np.sinh(alpha_xi)
        logp = self.logp_vals
        nP, nY, nZ = len(logp), len(yvals), len(zvals)

        if verbose:
            print(f"Building ρ-P table: n_xi={n_xi}, "
                  f"logP=[{logp[0]:.2f}, {logp[-1]:.2f}] "
                  f"(dlogP={logp[1]-logp[0]:.2f}, {nP} pts), "
                  f"logT=[{self.logt_min:.1f}, {self.logt_max:.1f}]")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {n_xi}×{nP}×{nY}×{nZ} = "
                  f"{n_xi*nP*nY*nZ:,}")

        # Step 1: compute rho bounds
        self.compute_rho_bounds_rhop(yvals, zvals, _zm, _za, _zr)
        rho_lo = self._rho_lo_rhop  # (nP, nY, nZ)
        rho_hi = self._rho_hi_rhop

        # Step 2: invert
        logt_tab = np.full((n_xi, nP, nY, nZ), np.nan, dtype=float)

        total = nY * nZ
        pbar = tqdm(total=total,
                     desc="Inverting P,T → ρ,P",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        prev_logt = np.full((n_xi, nP), np.nan)

        for iy, yp in enumerate(yvals):
            prev_logt[:] = np.nan

            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                use_newton = iz > 0

                for ip in range(nP):
                    lgp_i = logp[ip]
                    rlo_i = rho_lo[ip, iy, iz]
                    rhi_i = rho_hi[ip, iy, iz]

                    if (not np.isfinite(rlo_i)) or (not np.isfinite(rhi_i)):
                        continue
                    if rhi_i <= rlo_i:
                        continue

                    for ixi in range(n_xi):
                        rho_phys = rlo_i + xi_vals[ixi] * (rhi_i - rlo_i)

                        def err(lgt, _rho=rho_phys, _p=lgp_i):
                            try:
                                return float(self._logrho_pt(
                                    _p, lgt, yp, zv, _zm, _za, _zr)
                                    - _rho)
                            except (ZeroDivisionError,
                                    FloatingPointError):
                                return np.nan

                        guess = prev_logt[ixi, ip]
                        if not np.isfinite(guess):
                            guess = 0.5 * (1.5 + 7.0)
                        lgt_sol, ok = self._newton_1d(
                            err, guess, 1.5, 7.0)
                        if np.isfinite(lgt_sol):
                            logt_tab[ixi, ip, iy, iz] = lgt_sol
                            prev_logt[ixi, ip] = lgt_sol

        pbar.close()

        # Step 3: fill NaNs first (so Hampel has no gaps)
        n_nan = np.isnan(logt_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logt_tab = self._fill_table_nans(logt_tab)

        # Step 4: Hampel outlier filter on logT along ξ axis
        if verbose:
            print("Running Hampel outlier filter on logT along ξ axis ...")
        flat = logt_tab.reshape(n_xi, -1)
        n_outliers = 0
        for j in range(flat.shape[1]):
            col = flat[:, j]
            cleaned, n_rep = hampel_filter_1d(
                col, window=5, n_sigma=3.0)
            if n_rep > 0:
                changed = (cleaned != col) & np.isfinite(col)
                flat[changed, j] = cleaned[changed]
                n_outliers += n_rep
        if n_outliers > 0 and verbose:
            print(f"  Replaced {n_outliers} outlier cells in logT")

        logt_f32 = logt_tab.astype(np.float32)
        rho_lo_f32 = rho_lo.astype(np.float32)
        rho_hi_f32 = rho_hi.astype(np.float32)

        if verbose:
            mem_mb = logt_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        result = {
            'xi_vals':     xi_vals,
            'logpvals':    logp,
            'yvals':       yvals,
            'zvals':       zvals,
            'logt_rhop':   logt_f32,
            'rho_lo_rhop': rho_lo_f32,
            'rho_hi_rhop': rho_hi_f32,
            'logt_min':    self.logt_min,
            'logt_max':    self.logt_max,
        }

        # Load into this instance
        self._load_rhop_from_arrays(
            xi_vals, logp, yvals, zvals,
            logt_f32, rho_lo_f32, rho_hi_f32)

        if verbose:
            n_total = logt_tab.size
            n_good = np.isfinite(logt_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def _load_rhop_from_arrays(self, xi_vals, logp, yvals, zvals,
                                logt, rho_lo, rho_hi):
        """Build ρ-P RGI interpolators."""
        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._logt_rhop_rgi = RGI((xi_vals, logp, yvals, zvals),
                                   logt, **rgi_kw)
        self._rho_lo_rhop = rho_lo
        self._rho_hi_rhop = rho_hi
        self._yvals_rhop = yvals
        self._zvals_rhop = zvals
        self._rho_lo_rhop_rgi = RGI((logp, yvals, zvals), rho_lo, **rgi_kw)
        self._rho_hi_rhop_rgi = RGI((logp, yvals, zvals), rho_hi, **rgi_kw)

    def load_rhop_table(self, path):
        """Load a ρ-P → T table from NPZ."""
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        self._load_rhop_from_arrays(
            data['xi_vals'], data['logpvals'],
            data['yvals'], data['zvals'],
            data['logt_rhop'], data['rho_lo_rhop'], data['rho_hi_rhop'])

    def save_rhop_table(self, result, path=None):
        """Save a ρ-P table to NPZ."""
        if path is None:
            path = self._table_path('rhop')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # S-ρ inversion: P,T(S, ρ, Y', Z) — 2-D least-squares
    # =================================================================

    def compute_s_bounds_srho(self, yvals, zvals,
                              _zm=0.0, _za=0.0, _zr=0.0,
                              n_t_sweep=20, verbose=True):
        """Compute physical S bounds at each (logrho, Y', Z).

        At each (logrho, Y', Z), sweeps logT from logt_min to logt_max,
        inverts for logP via 1-D brentq (``get_logp_rhot``), evaluates
        S(P, T), and records the min/max over T.

        Stores ``self._s_lo_srho``, ``self._s_hi_srho`` as 3-D arrays
        of shape (nRho, nY, nZ) in kb/baryon.
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        logrho = self.logrho_vals
        nR, nY, nZ = len(logrho), len(yvals), len(zvals)

        logt_sweep = np.linspace(self.logt_min, self.logt_max, n_t_sweep)
        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]

        s_lo = np.full((nR, nY, nZ), np.inf)
        s_hi = np.full((nR, nY, nZ), -np.inf)

        pbar = tqdm(total=nY * nZ,
                     desc="S bounds (S-ρ)",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        for iy, yp in enumerate(yvals):
            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                for ir in range(nR):
                    rho_target = logrho[ir]

                    for lgt in logt_sweep:
                        # Find P that gives this (rho, T)
                        def err_p(lgp):
                            try:
                                return (self._logrho_pt(
                                    lgp, lgt, yp, zv, _zm, _za, _zr)
                                    - rho_target)
                            except (ZeroDivisionError, FloatingPointError):
                                return 1e30
                        try:
                            lgp = brentq(err_p, lgp_lo, lgp_hi,
                                         xtol=1e-6, maxiter=60)
                        except (ValueError, RuntimeError, ZeroDivisionError):
                            continue

                        s_val = (self._s_pt(
                            lgp, lgt, yp, zv, _zm, _za, _zr)
                            * erg_to_kbbar)

                        if np.isfinite(s_val):
                            s_lo[ir, iy, iz] = min(s_lo[ir, iy, iz], s_val)
                            s_hi[ir, iy, iz] = max(s_hi[ir, iy, iz], s_val)

        pbar.close()

        # Replace inf with NaN (cells with no valid T sweep)
        s_lo[~np.isfinite(s_lo)] = np.nan
        s_hi[~np.isfinite(s_hi)] = np.nan

        # Cap S_hi to prevent the ξ range from being dominated
        # by extreme values at low ρ / high T (same treatment as
        # compute_s_bounds_grid for the S-P table).
        for iy_i in range(nY):
            for iz_i in range(nZ):
                col = s_hi[:, iy_i, iz_i]
                finite = col[np.isfinite(col)]
                if len(finite) > 0:
                    cap = np.nanpercentile(finite, 75) * 1.5
                    cap = max(cap, np.nanmin(finite))
                    s_hi[:, iy_i, iz_i] = np.where(
                        np.isfinite(col), np.minimum(col, cap), col)

        # Smooth S bounds along the ρ axis
        for iy_i in range(nY):
            for iz_i in range(nZ):
                for arr in [s_hi, s_lo]:
                    col = arr[:, iy_i, iz_i]
                    if np.isfinite(col).sum() > 5:
                        col[:], _ = hampel_filter_1d(
                            col, window=10, n_sigma=3.0)
                        col[:] = gaussian_filter1d(col, sigma=3.0)

        # Pad bounds by 2%
        margin = 0.02 * (s_hi - s_lo)
        margin = np.where(np.isfinite(margin), margin, 0.0)
        s_lo -= margin
        s_hi += margin

        self._s_lo_srho = s_lo.astype(np.float32)
        self._s_hi_srho = s_hi.astype(np.float32)
        self._yvals_srho = yvals
        self._zvals_srho = zvals

        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._s_lo_srho_rgi = RGI((logrho, yvals, zvals),
                                   self._s_lo_srho, **rgi_kw)
        self._s_hi_srho_rgi = RGI((logrho, yvals, zvals),
                                   self._s_hi_srho, **rgi_kw)

    def s_to_xi_srho(self, _s_kb, _lgrho, _yp, _z):
        """Convert physical S → normalised ξ in the S-ρ rhomboid."""
        _lgrho_a = np.atleast_1d(_lgrho)
        _yp_a = np.atleast_1d(_yp)
        _z_a = np.atleast_1d(_z)
        _lgrho_a, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho_a, _yp_a, _z_a)
        pts = np.column_stack((_lgrho_a.ravel(), _yp_a.ravel(),
                                _z_a.ravel()))
        s_lo = self._s_lo_srho_rgi(pts).reshape(_lgrho_a.shape)
        s_hi = self._s_hi_srho_rgi(pts).reshape(_lgrho_a.shape)
        denom = s_hi - s_lo
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        xi = (_s_kb - s_lo) / denom
        if np.isscalar(_lgrho) and np.isscalar(_s_kb):
            return float(xi.ravel()[0])
        return xi

    def xi_to_s_srho(self, _xi, _lgrho, _yp, _z):
        """Convert normalised ξ → physical S in the S-ρ rhomboid."""
        _lgrho_a = np.atleast_1d(_lgrho)
        _yp_a = np.atleast_1d(_yp)
        _z_a = np.atleast_1d(_z)
        _lgrho_a, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho_a, _yp_a, _z_a)
        pts = np.column_stack((_lgrho_a.ravel(), _yp_a.ravel(),
                                _z_a.ravel()))
        s_lo = self._s_lo_srho_rgi(pts).reshape(_lgrho_a.shape)
        s_hi = self._s_hi_srho_rgi(pts).reshape(_lgrho_a.shape)
        return s_lo + _xi * (s_hi - s_lo)

    def get_logp_logt_srho(self, _s_kb, _lgrho, _yp, _z=0.0,
                            _zm=0.0, _za=0.0, _zr=0.0, **kw):
        """Pressure and temperature from (S, ρ) via 2-D least-squares.

        If a pre-computed S-ρ table has been loaded, the RGI is
        used instead.

        Parameters
        ----------
        _s_kb : float or array
            Entropy in kb/baryon.
        _lgrho : float or array
            log10 ρ [g/cm³].
        _yp : float
            Y' = Y/(1-Z).
        _z : float
            Total metal mass fraction.

        Returns
        -------
        logp, logt : float or array
            log10 P [dyn/cm²] and log10 T [K].
            NaN where the solver fails.
        """
        _yp = self._to_yprime(_yp, _z)
        # --- Fast path: pre-computed table ---
        if self._srho_rgi_p is not None:
            return self._lookup_srho_table(_s_kb, _lgrho, _yp, _z)

        # --- Slow path: per-point least_squares ---
        scalar = np.isscalar(_s_kb) and np.isscalar(_lgrho)
        _s_kb  = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _s_kb, _lgrho = np.broadcast_arrays(_s_kb, _lgrho)

        lgp_out = np.full_like(_s_kb, np.nan, dtype=float)
        lgt_out = np.full_like(_s_kb, np.nan, dtype=float)

        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]
        lgt_lo, lgt_hi = self.logt_min, self.logt_max
        lb = np.array([lgp_lo, lgt_lo])
        ub = np.array([lgp_hi, lgt_hi])

        # Warm-start: carry forward the last converged solution
        prev_x = None

        for idx in np.ndindex(_s_kb.shape):
            s_target = _s_kb[idx]
            rho_target = _lgrho[idx]

            s_scale = max(abs(s_target), 1.0)
            rho_scale = max(abs(rho_target), 1.0)

            def residuals(x):
                lgp, lgt = x
                s_test = (self._s_pt(
                    lgp, lgt, _yp, _z, _zm, _za, _zr)
                    * erg_to_kbbar)
                rho_test = self._logrho_pt(
                    lgp, lgt, _yp, _z, _zm, _za, _zr)
                if not (np.isfinite(s_test) and np.isfinite(rho_test)):
                    return np.array([1e30, 1e30])
                return np.array([(s_test - s_target) / s_scale,
                                 (rho_test - rho_target) / rho_scale])

            # Initial guess
            if prev_x is not None:
                x0 = prev_x.copy()
            else:
                # Seed: mid-T, then invert for P from rho
                lgt_seed = 0.5 * (lgt_lo + lgt_hi)
                try:
                    def err_p(lgp):
                        return (self._logrho_pt(
                            lgp, lgt_seed, _yp, _z, _zm, _za, _zr)
                            - rho_target)
                    lgp_seed = brentq(err_p, lgp_lo, lgp_hi,
                                      xtol=1e-5, maxiter=60)
                except (ValueError, RuntimeError, ZeroDivisionError):
                    lgp_seed = 0.5 * (lgp_lo + lgp_hi)
                x0 = np.array([lgp_seed, lgt_seed])

            x0 = np.clip(x0, lb, ub)

            try:
                sol = least_squares(residuals, x0,
                                     bounds=(lb, ub),
                                     method='trf',
                                     xtol=1e-10, ftol=1e-10,
                                     gtol=1e-10, max_nfev=200)
                if sol.success and np.all(np.isfinite(sol.x)):
                    lgp_out[idx] = sol.x[0]
                    lgt_out[idx] = sol.x[1]
                    prev_x = sol.x.copy()
            except Exception:
                pass

        if scalar:
            return lgp_out.item(), lgt_out.item()
        return lgp_out, lgt_out

    def _lookup_srho_table(self, _s_kb, _lgrho, _yp, _z):
        """Query the pre-computed (ξ, logrho, Y', Z) S-ρ RGI tables."""
        _s_kb  = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _s_kb, _lgrho, _yp_a, _z_a = np.broadcast_arrays(
            _s_kb, _lgrho, _yp_a, _z_a)

        xi = self.s_to_xi_srho(_s_kb, _lgrho, _yp_a, _z_a)

        lgp_out = np.full_like(xi, np.nan, dtype=float)
        lgt_out = np.full_like(xi, np.nan, dtype=float)
        good = (xi >= 0.0) & (xi <= 1.0) & np.isfinite(xi)

        if good.any():
            pts = np.column_stack((xi[good], _lgrho[good],
                                   _yp_a[good], _z_a[good]))
            lgp_out[good] = self._srho_rgi_p(pts)
            lgt_out[good] = self._srho_rgi_t(pts)

        if lgp_out.size == 1:
            return lgp_out.item(), lgt_out.item()
        return lgp_out, lgt_out

    def build_srho_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         n_xi=None, verbose=True):
        """Build logP and logT tables on a (ξ, logrho, Y', Z) grid.

        The ξ coordinate normalises the entropy axis at each
        (logrho, Y', Z) from 0 (S_lo) to 1 (S_hi), matching the
        S-P rhomboid approach.
        """
        if n_xi is None:
            n_xi = self.n_xi

        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        alpha_xi = 2.0
        t_uniform = np.linspace(0.0, 1.0, n_xi)
        xi_vals = np.sinh(alpha_xi * t_uniform) / np.sinh(alpha_xi)
        logrho = self.logrho_vals

        nR, nY, nZ = len(logrho), len(yvals), len(zvals)

        if verbose:
            print(f"Building S-ρ table: n_xi={n_xi}, "
                  f"logrho=[{logrho[0]:.2f}, {logrho[-1]:.2f}] "
                  f"(d={logrho[1]-logrho[0]:.2f}, {nR} pts), "
                  f"logT=[{self.logt_min:.1f}, {self.logt_max:.1f}]")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {n_xi}×{nR}×{nY}×{nZ} = "
                  f"{n_xi*nR*nY*nZ:,}")

        # Step 1: compute S bounds
        self.compute_s_bounds_srho(yvals, zvals, _zm, _za, _zr,
                                   verbose=verbose)
        s_lo = self._s_lo_srho  # (nR, nY, nZ)
        s_hi = self._s_hi_srho

        # Step 2: invert
        logp_tab = np.full((n_xi, nR, nY, nZ), np.nan, dtype=float)
        logt_tab = np.full((n_xi, nR, nY, nZ), np.nan, dtype=float)

        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]
        lgt_lo, lgt_hi = self.logt_min, self.logt_max
        lb = np.array([lgp_lo, lgt_lo])
        ub = np.array([lgp_hi, lgt_hi])

        total = nY * nZ
        pbar = tqdm(total=total,
                     desc="Inverting P,T → S,ρ",
                     disable=not verbose,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}]')

        # Warm-start caches:
        #   prev_z[ixi, ir]  — previous Z solution (reused across Z)
        #   prev_xi          — previous ξ solution (reused across ξ)
        #   prev_rho         — previous ρ solution (reused across ρ)
        prev_z = np.full((n_xi, nR, 2), np.nan)

        for iy, yp in enumerate(yvals):
            prev_z[:] = np.nan

            for iz, zv in enumerate(zvals):
                pbar.set_postfix_str(f"Y'={yp:.3f} Z={zv:.3f}")
                pbar.update(1)

                for ir in range(nR):
                    rho_target = logrho[ir]
                    slo_i = s_lo[ir, iy, iz]
                    shi_i = s_hi[ir, iy, iz]

                    if (not np.isfinite(slo_i)) or (not np.isfinite(shi_i)):
                        continue
                    if shi_i <= slo_i:
                        continue

                    rho_scale = max(abs(rho_target), 1.0)
                    prev_xi = np.array([np.nan, np.nan])

                    for ixi in range(n_xi):
                        s_phys = slo_i + xi_vals[ixi] * (shi_i - slo_i)
                        s_scale = max(abs(s_phys), 1.0)

                        def residuals(x, _s=s_phys, _ss=s_scale,
                                      _rho=rho_target, _rs=rho_scale):
                            lgp, lgt = x
                            try:
                                s_t = (self._s_pt(
                                    lgp, lgt, yp, zv, _zm, _za, _zr)
                                    * erg_to_kbbar)
                                rho_t = self._logrho_pt(
                                    lgp, lgt, yp, zv, _zm, _za, _zr)
                            except (ZeroDivisionError,
                                    FloatingPointError):
                                return np.array([1e30, 1e30])
                            if not (np.isfinite(s_t)
                                    and np.isfinite(rho_t)):
                                return np.array([1e30, 1e30])
                            return np.array([
                                (s_t - _s) / _ss,
                                (rho_t - _rho) / _rs])

                        # Guess priority:
                        #   1. Previous ξ (adjacent S, same ρ)
                        #   2. Previous Z (same ξ/ρ, adjacent Z)
                        #   3. Ideal EOS (cold start)
                        if np.all(np.isfinite(prev_xi)):
                            guess = prev_xi.copy()
                        elif np.all(np.isfinite(prev_z[ixi, ir])):
                            guess = prev_z[ixi, ir].copy()
                        else:
                            try:
                                pt_ideal = ideal_xy.get_pt_srho(
                                    s_phys, rho_target, yp)
                                guess = np.asarray(pt_ideal,
                                                   dtype=float)
                            except Exception:
                                guess = np.array([
                                    0.5 * (lgp_lo + lgp_hi),
                                    0.5 * (lgt_lo + lgt_hi)])

                        sol, ok = self._newton_2d(
                            residuals, guess, lb, ub)
                        if ok and np.all(np.isfinite(sol)):
                            logp_tab[ixi, ir, iy, iz] = sol[0]
                            logt_tab[ixi, ir, iy, iz] = sol[1]
                            prev_z[ixi, ir] = sol
                            prev_xi = sol

        pbar.close()

        # Step 3: fill NaNs first (so Hampel has no gaps)
        n_nan = np.isnan(logp_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logp_tab = self._fill_table_nans(logp_tab)
            logt_tab = self._fill_table_nans(logt_tab)

        # Step 4: Hampel outlier filter along ξ axis
        for arr, label in [(logp_tab, 'logP'), (logt_tab, 'logT')]:
            if verbose:
                print(f"Running Hampel outlier filter on {label} "
                      f"along ξ axis ...")
            flat = arr.reshape(n_xi, -1)
            n_out = 0
            for j in range(flat.shape[1]):
                col = flat[:, j]
                cleaned, n_rep = hampel_filter_1d(
                    col, window=5, n_sigma=3.0)
                if n_rep > 0:
                    changed = (cleaned != col) & np.isfinite(col)
                    flat[changed, j] = cleaned[changed]
                    n_out += n_rep
            if n_out > 0 and verbose:
                print(f"  Replaced {n_out} outlier cells in {label}")

        logp_f32 = logp_tab.astype(np.float32)
        logt_f32 = logt_tab.astype(np.float32)

        if verbose:
            mem_mb = (logp_f32.nbytes + logt_f32.nbytes) / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32, P+T)")

        result = {
            'xi_vals':    xi_vals,
            'logrhovals': logrho,
            'yvals':      yvals,
            'zvals':      zvals,
            'logp_srho':  logp_f32,
            'logt_srho':  logt_f32,
            's_lo_srho':  self._s_lo_srho,
            's_hi_srho':  self._s_hi_srho,
            'logt_min':   self.logt_min,
            'logt_max':   self.logt_max,
        }

        # Load into this instance
        self._load_srho_from_arrays(
            xi_vals, logrho, yvals, zvals,
            logp_f32, logt_f32,
            self._s_lo_srho, self._s_hi_srho)

        if verbose:
            n_total = logp_tab.size
            n_good = np.isfinite(logp_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def _load_srho_from_arrays(self, xi_vals, logrho, yvals, zvals,
                                logp, logt, s_lo, s_hi):
        """Build S-ρ RGI interpolators from arrays."""
        rgi_kw = dict(method='linear', bounds_error=False,
                      fill_value=None)
        self._srho_rgi_p = RGI((xi_vals, logrho, yvals, zvals),
                                logp, **rgi_kw)
        self._srho_rgi_t = RGI((xi_vals, logrho, yvals, zvals),
                                logt, **rgi_kw)
        self._s_lo_srho = s_lo
        self._s_hi_srho = s_hi
        self._yvals_srho = yvals
        self._zvals_srho = zvals
        self._s_lo_srho_rgi = RGI((logrho, yvals, zvals), s_lo, **rgi_kw)
        self._s_hi_srho_rgi = RGI((logrho, yvals, zvals), s_hi, **rgi_kw)

    def load_srho_table(self, path):
        """Load a pre-computed S-ρ table from NPZ."""
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        self._load_srho_from_arrays(
            data['xi_vals'], data['logrhovals'],
            data['yvals'], data['zvals'],
            data['logp_srho'], data['logt_srho'],
            data['s_lo_srho'], data['s_hi_srho'])

    def save_srho_table(self, result, path=None):
        """Save an S-ρ table to NPZ."""
        if path is None:
            path = self._table_path('srho')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # Thermodynamic derivatives (all from P-T basis)
    # =================================================================

    @staticmethod
    def _adaptive_dx(x, dx0=0.01):
        """Adjust finite-difference step to stay within [0, 1]."""
        x = np.asarray(x, dtype=float)
        dx = np.full_like(x, dx0)
        dx = np.minimum(dx, x)
        dx = np.minimum(dx, 1.0 - x)
        dx = np.maximum(dx, 1e-6)
        if dx.size == 1:
            return float(dx)
        return dx

    def _smooth_deriv(self, func, x0, h, n=5):
        """Savitzky-Golay derivative: fit degree-2 poly through n points.

        Returns df/dx at x0.  For n=5, stencil is
        [x0-2h, x0-h, x0, x0+h, x0+2h].
        """
        xs = np.linspace(x0 - (n // 2) * h, x0 + (n // 2) * h, n)
        ys = np.array([func(xi) for xi in xs])
        # Savitzky-Golay first-derivative coefficients for n=5, degree 2:
        # d/dx at center = (-2y_{-2} - y_{-1} + y_1 + 2y_2) / (10h)
        if n == 5:
            return (-2*ys[0] - ys[1] + ys[3] + 2*ys[4]) / (10 * h)
        # Generic fallback: polyfit
        coeffs = np.polyfit(xs - x0, ys, 2)
        return coeffs[1]  # linear coefficient = derivative at center

    def _pt_derivs(self, lgp, lgt, yp, z,
                   _zm=0.0, _za=0.0, _zr=0.0,
                   dlogt=1e-2, dlogp=1e-2,
                   dy=None, dz=None,
                   smooth=True, composition=False):
        """Compute all base P-T log-derivatives at a single point.

        Parameters
        ----------
        lgp, lgt : float
            log10 P [dyn/cm²], log10 T [K].
        yp : float
            Y' = Y/(1-Z).
        z : float
            Total metal mass fraction.
        dlogt, dlogp : float
            Step sizes in log space.
        dy, dz : float or None
            Step sizes for Y', Z.  None → adaptive.
        smooth : bool
            Use 5-point Savitzky-Golay stencil (True) or
            2-point central difference (False).
        composition : bool
            Also compute S_Y, S_Z, ρ_Y, ρ_Z, U_Y, U_Z, U_P, U_T.

        Returns
        -------
        d : dict with keys 'a', 'b', 'c', 'd', 'S', 'logrho',
            and optionally 'S_Y', 'S_Z', 'rho_Y', 'rho_Z',
            'U_Y', 'U_Z', 'U_P', 'U_T'.
        """
        yp = self._to_yprime(yp, z)  # convert Y→Y' if needed
        v = self.val
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)

        # Central value
        S0 = v.get_s_pt_val(lgp, lgt, yp, z, **kw)
        logS0 = np.log10(S0) if S0 > 0 else np.nan
        logrho0 = v.get_logrho_pt_val(lgp, lgt, yp, z, **kw)

        def _logS(lgp_i, lgt_i, yp_i, z_i):
            s = v.get_s_pt_val(lgp_i, lgt_i, yp_i, z_i, **kw)
            return np.log10(s) if s > 0 else np.nan

        def _logrho(lgp_i, lgt_i, yp_i, z_i):
            return v.get_logrho_pt_val(lgp_i, lgt_i, yp_i, z_i, **kw)

        if smooth:
            # a = dlogS/dlogT|_P
            a = self._smooth_deriv(
                lambda t: _logS(lgp, t, yp, z), lgt, dlogt)
            # b = dlogS/dlogP|_T
            b = self._smooth_deriv(
                lambda p: _logS(p, lgt, yp, z), lgp, dlogp)
            # c = dlogrho/dlogT|_P
            c = self._smooth_deriv(
                lambda t: _logrho(lgp, t, yp, z), lgt, dlogt)
            # d = dlogrho/dlogP|_T
            d = self._smooth_deriv(
                lambda p: _logrho(p, lgt, yp, z), lgp, dlogp)
        else:
            # 2-point central difference
            logS_Tp = _logS(lgp, lgt + dlogt, yp, z)
            logS_Tm = _logS(lgp, lgt - dlogt, yp, z)
            logS_Pp = _logS(lgp + dlogp, lgt, yp, z)
            logS_Pm = _logS(lgp - dlogp, lgt, yp, z)

            rho_Tp = _logrho(lgp, lgt + dlogt, yp, z)
            rho_Tm = _logrho(lgp, lgt - dlogt, yp, z)
            rho_Pp = _logrho(lgp + dlogp, lgt, yp, z)
            rho_Pm = _logrho(lgp - dlogp, lgt, yp, z)

            a = (logS_Tp - logS_Tm) / (2 * dlogt)
            b = (logS_Pp - logS_Pm) / (2 * dlogp)
            c = (rho_Tp - rho_Tm) / (2 * dlogt)
            d = (rho_Pp - rho_Pm) / (2 * dlogp)

        result = {'a': a, 'b': b, 'c': c, 'd': d,
                  'S': S0, 'logS': logS0, 'logrho': logrho0,
                  'lgp': lgp, 'lgt': lgt}

        if composition:
            if dy is None:
                dy = self._adaptive_dx(yp)
            if dz is None:
                dz = self._adaptive_dx(z)

            S_Yp = v.get_s_pt_val(lgp, lgt, yp + dy, z, **kw)
            S_Ym = v.get_s_pt_val(lgp, lgt, yp - dy, z, **kw)
            S_Zp = v.get_s_pt_val(lgp, lgt, yp, z + dz, **kw)
            S_Zm = v.get_s_pt_val(lgp, lgt, yp, z - dz, **kw)

            rho_Yp = v.get_logrho_pt_val(lgp, lgt, yp + dy, z, **kw)
            rho_Ym = v.get_logrho_pt_val(lgp, lgt, yp - dy, z, **kw)
            rho_Zp = v.get_logrho_pt_val(lgp, lgt, yp, z + dz, **kw)
            rho_Zm = v.get_logrho_pt_val(lgp, lgt, yp, z - dz, **kw)

            U_Yp = v.get_u_pt_val(lgp, lgt, yp + dy, z, **kw)
            U_Ym = v.get_u_pt_val(lgp, lgt, yp - dy, z, **kw)
            U_Zp = v.get_u_pt_val(lgp, lgt, yp, z + dz, **kw)
            U_Zm = v.get_u_pt_val(lgp, lgt, yp, z - dz, **kw)

            U_Pp = v.get_u_pt_val(lgp + dlogp, lgt, yp, z, **kw)
            U_Pm = v.get_u_pt_val(lgp - dlogp, lgt, yp, z, **kw)
            U_Tp = v.get_u_pt_val(lgp, lgt + dlogt, yp, z, **kw)
            U_Tm = v.get_u_pt_val(lgp, lgt - dlogt, yp, z, **kw)

            result['S_Y']   = (S_Yp - S_Ym) / (2 * dy)
            result['S_Z']   = (S_Zp - S_Zm) / (2 * dz)
            result['rho_Y'] = (rho_Yp - rho_Ym) / (2 * dy)
            result['rho_Z'] = (rho_Zp - rho_Zm) / (2 * dz)
            result['U_Y']   = (U_Yp - U_Ym) / (2 * dy)
            result['U_Z']   = (U_Zp - U_Zm) / (2 * dz)
            result['U_P']   = (U_Pp - U_Pm) / (2 * dlogp)
            result['U_T']   = (U_Tp - U_Tm) / (2 * dlogt)

        return result

    # ----- Vectorized derivative along T with post-smoothing -----

    def deriv_along_t(self, method_name, lgp, logt_arr, yp, z=0.0,
                      post_smooth_sigma=0, **kw):
        """Evaluate a derivative method over an array of logT values.

        Parameters
        ----------
        method_name : str
            Name of the getter (e.g. 'get_cp', 'get_nabla_ad',
            'get_dsdy_rhop').
        lgp : float
            log10 P [dyn/cm²] (fixed).
        logt_arr : 1-D array
            Array of log10 T values to evaluate at.
        yp, z : float
            Composition.
        post_smooth_sigma : float
            If > 0, apply a 1-D Gaussian filter with this sigma
            (in grid points) to the output.  This removes spikes
            from derivative-ratio singularities (e.g. near H₂
            dissociation where c→0).
        **kw :
            Passed to the derivative method.

        Returns
        -------
        result : 1-D array, same length as logt_arr.
        """
        func = getattr(self, method_name)
        out = np.array([func(lgp, lgt, yp, z, **kw) for lgt in logt_arr])

        if post_smooth_sigma > 0:
            good = np.isfinite(out)
            if good.sum() > 3:
                # Fill NaN gaps before smoothing, then restore
                filled = out.copy()
                filled[~good] = np.interp(
                    np.where(~good)[0],
                    np.where(good)[0], out[good])
                filled = gaussian_filter1d(filled, sigma=post_smooth_sigma)
                out[good] = filled[good]

        return out

    # ----- Individual getter methods -----
    # All accept scalar or array (lgp, lgt). When arrays are passed,
    # the derivative is computed element-wise.
    #
    # method='identity' (default): thermodynamic identity from P-T basis
    # method='finite_difference': direct FD on the appropriate inverted
    #   basis (uses pre-computed tables if loaded, otherwise on-the-fly)

    def _vec(self, func, lgp, lgt, *args, **kw):
        """Vectorize a scalar function over (lgp, lgt).

        Parameters
        ----------
        post_smooth : bool (keyword, extracted from **kw)
            If True, apply a 1-D Gaussian filter to the output
            array to remove spikes from derivative-ratio
            singularities (e.g. near H₂ dissociation).
        post_smooth_sigma : float (keyword, extracted from **kw)
            Sigma for the Gaussian filter (in grid points).
        """
        # Extract smoothing params so they don't reach scalar funcs
        post_smooth = kw.pop('post_smooth', False)
        post_smooth_sigma = kw.pop('post_smooth_sigma', 3)

        if np.isscalar(lgp) and np.isscalar(lgt):
            return func(lgp, lgt, *args, **kw)
        lgp_a = np.atleast_1d(lgp)
        lgt_a = np.atleast_1d(lgt)
        # Broadcast args (yp, z, ...) to same shape as lgp, lgt
        args_a = [np.broadcast_to(np.atleast_1d(a), lgp_a.shape)
                   if not np.isscalar(a) else a for a in args]
        lgp_a, lgt_a = np.broadcast_arrays(lgp_a, lgt_a)
        n = lgp_a.size
        out = np.empty(n)
        for i in range(n):
            a_i = [float(a.ravel()[i]) if not np.isscalar(a) else a
                   for a in args_a]
            out[i] = func(float(lgp_a.ravel()[i]),
                          float(lgt_a.ravel()[i]), *a_i, **kw)
        out = out.reshape(lgp_a.shape)

        if post_smooth and out.ndim >= 1 and out.size > 3:
            good = np.isfinite(out.ravel())
            if good.sum() > 3:
                flat = out.ravel().copy()
                bad = ~np.isfinite(flat)
                if bad.any():
                    flat[bad] = np.interp(
                        np.where(bad)[0],
                        np.where(~bad)[0], flat[~bad])
                flat = gaussian_filter1d(flat, sigma=post_smooth_sigma)
                out_flat = out.ravel()
                out_flat[good] = flat[good]
                out = out_flat.reshape(lgp_a.shape)

        return out

    # ---- FD scalar helpers (use inversions) ----
    # Each supports smooth=True for 5-point Savitzky-Golay stencil
    # (same as the identity method's internal smoothing).
    # When smooth=False, uses the standard 2-point central difference.

    def _fd_or_sg(self, func, x0, h, smooth=True):
        """Compute df/dx: Savitzky-Golay if smooth, else 2-pt central diff."""
        if smooth:
            return self._smooth_deriv(func, x0, h)
        return (func(x0 + h) - func(x0 - h)) / (2 * h)

    def _cp_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
               dt=1e-2, smooth=True):
        """C_P via direct FD: dS/d(lnT)|_P."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        f = lambda t: self._s_pt(lgp, t, yp, z, **kw)
        return self._fd_or_sg(f, lgt, dt, smooth) / log10_to_loge

    def _cv_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
               dt=1e-2, smooth=True):
        """C_V via direct FD: dS/d(lnT)|_ρ."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        def f(t):
            try:
                p = self.get_logp_rhot(rho0, t, yp, z, **kw)
                return self._s_pt(p, t, yp, z, **kw)
            except:
                return np.nan
        return self._fd_or_sg(f, lgt, dt, smooth) / log10_to_loge

    def _delta_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                  dt=1e-2, smooth=True):
        """δ via direct FD: -dlogρ/dlogT|_P."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        f = lambda t: self._logrho_pt(lgp, t, yp, z, **kw)
        return -self._fd_or_sg(f, lgt, dt, smooth)

    def _nabla_ad_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                     dp=1e-2, smooth=True):
        """∇_ad via direct FD on S-P inversion: dlogT/dlogP|_S."""
        s_kb = self._s_pt(lgp, lgt, yp, z, _zm, _za, _zr) * erg_to_kbbar
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        f = lambda p: self.get_logt_sp(s_kb, p, yp, z, **kw)
        result = self._fd_or_sg(f, lgp, dp, smooth)
        return result if np.isfinite(result) else np.nan

    def _gamma1_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                   dp=1e-2, smooth=True):
        """Γ₁ via direct FD: dlogP/dlogρ|_S."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        def f(p):
            t = self.get_logt_sp(s_kb, p, yp, z, **kw)
            if np.isfinite(t):
                return self._logrho_pt(p, t, yp, z, **kw)
            return np.nan
        drho_dp = self._fd_or_sg(f, lgp, dp, smooth)
        if np.isfinite(drho_dp) and abs(drho_dp) > 1e-30:
            return 1.0 / drho_dp
        return np.nan

    def _chi_T_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                  dt=1e-2, smooth=True):
        """χ_T via direct FD on ρ-T inversion: dlogP/dlogT|_ρ."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        f = lambda t: self.get_logp_rhot(rho0, t, yp, z, **kw)
        result = self._fd_or_sg(f, lgt, dt, smooth)
        return result if np.isfinite(result) else np.nan

    def _chi_rho_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                    dr=1e-2, smooth=True):
        """χ_ρ via direct FD on ρ-T inversion: dlogP/dlogρ|_T."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        f = lambda r: self.get_logp_rhot(r, lgt, yp, z, **kw)
        result = self._fd_or_sg(f, rho0, dr, smooth)
        return result if np.isfinite(result) else np.nan

    def _chi_Y_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                  dy=0.01, smooth=True):
        """χ_Y via direct FD on ρ-T inversion: dlogP/dY|_{ρ,T}."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dy = self._adaptive_dx(yp, dy)
        f = lambda y: self.get_logp_rhot(rho0, lgt, y, z, **kw)
        return self._fd_or_sg(f, yp, dy, smooth) * log10_to_loge

    def _chi_Z_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                  dz=0.01, smooth=True):
        """χ_Z via direct FD on ρ-T inversion: dlogP/dZ|_{ρ,T}."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dz = self._adaptive_dx(z, dz)
        f = lambda zv: self.get_logp_rhot(rho0, lgt, yp, zv, **kw)
        return self._fd_or_sg(f, z, dz, smooth) * log10_to_loge

    def _dtds_sp_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                    ds=0.1, smooth=True):
        """dT/dS|_P via direct FD on S-P inversion."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        f = lambda s: 10 ** self.get_logt_sp(s, lgp, yp, z, **kw)
        dTds_kb = self._fd_or_sg(f, s_kb, ds, smooth)
        return dTds_kb * erg_to_kbbar  # convert from dT/d(S_kb) to dT/d(S_cgs)

    def _dsdy_pt_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                    dy=0.01, smooth=True):
        """dS/dY|_{P,T} via direct FD."""
        dy = self._adaptive_dx(yp, dy)
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        f = lambda y: self._s_pt(lgp, lgt, y, z, **kw)
        return self._fd_or_sg(f, yp, dy, smooth)

    def _dsdz_pt_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0,
                    dz=0.01, smooth=True):
        """dS/dZ|_{P,T} via direct FD."""
        dz = self._adaptive_dx(z, dz)
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        f = lambda zv: self._s_pt(lgp, lgt, yp, zv, **kw)
        return self._fd_or_sg(f, z, dz, smooth)

    # ---- Identity scalar helpers ----

    def _cp_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return d['S'] * d['a']

    def _cv_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return d['S'] * (d['a'] - d['b'] * d['c'] / d['d'])

    def _delta_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return -d['c']

    def _nabla_ad_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return -d['b'] / d['a']

    def _chi_T_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return -d['c'] / d['d']

    def _chi_rho_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return 1.0 / d['d']

    def _gamma1_id(self, lgp, lgt, yp, z, **kw):
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return 1.0 / (d['d'] - d['c'] * d['b'] / d['a'])

    def _chi_Y_id(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return -d['rho_Y'] / d['d']

    def _chi_Z_id(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return -d['rho_Z'] / d['d']

    def _dtds_sp_id(self, lgp, lgt, yp, z, **kw):
        # dT/dS|_P = 1/(dS/dT|_P) = T/C_P
        # since C_P = dS/d(lnT)|_P = S·a, and dS/dT = C_P/T
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        cp = d['S'] * d['a']
        return 10.0**lgt / cp

    def _dsdy_pt_id(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return d['S_Y']

    def _dsdz_pt_id(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        return d['S_Z']

    # ---- Public getters with method dispatch ----

    # ORCHARD-specific kwargs that should not be passed to scalar helpers
    _ORCHARD_KW = frozenset({'tab', 'ideal_guess', 'arr_guess',
                              'arr_p_guess', 'arr_t_guess',
                              '_frock', 'frock',
                              'dy', 'dz', 'ds', 'dt'})

    # Vectorized FD implementations (array-safe, no Python loop).
    # These call the forward model or table lookups with full arrays
    # in a single RGI evaluation — orders of magnitude faster than
    # the _vec per-element loop.

    def _vec_fd_cp(self, lgp, lgt, yp, z, h=1e-2, **kw):
        return (self.get_s_pt_tab(lgp, lgt + h, yp, z)
                - self.get_s_pt_tab(lgp, lgt - h, yp, z)) / (2*h*log10_to_loge)

    def _vec_fd_delta(self, lgp, lgt, yp, z, h=1e-2, **kw):
        return -(self.get_logrho_pt_tab(lgp, lgt + h, yp, z)
                 - self.get_logrho_pt_tab(lgp, lgt - h, yp, z)) / (2*h)

    def _vec_fd_nabla_ad(self, lgp, lgt, yp, z, h=1e-2, **kw):
        s_kb = self.get_s_pt_tab(lgp, lgt, yp, z) * erg_to_kbbar
        t1 = self.get_logt_sp(s_kb, lgp - h, yp, z)
        t2 = self.get_logt_sp(s_kb, lgp + h, yp, z)
        return (t2 - t1) / (2*h)

    def _vec_fd_gamma1(self, lgp, lgt, yp, z, h=1e-2, **kw):
        s_kb = self.get_s_pt_tab(lgp, lgt, yp, z) * erg_to_kbbar
        t1 = self.get_logt_sp(s_kb, lgp - h, yp, z)
        t2 = self.get_logt_sp(s_kb, lgp + h, yp, z)
        r1 = self._logrho_pt(lgp - h, t1, yp, z)
        r2 = self._logrho_pt(lgp + h, t2, yp, z)
        result = (2*h) / (r2 - r1)
        return np.where(np.isfinite(result), result, np.nan)

    def _vec_fd_chi_T(self, lgp, lgt, yp, z, h=1e-2, **kw):
        rho0 = self.get_logrho_pt_tab(lgp, lgt, yp, z)
        p1 = self.get_logp_rhot(rho0, lgt - h, yp, z)
        p2 = self.get_logp_rhot(rho0, lgt + h, yp, z)
        return (p2 - p1) / (2*h)

    def _vec_fd_chi_rho(self, lgp, lgt, yp, z, h=1e-2, **kw):
        rho0 = self.get_logrho_pt_tab(lgp, lgt, yp, z)
        p1 = self.get_logp_rhot(rho0 - h, lgt, yp, z)
        p2 = self.get_logp_rhot(rho0 + h, lgt, yp, z)
        return (p2 - p1) / (2*h)

    def _vec_fd_cv(self, lgp, lgt, yp, z, h=1e-2, **kw):
        rho0 = self.get_logrho_pt_tab(lgp, lgt, yp, z)
        p1 = self.get_logp_rhot(rho0, lgt - h, yp, z)
        p2 = self.get_logp_rhot(rho0, lgt + h, yp, z)
        s1 = self._s_pt(p1, lgt - h, yp, z)
        s2 = self._s_pt(p2, lgt + h, yp, z)
        return (s2 - s1) / (2*h*log10_to_loge)

    def _vec_fd_dsdy_pt(self, lgp, lgt, yp, z, h=0.01, **kw):
        return (self.get_s_pt_tab(lgp, lgt, yp + h, z)
                - self.get_s_pt_tab(lgp, lgt, yp - h, z)) / (2*h)

    def _vec_fd_dsdz_pt(self, lgp, lgt, yp, z, h=0.01, **kw):
        return (self.get_s_pt_tab(lgp, lgt, yp, z + h)
                - self.get_s_pt_tab(lgp, lgt, yp, z - h)) / (2*h)

    def _vec_fd_dtds_sp(self, lgp, lgt, yp, z, h=0.1, **kw):
        s_kb = self.get_s_pt_tab(lgp, lgt, yp, z) * erg_to_kbbar
        t1 = 10 ** self.get_logt_sp(s_kb - h, lgp, yp, z)
        t2 = 10 ** self.get_logt_sp(s_kb + h, lgp, yp, z)
        return (t2 - t1) / (2*h / erg_to_kbbar)

    def _vec_fd_chi_Y(self, lgp, lgt, yp, z, h=0.01, **kw):
        rho0 = self.get_logrho_pt_tab(lgp, lgt, yp, z)
        p1 = self.get_logp_rhot(rho0, lgt, yp - h, z)
        p2 = self.get_logp_rhot(rho0, lgt, yp + h, z)
        return (p2 - p1) * log10_to_loge / (2*h)

    def _vec_fd_chi_Z(self, lgp, lgt, yp, z, h=0.01, **kw):
        rho0 = self.get_logrho_pt_tab(lgp, lgt, yp, z)
        p1 = self.get_logp_rhot(rho0, lgt, yp, z - h)
        p2 = self.get_logp_rhot(rho0, lgt, yp, z + h)
        return (p2 - p1) * log10_to_loge / (2*h)

    def _vec_fd_dsdy_rhop(self, lgp, lgt, yp, z, h=0.01, **kw):
        """Ledoux dS/dY|_{ρ,P} vectorized via identity with clamped c."""
        ht = 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        logS_Tp = np.log10(self.get_s_pt_tab(lgp, lgt + ht, yp, z))
        logS_Tm = np.log10(self.get_s_pt_tab(lgp, lgt - ht, yp, z))
        a = (logS_Tp - logS_Tm) / (2 * ht)
        rho_Tp = self.get_logrho_pt_tab(lgp, lgt + ht, yp, z)
        rho_Tm = self.get_logrho_pt_tab(lgp, lgt - ht, yp, z)
        c = (rho_Tp - rho_Tm) / (2 * ht)
        S_Y = (self.get_s_pt_tab(lgp, lgt, yp + h, z)
               - self.get_s_pt_tab(lgp, lgt, yp - h, z)) / (2 * h)
        rho_Y = (self.get_logrho_pt_tab(lgp, lgt, yp + h, z)
                 - self.get_logrho_pt_tab(lgp, lgt, yp - h, z)) / (2 * h)
        # Clamp c to avoid 1/c singularity at dissociation
        c_safe = np.where(np.abs(c) > 0.02, c, np.sign(c) * 0.02)
        c_safe = np.where(c_safe == 0, 0.02, c_safe)
        return S_Y - S0 * a * log10_to_loge * rho_Y / c_safe

    def _vec_fd_dsdz_rhop(self, lgp, lgt, yp, z, h=0.01, **kw):
        """Ledoux dS/dZ|_{ρ,P} vectorized via identity with clamped c."""
        ht = 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        logS_Tp = np.log10(self.get_s_pt_tab(lgp, lgt + ht, yp, z))
        logS_Tm = np.log10(self.get_s_pt_tab(lgp, lgt - ht, yp, z))
        a = (logS_Tp - logS_Tm) / (2 * ht)
        rho_Tp = self.get_logrho_pt_tab(lgp, lgt + ht, yp, z)
        rho_Tm = self.get_logrho_pt_tab(lgp, lgt - ht, yp, z)
        c = (rho_Tp - rho_Tm) / (2 * ht)
        S_Z = (self.get_s_pt_tab(lgp, lgt, yp, z + h)
               - self.get_s_pt_tab(lgp, lgt, yp, z - h)) / (2 * h)
        rho_Z = (self.get_logrho_pt_tab(lgp, lgt, yp, z + h)
                 - self.get_logrho_pt_tab(lgp, lgt, yp, z - h)) / (2 * h)
        c_safe = np.where(np.abs(c) > 0.02, c, np.sign(c) * 0.02)
        c_safe = np.where(c_safe == 0, 0.02, c_safe)
        return S_Z - S0 * a * log10_to_loge * rho_Z / c_safe

    def _vec_fd_dtdy_srho(self, lgp, lgt, yp, z, h=0.01, **kw):
        """dT/dY|_{S,ρ} vectorized via identity."""
        ht, hp = 1e-2, 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        a = (np.log10(self._s_pt(lgp, lgt+ht, yp, z))
             - np.log10(self._s_pt(lgp, lgt-ht, yp, z))) / (2*ht)
        b = (np.log10(self._s_pt(lgp+hp, lgt, yp, z))
             - np.log10(self._s_pt(lgp-hp, lgt, yp, z))) / (2*hp)
        c = (self._logrho_pt(lgp, lgt+ht, yp, z)
             - self._logrho_pt(lgp, lgt-ht, yp, z)) / (2*ht)
        d = (self._logrho_pt(lgp+hp, lgt, yp, z)
             - self._logrho_pt(lgp-hp, lgt, yp, z)) / (2*hp)
        S_Y = (self._s_pt(lgp, lgt, yp+h, z)
               - self._s_pt(lgp, lgt, yp-h, z)) / (2*h)
        rho_Y = (self._logrho_pt(lgp, lgt, yp+h, z)
                 - self._logrho_pt(lgp, lgt, yp-h, z)) / (2*h)
        T = 10.0 ** lgt
        det = b*c - a*d
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        return T * (d*S_Y - S0*b*log10_to_loge*rho_Y) / (S0 * det)

    def _vec_fd_dtdz_srho(self, lgp, lgt, yp, z, h=0.01, **kw):
        """dT/dZ|_{S,ρ} vectorized via identity."""
        ht, hp = 1e-2, 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        a = (np.log10(self._s_pt(lgp, lgt+ht, yp, z))
             - np.log10(self._s_pt(lgp, lgt-ht, yp, z))) / (2*ht)
        b = (np.log10(self._s_pt(lgp+hp, lgt, yp, z))
             - np.log10(self._s_pt(lgp-hp, lgt, yp, z))) / (2*hp)
        c = (self._logrho_pt(lgp, lgt+ht, yp, z)
             - self._logrho_pt(lgp, lgt-ht, yp, z)) / (2*ht)
        d = (self._logrho_pt(lgp+hp, lgt, yp, z)
             - self._logrho_pt(lgp-hp, lgt, yp, z)) / (2*hp)
        S_Z = (self._s_pt(lgp, lgt, yp, z+h)
               - self._s_pt(lgp, lgt, yp, z-h)) / (2*h)
        rho_Z = (self._logrho_pt(lgp, lgt, yp, z+h)
                 - self._logrho_pt(lgp, lgt, yp, z-h)) / (2*h)
        T = 10.0 ** lgt
        det = b*c - a*d
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        return T * (d*S_Z - S0*b*log10_to_loge*rho_Z) / (S0 * det)

    def _vec_fd_dudy_srho(self, lgp, lgt, yp, z, h=0.01, **kw):
        """dU/dY|_{S,ρ} vectorized via identity."""
        ht, hp = 1e-2, 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        a = (np.log10(self._s_pt(lgp, lgt+ht, yp, z))
             - np.log10(self._s_pt(lgp, lgt-ht, yp, z))) / (2*ht)
        b = (np.log10(self._s_pt(lgp+hp, lgt, yp, z))
             - np.log10(self._s_pt(lgp-hp, lgt, yp, z))) / (2*hp)
        c = (self._logrho_pt(lgp, lgt+ht, yp, z)
             - self._logrho_pt(lgp, lgt-ht, yp, z)) / (2*ht)
        d = (self._logrho_pt(lgp+hp, lgt, yp, z)
             - self._logrho_pt(lgp-hp, lgt, yp, z)) / (2*hp)
        S_Y = (self._s_pt(lgp, lgt, yp+h, z)
               - self._s_pt(lgp, lgt, yp-h, z)) / (2*h)
        rho_Y = (self._logrho_pt(lgp, lgt, yp+h, z)
                 - self._logrho_pt(lgp, lgt, yp-h, z)) / (2*h)
        U_Y = (self.val.get_u_pt_val(lgp, lgt, yp+h, z)
               - self.val.get_u_pt_val(lgp, lgt, yp-h, z)) / (2*h)
        U_P = (self.val.get_u_pt_val(lgp+hp, lgt, yp, z)
               - self.val.get_u_pt_val(lgp-hp, lgt, yp, z)) / (2*hp)
        U_T = (self.val.get_u_pt_val(lgp, lgt+ht, yp, z)
               - self.val.get_u_pt_val(lgp, lgt-ht, yp, z)) / (2*ht)
        T = 10.0 ** lgt; P = 10.0 ** lgp; ln10 = log10_to_loge
        det = b*c - a*d
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        dTdY = T * (d*S_Y - S0*b*ln10*rho_Y) / (S0 * det)
        dPdY = P * (S0*a*ln10*rho_Y - c*S_Y) / (S0 * det)
        return U_P * dPdY/(P*ln10) + U_T * dTdY/(T*ln10) + U_Y

    def _vec_fd_dudz_srho(self, lgp, lgt, yp, z, h=0.01, **kw):
        """dU/dZ|_{S,ρ} vectorized via identity."""
        ht, hp = 1e-2, 1e-2
        S0 = self.get_s_pt_tab(lgp, lgt, yp, z)
        a = (np.log10(self._s_pt(lgp, lgt+ht, yp, z))
             - np.log10(self._s_pt(lgp, lgt-ht, yp, z))) / (2*ht)
        b = (np.log10(self._s_pt(lgp+hp, lgt, yp, z))
             - np.log10(self._s_pt(lgp-hp, lgt, yp, z))) / (2*hp)
        c = (self._logrho_pt(lgp, lgt+ht, yp, z)
             - self._logrho_pt(lgp, lgt-ht, yp, z)) / (2*ht)
        d = (self._logrho_pt(lgp+hp, lgt, yp, z)
             - self._logrho_pt(lgp-hp, lgt, yp, z)) / (2*hp)
        S_Z = (self._s_pt(lgp, lgt, yp, z+h)
               - self._s_pt(lgp, lgt, yp, z-h)) / (2*h)
        rho_Z = (self._logrho_pt(lgp, lgt, yp, z+h)
                 - self._logrho_pt(lgp, lgt, yp, z-h)) / (2*h)
        U_Z = (self.val.get_u_pt_val(lgp, lgt, yp, z+h)
               - self.val.get_u_pt_val(lgp, lgt, yp, z-h)) / (2*h)
        U_P = (self.val.get_u_pt_val(lgp+hp, lgt, yp, z)
               - self.val.get_u_pt_val(lgp-hp, lgt, yp, z)) / (2*hp)
        U_T = (self.val.get_u_pt_val(lgp, lgt+ht, yp, z)
               - self.val.get_u_pt_val(lgp, lgt-ht, yp, z)) / (2*ht)
        T = 10.0 ** lgt; P = 10.0 ** lgp; ln10 = log10_to_loge
        det = b*c - a*d
        det = np.where(np.abs(det) < 1e-20, 1e-20, det)
        dTdZ = T * (d*S_Z - S0*b*ln10*rho_Z) / (S0 * det)
        dPdZ = P * (S0*a*ln10*rho_Z - c*S_Z) / (S0 * det)
        return U_P * dPdZ/(P*ln10) + U_T * dTdZ/(T*ln10) + U_Z

    _VEC_FD_MAP = {
        'cp': '_vec_fd_cp', 'delta': '_vec_fd_delta',
        'nabla_ad': '_vec_fd_nabla_ad', 'gamma1': '_vec_fd_gamma1',
        'chi_T': '_vec_fd_chi_T', 'chi_rho': '_vec_fd_chi_rho',
        'chi_Y': '_vec_fd_chi_Y', 'chi_Z': '_vec_fd_chi_Z',
        'cv': '_vec_fd_cv',
        'dsdy_pt': '_vec_fd_dsdy_pt', 'dsdz_pt': '_vec_fd_dsdz_pt',
        'dsdy_rhop': '_vec_fd_dsdy_rhop', 'dsdz_rhop': '_vec_fd_dsdz_rhop',
        'dtdy_srho': '_vec_fd_dtdy_srho', 'dtdz_srho': '_vec_fd_dtdz_srho',
        'dudy_srho': '_vec_fd_dudy_srho', 'dudz_srho': '_vec_fd_dudz_srho',
        'dtds_sp': '_vec_fd_dtds_sp',
    }

    def _dispatch(self, name, lgp, lgt, yp, z, method, **kw):
        """Route to identity or FD scalar, then vectorize."""
        yp = self._to_yprime(yp, z)
        # Strip ORCHARD-specific kwargs that scalar helpers don't accept
        clean_kw = {k: v for k, v in kw.items()
                    if k not in self._ORCHARD_KW}

        # Fast vectorized path for common FD derivatives (no Python loop)
        if method == 'finite_difference' and name in self._VEC_FD_MAP:
            vec_func = getattr(self, self._VEC_FD_MAP[name])
            return vec_func(lgp, lgt, yp, z, **clean_kw)

        if method == 'finite_difference':
            func = getattr(self, f'_{name}_fd')
        else:
            func = getattr(self, f'_{name}_id')
        return self._vec(func, lgp, lgt, yp, z, **clean_kw)

    def get_cp(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """C_P  [erg/(g·K)]."""
        return self._dispatch('cp', lgp, lgt, yp, z, method, **kw)

    def get_cv(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """C_V  [erg/(g·K)]."""
        return self._dispatch('cv', lgp, lgt, yp, z, method, **kw)

    def get_delta(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """δ = −(∂logρ/∂logT)|_P  [dimensionless]."""
        return self._dispatch('delta', lgp, lgt, yp, z, method, **kw)

    def get_nabla_ad(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """∇_ad = dlogT/dlogP|_S  [dimensionless]."""
        return self._dispatch('nabla_ad', lgp, lgt, yp, z, method, **kw)

    def get_chi_T(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """χ_T = dlogP/dlogT|_ρ  [dimensionless]."""
        return self._dispatch('chi_T', lgp, lgt, yp, z, method, **kw)

    def get_chi_rho(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """χ_ρ = dlogP/dlogρ|_T  [dimensionless]."""
        return self._dispatch('chi_rho', lgp, lgt, yp, z, method, **kw)

    def get_gamma1(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """Γ₁ = dlogP/dlogρ|_S  [dimensionless]."""
        return self._dispatch('gamma1', lgp, lgt, yp, z, method, **kw)

    def get_chi_Y(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """χ_Y = dlogP/dY|_{ρ,T}."""
        return self._dispatch('chi_Y', lgp, lgt, yp, z, method, **kw)

    def get_chi_Z(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """χ_Z = dlogP/dZ|_{ρ,T}."""
        return self._dispatch('chi_Z', lgp, lgt, yp, z, method, **kw)

    def get_dtds_sp(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """dT/dS|_P  [K·g·K/erg]."""
        return self._dispatch('dtds_sp', lgp, lgt, yp, z, method, **kw)

    def get_dsdy_pt(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """dS/dY|_{P,T}  [erg/(g·K) per Y]."""
        return self._dispatch('dsdy_pt', lgp, lgt, yp, z, method, **kw)

    def get_dsdz_pt(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """dS/dZ|_{P,T}  [erg/(g·K) per Z]."""
        return self._dispatch('dsdz_pt', lgp, lgt, yp, z, method, **kw)

    def _ledoux_dsdy_fd(self, lgp, lgt, yp, z, _zm, _za, _zr, dy):
        """Direct finite-difference fallback for dS/dY|_{ρ,P}.

        Finds T(Y±dY) at constant (P, ρ) via brentq, then
        differentiates S.
        """
        rho0 = self._logrho_pt(lgp, lgt, yp, z, _zm, _za, _zr)
        if dy is None:
            dy = self._adaptive_dx(yp)

        def get_t(yp_i):
            def err(lgt_i):
                try:
                    return self._logrho_pt(
                        lgp, lgt_i, yp_i, z, _zm, _za, _zr) - rho0
                except (ZeroDivisionError, FloatingPointError):
                    return 1e30
            return brentq(err, self.logt_min, self.logt_max, xtol=1e-8)

        try:
            lgt_p = get_t(yp + dy)
            lgt_m = get_t(yp - dy)
            return (self._s_pt(lgp, lgt_p, yp + dy, z, _zm, _za, _zr) -
                    self._s_pt(lgp, lgt_m, yp - dy, z, _zm, _za, _zr)) / (2 * dy)
        except (ValueError, RuntimeError, ZeroDivisionError):
            return np.nan

    def _ledoux_dsdz_fd(self, lgp, lgt, yp, z, _zm, _za, _zr, dz):
        """Direct finite-difference fallback for dS/dZ|_{ρ,P}."""
        rho0 = self._logrho_pt(lgp, lgt, yp, z, _zm, _za, _zr)
        if dz is None:
            dz = self._adaptive_dx(z)

        def get_t(z_i):
            def err(lgt_i):
                try:
                    return self._logrho_pt(
                        lgp, lgt_i, yp, z_i, _zm, _za, _zr) - rho0
                except (ZeroDivisionError, FloatingPointError):
                    return 1e30
            return brentq(err, self.logt_min, self.logt_max, xtol=1e-8)

        try:
            lgt_p = get_t(z + dz)
            lgt_m = get_t(z - dz)
            return (self._s_pt(lgp, lgt_p, yp, z + dz, _zm, _za, _zr) -
                    self._s_pt(lgp, lgt_m, yp, z - dz, _zm, _za, _zr)) / (2 * dz)
        except (ValueError, RuntimeError, ZeroDivisionError):
            return np.nan

    def _dsdy_rhop_scalar(self, lgp, lgt, yp, z=0.0,
                          _zm=0.0, _za=0.0, _zr=0.0, c_guard=0.02, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, _zm=_zm, _za=_za, _zr=_zr, **kw)
        if abs(d['c']) > c_guard:
            return d['S_Y'] - d['S'] * d['a'] * log10_to_loge * d['rho_Y'] / d['c']
        return self._ledoux_dsdy_fd(lgp, lgt, yp, z, _zm, _za, _zr,
                                     dy=kw.get('dy', None))

    def get_dsdy_rhop(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """Ledoux: dS/dY|_{ρ,P}  [erg/(g·K) per Y].

        method='identity': S_Y − S·a·ln(10)·ρ_Y / c (with c_guard fallback)
        method='finite_difference': direct constrained FD at constant (ρ,P)
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(lambda p, t, yp, z, **k:
                self._ledoux_dsdy_fd(p, t, yp, z,
                    k.get('_zm', 0.), k.get('_za', 0.), k.get('_zr', 0.),
                    k.get('dy', None)),
                lgp, lgt, yp, z, **kw)
        return self._vec(self._dsdy_rhop_scalar, lgp, lgt, yp, z, **kw)

    def _dsdz_rhop_scalar(self, lgp, lgt, yp, z=0.0,
                          _zm=0.0, _za=0.0, _zr=0.0, c_guard=0.02, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, _zm=_zm, _za=_za, _zr=_zr, **kw)
        if abs(d['c']) > c_guard:
            return d['S_Z'] - d['S'] * d['a'] * log10_to_loge * d['rho_Z'] / d['c']
        return self._ledoux_dsdz_fd(lgp, lgt, yp, z, _zm, _za, _zr,
                                     dz=kw.get('dz', None))

    def get_dsdz_rhop(self, lgp, lgt, yp, z=0.0, _frock=0.0, method='finite_difference', **kw):
        """Ledoux: dS/dZ|_{ρ,P}  [erg/(g·K) per Z].

        method='identity': S_Z − S·a·ln(10)·ρ_Z / c (with c_guard fallback)
        method='finite_difference': direct constrained FD at constant (ρ,P)
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(lambda p, t, yp, z, **k:
                self._ledoux_dsdz_fd(p, t, yp, z,
                    k.get('_zm', 0.), k.get('_za', 0.), k.get('_zr', 0.),
                    k.get('dz', None)),
                lgp, lgt, yp, z, **kw)
        return self._vec(self._dsdz_rhop_scalar, lgp, lgt, yp, z, **kw)

    def _dtdy_srho_scalar(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        T = 10.0 ** lgt
        det = d['b'] * d['c'] - d['a'] * d['d']
        return T * (d['d'] * d['S_Y']
                    - d['S'] * d['b'] * log10_to_loge * d['rho_Y']) \
               / (d['S'] * det)

    def _dtdy_srho_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0, dy=0.01):
        """dT/dY|_{S,ρ} via 2D S-ρ inversion."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dy = self._adaptive_dx(yp, dy)
        _, t1 = self.get_logp_logt_srho(s_kb, rho0, yp - dy, z, **kw)
        _, t2 = self.get_logp_logt_srho(s_kb, rho0, yp + dy, z, **kw)
        if np.isfinite(t1) and np.isfinite(t2):
            return (10**t2 - 10**t1) / (2 * dy)
        return np.nan

    def _dtdz_srho_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0, dz=0.01):
        """dT/dZ|_{S,ρ} via 2D S-ρ inversion."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dz = self._adaptive_dx(z, dz)
        _, t1 = self.get_logp_logt_srho(s_kb, rho0, yp, z - dz, **kw)
        _, t2 = self.get_logp_logt_srho(s_kb, rho0, yp, z + dz, **kw)
        if np.isfinite(t1) and np.isfinite(t2):
            return (10**t2 - 10**t1) / (2 * dz)
        return np.nan

    def _dudy_srho_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0, dy=0.01):
        """dU/dY|_{S,ρ} via 2D S-ρ inversion."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dy = self._adaptive_dx(yp, dy)
        p1, t1 = self.get_logp_logt_srho(s_kb, rho0, yp - dy, z, **kw)
        p2, t2 = self.get_logp_logt_srho(s_kb, rho0, yp + dy, z, **kw)
        if np.isfinite(p1) and np.isfinite(p2):
            u1 = self.val.get_u_pt_val(p1, t1, yp - dy, z, **kw)
            u2 = self.val.get_u_pt_val(p2, t2, yp + dy, z, **kw)
            return (u2 - u1) / (2 * dy)
        return np.nan

    def _dudz_srho_fd(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0, dz=0.01):
        """dU/dZ|_{S,ρ} via 2D S-ρ inversion."""
        kw = dict(_zm=_zm, _za=_za, _zr=_zr)
        s_kb = self._s_pt(lgp, lgt, yp, z, **kw) * erg_to_kbbar
        rho0 = self._logrho_pt(lgp, lgt, yp, z, **kw)
        dz = self._adaptive_dx(z, dz)
        p1, t1 = self.get_logp_logt_srho(s_kb, rho0, yp, z - dz, **kw)
        p2, t2 = self.get_logp_logt_srho(s_kb, rho0, yp, z + dz, **kw)
        if np.isfinite(p1) and np.isfinite(p2):
            u1 = self.val.get_u_pt_val(p1, t1, yp, z - dz, **kw)
            u2 = self.val.get_u_pt_val(p2, t2, yp, z + dz, **kw)
            return (u2 - u1) / (2 * dz)
        return np.nan

    def get_dtdy_srho(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """dT/dY|_{S,ρ}  [K per Y].

        method='identity': 2×2 implicit function theorem from P-T basis
        method='finite_difference': direct FD via 2D S-ρ inversion
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(self._dtdy_srho_fd, lgp, lgt, yp, z, **kw)
        return self._vec(self._dtdy_srho_scalar, lgp, lgt, yp, z, **kw)

    def _dtdz_srho_scalar(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        T = 10.0 ** lgt
        det = d['b'] * d['c'] - d['a'] * d['d']
        return T * (d['d'] * d['S_Z']
                    - d['S'] * d['b'] * log10_to_loge * d['rho_Z']) \
               / (d['S'] * det)

    def get_dtdz_srho(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """dT/dZ|_{S,ρ}  [K per Z].

        method='identity': 2×2 implicit function theorem from P-T basis
        method='finite_difference': direct FD via 2D S-ρ inversion
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(self._dtdz_srho_fd, lgp, lgt, yp, z, **kw)
        return self._vec(self._dtdz_srho_scalar, lgp, lgt, yp, z, **kw)

    def _dudy_srho_scalar(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        T = 10.0 ** lgt
        P = 10.0 ** lgp
        ln10 = log10_to_loge
        det = d['b'] * d['c'] - d['a'] * d['d']
        dTdY = T * (d['d'] * d['S_Y']
                    - d['S'] * d['b'] * ln10 * d['rho_Y']) \
               / (d['S'] * det)
        dPdY = P * (d['S'] * d['a'] * ln10 * d['rho_Y']
                    - d['c'] * d['S_Y']) \
               / (d['S'] * det)
        dlogPdY = dPdY / (P * ln10)
        dlogTdY = dTdY / (T * ln10)
        return d['U_P'] * dlogPdY + d['U_T'] * dlogTdY + d['U_Y']

    def get_dudy_srho(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """dU/dY|_{S,ρ}  [erg/g per Y].

        Chain rule:
          dU/dY = (dU/dlogP)·(dlogP/dY) + (dU/dlogT)·(dlogT/dY) + U_Y
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(self._dudy_srho_fd, lgp, lgt, yp, z, **kw)
        return self._vec(self._dudy_srho_scalar, lgp, lgt, yp, z, **kw)

    def _dudz_srho_scalar(self, lgp, lgt, yp, z, **kw):
        kw['composition'] = True
        d = self._pt_derivs(lgp, lgt, yp, z, **kw)
        T = 10.0 ** lgt
        P = 10.0 ** lgp
        ln10 = log10_to_loge
        det = d['b'] * d['c'] - d['a'] * d['d']
        dTdZ = T * (d['d'] * d['S_Z']
                    - d['S'] * d['b'] * ln10 * d['rho_Z']) \
               / (d['S'] * det)
        dPdZ = P * (d['S'] * d['a'] * ln10 * d['rho_Z']
                    - d['c'] * d['S_Z']) \
               / (d['S'] * det)
        dlogPdZ = dPdZ / (P * ln10)
        dlogTdZ = dTdZ / (T * ln10)
        return d['U_P'] * dlogPdZ + d['U_T'] * dlogTdZ + d['U_Z']

    def get_dudz_srho(self, lgp, lgt, yp, z=0.0, method='finite_difference', **kw):
        """dU/dZ|_{S,ρ}  [erg/g per Z].

        Same chain rule as get_dudy_srho with Z replacing Y.
        """
        yp = self._to_yprime(yp, z)
        if method == 'finite_difference':
            return self._vec(self._dudz_srho_fd, lgp, lgt, yp, z, **kw)
        return self._vec(self._dudz_srho_scalar, lgp, lgt, yp, z, **kw)

    # =================================================================
    # Convenience: rhomboid plotting / diagnostics
    # =================================================================

    def get_s_bounds_at(self, _lgp):
        """Return (S_lo, S_hi) in kb/baryon at the given logP."""
        return self._get_bounds(_lgp)

    # =================================================================
    # ORCHARD compatibility aliases
    # =================================================================
    # The old `mixtures` class uses different method names and passes
    # extra kwargs (frock, ideal_guess, arr_guess, tab) that are
    # ignored here. These aliases let ORCHARD call hhe_z_mixtures
    # as a drop-in replacement.

    def adaptive_dx(self, x, dx0=0.01):
        """Alias for _adaptive_dx (old mixtures interface)."""
        return self._adaptive_dx(x, dx0)

    def get_s_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """S(P, T, Y, Z).  Delegates to get_s_pt_tab."""
        return self.get_s_pt_tab(_lgp, _lgt, _y, _z, **kw)

    def get_logrho_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """log10 rho(P, T, Y, Z).  Delegates to get_logrho_pt_tab."""
        return self.get_logrho_pt_tab(_lgp, _lgt, _y, _z, **kw)

    def get_logu_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """log10 U(P, T, Y, Z).  Delegates to get_logu_pt_tab."""
        return self.get_logu_pt_tab(_lgp, _lgt, _y, _z, **kw)

    # --- Derivative aliases (old names → new names) ---

    def get_cp_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """C_P.  Old mixtures name for get_cp."""
        return self.get_cp(_lgp, _lgt, _y, _z)

    def get_cv_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, **kw):
        """C_V.  Old mixtures name for get_cv.

        Inverts rho→P via table, then computes C_V = dS/d(lnT)|_ρ
        using vectorized FD on the ρ-T table.
        """
        _y = self._to_yprime(_y, _z)
        lgp = self.get_logp_rhot(_lgrho, _lgt, _y, _z)
        return self._vec_fd_cv(lgp, _lgt, _y, _z)

    def get_dlogrho_dlogt_py(self, _lgp, _lgt, _y, _z, _frock=0.0,
                              dt=1e-2, **kw):
        """dlogρ/dlogT|_P (= −δ).  Old mixtures convention."""
        return -self.get_delta(_lgp, _lgt, _y, _z)

    def get_dlogrho_ds_py(self, _s, _lgp, _y, _z, _frock=0.0,
                           ds=0.1, **kw):
        """dlogρ/dS|_P.  Brunt coefficient in drho space.

        Vectorized: uses S-P table to get T(S±ds, P), then
        evaluates ρ(P, T) and differences.
        """
        _y = self._to_yprime(_y, _z)
        s_kb = _s  # already in kb/baryon
        lgt1 = self.get_logt_sp(s_kb - ds, _lgp, _y, _z)
        lgt2 = self.get_logt_sp(s_kb + ds, _lgp, _y, _z)
        rho1 = self._logrho_pt(_lgp, lgt1, _y, _z)
        rho2 = self._logrho_pt(_lgp, lgt2, _y, _z)
        return (rho2 - rho1) * log10_to_loge / (2 * ds / erg_to_kbbar)

    def get_dlogt_dy_rhop_rhot(self, _lgrho, _lgt, _y, _z,
                                _frock=0.0, **kw):
        """dlogT/dY|_{ρ,P} = χ_Y / χ_T.  Old mixtures interface."""
        _y = self._to_yprime(_y, _z)
        lgp = self.get_logp_rhot(_lgrho, _lgt, _y, _z)
        if np.isscalar(lgp) and not np.isfinite(lgp):
            return np.nan
        chi_y = self.get_chi_Y(lgp, _lgt, _y, _z)
        chi_t = self.get_chi_T(lgp, _lgt, _y, _z)
        if np.isscalar(chi_t) and abs(chi_t) < 1e-30:
            return np.nan
        return chi_y / chi_t

    def get_dlogt_dz_rhop_rhot(self, _lgrho, _lgt, _y, _z,
                                _frock=0.0, **kw):
        """dlogT/dZ|_{ρ,P} = χ_Z / χ_T.  Old mixtures interface."""
        _y = self._to_yprime(_y, _z)
        lgp = self.get_logp_rhot(_lgrho, _lgt, _y, _z)
        if np.isscalar(lgp) and not np.isfinite(lgp):
            return np.nan
        chi_z = self.get_chi_Z(lgp, _lgt, _y, _z)
        chi_t = self.get_chi_T(lgp, _lgt, _y, _z)
        if np.isscalar(chi_t) and abs(chi_t) < 1e-30:
            return np.nan
        return chi_z / chi_t

    def get_dpdt_rhot_rhoy(self, _lgrho, _lgt, _y, _z,
                            _frock=0.0, **kw):
        """dP/dT|_{ρ,Y} in dyn/cm²/K.  Old mixtures interface."""
        _y = self._to_yprime(_y, _z)
        lgp = self.get_logp_rhot(_lgrho, _lgt, _y, _z)
        if np.isscalar(lgp) and not np.isfinite(lgp):
            return np.nan
        chi_t = self.get_chi_T(lgp, _lgt, _y, _z)
        P = 10.0 ** lgp
        T = 10.0 ** _lgt
        return chi_t * P / T

    # --- S-P basis wrappers (ORCHARD calls these with (S, P, Y, Z, frock)) ---

    # The P-T basis getters are available as get_nabla_ad_pt, get_gamma1_pt, etc.
    # The bare names (get_nabla_ad, get_gamma1, ...) are overridden below
    # to accept the OLD ORCHARD calling convention: (S, P, Y, Z, frock, ...)
    # which inverts S→T first, then delegates to the P-T getter.
    # To call the P-T version directly, use get_nabla_ad_pt(lgp, lgt, yp, z).

    def get_nabla_ad(self, _s_or_lgp, _lgp_or_lgt, _y, _z=0.0,
                      _frock=0.0, **kw):
        """∇_ad.  Accepts old ORCHARD (S, P, Y, Z) or new (P, T, Y, Z).

        Auto-detects: if called with 5+ positional args (has _frock),
        or with keyword args like tab/ideal_guess, uses old S-P path.
        Otherwise uses new P-T path.
        """
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            # Old ORCHARD calling convention: (S, P, Y, Z, frock, ...)
            _s, _lgp = _s_or_lgp, _lgp_or_lgt
            _y = self._to_yprime(_y, _z)
            lgt = self.get_logt_sp(_s, _lgp, _y, _z)
            return self._dispatch('nabla_ad', _lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        # New P-T calling convention: (lgp, lgt, Y, Z)
        return self._dispatch('nabla_ad', _s_or_lgp, _lgp_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_gamma1(self, _s_or_lgp, _lgp_or_lgt, _y, _z=0.0,
                    _frock=0.0, **kw):
        """Γ₁.  Accepts old ORCHARD (S, P, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgp = _s_or_lgp, _lgp_or_lgt
            _y = self._to_yprime(_y, _z)
            lgt = self.get_logt_sp(_s, _lgp, _y, _z)
            return self._dispatch('gamma1', _lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('gamma1', _s_or_lgp, _lgp_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_dtds_sp(self, _s_or_lgp, _lgp_or_lgt, _y, _z=0.0,
                     _frock=0.0, **kw):
        """dT/dS|_P.  Accepts old (S, P, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgp = _s_or_lgp, _lgp_or_lgt
            _y = self._to_yprime(_y, _z)
            lgt = self.get_logt_sp(_s, _lgp, _y, _z)
            return self._dispatch('dtds_sp', _lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('dtds_sp', _s_or_lgp, _lgp_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_dtdy_srho(self, _s_or_lgp, _lgrho_or_lgt, _y, _z=0.0,
                       _frock=0.0, **kw):
        """dT/dY|_{S,ρ}.  Accepts old (S, ρ, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgrho = _s_or_lgp, _lgrho_or_lgt
            _y = self._to_yprime(_y, _z)
            lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
            if np.isscalar(lgp) and not np.isfinite(lgp):
                return np.nan
            return self._dispatch('dtdy_srho', lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('dtdy_srho', _s_or_lgp, _lgrho_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_dtdz_srho(self, _s_or_lgp, _lgrho_or_lgt, _y, _z=0.0,
                       _frock=0.0, **kw):
        """dT/dZ|_{S,ρ}.  Accepts old (S, ρ, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgrho = _s_or_lgp, _lgrho_or_lgt
            _y = self._to_yprime(_y, _z)
            lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
            if np.isscalar(lgp) and not np.isfinite(lgp):
                return np.nan
            return self._dispatch('dtdz_srho', lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('dtdz_srho', _s_or_lgp, _lgrho_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_dudy_srho(self, _s_or_lgp, _lgrho_or_lgt, _y, _z=0.0,
                       _frock=0.0, **kw):
        """dU/dY|_{S,ρ}.  Accepts old (S, ρ, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgrho = _s_or_lgp, _lgrho_or_lgt
            _y = self._to_yprime(_y, _z)
            lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
            if np.isscalar(lgp) and not np.isfinite(lgp):
                return np.nan
            return self._dispatch('dudy_srho', lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('dudy_srho', _s_or_lgp, _lgrho_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    def get_dudz_srho(self, _s_or_lgp, _lgrho_or_lgt, _y, _z=0.0,
                       _frock=0.0, **kw):
        """dU/dZ|_{S,ρ}.  Accepts old (S, ρ, Y, Z) or new (P, T, Y, Z)."""
        if 'tab' in kw or 'ideal_guess' in kw or (np.isscalar(_frock) and _frock != 0.0) or (not np.isscalar(_frock)):
            _s, _lgrho = _s_or_lgp, _lgrho_or_lgt
            _y = self._to_yprime(_y, _z)
            lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
            if np.isscalar(lgp) and not np.isfinite(lgp):
                return np.nan
            return self._dispatch('dudz_srho', lgp, lgt, _y, _z,
                                   kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})
        return self._dispatch('dudz_srho', _s_or_lgp, _lgrho_or_lgt, _y, _z,
                               kw.pop('method', 'finite_difference'),
                                   **{k:v for k,v in kw.items()
                                      if k not in ('tab','ideal_guess','arr_guess',
                                                   'arr_p_guess','arr_t_guess',
                                                   'dy','dz','ds','dt','_frock')})

    # --- S-ρ basis wrappers (ORCHARD calls these with (S, rho, Y, Z, frock)) ---

    def get_dsdy_rhop_srho(self, _s, _lgrho, _y, _z,
                            _frock=0.0, **kw):
        """dS/dY|_{ρ,P} (Ledoux).  Old interface takes (S, ρ).

        Vectorized: inverts (S,ρ)→(P,T) via table, then uses
        the vectorized Ledoux identity on the resulting (P,T).
        """
        _y = self._to_yprime(_y, _z)
        lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
        return self._vec_fd_dsdy_rhop(lgp, lgt, _y, _z)

    def get_dsdz_rhop_srho(self, _s, _lgrho, _y, _z,
                            _frock=0.0, **kw):
        """dS/dZ|_{ρ,P} (Ledoux).  Old interface takes (S, ρ).

        Vectorized: inverts (S,ρ)→(P,T) via table, then uses
        the vectorized Ledoux identity on the resulting (P,T).
        """
        _y = self._to_yprime(_y, _z)
        lgp, lgt = self.get_logp_logt_srho(_s, _lgrho, _y, _z)
        return self._vec_fd_dsdz_rhop(lgp, lgt, _y, _z)


class mixtures(hhe_eos):
    def __init__(self, hhe_eos,
                    z_eos = 'aqua',
                    zmix_eos1 = 'aqua',
                    zmix_eos2 = 'ppv2',
                    zmix_eos3 = 'iron2',
                    zmix_eos4 = 'methane',
                    zmix_eos5 = 'ammonia',
                    hg=True,
                    y_prime=False,
                    interp_method='linear',
                    new_z_mix=False,
                    rhot_sp_inv = False,
                    srho_rhop_inv = False,
                    smooth_hhe = False
                    ):
        if hhe_eos in ['cms', 'cd']:
            super().__init__(hhe_eos=hhe_eos, smooth_hhe=smooth_hhe)

        self.y_prime = y_prime
        self.hg = hg
        self.z_eos = z_eos
        self.new_z_mix = new_z_mix

        if 'ice' in z_eos and zmix_eos1 == 'aqua_mlcp':
            self.ices = ice_eos.ice_eos(use_mlcp=True) # whether to use the updated MLCP 2021 water tables
        elif 'ice' in z_eos and zmix_eos1 == 'aqua':
            self.ices = ice_eos.ice_eos() # original AQUA table

        if self.z_eos == 'mixture' or self.z_eos == 'total_mixture':
            self.zmix_eos1 = zmix_eos1
            self.zmix_eos2 = zmix_eos2
            self.zmix_eos3 = zmix_eos3
            self.zmix_eos4 = zmix_eos4
            self.zmix_eos5 = zmix_eos5
            # self.z_methane = z_methane
            # self.z_ammonia = z_ammonia
            # self.z_ppv = z_ppv
            # self.z_fe = z_fe

            # z_water is not defined because it is defined as what is left over from the sum of methane, ammonia, ppv, and iron, just like the hydrogen fraction is defined

        self.interp_method = interp_method

        if not new_z_mix:
            # IF TRUE THEN THIS MODE IS USED FOR BRAND NEW Z MIXTURES. NO TABLES YET EXIST.
            if self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
                z_eos_pt = 'aqua'
            elif self.z_eos == 'aqua':
                z_eos_pt = 'aqua'

            else:
                z_eos_pt = self.z_eos
            self.pt_data = np.load('eos/{}/{}_{}_pt.npz'.format(hhe_eos, hhe_eos, z_eos_pt))

            # RGI interpolation functions
            rgi_args = {'method': self.interp_method, 'bounds_error': False, 'fill_value': None}
            # 1-D independent grids (P, T)
            self.logpvals = self.pt_data['logpvals'] # these are shared. Units: log10 dyn/cm^2
            self.logtvals = self.pt_data['logtvals'] # log10 K
            self.yvals_pt = self.pt_data['yvals'] # mass fraction -- yprime
            self.zvals_pt = self.pt_data['zvals'] # mass fraction
            # 4-D dependent grids (P, T)
            self.s_pt_tab = self.pt_data['s_pt'] # erg/g/K
            self.logrho_pt_tab = self.pt_data['logrho_pt'] # log10 g/cc
            self.logu_pt_tab = self.pt_data['logu_pt'] # log10 erg/g

            self.s_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt),
                                    self.s_pt_tab, **rgi_args)
            self.logrho_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt),
                                    self.logrho_pt_tab, **rgi_args)
            self.logu_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt),
                                    self.logu_pt_tab, **rgi_args)

            if not rhot_sp_inv:
                # IF TRUE THEN MODE IS USED WHEN INVERTING FOR RHO-T AND S-P USING EXISTING P-T TABLES.
                if self.z_eos == 'aqua_smooth2':
                    z_eos_rhot = 'aqua_smooth' # use the same one because it wasn't smoothed along pressure space
                elif self.z_eos == 'aqua_smooth':
                    z_eos_rhot = 'aqua_smooth'
                elif self.z_eos == 'aqua':
                    z_eos_rhot = 'aqua'
                else:
                    z_eos_rhot = self.z_eos
                self.rhot_data = np.load('eos/{}/{}_{}_rhot.npz'.format(hhe_eos, hhe_eos, z_eos_rhot))

                # S, P table can be aqua_smooth (output of smoothed inversion) or aqua_smooth2 (output of pressure smoothing)
                self.sp_data = np.load('eos/{}/{}_{}_sp.npz'.format(hhe_eos, hhe_eos, self.z_eos))
                # # 1-D independent grids (S, P)
                self.svals_sp = self.sp_data['s_vals'] # kb/baryon
                self.logpvals_sp = self.sp_data['logpvals']
                self.yvals_sp = self.sp_data['yvals']
                self.zvals_sp = self.sp_data['zvals']
                # 4-D dependent grids (S, P)
                self.logt_sp_tab = self.sp_data['logt_sp']
                self.logrho_sp_tab = self.sp_data['logrho_sp']

                # # 1-D independent grids (rho, T)
                self.logrhovals_rhot = self.rhot_data['logrhovals'] # log10 g/cc
                self.logtvals_rhot = self.rhot_data['logtvals']
                self.yvals_rhot = self.rhot_data['yvals']
                self.zvals_rhot = self.rhot_data['zvals']
                # 4-D dependent grids (rho T)
                self.s_rhot_tab = self.rhot_data['s_rhot'] # erg/g/K
                self.logp_rhot_tab = self.rhot_data['logp_rhot']

                self.s_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.yvals_rhot, self.zvals_rhot),
                                        self.s_rhot_tab, **rgi_args)
                self.logp_rhot_rgi = RGI((self.logrhovals_rhot, self.logtvals_rhot, self.yvals_rhot, self.zvals_rhot),
                                        self.logp_rhot_tab, **rgi_args)
                # If it is aqua_smooth2, then the S-P table is smoothed along pressure space.
                # this means that the axes are changed because I iterated first along Z, then Y, S, and P
                if self.z_eos == 'aqua_smooth2':
                    self.logt_sp_rgi = RGI((self.zvals_sp, self.yvals_sp, self.svals_sp, self.logpvals_sp),
                                            self.logt_sp_tab, **rgi_args)
                    self.logrho_sp_rgi = RGI((self.zvals_sp, self.yvals_sp, self.svals_sp, self.logpvals_sp),
                                            self.logrho_sp_tab, **rgi_args)

                else:
                    self.logt_sp_rgi = RGI((self.svals_sp, self.logpvals_sp, self.yvals_sp, self.zvals_sp),
                                        self.logt_sp_tab, **rgi_args)
                    self.logrho_sp_rgi = RGI((self.svals_sp, self.logpvals_sp, self.yvals_sp, self.zvals_sp),
                                        self.logrho_sp_tab, **rgi_args)

            if not srho_rhop_inv:
                # IF TRUE THEN MODE IS USED WHEN INVERTING FOR S-RHO AND RHO-P USING EXISTING P-T, S-P, AND RHO-T TABLES.

                if self.z_eos == 'aqua_smooth2':
                    z_eos_rhop = 'aqua' # use the same one because rho,P table was not updated nor is it used in evolution for now
                elif self.z_eos == 'aqua_smooth':
                    z_eos_rhop = 'aqua'
                elif self.z_eos == 'aqua':
                    z_eos_rhop = 'aqua'
                else:
                    z_eos_rhop = self.z_eos
                # elif self.z_eos == 'aqua_smooth':
                #     z_eos_srho = 'aqua_smooth'
                # elif self.z_eos == 'aqua':
                #     z_eos_srho = 'aqua'

                #self.rhop_data = np.load('eos/{}/{}_{}_rhop.npz'.format(hhe_eos, hhe_eos, z_eos_rhop))
                self.srho_data = np.load('eos/{}/{}_{}_srho.npz'.format(hhe_eos, hhe_eos, self.z_eos))
                # 1-D independent grids (rho, P)
                # self.logpvals_rhop = self.rhop_data['logpvals']
                # self.logrhovals_rhop = self.rhop_data['logrhovals'] # log10 g/cc -- rho, P table range
                # self.yvals_rhop = self.rhop_data['yvals']
                # self.zvals_rhop = self.rhop_data['zvals']
                # 4-D dependent grids (rho, P)
                # self.s_rhop_tab = self.rhop_data['s_rhop'] # erg/g/K
                # self.logt_rhop_tab = self.rhop_data['logt_rhop']

                # # 1-D independent grids (S, rho)
                self.svals_srho = self.srho_data['s_vals'] # kb/baryon
                self.logrhovals_srho = self.srho_data['logrhovals'] # log10 g/cc -- rho, P table range
                self.yvals_srho = self.srho_data['yvals']
                self.zvals_srho = self.srho_data['zvals']
                # 4-D dependent grids (S, rho)
                self.logp_srho_tab = self.srho_data['logp_srho']
                self.logt_srho_tab = self.srho_data['logt_srho']

                # self.s_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals_rhop, self.yvals_rhop, self.zvals_rhop),
                #                         self.s_rhop_tab, **rgi_args)
                # self.logt_rhop_rgi = RGI((self.logrhovals_rhop, self.logpvals_rhop, self.yvals_rhop, self.zvals_rhop),
                #                         self.logt_rhop_tab, **rgi_args)

                if self.z_eos == 'aqua_smooth2':
                    self.logp_srho_rgi = RGI((self.zvals_srho, self.yvals_srho, self.svals_srho, self.logrhovals_srho),
                                        self.logp_srho_tab, **rgi_args)
                    self.logt_srho_rgi = RGI((self.zvals_srho, self.yvals_srho, self.svals_srho, self.logrhovals_srho),
                                        self.logt_srho_tab, **rgi_args)
                else:
                    self.logp_srho_rgi = RGI((self.svals_srho, self.logrhovals_srho, self.yvals_srho, self.zvals_srho),
                                            self.logp_srho_tab, **rgi_args)
                    self.logt_srho_rgi = RGI((self.svals_srho, self.logrhovals_srho, self.yvals_srho, self.zvals_srho),
                                            self.logt_srho_tab, **rgi_args)


    def Y_to_n(self, _y):
        ''' Change between mass and number fraction OF HELIUM'''
        return ((_y/mhe)/(((1 - _y)/mh) + (_y/mhe)))

    def n_to_Y(self, x):
        ''' Change between number and mass fraction OF HELIUM'''
        return (mhe * x)/(1 + 3.0026*x)

    def x_H(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (1-yeff)*(1-_z)/mh/Ntot

    def x_Z(self, _y, _z, mz):
        yeff = _y#/(1 - _z)
        Ntot = (1-yeff)*(1-_z)/mh + (yeff*(1-_z)/mhe) + _z/mz
        return (_z/mz)/Ntot

    def guarded_log(self, x):
        ''' Used to calculate ideal enetropy of mixing: xlogx'''
        if np.isscalar(x):
            if x == 0:
                return 0
            elif x  < 0:
                raise ValueError('Number fraction went negative.')
            return x * np.log(x)
        return np.array([self.guarded_log(x_) for x_ in x])

    def get_smix_id_y(self, Y):
        xhe = self.Y_to_n(Y)
        xh = 1 - xhe
        q = mh*xh + mhe*xhe
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe)) / q

    def get_smix_id_yz(self, Y, Z, mz):
        xh = self.x_H(Y, Z, mz)
        xz = self.x_Z(Y, Z, mz)
        xhe = 1 - xh - xz
        q = mh*xh + mhe*xhe + mz*xz
        return -1*(self.guarded_log(xh) + self.guarded_log(xhe) + self.guarded_log(xz)) / q

    def get_smix_ideal(self, _y, _zw, _zm, _za, _zr, _zfe):
        # Mass fractions:
        f_h       = (1 - _y) * (1 - _zw) * (1 - _zm) * (1 - _za) * (1 - _zr) * (1 - _zfe)
        f_he      = _y * (1 - _zw) * (1 - _zm) * (1 - _za) * (1 - _zr) * (1 - _zfe)
        f_water   = _zw * (1 - _zm) * (1 - _za) * (1 - _zr) * (1 - _zfe)
        f_methane = _zm * (1 - _za) * (1 - _zr) * (1 - _zfe)
        f_ammonia = _za * (1 - _zr) * (1 - _zfe)
        f_rock    = _zr * (1 - _zfe)
        f_iron    = _zfe

        # # Compute ideal mixing entropy:
        m_h = 1.0
        m_he = 4.0026
        m_water = 18.015
        m_methane = 16.04     # CH4
        m_ammonia = 17.031    # NH3
        m_rock = 100.3887
        m_iron = 55.845

        n_h       = f_h / m_h
        n_he      = f_he / m_he
        n_water   = f_water   / m_water
        n_methane = f_methane / m_methane
        n_ammonia = f_ammonia / m_ammonia
        n_rock    = f_rock    / m_rock
        n_iron    = f_iron    / m_iron
        Ntot = n_h + n_he + n_water + n_methane + n_ammonia + n_rock + n_iron

        # Number fractions:
        x_h       = n_h       / Ntot
        x_he      = n_he      / Ntot
        x_water   = n_water   / Ntot
        x_methane = n_methane / Ntot
        x_ammonia = n_ammonia / Ntot
        x_rock    = n_rock    / Ntot
        x_iron    = n_iron    / Ntot

        #Compute average molecular weight weighted by these number fractions:
        q = m_h * x_h + m_he * x_he + m_water * x_water + m_methane * x_methane + m_ammonia * x_ammonia + m_rock * x_rock + m_iron * x_iron

        #Ideal entropy of mixing term (using natural logs and a guarded logarithm):
        s_mix_id = - (self.guarded_log(x_h) + self.guarded_log(x_he) + self.guarded_log(x_water) + self.guarded_log(x_methane) + self.guarded_log(x_ammonia) \
                        + self.guarded_log(x_rock) + self.guarded_log(x_iron)) / q

        return s_mix_id

    def get_smix_ideal_abs(self, f_h, f_he, f_water, f_methane, f_ammonia, f_rock=0.0, f_iron=0.0):
        """
        Ideal entropy of mixing from absolute (physical) mass fractions.
        Unlike get_smix_ideal, this takes the actual mass fractions directly
        rather than nested conditional fractions.
        """
        m_h = 1.0
        m_he = 4.0026
        m_water = 18.015
        m_methane = 16.04     # CH4
        m_ammonia = 17.031    # NH3
        m_rock = 100.3887
        m_iron = 55.845

        n_h       = f_h / m_h
        n_he      = f_he / m_he
        n_water   = f_water / m_water
        n_methane = f_methane / m_methane
        n_ammonia = f_ammonia / m_ammonia
        n_rock    = f_rock / m_rock
        n_iron    = f_iron / m_iron
        Ntot = n_h + n_he + n_water + n_methane + n_ammonia + n_rock + n_iron

        x_h       = n_h       / Ntot
        x_he      = n_he      / Ntot
        x_water   = n_water   / Ntot
        x_methane = n_methane / Ntot
        x_ammonia = n_ammonia / Ntot
        x_rock    = n_rock    / Ntot
        x_iron    = n_iron    / Ntot

        q = (m_h * x_h + m_he * x_he + m_water * x_water
             + m_methane * x_methane + m_ammonia * x_ammonia
             + m_rock * x_rock + m_iron * x_iron)

        s_mix_id = -(self.guarded_log(x_h) + self.guarded_log(x_he)
                     + self.guarded_log(x_water) + self.guarded_log(x_methane)
                     + self.guarded_log(x_ammonia) + self.guarded_log(x_rock)
                     + self.guarded_log(x_iron)) / q

        return s_mix_id

    ####### Volume-Addition Law #######

    def get_s_pt_val(self, _lgp, _lgt, _y_prime, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0):
        """
        This calculates the entropy for a metallicity mixture using the volume addition law.
        These terms contain the ideal entropy of mixing, so
        for metal mixtures, we subtract the H-He ideal entropy of mixing and
        add back the metal mixture entropy of mixing plus the non-ideal
        correction from Howard & Guillot (2023a).

        The _y_prime parameter is the Y in a pure H-He EOS. Therefore, it
        is Y/(1 - Z). So the y value that should be
        used to calculate the entropy of mixing should be Y*(1 - Z).
        """

        logt_arr = np.atleast_1d(_lgt)
        logp_arr = np.atleast_1d(_lgp)
        pts = np.column_stack((logt_arr, logp_arr))

        def validate_mass_fractions(_y_prime, _z):
            if (
                (np.isscalar(_y_prime) and _y_prime > 1.0)
                or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
                or (np.isscalar(_z) and _z > 1.0)
                or ((not np.isscalar(_z)) and np.any(_z > 1.0))
            ):
                raise ValueError('Invalid mass fractions: X + Y + Z > 1.')

        def get_mz(z_eos):
            if z_eos == 'aqua' or z_eos == 'aneos_mlcp' or z_eos == 'ice_aneos' or z_eos == 'aqua_mlcp' or z_eos == 'aqua_smooth':
                return 18.015
            elif z_eos == 'ppv' or z_eos == 'ppv2':
                return 100.3887
            elif z_eos == 'iron':
                return 55.845
            elif z_eos == 'mixture':
                return 18.015 * (1 - self.f_ppv) + 100.3887 * self.f_ppv# + 55.845
            else:
                raise ValueError('Only water (aqua or mazevet+19 (mlcp)), ppv, and iron supported for now.')

        #_y = _y_prime * (1 - _z)

        validate_mass_fractions(_y_prime, _z)

        smix_xy_ideal = self.get_smix_id_y(_y_prime) / erg_to_kbbar
        smix_xy_nonideal = 0.0
        if self.hg:
            if self.hhe_eos == 'cms':
                smix_xy_nonideal = self.smix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime - smix_xy_ideal

        if self.hhe_eos in ['cms', 'cd']:
            s_x = 10 ** self.get_s_h(_lgp, _lgt)
            s_y = 10 ** self.get_s_he(_lgp, _lgt)
            s_xy = s_x * (1 - _y_prime) + s_y * _y_prime

        elif self.hhe_eos == 'scvh':
            s_xy = scvh_eos.get_s_pt_tab(_lgp, _lgt, _y_prime) - smix_xy_ideal # subtract ideal entropy of mixing

        if self.z_eos == 'mixture':
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=_zr, f_fe=0.0,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _z, _zm=0.0, _za=0.0, _zr=_zr, _zfe=0.0) / erg_to_kbbar
        elif 'ice_mixture' in self.z_eos:
            s_z = self.ices.get_s_pt_val(_lgp, _lgt, _zm, _za)
            # Physical mass fractions: H-He make up (1-Z), ices make up Z
            smix_xyz_ideal = self.get_smix_ideal_abs(
                f_h       = (1 - _y_prime) * (1 - _z),
                f_he      = _y_prime * (1 - _z),
                f_water   = (1 - _zm) * (1 - _za) * _z,
                f_methane = _zm * (1 - _za) * _z,
                f_ammonia = _za * _z,
            ) / erg_to_kbbar

        elif 'ice_rock' in self.z_eos:
            s_z_ice = self.ices.get_s_pt_val(_lgp, _lgt, _zm, _za)
            s_z_rock = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ppv2')

            s_z = (1 - _zr) * s_z_ice + _zr * s_z_rock
            # Physical mass fractions: H-He at (1-Z), ices at (1-_zr)*Z, rock at _zr*Z
            smix_xyz_ideal = self.get_smix_ideal_abs(
                f_h       = (1 - _y_prime) * (1 - _z),
                f_he      = _y_prime * (1 - _z),
                f_water   = (1 - _zm) * (1 - _za) * (1 - _zr) * _z,
                f_methane = _zm * (1 - _za) * (1 - _zr) * _z,
                f_ammonia = _za * (1 - _zr) * _z,
                f_rock    = _zr * _z,
            ) / erg_to_kbbar

        elif self.z_eos == 'aqua' or self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
            self.z_eos = 'aqua'
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _zw=_z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0) / erg_to_kbbar

        elif self.z_eos == 'ppv' or self.z_eos == 'ppv2':
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _zw=0.0, _zm=0.0, _za=0.0, _zr=1.0, _zfe=0.0) / erg_to_kbbar

        elif self.z_eos == 'iron' or self.z_eos == 'iron2':
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _zw=0.0, _zm=0.0, _za=0.0, _zr=0.0, _zfe=1.0) / erg_to_kbbar
        else:
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos)

        return (
            s_xy * (1 - _z)
            + s_z * _z
            + smix_xyz_ideal
            + smix_xy_nonideal * (1 - _z)
        )

    def get_logrho_pt_val(self, _lgp, _lgt, _y_prime, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0):
        """
        This function calculates the density of a H-He-Z mixture using the volume addition law.
        When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a).

        Parameters:
            _lgp (float): Logarithm of pressure.
            _lgt (float): Logarithm of temperature.
            _y_prime (float): Helium mass fraction in a pure H-He EOS.
            _z (float): Metallicity.

        Returns:
            float: Logarithm of the density.
        """

        def validate_mass_fractions(_y_prime, _z):
            if (
                (np.isscalar(_y_prime) and _y_prime > 1.0)
                or ((not np.isscalar(_y_prime)) and np.any(_y_prime > 1.0))
                or (np.isscalar(_z) and _z > 1.0)
                or ((not np.isscalar(_z)) and np.any(_z > 1.0))
            ):
                raise ValueError('Invalid mass fractions: X + Y + Z > 1.')

        def calculate_vmix(_lgp, _lgt, _y_prime):
            if self.hg and self.hhe_eos == 'cms':
                return self.vmix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime
            return 0.0

        validate_mass_fractions(_y_prime, _z)
        vmix = calculate_vmix(_lgp, _lgt, _y_prime)

        if self.hhe_eos in ['cms', 'cd']:

            rho_h = 10 ** self.get_logrho_h(_lgp, _lgt)
            rho_he = 10 ** self.get_logrho_he(_lgp, _lgt)
            v_xy = (1 - _y_prime) / rho_h + _y_prime / rho_he + vmix

        elif self.hhe_eos == 'scvh':
            rho_xy = 10 ** scvh_eos.get_rho_pt_tab(_lgp, _lgt, _y_prime)  # rho_xy is in g/cc
            v_xy = 1 / rho_xy + vmix # vmix is zero since there are no interaction terms

        if self.z_eos == 'mixture':
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=_zr, f_fe=_zfe,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
        elif 'ice_mixture' in self.z_eos:
            rho_z = 10 ** self.ices.get_logrho_pt_val(_lgp, _lgt, _zm, _za)
        elif 'ice_rock' in self.z_eos:
            rho_z_ice = 10 ** self.ices.get_logrho_pt_val(_lgp, _lgt, _zm, _za)
            rho_z_rock = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos='ppv2')

            rho_z = 1 / ((1 - _zr) / rho_z_ice + _zr / rho_z_rock)

        elif self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
            self.z_eos = 'aqua'
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)
        else:
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)

        #mixture_density = (1 - _y_prime) * (1 - _z) / rho_h + _y_prime * (1 - _z) / rho_he + vmix * (1 - _z) + _z / rho_z
        mixture_density = v_xy * (1 - _z) + _z / rho_z

        return np.log10(1 / mixture_density)

    def get_u_pt_val(self, _lgp, _lgt, _y_prime, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0):
        """
        This function calculates the internal energy per unit mass of a H-He-Z mixture using the volume addition law.
        When including the non-ideal corrections, this function adds the volume of mixing from Howard & Guillot (2023a).

        Parameters:
            _lgp (float): Logarithm of pressure.
            _lgt (float): Logarithm of temperature.
            _y_prime (float): Helium mass fraction in a pure H-He EOS.
            _z (float): Metallicity.

        Returns:
            float: Logarithm of the internal energy per unit mass.
        """

        def calculate_umix(_lgp, _lgt, _y_prime):
            if self.hg and self.hhe_eos == 'cms':
                return self.umix_interp(_lgp, _lgt) * (1 - _y_prime) * _y_prime
            return 0.0

        umix = calculate_umix(_lgp, _lgt, _y_prime)

        if self.hhe_eos in ['cms', 'cd']:
            u_h = 10 ** self.get_logu_h(_lgp, _lgt)
            u_he = 10 ** self.get_logu_he(_lgp, _lgt)
            u_xy = u_h * (1 - _y_prime) + u_he * _y_prime + umix

        elif self.hhe_eos == 'scvh':
            u_xy = 10 ** scvh_eos.get_u_pt(_lgp, _lgt, _y_prime) + umix # umix is zero since there are no interaction terms
        if self.z_eos == 'mixture':
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=_zr, f_fe=_zfe,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
        elif 'ice_mixture' in self.z_eos:
            u_z = self.ices.get_u_pt_val(_lgp, _lgt, _zm, _za)

        elif 'ice_rock' in self.z_eos:
            u_z_ice = self.ices.get_u_pt_val(_lgp, _lgt, _zm, _za)
            u_z_rock = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos='ppv2')

            u_z = (1 - _zr) * u_z_ice + _zr * u_z_rock

        elif self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
            self.z_eos = 'aqua'
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)
        else:
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)

        # mixture_energy = (
        #     u_h * (1 - _y_prime) * (1 - _z)
        #     + u_he * _y_prime * (1 - _z)
        #     + umix * (1 - _z)
        #     + u_z * _z
        # )

        mixture_energy = (
            u_xy * (1 - _z)
            + u_z * _z
        )

        return mixture_energy


    ####### EOS table calls #######

    # logp, logt tables
    def get_s_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logrho_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logu_pt_tab(self, _lgp, _lgt, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgp, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.logu_pt_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logt tables
    def get_s_rhot_tab(self, _lgrho, _lgt, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result = self.s_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logp_rhot_tab(self, _lgrho, _lgt, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgrho, _lgt, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_rhot_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # S, logp tables
    def get_logt_sp_tab(self, _s, _lgp, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        if self.z_eos == 'aqua_smooth2':
            args = (_z, _y, _s, _lgp)
        else:
            args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logrho_sp_tab(self, _s, _lgp, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        if self.z_eos == 'aqua_smooth2':
            args = (_z, _y, _s, _lgp)
        else:
            args = (_s, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logrho_sp_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    # logrho, logp tables

    def get_logt_rhop_tab(self, _lgrho, _lgp, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_s_rhop_tab(self, _lgrho, _lgp, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        args = (_lgrho, _lgp, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.s_rhop_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result
        r#eturn self.get_s_pt_tab(_lgp, self.get_logt_rhop_tab(*args), _y, _z)

    # S, logrho tables
    def get_logp_srho_tab(self, _s, _lgrho, _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        if self.z_eos == 'aqua_smooth2':
            args = (_z, _y, _s, _lgrho)
        else:
            args = (_s, _lgrho, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logp_srho_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result

    def get_logt_srho_tab(self, _s, _lgrho,  _y, _z, _frock=0.0):

        _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        if self.z_eos == 'aqua_smooth2':
            args = (_z, _y, _s, _lgrho)
        else:
            args = (_s, _lgrho, _y, _z)
        v_args = [np.atleast_1d(arg) for arg in args]
        pts = np.column_stack(v_args)
        result =  self.logt_srho_rgi(pts)
        if all(np.isscalar(arg) for arg in args):
            return result.item()
        else:
            return result


    ### Inversion Functions ###

    def get_logt_sp_inv(self, _s, _lgp, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _s = np.atleast_1d(_s)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        _zr = np.atleast_1d(_zr)
        _zfe = np.atleast_1d(_zfe)

        # _y = _y if self.y_prime else _y * (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgp, _y, _z, _zm, _za, _zr, _zfe = np.broadcast_arrays(_s, _lgp, _y, _z, _zm, _za, _zr, _zfe)

        if ideal_guess:
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            guess = arr_guess

    # Define a function to compute root and capture convergence
        def root_func(s_i, lgp_i, y_i, z_i, zm_i, za_i, zr_i, zfe_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                #
                if self.new_z_mix:
                    s_test = self.get_s_pt_val(lgp_i, _lgt, y_i, z_i, zm_i, za_i, zr_i, zfe_i) * erg_to_kbbar
                else:
                    s_test = self.get_s_pt_tab(lgp_i, _lgt, y_i, z_i) * erg_to_kbbar
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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

                except OverflowError:
                    print('Failed at s={}, logp={}, y={}, z={}'.format(s_i, lgp_i, y_i, z_i))
                    raise
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgp, _y, _z, _zm, _za, _zr, _zfe, guess)

        return temperatures, converged

    def get_logrho_sp_inv(self, _s, _lgp, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logt, conv = self.get_logt_sp_inv( _s, _lgp, _y, _z, _zm, _za, _zr, _zfe=0.0, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        return self.get_logrho_pt_tab(_lgp, logt, _y, _z)

    def get_logp_rhot_inv(self, _lgrho, _lgt, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgt, _y, _z = np.broadcast_arrays(_lgrho, _lgt, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
       # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgt_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                _y_call = y_i if self.y_prime else y_i / (1 - z_i)
                logrho_test = self.get_logrho_pt_tab(_lgp, lgt_i, _y_call, z_i)
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
        pressure, converged = vectorized_root_func(_lgrho, _lgt, _y, _z, guess)

        return pressure, converged

    def get_s_rhot_inv(self, _lgrho, _lgt, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logp, conv = self.get_logp_rhot_inv(_lgrho, _lgt, _y, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=_zfe, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        return self.get_s_pt_tab(logp, _lgt, _y, _z)

    def get_logp_srho_inv(self, _s, _lgrho, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the pressure given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_p_srho(_s, _lgrho, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                logrho_test = self.get_logrho_sp_tab(s_i, _lgp, y_i, z_i)
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, guess)

        return temperatures, converged

    def get_logt_srho_inv(self, _s, _lgrho, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

       # _y = _y if self.y_prime else _y / (1 - _z+1e-6)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_srho(_s, _lgrho, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
    # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                s_test = self.get_s_rhot_tab(lgrho_i, _lgt, y_i, z_i) * erg_to_kbbar
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, guess)

        return temperatures, converged

    def get_logp_logt_srho_2Dinv(self, _s, _lgrho, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='root'):
        """
        Compute temperature and pressure given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            arr_guess (tuple of array_like, optional): User-provided initial guesses for log temperature and log pressure when `ideal_guess` is False.

        Returns:
            logt_values (ndarray): Computed log10 temperature values.
            logp_values (ndarray): Computed log10 pressure values.
            converged (ndarray): Boolean array indicating convergence for each point.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z = np.broadcast_arrays(_s, _lgrho, _y, _z)

        # Prepare output arrays
        shape = _s.shape
        logt_values = np.empty(shape)
        logp_values = np.empty(shape)
        converged = np.zeros(shape, dtype=bool)

        # Initial guesses for log temperature and log pressure
        if ideal_guess:
            # Use the ideal EOS for the initial guesses
            #pdb.set_trace()
            guess_lgp, guess_lgt = ideal_xy.get_pt_srho(_s, _lgrho, _y).T
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            else:
                guess_lgp, guess_lgt = arr_guess

        # Flatten arrays for iteration
        lgrho_flat = _lgrho.flatten()
        s_flat = _s.flatten()
        y_flat = _y.flatten()
        z_flat = _z.flatten()
        guess_lgp_flat = guess_lgp.flatten()
        guess_lgt_flat = guess_lgt.flatten()

        # Iterate over each element
        for idx in range(len(s_flat)):
            lgrho_i = lgrho_flat[idx]
            s_i = s_flat[idx]
            y_i = y_flat[idx]
            z_i = z_flat[idx]
            guess_lgp_i = guess_lgp_flat[idx]
            guess_lgt_i = guess_lgt_flat[idx]
            if method == 'root':

                def opt(vars):
                    lgp, lgt = vars
                    s_calc = self.get_s_pt_tab(lgp, lgt, y_i, z_i) * erg_to_kbbar
                    lgrho_calc = self.get_logrho_pt_tab(lgp, lgt, y_i, z_i)

                    # Convert s_calc and lgrho_calc to scalars if they are arrays
                    if isinstance(s_calc, np.ndarray):
                        s_calc = s_calc.item()
                    if isinstance(lgrho_calc, np.ndarray):
                        lgrho_calc = lgrho_calc.item()

                    err1 = (s_calc/s_i) - 1
                    err2 = (lgrho_calc/lgrho_i) - 1
                    return np.array([err1, err2])

                try:
                    sol = root(
                        opt, [guess_lgp_i, guess_lgt_i], method='hybr', tol=1e-6
                    )
                    if sol.success:
                        logp_values.flat[idx], logt_values.flat[idx] = sol.x
                        converged.flat[idx] = True
                    else:
                        logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                        converged.flat[idx] = False
                except Exception as e:
                    logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                    converged.flat[idx] = False

            elif method == 'nelder-mead':
                def opt(vars):

                    lgp, lgt = vars
                    s_calc = self.get_s_pt_tab(lgp, lgt, y_i, z_i) * erg_to_kbbar
                    lgrho_calc = self.get_logrho_pt_tab(lgp, lgt, y_i, z_i)

                    # Convert s_calc and lgrho_calc to scalars if they are arrays
                    if isinstance(s_calc, np.ndarray):
                        s_calc = s_calc.item()
                    if isinstance(lgrho_calc, np.ndarray):
                        lgrho_calc = lgrho_calc.item()

                    err1 = (s_calc / s_i) - 1
                    err2 = (lgrho_calc / lgrho_i) - 1
                    return err1**2 + err2**2  # Return a scalar

                try:
                    sol = minimize(
                        opt, [guess_lgp_i, guess_lgt_i], method='nelder-mead'
                    )
                    if sol.success:
                        logp_values.flat[idx], logt_values.flat[idx] = sol.x
                        converged.flat[idx] = True
                    else:
                        logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                        converged.flat[idx] = False
                except Exception as e:
                    logp_values.flat[idx], logt_values.flat[idx] = np.nan, np.nan
                    converged.flat[idx] = False


        # Reshape output arrays to original shape
        logp_values = logp_values.reshape(shape)
        logt_values = logt_values.reshape(shape)
        converged = converged.reshape(shape)

        return logp_values, logt_values, converged


    def get_logp_logt_srho_inv(self, _s, _lgrho, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logp, convp = self.get_logp_srho_inv(_s, _lgrho, _y, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=_zfe, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        logt, convt = self.get_logt_sp_inv(_s, logp, _y, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=_zfe, ideal_guess=True, arr_guess=None, method=method)
        return logp, logt

    def get_logu_srho(self, _s, _lgrho, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):

        if tab:
            if self.z_eos == 'aqua_smooth2':
                logp, logt = self.get_logp_srho_tab(_s, _lgrho, _y, _z), self.get_logt_srho_tab(_s, _lgrho, _y, _z)
                logu = self.get_logu_pt_tab(logp, logt, _y, _z)
                # logu = self.return_noglitch(logp, logu)
                # logu = self.fill_nans_1d(logu, kind='linear')
                logu[(_lgrho > -4.5) & (_lgrho < -0.5)] = gaussian_filter1d(logu[(_lgrho > -4.5) & (_lgrho < -0.5)],
                                                                                   sigma=2.0, mode='reflect')
                return logu
            else:
                logp, logt = self.get_logp_srho_tab(_s, _lgrho, _y, _z), self.get_logt_srho_tab(_s, _lgrho, _y, _z)
                return self.get_logu_pt_tab(logp, logt, _y, _z)
        else:
            #_y = _y if self.y_prime else _y / (1 - _z+1e-6)
            # WARNING: do not rely on in-situ derivatives because the y prime is not implemented here (yet)
            logp, convp = self.get_logp_srho_inv( _s, _lgrho, _y, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=_zfe, ideal_guess=ideal_guess, arr_guess=arr_p_guess, method=method)
            logt, convt = self.get_logt_sp_inv( _s, logp, _y, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=_zfe, ideal_guess=ideal_guess, arr_guess=arr_t_guess, method=method)
            return self.get_logu_pt_tab(logp, logt, _y, _z)


    def get_logt_rhop_inv(self, _lgrho, _lgp, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given density, pressure, helium abundance, and metallicity.

        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _lgrho = np.atleast_1d(_lgrho)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

        if ideal_guess:
            guess = ideal_xy.get_t_rhop(_lgrho, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            else:
                guess = arr_guess
        # sol = root(err, guess, tol=1e-6)
        # return sol.x, sol.success

    # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
            def err(_lgt):
                logrho_test = self.get_logrho_pt_tab(lgp_i, _lgt, y_i, z_i)
                return (logrho_test / lgrho_i) - 1


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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

        return temperatures, converged


    def get_s_rhop_inv(self, _lgrho, _lgp, _y, _z, _zm=0.0, _za=0.0, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        """
        Compute the entropy given density, pressure, helium abundance, and metallicity.

        Parameters:
            _lgrho (array_like): Log10 density values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            arr_guess (array_like, optional): User-provided initial guess for log entropy when `ideal_guess` is False.
            method (str, optional): Method to use for root finding ('root', 'newton', or 'brentq').

        Returns:
            ndarray: Computed entropy values.
            ndarray: Convergence status for each element.
        """

        _lgrho = np.atleast_1d(_lgrho)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgp, _y, _z = np.broadcast_arrays(_lgrho, _lgp, _y, _z)

        if ideal_guess:
            #y_call = _y if self.y_prime else _y / (1 - _z+1e-6)
            guess = ideal_xy.get_s_rhop(_lgrho, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            else:
                guess = arr_guess

        def root_func(lgrho_i, lgp_i, y_i, z_i, guess_i):
            def err(_s):
                logrho_test = self.get_logrho_sp_tab(_s, lgp_i, y_i, z_i)
                return (logrho_test / lgrho_i) - 1

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
                    # Convergence failed
                    return np.nan, False
                except Exception as e:
                    # Handle other exceptions
                    return np.nan, False

            elif method == 'brentq':
                try:
                    a, b = guess_i - 1, guess_i + 1  # Initial bracket
                    fa, fb = err(a), err(b)
                    factor = 1.5
                    delta = 0.1
                    while fa * fb > 0:
                        a -= delta * factor
                        b += delta * factor
                        delta *= factor
                        fa, fb = err(a), err(b)
                        if np.isnan(fa) or np.isnan(fb):
                            raise ValueError("Function returned NaN.")
                    sol_root = brentq(err, a, b, xtol=1e-5, maxiter=100)
                    return sol_root, True
                except ValueError:
                    return np.nan, False

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

            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")

        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        entropies, converged = vectorized_root_func(_lgrho, _lgp, _y, _z, guess)

        return entropies / erg_to_kbbar, converged

    # adaptive delta function for z and y derivatives
    def adaptive_dx(self, x_profile, initial_dx=0.01, tolerance=1e-3):
        # Initialize dx as an array with an initial value
        dx = np.full_like(x_profile, initial_dx, dtype=float)

        # Adjust each dz based on z_profile constraints
        for i in range(len(x_profile)):
            # Adjust dz so z_profile[i] - dz[i] >= 0
            if x_profile[i] - dx[i] < 0:
                dx[i] = x_profile[i]  # Set dz to the maximum allowed value to keep z_profile - dz non-negative

            # Adjust dz so z_profile[i] + dz[i] <= 1
            elif x_profile[i] + dx[i] > 1:
                dx[i] = 1 - x_profile[i]  # Set dz to the maximum allowed value to keep z_profile + dz <= 1

            # Add a tolerance check to prevent overshooting the bounds
            if x_profile[i] - dx[i] < 0:
                dx[i] = max(dx[i] - tolerance, 0)
            if x_profile[i] + dx[i] > 0.999:
                dx[i] = min(dx[i] - tolerance, 1 - x_profile[i])

        return dx


    def interpolate_non_converged_temperatures_1d(self, _lgrho, temperatures, converged, interp_kind='linear'):

        # Get converged and non-converged indices
        converged_indices = np.where(converged)
        non_converged_indices = np.where(~converged)

        # Extract converged data
        lgrho_converged = _lgrho[converged_indices]
        temperatures_converged = temperatures[converged_indices]

        # Sort data for interpolation
        sorted_indices = np.argsort(lgrho_converged)
        lgrho_converged_sorted = lgrho_converged[sorted_indices]
        temperatures_converged_sorted = temperatures_converged[sorted_indices]

        # Create interpolation function
        interp_func = interp1d(
            lgrho_converged_sorted, temperatures_converged_sorted, kind=interp_kind, fill_value="extrapolate"
        )

        # Interpolate temperatures for non-converged points
        temperatures_interpolated = temperatures.copy()
        temperatures_interpolated[non_converged_indices] = interp_func(_lgrho[non_converged_indices])

        return temperatures_interpolated


    def adaptive_hampel_filter(self, y, min_window=3, max_window=15, n_sigmas=3):
        y = np.array(y)
        n = len(y)
        y_filtered = y.copy()
        outlier_indices = []

        for i in range(n):
            # Determine the optimal window size at position i
            local_window_size = self.determine_optimal_window(y, i, min_window, max_window)
            window_range = range(max(0, i - local_window_size), min(n, i + local_window_size + 1))
            window = y[window_range]
            median = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - median))
            deviation = np.abs(y[i] - median)
            if deviation > n_sigmas * mad:
                y_filtered[i] = median
                outlier_indices.append(i)
        return y_filtered, outlier_indices

    def determine_optimal_window(self, y, index, min_window, max_window):
        # Custom logic to determine window size based on local data properties
        # Placeholder for actual implementation
        return min_window  # Or any logic to vary window size

    def remove_outliers(self, x, y, outlier_indices):
        """
        Removes outliers from the data arrays.

        Parameters:
            x (array-like): The independent variable array.
            y (array-like): The dependent variable array.
            outlier_indices (list): Indices of the outliers to remove.

        Returns:
            x_clean (np.array): The x array without outliers.
            y_clean (np.array): The y array without outliers.
        """
        x_clean = np.delete(x, outlier_indices)
        y_clean = np.delete(y, outlier_indices)
        return x_clean, y_clean

    def interpolate_missing(self, x_clean, y_clean, x_original, kind='linear'):
        """
        Interpolates the missing data points.

        Parameters:
            x_clean (array-like): The x array without outliers.
            y_clean (array-like): The y array without outliers.
            x_original (array-like): The original x array (including outlier positions).
            kind (str): Type of interpolation ('linear', 'quadratic', 'cubic', etc.).

        Returns:
            y_interpolated (np.array): The y array with interpolated values at missing points.
        """
        interp_func = interp1d(x_clean, y_clean, kind=kind, fill_value='extrapolate')
        y_interpolated = interp_func(x_original)
        return y_interpolated

    def fill_nans_1d(self, arr, kind='linear'):
        """
        Fill NaNs in a 1D array by interpolation of any specified 'kind'
        recognized by scipy.interpolate.interp1d:
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.

        Parameters
        ----------
        arr : 1D numpy array
            Array that may contain NaNs.
        kind : str
            Interpolation type to pass to interp1d (e.g. 'linear', 'quadratic').

        Returns
        -------
        arr_filled : 1D numpy array
            A copy of arr with NaNs replaced by interpolation of the specified kind.
            If there are not enough valid points to interpolate (e.g., all NaN),
            we simply return arr unchanged.
        """

        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError("This function only handles 1D arrays.")

        x = np.arange(len(arr))
        valid_mask = ~np.isnan(arr)

        # If everything is NaN, or only 1 valid point, we can’t do a real polynomial interpolation.
        if np.count_nonzero(valid_mask) < 2:
            return arr  # or decide on a default fill approach

        # Build the interpolator.  'fill_value="extrapolate"' lets us fill beyond the data range.
        f = interp1d(
            x[valid_mask],
            arr[valid_mask],
            kind=kind,
            fill_value="extrapolate"
        )

        # Create a copy for the result
        arr_filled = arr.copy()
        # Where arr is NaN, replace with the interpolation
        arr_filled[~valid_mask] = f(x[~valid_mask])

        return arr_filled

    def return_noglitch(self, x, y):

        y_filtered, outlier_indices = self.adaptive_hampel_filter(y, min_window=3, max_window=15, n_sigmas=3)
        x_clean, y_clean = self.remove_outliers(x, y, outlier_indices)
        y_interpolated = self.interpolate_missing(x_clean, y_clean, x, kind='linear')

        return y_interpolated

    def glitch_correct_1d(self, y, window=5, n_sigmas=3):
        """
        1) Identify outliers in y with a hampel filter, window +/- 'window'.
        2) Mark them as nan in a copy.
        3) Interpolate the nans.
        Returns y_corrected.
        """
        y_copy = np.asarray(y, float).copy()
        n = len(y_copy)
        outlier_inds = []
        for i in range(n):
            w_start = max(0, i-window)
            w_end   = min(n, i+window+1)
            local_y = y_copy[w_start:w_end]
            med = np.median(local_y)
            mad = 1.4826 * np.median(np.abs(local_y - med)) or 1e-6
            if abs(y_copy[i] - med) > n_sigmas*mad:
                outlier_inds.append(i)

        # set them to nan
        y_copy[outlier_inds] = np.nan

        # now interpolate the nan
        y_fixed = self.fill_nans_1d(y_copy, kind='linear')
        return y_fixed

    def inversion(self, a_arr, b_arr, y_arr, z_arr, basis, inversion_method='newton_brentq', twoD_inv=False, calc_derivatives=False, gauss_smooth=False):

        """
        Invert the EOS table to compute new thermodynamic quantities.

        Parameters:
            a_arr (array_like): Array of 'a' values (e.g., entropy).
            b_arr (array_like): Array of 'b' values (e.g., pressure).
            y_arr (array_like): Array of helium mass fraction values.
            z_arr (array_like): Array of metallicity values.
            basis (str): The basis for inversion ('sp', 'rhot', 'srho', 'rhop').

        Returns:
            Two arrays of the inverted quantities.
        """

        res1_list = []
        res2_list = []

        for a_ in tqdm(a_arr):
            res1_b = []
            res2_b = []
            for b_ in b_arr:
                res1_y = []
                res2_y = []
                prev_res1_temp = None  # Initialize previous res1_temp to None
                prev_res2_temp = None # For double inversion
                for y_ in y_arr:
                    a_const = np.full_like(z_arr, a_)
                    b_const = np.full_like(z_arr, b_)
                    y_const = np.full_like(z_arr, y_)
                    if basis == 'sp':
                        try:
                            if prev_res1_temp is None:
                                res1_temp, conv = self.get_logt_sp_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                            else:
                                res1_temp, conv = self.get_logt_sp_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )
                        except:
                            print('Failed at s={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        try:
                            res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='linear'
                            )
                        except:
                            # import pdb
                            # pdb.set_trace()
                            raise Exception('Failed interpolation at s={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))

                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1_noglitch2 = self.return_noglitch(z_arr, res1_noglitch)
                        # last line of defense against nans in inversion ...
                        res1 = self.fill_nans_1d(res1_noglitch2, kind='linear')

                        if gauss_smooth:
                            if a_ <= 3.0: # smooth only the coldest regions
                                res1 = gaussian_filter1d(res1, sigma=3.0)

                        res2 = self.get_logrho_pt_tab(b_const, res1, y_const, z_arr)




                        prev_res1_temp = res1 # Update prev_res1_temp for the next iteration

                    elif basis == 'rhot':

                        try:
                            if prev_res1_temp is None:

                                res1_temp, conv = self.get_logp_rhot_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                            else:
                                res1_temp, conv = self.get_logp_rhot_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )

                        except:
                            print('Failed at rho={}, logt={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_interp = self.interpolate_non_converged_temperatures_1d(
                            z_arr, res1_temp, conv, interp_kind='quadratic'
                        )

                        # two passes through no-glitch filter... some have more than two glitches that the first pass does
                        # not catch.
                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1_noglitch2 = self.return_noglitch(z_arr, res1_noglitch)
                        # last line of defense against nans in inversion ...
                        res1 = self.fill_nans_1d(res1_noglitch2, kind='linear')

                        res2 = self.get_s_pt_tab(res1, b_const, y_const, z_arr)

                        prev_res1_temp = res1

                    elif basis == 'srho':

                        # if twoD_inv:
                        #     try:
                        #         if prev_res1_temp is None:
                        #             res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
                        #                 a_const, b_const, y_const, z_arr, ideal_guess=True, method='root'
                        #                 )
                        #         else:
                        #             res1_temp, res2_temp, conv = self.get_logp_logt_srho_2Dinv(
                        #                 a_const, b_const, y_const, z_arr, ideal_guess=False, method='root', arr_guess=[prev_res1_temp, prev_res2_temp]
                        #                 )
                        #     except:
                        #         print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                        #         raise

                        #     res1_interp = self.interpolate_non_converged_temperatures_1d(
                        #         z_arr, res1_temp, conv, interp_kind='quadratic'
                        #     )

                        #     res2_interp = self.interpolate_non_converged_temperatures_1d(
                        #         z_arr, res2_temp, conv, interp_kind='quadratic'
                        #     )

                        #     res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        #     res1 = self.return_noglitch(z_arr, res1_noglitch)

                        #     res2_noglitch = self.return_noglitch(z_arr, res2_interp)
                        #     res2 = self.return_noglitch(z_arr, res2_noglitch)

                        #     prev_res1_temp = res1
                        #     prev_res2_temp = res2

                        # else: # uses 1-D inversion via SP inverted table

                        try:
                            if prev_res1_temp is None:

                                res1_temp, conv = self.get_logp_srho_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                    z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )

                                # res2_temp, conv2_1 = self.get_logt_sp_inv(
                                #     a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=True
                                #     )
                                # res2_interp = self.interpolate_non_converged_temperatures_1d(
                                #     z_arr, res2_temp, conv, interp_kind='quadratic'
                                #     )


                            else:
                                res1_temp, conv = self.get_logp_srho_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                    z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )
                                # res2_temp, conv2_1 = self.get_logt_sp_inv(
                                #     a_const, res1_interp, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res2_temp
                                #     )

                                # res2_interp = self.interpolate_non_converged_temperatures_1d(
                                #     z_arr, res2_temp, conv2_1, interp_kind='quadratic'
                                #     )

                        except:
                            print('Failed at s={}, rho={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1_noglitch2 = self.return_noglitch(z_arr, res1_noglitch)

                        res1 = self.fill_nans_1d(res1_noglitch2, kind='linear')

                        # res2_noglitch = self.return_noglitch(z_arr, res2_interp)
                        # res2 = self.return_noglitch(z_arr, res2_noglitch)

                        if gauss_smooth:
                            if a_ <= 4.0: # smooth only the coldest regions
                                res1 = gaussian_filter1d(res1, sigma=3.0)

                        res2 = self.get_logt_sp_tab(
                            a_const, res1, y_const, z_arr
                            )

                        prev_res1_temp = res1
                        prev_res2_temp = res2

                    elif basis == 'rhop':
                        try:
                            if prev_res1_temp is None:


                                # inverting the table along entropy instead of temperature instead....
                                res1_temp, conv = self.get_s_rhop_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=True
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )
                            else:
                                res1_temp, conv = self.get_s_rhop_inv(
                                    a_const, b_const, y_const, z_arr, method=inversion_method, ideal_guess=False, arr_guess=prev_res1_temp
                                    )
                                res1_interp = self.interpolate_non_converged_temperatures_1d(
                                z_arr, res1_temp, conv, interp_kind='quadratic'
                                    )

                        except:
                            print('Failed at rho={}, logp={}, y={}'.format(a_const[0], b_const[0], y_const[0]))
                            raise

                        res1_noglitch = self.return_noglitch(z_arr, res1_interp)
                        res1 = self.return_noglitch(z_arr, res1_noglitch)

                        res2 = self.get_logt_sp_tab(res1*erg_to_kbbar, b_const, y_const, z_arr)


                        prev_res1_temp = res1
                        prev_res2_temp = res2


                    else:
                        raise ValueError('Unknown inversion basis. Please choose sp, rhot, srho, or rhop')

                    res1_y.append(res1)
                    res2_y.append(res2)

                res1_b.append(res1_y)
                res2_b.append(res2_y)

            res1_list.append(res1_b)
            res2_list.append(res2_b)

        return np.array(res1_list), np.array(res2_list)

    ################################################ Wrapper Functions ################################################

    def get_logt_sp(self, _s, _lgp, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_sp_tab(_s, _lgp, _y, _z)

        else:
            return self.get_logt_sp_inv(_s, _lgp, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

    def get_logrho_sp(self, _s, _lgp, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logrho_sp_tab(_s, _lgp, _y, _z)

        else:
            return self.get_logrho_sp_inv(_s, _lgp, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)

    def get_logp_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logp_rhot_tab(_lgrho, _lgt, _y, _z)

        else:
            return self.get_logp_rhot_inv(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

    def get_s_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_s_rhot_tab(_lgrho, _lgt, _y, _z)

        else:
            return self.get_s_rhot_inv(_lgrho, _lgt, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)

    def get_logt_rhop(self, _lgrho, _lgp, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_rhop_tab(_lgrho, _lgp, _y, _z)

        else:
            return self.get_logt_rhop_inv(_lgrho, _lgp, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

    def get_s_rhop(self, _lgrho, _lgp, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_s_rhop_tab(_lgrho, _lgp, _y, _z)

        else:
            return self.get_s_rhop_inv(_lgrho, _lgp, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

    def get_logp_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logp_srho_tab(_s, _lgrho, _y, _z)

        else:
            return self.get_logp_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

    def get_logt_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        if tab:
            return self.get_logt_srho_tab(_s, _lgrho, _y, _z)

        else:
            # return self.get_logp_logt_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess,
            #                             arr_guess=arr_guess, method=method)[-1]
            return self.get_logt_srho_inv(_s, _lgrho, _y, _z, ideal_guess=ideal_guess,
                                        arr_guess=arr_guess, method=method)[0]

#    P, T wrappers
    def get_logrho_pt(self, _lgp, _lgt, _y, _z, _frock=0.0):
        return self.get_logrho_pt_tab(_lgp, _lgt, _y, _z)
    def get_s_pt(self, _lgp, _lgt, _y, _z, _frock=0.0):
        return self.get_s_pt_tab(_lgp, _lgt, _y, _z)
    def get_logu_pt(self, _lgp, _lgt, _y, _z, _frock=0.0):
        return self.get_logu_pt_tab(_lgp, _lgt, _y, _z)

    # obtains adiabatic entropy profile based on a P, T, Y, and Z profile:
    def err_grad(self, s_trial, _lgp, _y, _z):
        grad_a = self.get_nabla_ad(s_trial, _lgp, _y, _z)
        logt = self.get_logt_sp_tab(s_trial, _lgp, _y, _z)
        grad_prof = np.gradient(logt)/np.gradient(_lgp)
        return (grad_a/grad_prof) - 1

    def get_s_ad(self, _lgp, _lgt, _y, _z):
        """This function returns the entropy value
        required for nabla - nabla_a = 0 at
        pressure and temperature profiles"""

        # if y_tot:
        #     _y /= (1 - _z)

        guess = self.get_s_pt_tab(_lgp, _lgt, _y, _z) * const.erg_to_kbbar

        sol = root(self.err_grad, guess, tol=1e-8, method='hybr', args=(_lgp, _y, _z))
        return sol.x

    ################################################ Derivatives ################################################

    ########### Convection Derivatives ###########

    # Specific heat at constant pressure
    def get_cp_sp(self, _s, _lgp, _y, _z, _frock=0.0, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cp_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dt=0.1):

        s1 = self.get_s_pt_tab(_lgp, _lgt - dt, _y, _z, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt + dt, _y, _z, _frock)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    def get_cp2_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dt=0.1):
        s0 = self.get_s_pt_tab(_lgp, _lgt, _y, _z, _frock)
        s1 = self.get_s_pt_tab(_lgp, _lgt - dt, _y, _z, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt + dt, _y, _z, _frock)

        d2sdlnT2 = (s2 - 2 * s0 + s1) / (dt * log10_to_loge) ** 2

        return d2sdlnT2

    # Specific heat at constant volume
    def get_cv_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cv_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        s1 = self.get_s_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        s2 = self.get_s_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    def get_cv2_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        s0 = self.get_s_rhot(_lgrho, _lgt, _y, _z, _frock, **kwargs)
        s1 = self.get_s_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        s2 = self.get_s_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)

        d2sdlnT2 = (s2 - 2 * s0 + s1) / (dt * log10_to_loge) ** 2

        return d2sdlnT2

    # Adiabatic temperature gradient
    def get_nabla_ad(self, _s, _lgp, _y, _z, _frock=0.0, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        lgt1 = self.get_logt_sp(_s, _lgp - dp, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_sp(_s, _lgp + dp, _y, _z, _frock, **kwargs)
        return (lgt2 - lgt1)/(2 * dp)

    def get_dpdt_rhot_rhoy(self, _lgrho, _lgt, _y, _z, _frock=0.0, dT=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dt = _lgt*0.1 if dt is None else dt

        T0 = 10**_lgt
        T1 = T0*(1 - dT)
        T2 = T0*(1 + dT)

        P1 = 10**self.get_logp_rhot(_lgrho, np.log10(T1), _y, _z, _frock, **kwargs)
        P2 = 10**self.get_logp_rhot(_lgrho, np.log10(T2), _y, _z, _frock, **kwargs)

        return (P2 - P1)/(T2 - T1)

    # DS/DX|_P, T - DERIVATIVES NECESSARY FOR THE SCHWARZSCHILD CONDITION
    def get_dsdy_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dy=0.1):
        dy = _y*0.1 if dy is None else dy
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y - dy, _z, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y + dy, _z, _frock)
        return (s2 - s1)/(2 * dy)

    def get_d2sdy2_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dy=0.1):
        dy = _y*0.1 if dy is None else dy
        s0 = self.get_s_pt_tab(_lgp, _lgt, _y, _z, _frock)
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y - dy, _z, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y + dy, _z, _frock)
        return (s2 - 2 * s0 + s1)/(dy * log10_to_loge) ** 2

    def get_dsdz_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dz=0.1):
        dz = _z*0.1 if dz is None else dz
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y, _z - dz, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y, _z + dz, _frock)
        return (s2 - s1)/(2 * dz)

    def get_d2sdz2_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dz=0.1):
        dz = _z*0.1 if dz is None else dz
        s0 = self.get_s_pt_tab(_lgp, _lgt, _y, _z, _frock)
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y, _z - dz, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y, _z + dz, _frock)
        return (s2 - 2 * s0 + s1)/(dz * log10_to_loge) ** 2

    # def get_dsdy_rhop(self, _lgrho, _lgp, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     dy = _y*0.1 if dy is None else dy
    #     s1 = self.get_s_rhop(_lgrho, _lgp, _y - dy, _z, **kwargs)
    #     s2 = self.get_s_rhop(_lgrho, _lgp, _y + dy, _z, **kwargs)
    #     return (s2 - s1)/(2 * dy)

    # def get_dsdz_rhop(self, _lgrho, _lgp, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     dz = _z*0.1 if dz is None else dz
    #     s1 = self.get_s_rhop(_lgrho, _lgp, _y, _z - dz, **kwargs)
    #     s2 = self.get_s_rhop(_lgrho, _lgp, _y, _z + dz, **kwargs)
    #     return (s2 - s1)/(2 * dz)

    def get_gamma1(self, _s, _lgp, _y, _z, _frock=0.0, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # dlnP/dlnrho_S, Y, Z = dlogP/dlogrho_S, Y, Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho1 = self.get_logrho_sp(_s, _lgp - dp, _y, _z, _frock, **kwargs)
        lgrho2 = self.get_logrho_sp(_s, _lgp + dp, _y, _z, _frock, **kwargs)
        return (2*dp)/(lgrho2 - lgrho1)

    # Brunt coefficient when computing in drho space
    def get_dlogrho_ds_py(self, _s, _lgp, _y, _z, _frock=0.0, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho2 = self.get_logrho_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)
        lgrho1 = self.get_logrho_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        return ((lgrho2 - lgrho1) * log10_to_loge) / (2 * ds / erg_to_kbbar)

    # Chi_T/Chi_rho
    # aka "delta" in MLT flux
    def get_dlogrho_dlogt_py(self, _lgp, _lgt, _y, _z, _frock=0.0, dt=1e-2):

        lgrho1 = self.get_logrho_pt_tab(_lgp, _lgt - dt, _y, _z, _frock)
        lgrho2 = self.get_logrho_pt_tab(_lgp, _lgt + dt, _y, _z, _frock)

        return (lgrho2 - lgrho1)/(2 * dt)

    def get_dlogp_dy_rhot(self, _lgrho, _lgt,  _y, _z, _frock=0.0, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Y
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt, _y - dy, _z, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt, _y + dy, _z, _frock, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dy)

    def get_dlogp_dz_rhot(self, _lgrho, _lgt,  _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt, _y, _z - dz, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt, _y, _z + dz, _frock, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dz)

    def get_dlogp_dlogt_rhoy_rhot(self, _lgrho, _lgt,  _y, _z, _frock=0.0, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_T
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)

        return (lgp2 - lgp1)/(2 * dt)

    # Chi_Y/Chi_T
    def get_dlogt_dy_rhop_rhot(self, _lgrho, _lgt,  _y, _z, _frock=0.0, dy=0.1, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Y = self.get_dlogp_dy_rhot(_lgrho, _lgt,  _y, _z, _frock, dy=dy, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, _frock, dt=dt, **kwargs)

        return Chi_Y/Chi_T

    # Chi_Z/Chi_T
    def get_dlogt_dz_rhop_rhot(self, _lgrho, _lgt,  _y, _z, _frock=0.0, dz=0.1, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Z = self.get_dlogp_dz_rhot(_lgrho, _lgt,  _y, _z,_frock, dz=dz, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, _frock, dt=dt, **kwargs)

        return Chi_Z/Chi_T

    #### Triple Product Rule Derivatives ###*

    def get_dpds_rhoy_srho(self, _s, _lgrho, _y, _z,_frock=0.0, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        ds = _s*0.1 if ds is None else ds
        p1 = 10**self.get_logp_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (p2 - p1) / (2 * ds / erg_to_kbbar)

    def get_dpdy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y - dy, _z, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y + dy, _z, **kwargs)

        return (p2 - p1) / (2 * dy)


    def get_dpdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (p2 - p1) / (2 * dz)

    ########### Triple product rule dsdx_rhop version ###########

    # DS/DX|_rho, P - DERIVATIVES NECESSARY FOR THE LEDOUX CONDITION
    def get_dsdy_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdy_srho = self.get_dpdy_srho(_s, _lgrho, _y, _z, _frock, dy=dy, **kwargs)

        #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdy_rhopy = -dpdy_srho/dpds_rhoy_srho # triple product rule

        return dsdy_rhopy

    def get_d2sdy2_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        dsdy_rhopy1 = self.get_dsdy_rhop_srho(_s, _lgrho, _y - dy, _z, _frock, ds=ds, dy=dy, **kwargs)
        dsdy_rhopy2 = self.get_dsdy_rhop_srho(_s, _lgrho, _y + dy, _z, _frock, ds=ds, dy=dy, **kwargs)

        return (dsdy_rhopy2 - dsdy_rhopy1) / (2 * dy)

    def get_d2sdzdy_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dsdy_rhopz1 = self.get_dsdy_rhop_srho(_s, _lgrho, _y, _z - dz, _frock, ds=ds, dy=dy, **kwargs)
        dsdy_rhopz2 = self.get_dsdy_rhop_srho(_s, _lgrho, _y, _z + dz, _frock, ds=ds, dy=dy, **kwargs)
        return (dsdy_rhopz2 - dsdy_rhopz1) / (2 * dz)

    def get_d2sdsdy_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dsdy_rhops1 = self.get_dsdy_rhop_srho(_s - ds, _lgrho, _y, _z, _frock, ds=ds, dy=dy, **kwargs)
        dsdy_rhops2 = self.get_dsdy_rhop_srho(_s + ds, _lgrho, _y, _z, _frock, ds=ds, dy=dy, **kwargs)
        return (dsdy_rhops2 - dsdy_rhops1) / (2 * ds / erg_to_kbbar)


    def get_dsdz_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdz_srho = self.get_dpdz_srho(_s, _lgrho, _y, _z, _frock, dz=dz, **kwargs)

        #dSdZ|{rho, P, Z} = -dPdZ|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdz_rhopy = -dpdz_srho/dpds_rhoy_srho # triple product rule

        return dsdz_rhopy

    def get_d2sdz2_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dsdz_rhopy1 = self.get_dsdz_rhop_srho(_s, _lgrho, _y, _z - dz, _frock, ds=ds, dz=dz, **kwargs)
        dsdz_rhopy2 = self.get_dsdz_rhop_srho(_s, _lgrho, _y, _z + dz, _frock, ds=ds, dz=dz, **kwargs)
        return (dsdz_rhopy2 - dsdz_rhopy1) / (2 * dz)

    def get_d2sdydz_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dsdz_rhopz1 = self.get_dsdz_rhop_srho(_s, _lgrho, _y - dy, _z, _frock, ds=ds, dz=dz, **kwargs)
        dsdz_rhopz2 = self.get_dsdz_rhop_srho(_s, _lgrho, _y + dy, _z, _frock, ds=ds, dz=dz, **kwargs)
        return (dsdz_rhopz2 - dsdz_rhopz1) / (2 * dy)

    def get_d2sdsdz_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, dy=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dsdz_rhops1 = self.get_dsdz_rhop_srho(_s - ds, _lgrho, _y, _z, _frock, ds=ds, dz=dz, **kwargs)
        dsdz_rhops2 = self.get_dsdz_rhop_srho(_s + ds, _lgrho, _y, _z, _frock, ds=ds, dz=dz, **kwargs)
        return (dsdz_rhops2 - dsdz_rhops1) / (2 * ds / erg_to_kbbar)

    # def get_drhods_rhoy_sp(self, _s, _lgp, _y, _z, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     ds = _s*0.1 if ds is None else ds
    #     rho1 = 10**self.get_logrho_sp(_s - ds, _lgp, _y, _z, **kwargs)
    #     rho2 = 10**self.get_logrho_sp(_s + ds, _lgp, _y, _z, **kwargs)

    #     return (rho2 - rho1) / (2 * ds / erg_to_kbbar)

    # def get_drhody_sp(self, _s, _lgp, _y, _z, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     dy = _y*0.1 if dy is None else dy
    #     rho1 = 10**self.get_logrho_sp(_s, _lgp, _y - dy, _z, **kwargs)
    #     rho2 = 10**self.get_logrho_sp(_s, _lgp, _y + dy, _z, **kwargs)

    #     return (rho2 - rho1) / (2 * dy)


    # def get_drhodz_sp(self, _s, _lgp, _y, _z, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     dz = _z*0.1 if dz is None else dz
    #     rho1 = 10**self.get_logrho_sp(_s, _lgp, _y, _z - dz, **kwargs)
    #     rho2 = 10**self.get_logrho_sp(_s, _lgp, _y, _z + dz, **kwargs)

    #     return (rho2 - rho1) / (2 * dz)

    # def get_dsdy_rhop_sp(self, _s, _lgp, _y, _z, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     #dPdS|{rho, Y, Z}:
    #     drhods_rhoy_srho = self.get_drhods_rhoy_sp(_s, _lgp, _y, _z, ds=ds, **kwargs)
    #     #dPdY|{S, rho, Y}:
    #     drhody_sp = self.get_drhody_sp(_s, _lgp, _y, _z, dy=dy, **kwargs)

    #     #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
    #     dsdy_rhop = -drhody_sp/drhods_rhoy_srho # triple product rule

    #     return dsdy_rhop


    # def get_dsdz_rhop_sp(self, _s, _lgp, _y, _z, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
    #     kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
    #     #drhodS|{P, Y, Z}:
    #     drhods_rhoy_srho = self.get_drhods_rhoy_sp(_s, _lgp, _y, _z, ds=ds, **kwargs)
    #     #drhodZ|{P, rho, Y}:
    #     drhodz_sp = self.get_drhodz_sp(_s, _lgp, _y, _z, dz=dz, **kwargs)

    #     #dSdZ|{rho, P, Z} = -drhodZ|{S, rho, Y} / drhodS|{rho, Y, Z}
    #     dsdz_rhop = -drhodz_sp/drhods_rhoy_srho # triple product rule

    #     return dsdz_rhop


    ########### Chemical Potential Terms ###########

    def get_dudy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.1, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_p_guess': arr_p_guess, 'arr_t_guess': arr_p_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y - dy, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y + dy, _z, _frock, **kwargs)

        return (u2 - u1)/(2 * dy)

    def get_dudz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_p_guess': arr_p_guess, 'arr_t_guess': arr_p_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (u2 - u1)/(2 * dz)

    ########### Conductive Flux Terms ###########

    def get_dtdy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y - dy, _z, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y + dy, _z, _frock, **kwargs)

        return (t2 - t1)/(2 * dy)

    def get_dtdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (t2 - t1)/(2 * dz)

    ########## Thermodynamic Consistency Test ###########

    # du/ds_(rho, Y) = T
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        u1 = 10**self.get_logu_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (u2 - u1)/(2 * ds / erg_to_kbbar)

    # -du/dV_(S, Y) = P
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, drho=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        R0 = 10 **_lgrho
        R1 = R0*(1-drho)
        R2 = R0*(1+drho)

        u1 = 10**self.get_logu_srho(_s, np.log10(R1), _y, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, np.log10(R2), _y, _z, _frock, **kwargs)

        return (u2 - u1)/((1/R1) - (1/R2))

    ########## Atmospheric update derivative ###########

    def get_dtds_sp(self, _s, _lgp, _y, _z, _frock=0.0, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        t2 = 10**self.get_logt_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)

        return (t2 - t1)/(2 * ds / erg_to_kbbar)


################################################################################
# Now define the multi‐rock‐fraction class that *inherits* from mixtures
################################################################################

class multifraction_mixtures(mixtures):
    """
    This class loads multiple precomputed H-He-Z EoS tables corresponding
    to different fractions of rock (f_ppv). It then provides 5D interpolators
    for each table type (pt, rhot, sp, srho), with coordinate order
    (x1, x2, x3, x4, f_ppv).

    Usage:
        obj = MultiFractionMixtures(hhe_eos='cd')
        val_s = obj.get_s_pt_tab(logP, logT, _y, _z, _frock=0.75)
        val_logrho = obj.get_logrho_pt_tab(logP, logT, _y, _z, _frock=0.75)
        ...
    """

    def __init__(self,
                #  zmix_eos1,
                #  zmix_eos2,
                #  zmix_eos3,
                 hhe_eos = 'cd',
                 z_eos_list: list = None,
                 f_ppv_vals: np.ndarray = None,
                 f_ppv: float = 0.0,
                 f_fe: float = 0.0,
                 hg: bool = False,
                 y_prime: bool = False,
                 interp_method: str = 'linear',
                 new_z_mix: bool = False):
        """
        Initialize the MultiFractionMixtures class.

        Parameters:
            hhe_eos (str): H-He EOS identifier.
            z_eos (str): Z EOS identifier.
            z_eos_list (list): List of H-He-ice/rock mixture EOSes.
            f_ppv_vals (np.ndarray): Array of f_ppv values.
            zmix_eos1 (str): First Z mixture EOS identifier.
            zmix_eos2 (str): Second Z mixture EOS identifier.
            zmix_eos3 (str): Third Z mixture EOS identifier.
            f_ppv (float): Fraction of ppv.
            f_fe (float): Fraction of Fe.
            hg (bool): Flag for HG.
            y_prime (bool): Flag for Y prime.
            interp_method (str): Interpolation method.
            new_z_mix (bool): Flag for new Z mix.
        """

        # super().__init__(hhe_eos=hhe_eos,
        #                  z_eos=z_eos,
        #                  zmix_eos1=zmix_eos1,
        #                  zmix_eos2=zmix_eos2,
        #                  zmix_eos3=zmix_eos3,
        #                  f_ppv=f_ppv,
        #                  f_fe=f_fe,
        #                  hg=hg,
        #                  y_prime=y_prime,
        #                  interp_method=interp_method,
        #                  new_z_mix=new_z_mix)

        # # If user doesn't provide a list, use a default range of fractions:
        # if z_eos_list is None:
        #     z_eos_list = [
        #         f'{zmix_eos1}_{zmix_eos2}_0.0',
        #         f'{zmix_eos1}_{zmix_eos2}_0.25',
        #         f'{zmix_eos1}_{zmix_eos2}_0.5',
        #         f'{zmix_eos1}_{zmix_eos2}_0.75',
        #         f'{zmix_eos1}_{zmix_eos2}_1.0'
        #     ]

        if z_eos_list is None:
            z_eos_list = [
                '1.0_0.0_ice_rock_mixture',
                '0.75_0.25_ice_rock_mixture',
                '0.5_0.5_ice_rock_mixture',
                '0.25_0.75_ice_rock_mixture',
                '0.0_1.0_ice_rock_mixture',
            ]
        if f_ppv_vals is None:
            f_ppv_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        self.hhe_eos = hhe_eos
        self.y_prime = y_prime
        self.z_eos_list   = z_eos_list
        self.f_ppv_vals   = f_ppv_vals
        self.table_types  = ['pt', 'rhot', 'sp', 'srho']  # or add 'rhop' if needed
        self.interp_method = interp_method

        # We'll store final results in self.data_combined
        self.data_combined = {}

        # Now build the multi-fraction tables
        self._build_multi_rock_tables()

    def _build_multi_rock_tables(self):
        """
        Private helper to load .npz data for each fraction and table type,
        combine them along the fraction dimension, and build 5D interpolators.
        """
        # 1. Define table metadata: coordinate and dependent variable NPZ keys
        table_defs = {
            'pt': {
                'coords_names': ['logpvals', 'logtvals', 'yvals', 'zvals'],
                'data_names':   ['s_pt', 'logrho_pt', 'logu_pt']
            },
            'rhot': {
                'coords_names': ['logrhovals', 'logtvals', 'yvals', 'zvals'],
                'data_names':   ['s_rhot', 'logp_rhot']
            },
            'sp': {
                'coords_names': ['s_vals', 'logpvals', 'yvals', 'zvals'],
                'data_names':   ['logt_sp', 'logrho_sp']
            },
            'srho': {
                'coords_names': ['s_vals', 'logrhovals', 'yvals', 'zvals'],
                'data_names':   ['logp_srho', 'logt_srho']
            },
        }

        # 2. Load & combine for each table type
        for table_type in self.table_types:
            # Grab metadata
            cinfo = table_defs[table_type]
            coords_names = cinfo['coords_names']
            data_names   = cinfo['data_names']

            # We'll collect 4D arrays for each fraction
            data0_list = []
            data1_list = []
            data2_list = []

            # 2A. Load the first file
            first_fname = f'eos/{self.hhe_eos}/{self.hhe_eos}_{self.z_eos_list[0]}_{table_type}.npz'
            arrays_0    = np.load(first_fname)

            coords_4d = [arrays_0[nm] for nm in coords_names]  # 4 coordinate arrays
            dep0_0    = arrays_0[data_names[0]]  # shape: (n_x1, n_x2, n_x3, n_x4)
            dep1_0    = arrays_0[data_names[1]]

            data0_list.append(dep0_0)
            data1_list.append(dep1_0)
            if table_type == 'pt':
                dep2_0 = arrays_0[data_names[2]]
                data2_list.append(dep2_0)

            # 2B. Load subsequent files
            for i in range(1, len(self.z_eos_list)):
                fname_i = f'eos/{self.hhe_eos}/{self.hhe_eos}_{self.z_eos_list[i]}_{table_type}.npz'
                arr_i   = np.load(fname_i)

                d0_i = arr_i[data_names[0]]
                d1_i = arr_i[data_names[1]]
                data0_list.append(d0_i)
                data1_list.append(d1_i)
                if table_type == 'pt':
                    d2_i = arr_i[data_names[2]]
                    data2_list.append(d2_i)

            # 2C. Stack along last axis => shape: (n_x1, n_x2, n_x3, n_x4, n_f)
            data0_5d = np.stack(data0_list, axis=-1)
            data1_5d = np.stack(data1_list, axis=-1)

            # 2D. Create final 5D coordinates
            coords_5d = tuple(coords_4d) + (self.f_ppv_vals,)

            # 2E. Build interpolators
            interp_0 = RGI(
                coords_5d, data0_5d,
                method=self.interp_method,
                bounds_error=False,
                fill_value=None
            )
            interp_1 = RGI(
                coords_5d, data1_5d,
                method=self.interp_method,
                bounds_error=False,
                fill_value=None
            )

            # 2F. Store in self.data_combined
            if table_type == 'pt':
                data2_5d = np.stack(data2_list, axis=-1)

                interp_2 = RGI(
                    coords_5d, data2_5d,
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=None
                )

                self.data_combined[table_type] = {
                    'coords':   coords_5d,
                    'data0_5d': data0_5d,
                    'data1_5d': data1_5d,
                    'interp_0': interp_0,
                    'interp_1': interp_1,
                    'data2_5d': data2_5d,
                    'interp_2': interp_2
                }

            else:
                self.data_combined[table_type] = {
                    'coords':   coords_5d,
                    'data0_5d': data0_5d,
                    'data1_5d': data1_5d,
                    'interp_0': interp_0,
                    'interp_1': interp_1
                }

    ############################################################################
    # 3. Define “getter” methods to query each table type
    #
    #    Each method calls the corresponding interpolator in self.data_combined
    #    with the correct ordering of arguments:
    #      (x1, x2, y, z, f_ppv)
    ############################################################################

    # --- pt: s_pt, logrho_pt
    def get_s_pt(self, _lgp, _lgt, _y, _z, _frock):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['pt']['interp_0'](( _lgp, _lgt, _y, _z, _frock))

    def get_logrho_pt(self, _lgp, _lgt, _y, _z, _frock):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['pt']['interp_1'](( _lgp, _lgt, _y, _z, _frock))

    def get_logu_pt(self, _lgp, _lgt, _y, _z, _frock):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['pt']['interp_2'](( _lgp, _lgt, _y, _z, _frock))

    # --- rhot: s_rhot, logp_rhot
    def get_s_rhot(self, _lgrho, _lgt, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['rhot']['interp_0']((_lgrho, _lgt, _y, _z, _frock))

    def get_logp_rhot(self, _lgrho, _lgt, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['rhot']['interp_1']((_lgrho, _lgt, _y, _z, _frock))

    # --- sp: logt_sp, logrho_sp
    def get_logt_sp(self, _s, _lgp, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['sp']['interp_0']((_s, _lgp, _y, _z, _frock))

    def get_logrho_sp(self, _s, _lgp, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['sp']['interp_1']((_s, _lgp, _y, _z, _frock))

    # --- srho: logp_srho, logt_srho
    def get_logp_srho(self, _s, _lgrho, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['srho']['interp_0']((_s, _lgrho, _y, _z, _frock))

    def get_logt_srho(self, _s, _lgrho, _y, _z, _frock,
                        ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        _y = _y if self.y_prime else _y / (1 - _z+1e-6)
        return self.data_combined['srho']['interp_1']((_s, _lgrho, _y, _z, _frock))

    def get_logu_srho(self, _s, _lgrho, _y, _z, _frock,
                        ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):

        logp = self.get_logp_srho(_s, _lgrho, _y, _z, _frock)
        logt = self.get_logt_srho(_s, _lgrho, _y, _z, _frock)

        return self.get_logu_pt(logp, logt, _y, _z, _frock)

    ################################################ Derivatives ################################################

    ########### Convection Derivatives ###########

    # Specific heat at constant pressure
    def get_cp_sp(self, _s, _lgp, _y, _z, _frock, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cp_pt(self, _lgp, _lgt, _y, _z, _frock, dt=0.1):

        s1 = self.get_s_pt(_lgp, _lgt - dt, _y, _z, _frock)
        s2 = self.get_s_pt(_lgp, _lgt + dt, _y, _z, _frock)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    # Specific heat at constant volume
    def get_cv_srho(self, _s, _lgrho, _y, _z, _frock, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        ds = _s*0.1 if ds is None else ds

        lgt1 = self.get_logt_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (2 * ds / erg_to_kbbar) / ((lgt2 - lgt1) * log10_to_loge)

    def get_cv_rhot(self, _lgrho, _lgt, _y, _z, _frock, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        s1 = self.get_s_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        s2 = self.get_s_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)

        return (s2 - s1) / (2 * dt * log10_to_loge)

    # Adiabatic temperature gradient
    def get_nabla_ad(self, _s, _lgp, _y, _z, _frock, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}

        lgt1 = self.get_logt_sp(_s, _lgp - dp, _y, _z, _frock, **kwargs)
        lgt2 = self.get_logt_sp(_s, _lgp + dp, _y, _z, _frock, **kwargs)
        return (lgt2 - lgt1)/(2 * dp)

    def get_dpdt_rhot_rhoy(self, _lgrho, _lgt, _y, _z, _frock, dt=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dt = _lgt*0.1 if dt is None else dt
        p1 = 10**self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        p2 = 10**self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)
        return (p2 - p1)/(2 * dt)

    # DS/DX|_P, T - DERIVATIVES NECESSARY FOR THE SCHWARZSCHILD CONDITION
    def get_dsdy_pt(self, _lgp, _lgt, _y, _z, _frock, dy=0.1):
        dy = _y*0.1 if dy is None else dy
        s1 = self.get_s_pt(_lgp, _lgt, _y - dy, _z, _frock)
        s2 = self.get_s_pt(_lgp, _lgt, _y + dy, _z, _frock)
        return (s2 - s1)/(2 * dy)

    def get_dsdz_pt(self, _lgp, _lgt, _y, _z, _frock, dz=0.1):
        dz = _z*0.1 if dz is None else dz
        s1 = self.get_s_pt(_lgp, _lgt, _y, _z - dz,  _frock)
        s2 = self.get_s_pt(_lgp, _lgt, _y, _z + dz,  _frock)
        return (s2 - s1)/(2 * dz)

    def get_gamma1(self, _s, _lgp, _y, _z, _frock, dp=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # dlnP/dlnrho_S, Y, Z = dlogP/dlogrho_S, Y, Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho1 = self.get_logrho_sp(_s, _lgp - dp, _y, _z, _frock, **kwargs)
        lgrho2 = self.get_logrho_sp(_s, _lgp + dp, _y, _z, _frock, **kwargs)
        return (2*dp)/(lgrho2 - lgrho1)

    # Brunt coefficient when computing in drho space
    def get_dlogrho_ds_py(self, _s, _lgp, _y, _z, _frock, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgrho2 = self.get_logrho_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)
        lgrho1 = self.get_logrho_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        return ((lgrho2 - lgrho1) * log10_to_loge) / (2 * ds / erg_to_kbbar)

    # Chi_T/Chi_rho
    # aka "delta" in MLT flux
    def get_dlogrho_dlogt_py(self, _lgp, _lgt, _y, _z, _frock, dt=1e-2):

        lgrho1 = self.get_logrho_pt(_lgp, _lgt - dt, _y, _z, _frock)
        lgrho2 = self.get_logrho_pt(_lgp, _lgt + dt, _y, _z, _frock)

        return (lgrho2 - lgrho1)/(2 * dt)

    def get_dlogp_dy_rhot(self, _lgrho, _lgt,  _y, _z, _frock, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Y
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt, _y - dy, _z, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt, _y + dy, _z, _frock, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dy)

    def get_dlogp_dz_rhot(self, _lgrho, _lgt,  _y, _z, _frock, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_Z
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt, _y, _z - dz, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt, _y, _z + dz, _frock, **kwargs)

        return ((lgp2 - lgp1) * log10_to_loge)/(2 * dz)

    def get_dlogp_dlogt_rhoy_rhot(self, _lgrho, _lgt,  _y, _z, _frock, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        # Chi_T
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        lgp1 = self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _frock, **kwargs)
        lgp2 = self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _frock, **kwargs)

        return (lgp2 - lgp1)/(2 * dt)

    # Chi_Y/Chi_T
    def get_dlogt_dy_rhop_rhot(self, _lgrho, _lgt,  _y, _z, _frock, dy=0.1, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Y = self.get_dlogp_dy_rhot(_lgrho, _lgt,  _y, _z, _frock, dy=dy, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, _frock, dt=dt, **kwargs)

        return Chi_Y/Chi_T

    # Chi_Z/Chi_T
    def get_dlogt_dz_rhop_rhot(self, _lgrho, _lgt,  _y, _z, _frock, dz=0.1, dt=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        Chi_Z = self.get_dlogp_dz_rhot(_lgrho, _lgt,  _y, _z, _frock, dz=dz, **kwargs)
        Chi_T = self.get_dlogp_dlogt_rhoy_rhot(_lgrho, _lgt,  _y, _z, _frock, dt=dt, **kwargs)

        return Chi_Z/Chi_T

    #### Triple Product Rule Derivatives ###*


    def get_dpds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        ds = _s*0.1 if ds is None else ds
        p1 = 10**self.get_logp_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (p2 - p1) / (2 * ds / erg_to_kbbar)

    def get_dpdy_srho(self, _s, _lgrho, _y, _z, _frock, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y - dy, _z, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y + dy, _z, _frock, **kwargs)

        return (p2 - p1) / (2 * dy)


    def get_dpdz_srho(self, _s, _lgrho, _y, _z, _frock, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (p2 - p1) / (2 * dz)

    ########### Triple product rule dsdx_rhop version ###########

    # DS/DX|_rho, P - DERIVATIVES NECESSARY FOR THE LEDOUX CONDITION
    def get_dsdy_rhop_srho(self, _s, _lgrho, _y, _z, _frock, ds=0.1, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdy_srho = self.get_dpdy_srho(_s, _lgrho, _y, _z, _frock, dy=dy, **kwargs)

        #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdy_rhopy = -dpdy_srho/dpds_rhoy_srho # triple product rule

        return dsdy_rhopy


    def get_dsdz_rhop_srho(self, _s, _lgrho, _y, _z, _frock, ds=0.1, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        #dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kwargs)
        #dPdY|{S, rho, Y}:
        dpdz_srho = self.get_dpdz_srho(_s, _lgrho, _y, _z, _frock, dz=dz, **kwargs)

        #dSdZ|{rho, P, Z} = -dPdZ|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdz_rhopy = -dpdz_srho/dpds_rhoy_srho # triple product rule

        return dsdz_rhopy


    ########### Chemical Potential Terms ###########

    def get_dudy_srho(self, _s, _lgrho, _y, _z, _frock, dy=0.1, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_p_guess': arr_p_guess, 'arr_t_guess': arr_p_guess, 'method': method, 'tab':tab}
        dy = _y*0.1 if dy is None else dy
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y - dy, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y + dy, _z, _frock, **kwargs)

        return (u2 - u1)/(2 * dy)

    def get_dudz_srho(self, _s, _lgrho, _y, _z, _frock, dz=0.1, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_p_guess': arr_p_guess, 'arr_t_guess': arr_p_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (u2 - u1)/(2 * dz)

    ########### Conductive Flux Terms ###########

    def get_dtdy_srho(self, _s, _lgrho, _y, _z, _frock, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y - dy, _z, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y + dy, _z, _frock, **kwargs)

        return (t2 - t1)/(2 * dy)

    def get_dtdz_srho(self, _s, _lgrho, _y, _z, _frock, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y, _z - dz, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y, _z + dz, _frock, **kwargs)

        return (t2 - t1)/(2 * dz)

    ########## Thermodynamic Consistency Test ###########

    # du/ds_(rho, Y) = T
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock, ds=1e-2, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        u1 = 10**self.get_logu_srho(_s - ds, _lgrho, _y, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s + ds, _lgrho, _y, _z, _frock, **kwargs)

        return (u2 - u1)/(2 * ds / erg_to_kbbar)

    # -du/dV_(S, Y) = P
    def get_duds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock, drho=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        R0 = 10 **_lgrho
        R1 = R0*(1-drho)
        R2 = R0*(1+drho)

        u1 = 10**self.get_logu_srho(_s, np.log10(R1), _y, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, np.log10(R2), _y, _z, _frock, **kwargs)

        return (u2 - u1)/((1/R1) - (1/R2))

    ########## Atmospheric update derivative ###########

    def get_dtds_sp(self, _s, _lgp, _y, _z, _frock, ds=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        t1 = 10**self.get_logt_sp(_s - ds, _lgp, _y, _z, _frock, **kwargs)
        t2 = 10**self.get_logt_sp(_s + ds, _lgp, _y, _z, _frock, **kwargs)

        return (t2 - t1)/(2 * ds / erg_to_kbbar)


class total_eos(mixtures):

    def __init__(self,
                 hhe_eos = 'cd',
                 z_eos = 'total_mixture',
                 hg: bool = False,
                 y_prime: bool = False,
                 interp_method: str = 'linear',
                 pt_only: bool = False, # for new inversions PT is calculated first to get Rho, T and S, P later
                 srho_table: bool = True, # To obtain S, Rho table, we need Rho, T and S, P tables first or perform a double inversion
                 new_z_mix: bool = True,
                 smooth_hhe: bool = False
                    ):

        super().__init__(hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_prime=y_prime, interp_method=interp_method, new_z_mix=new_z_mix, smooth_hhe=smooth_hhe)

        self.hhe_eos = hhe_eos
        self.z_eos = z_eos
        self.y_prime = y_prime
        self.hg = hg
        self.interp_method = interp_method
        self.table_types  = ['pt', 'rhot', 'sp', 'srho']  # or add 'rhop' if needed
        self.interp_method = interp_method

        self.pt_data = np.load('eos/total_mixture_eos/merged_eos_pt.npz')

        # RGI interpolation functions

        ####### P, T ####### tables
        rgi_args = {'method': self.interp_method, 'bounds_error': False, 'fill_value': None}
        # 1-D independent grids (P, T)
        self.logpvals = self.pt_data['logpvals'] # Units: log10 dyn/cm^2
        self.logtvals = self.pt_data['logtvals'] # log10 K
        self.yvals_pt = self.pt_data['yvals'] # mass fraction -- yprime
        self.zvals_pt = self.pt_data['zvals'] # mass fraction
        self.zmvals_pt = self.pt_data['zmvals']
        self.zavals_pt = self.pt_data['zavals']
        #self.zrvals_pt = self.pt_data['zrvals']

        # 7-D dependent grids (P, T)
        self.s_pt_tab = self.pt_data['s'] # erg/g/K
        self.logrho_pt_tab = self.pt_data['logrho'] # log10 g/cc
        self.u_pt_tab = self.pt_data['u'] # log10 erg/g

        self.s_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt, self.zmvals_pt, self.zavals_pt,
                                    #self.zrvals_pt
                                    ),
                                self.s_pt_tab, **rgi_args)
        self.logrho_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt, self.zmvals_pt, self.zavals_pt,
                                    #self.zrvals_pt
                                    ),
                                self.logrho_pt_tab, **rgi_args)
        self.u_pt_rgi = RGI((self.logpvals, self.logtvals, self.yvals_pt, self.zvals_pt, self.zmvals_pt, self.zavals_pt,
                                #self.zrvals_pt
                                ),
                                self.u_pt_tab, **rgi_args)

        if not pt_only:
            # RGI interpolation functions

            ####### Rho, T ####### tables
            # self.rhot_data = np.load('eos/total_mixture_eos/merged_eos_rhot.npz'))
            # self.logrhovals = self.rhot_data['logrhovals'] # log10 g/cc
            # self.logtvals_rhot = self.rhot_data['logtvals'] # log10 K
            # self.yvals_rhot = self.rhot_data['yvals'] # mass fraction -- yprime
            # self.zvals_rhot = self.rhot_data['zvals'] # mass fraction
            # self.zmvals_rhot = self.rhot_data['zmvals']
            # self.zavals_rhot = self.rhot_data['zavals']
            ## self.zrvals_rhot = self.rhot_data['zrvals']

            # # 7-D dependent grids (Rho, T)-- S(rho, T) can be calculated with S(P(rho, T), T)
            # self.logp_rhot_tab = self.rhot_data['logP']

            # self.logp_rhot_rgi = RGI((self.logrhovals, self.logtvals_rhot, self.yvals_rhot, self.zvals_rhot,
            #                         self.zmvals_rhot, self.zavals_rhot,
            #                         self.logp_rhot_tab, **rgi_args)

            ####### S, P ####### tables
            self.sp_data = np.load('eos/total_mixture_eos/merged_eos_sp.npz')
            self.svals_sp = self.sp_data['svals'] # erg/g/K
            self.logpvals_sp = self.sp_data['logpvals'] # log10 dyn/cm^2
            self.yvals_sp = self.sp_data['yvals'] # mass fraction -- yprime
            self.zvals_sp = self.sp_data['zvals'] # mass fraction
            self.zmvals_sp = self.sp_data['zmvals']
            self.zavals_sp = self.sp_data['zavals']
            #self.zrvals_sp = self.sp_data['zrvals']

            # 7-D dependent grids (S, P)-- Rho(S, P) can be calculated with Rho(P, T(S, P))
            self.logt_sp_tab = self.sp_data['logT'] # log10 K

            self.logt_sp_rgi = RGI((self.svals_sp, self.logpvals_sp, self.yvals_sp, self.zvals_sp,
                                    self.zmvals_sp, self.zavals_sp),
                                    self.logt_sp_tab, **rgi_args)

            if srho_table:
                # S, Rho tables
                self.srho_data = np.load('eos/total_mixture_eos/merged_eos_srho.npz')
                self.svals_srho = self.srho_data['svals']
                self.logrhovals_srho = self.srho_data['logrhovals']
                self.yvals_srho = self.srho_data['yvals'] # mass fraction -- yprime
                self.zvals_srho = self.srho_data['zvals']
                self.zmvals_srho = self.srho_data['zmvals']
                self.zavals_srho = self.srho_data['zavals']

                # 7-D dependent grids (S, Rho)-- T(S, Rho) can be calculated with T(S, P(S, Rho))
                self.logp_srho_tab = self.srho_data['logP'] # log10 dyn/cm^2

                self.logp_srho_rgi = RGI((self.svals_srho, self.logrhovals_srho, self.yvals_srho, self.zvals_srho,
                                        self.zmvals_srho, self.zavals_srho, self.zrvals_srho),
                                        self.logp_srho_tab, **rgi_args)

    ######## P, T ####### getters
    def get_s_pt(self, _lgp, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            return self.s_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            #_za = _z / (1 - _zr+1e-6)
            return self.s_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))


    def get_logrho_pt(self, _lgp, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            return self.logrho_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            #_za = _z / (1 - _zr+1e-6)
            return self.logrho_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))

    def get_u_pt(self, _lgp, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            return self.u_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            #_za = _z / (1 - _zr+1e-6)
            return self.u_pt_rgi(( _lgp, _lgt, _y, _z, _zm, _za))

    ######## Rho, T ####### getters
    def get_logp_rhot(self, _lgrho, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            return self.logp_rhot_rgi(( _lgrho, _lgt, _y, _z, _zm, _za))
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zr+1e-6)
            return self.logp_rhot_rgi(( _lgrho, _lgt, _y, _z, _zm, _za))

    def get_s_rhot(self, _lgrho, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            logp = self.get_logp_rhot( _lgrho, _lgt, _y, _z, _zm, _za)
            return self.get_s_pt(logp, _lgt, _y, _z, _zm, _za)
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zr+1e-6)

            logp = self.get_logp_rhot( _lgrho, _lgt, _y, _z, _zm, _za)
            return self.get_s_pt(logp, _lgt, _y, _z, _zm, _za)

    def get_u_rhot(self, _lgrho, _lgt, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            logp = self.get_logp_rhot( _lgrho, _lgt, _y, _z, _zm, _za)
            return self.get_u_pt(logp, _lgt, _y, _z, _zm, _za)
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zr+1e-6)

            logp = self.get_logp_rhot( _lgrho, _lgt, _y, _z, _zm, _za)
            return self.get_u_pt(logp, _lgt, _y, _z, _zm, _za)

    ######## S, P ####### getters
    def get_logt_sp(self, _s, _lgp, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            return self.logt_sp_rgi(( _s, _lgp, _y, _z, _zm, _za))
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zr+1e-6)
            return self.logt_sp_rgi(( _s, _lgp, _y, _z, _zm, _za))

    def get_logrho_sp(self, _s, _lgp, _y, _z, _zm, _za, _zr=0.0):
        if self.y_prime:
            logt = self.get_logt_sp( _s, _lgp, _y, _z, _zm, _za)
            return self.get_logrho_pt(_lgp, logt, _y, _z, _zm, _za)
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zr+1e-6)

            logt = self.get_logt_sp( _s, _lgp, _y, _z, _zm, _za)
            return self.get_logrho_pt(_lgp, logt, _y, _z, _zm, _za)

    def get_logu_sp(self, _s, _lgp, _y, _z, _zm, _za):
        if self.y_prime:
            logt = self.get_logt_sp( _s, _lgp, _y, _z, _zm, _za)
            return self.get_u_pt(_lgp, logt, _y, _z, _zm, _za)
        else:
            _y = _y / (1 - _z+1e-6)
            _zm = _z / (1 - _za+1e-6)
            _za = _z / (1 - _zm+1e-6)

            logt = self.get_logt_sp( _s, _lgp, _y, _z, _zm, _za)
            return self.get_u_pt(_lgp, logt, _y, _z, _zm, _za)

   ### Inversion Functions ###

    def get_logt_sp_inv(self, _s, _lgp, _y, _z, _zm, _za, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the temperature given entropy, pressure, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgp (array_like): Log10 pressure values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _s = np.atleast_1d(_s)
        _lgp = np.atleast_1d(_lgp)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        _zr = np.atleast_1d(_zr)
        _zfe = np.atleast_1d(_zfe)

        # _y = _y if self.y_prime else _y * (1 - _z)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgp, _y, _z, _zm, _za, _zr, _zfe = np.broadcast_arrays(_s, _lgp, _y, _z, _zm, _za, _zr, _zfe)

        if ideal_guess:
            guess = ideal_xy.get_t_sp(_s, _lgp, _y)
        else:
            if arr_guess is None:
                raise ValueError("arr_guess must be provided when ideal_guess is False.")
            guess = arr_guess

    # Define a function to compute root and capture convergence
        def root_func(s_i, lgp_i, y_i, z_i, zm_i, za_i, zr_i, zfe_i, guess_i):
            def err(_lgt):
                # Error function for logt(S, logp)
                #s_test = self.get_s_pt(lgp_i, _lgt, y_i, z_i, zm_i, za_i) * erg_to_kbbar
                s_test = self.get_s_pt_val(lgp_i, _lgt, y_i, z_i, zm_i, za_i) * erg_to_kbbar
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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

                except OverflowError:
                    print('Failed at s={}, logp={}, y={}, z={}'.format(s_i, lgp_i, y_i, z_i, zm_i, za_i, zr_i))
                    raise
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgp, _y, _z, _zm, _za, _zr, _zfe, guess)

        return temperatures, converged

    def get_logrho_sp_inv(self, _s, _lgp, _y, _z, _zm, _za, _zr, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):
        logt, conv = self.get_logt_sp_inv( _s, _lgp, _y, _z, _zm, _za, _zr, _zfe=0.0, ideal_guess=ideal_guess, arr_guess=arr_guess, method=method)
        return self.get_logrho_pt(_lgp, logt, _y, _z, _zm, _za, _zr)

    def get_logp_rhot_inv(self, _lgrho, _lgt, _y, _z, _zm, _za, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

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
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        _zr = np.atleast_1d(_zr)
        _zfe = np.atleast_1d(_zfe)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)
        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _lgrho, _lgt, _y, _z, _zm, _za, _zr, _zfe = np.broadcast_arrays(_lgrho, _lgt, _y, _z, _zm, _za, _zr, _zfe)

        if ideal_guess:
            guess = ideal_xy.get_p_rhot(_lgrho, _lgt, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
       # Define a function to compute root and capture convergence
        def root_func(lgrho_i, lgt_i, y_i, z_i, zm_i, za_i, zr_i, zfe_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                # _y_call = y_i if self.y_prime else y_i / (1 - z_i)
                logrho_test = self.get_logrho_pt(_lgp, lgt_i, y_i, z_i, zm_i, za_i)
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
        pressure, converged = vectorized_root_func(_lgrho, _lgt, _y, _z, _zm, _za, _zr, _zfe, guess)

        return pressure, converged


    def get_logp_srho_inv(self, _s, _lgrho, _y, _z, _zm, _za, _zr=0.0, _zfe=0.0, ideal_guess=True, arr_guess=None, method='newton_brentq'):

        """
        Compute the pressure given entropy, density, helium abundance, and metallicity.

        Parameters:
            _s (array_like): Entropy values.
            _lgrho (array_like): Log10 density values.
            _y (array_like): Helium mass fraction values.
            _z (array_like): Heavy metal mass fraction values.
            ideal_guess (bool, optional): If True, use the ideal EOS for the initial guess (default is True).
            logt_guess (array_like, optional): User-provided initial guess for log temperature when `ideal_guess` is False.

        Returns:
            ndarray: Computed temperature values.
        """

        _s = np.atleast_1d(_s)
        _lgrho = np.atleast_1d(_lgrho)
        _y = np.atleast_1d(_y)
        _z = np.atleast_1d(_z)
        _zm = np.atleast_1d(_zm)
        _za = np.atleast_1d(_za)
        _zr = np.atleast_1d(_zr)
        _zfe = np.atleast_1d(_zfe)

        #_y = _y if self.y_prime else _y / (1 - _z+1e-6)

        # Ensure inputs are numpy arrays and broadcasted to the same shape
        _s, _lgrho, _y, _z, _zm, _za, _zr, _zfe = np.broadcast_arrays(_s, _lgrho, _y, _z, _zm, _za, _zr, _zfe)

        if ideal_guess:
            guess = ideal_xy.get_p_srho(_s, _lgrho, _y)
        else:
            if arr_guess is None:
                raise ValueError("logt_guess must be provided when ideal_guess is False.")
            guess = arr_guess
        # Define a function to compute root and capture convergence
        def root_func(s_i, lgrho_i, y_i, z_i, zm_i, za_i, zr_i, zfe_i, guess_i):
            def err(_lgp):
                # Error function for logt(S, logp)
                logrho_test = self.get_logrho_sp(s_i, _lgp, y_i, z_i, zm_i, za_i)
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

                # If no valid interval is found after max_attempts
                return np.nan, False

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
            else:
                raise ValueError("Invalid method specified. Use 'root', 'newton', or 'brentq'.")
        # Vectorize the root_func
        vectorized_root_func = np.vectorize(root_func, otypes=[np.float64, bool])

        # Apply the vectorized function
        temperatures, converged = vectorized_root_func(_s, _lgrho, _y, _z, _zm, _za, _zr, _zfe, guess)

        return temperatures, converged


    def glitch_correct_1d(self, y, window=5, n_sigmas=3):
        """
        1) Identify outliers in y with a hampel filter, window +/- 'window'.
        2) Mark them as nan in a copy.
        3) Interpolate the nans.
        Returns y_corrected.
        """
        y_copy = np.asarray(y, float).copy()
        n = len(y_copy)
        outlier_inds = []
        for i in range(n):
            w_start = max(0, i-window)
            w_end   = min(n, i+window+1)
            local_y = y_copy[w_start:w_end]
            med = np.median(local_y)
            mad = 1.4826 * np.median(np.abs(local_y - med)) or 1e-6
            if abs(y_copy[i] - med) > n_sigmas*mad:
                outlier_inds.append(i)

        # set them to nan
        y_copy[outlier_inds] = np.nan

        # now interpolate the nan
        y_fixed = self.fill_nans_1d(y_copy, kind='linear')
        return y_fixed

    def fill_nans_1d(self, arr, kind='linear'):
        """
        Fill NaNs in a 1D array by interpolation of any specified 'kind'
        recognized by scipy.interpolate.interp1d:
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.

        Parameters
        ----------
        arr : 1D numpy array
            Array that may contain NaNs.
        kind : str
            Interpolation type to pass to interp1d (e.g. 'linear', 'quadratic').

        Returns
        -------
        arr_filled : 1D numpy array
            A copy of arr with NaNs replaced by interpolation of the specified kind.
            If there are not enough valid points to interpolate (e.g., all NaN),
            we simply return arr unchanged.
        """

        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError("This function only handles 1D arrays.")

        x = np.arange(len(arr))
        valid_mask = ~np.isnan(arr)

        # If everything is NaN, or only 1 valid point, we can’t do a real polynomial interpolation.
        if np.count_nonzero(valid_mask) < 2:
            return arr  # or decide on a default fill approach

        # Build the interpolator.  'fill_value="extrapolate"' lets us fill beyond the data range.
        f = interp1d(
            x[valid_mask],
            arr[valid_mask],
            kind=kind,
            fill_value="extrapolate"
        )

        # Create a copy for the result
        arr_filled = arr.copy()
        # Where arr is NaN, replace with the interpolation
        arr_filled[~valid_mask] = f(x[~valid_mask])

        return arr_filled
