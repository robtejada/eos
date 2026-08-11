import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import eos.const as const
import pdb
import pandas as pd
from tqdm import tqdm
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
      - 'iron'     : Fe combined EOS (Gonzalez-Cataldo & Militzer 2023), P-T basis only

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
        'iron':    55.845,   # Fe
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
        elif species == 'iron':
            self._load_iron_gonzalez(smooth_z)
        else:
            raise ValueError(f"Unknown z_eos species '{species}'. "
                             f"Use 'water', 'methane', 'ammonia', 'mg2sio4', or 'iron'.")

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
    # Iron (Gonzalez-Cataldo & Militzer 2023) loader
    # -----------------------------------------------------------------
    def _load_iron_gonzalez(self, smooth_z):
        """Load pre-computed combined solid+liquid iron EOS table.

        Table: gonzalez_iron_eos/gonzalez_iron_combined_pt.npz
        Keys: p_vals (GPa), t_vals (K), rho_grid (g/cm³),
              s_grid (erg/g/K), u_grid (erg/g) — all already CGS.
        """
        data = np.load(f'{CURR_DIR}/gonzalez_iron_eos/gonzalez_iron_combined_pt.npz')

        P_GPa = np.asarray(data['p_vals'], dtype=float)
        T_K = np.asarray(data['t_vals'], dtype=float)
        n_P, n_T = P_GPa.size, T_K.size

        rho_raw = np.asarray(data['rho_grid'], dtype=float)
        s_raw = np.asarray(data['s_grid'], dtype=float)
        u_raw = np.asarray(data['u_grid'], dtype=float)

        # Table is stored as (n_P, n_T) by construction.
        # Skip the n_T==n_P shape ambiguity check — axis order is known.

        # Convert axes to log CGS
        logp_cgs = np.log10(P_GPa * self._GPa_to_dyncm2)
        logt_1d = np.log10(T_K)

        # Convert rho, S to log form (handle negative/zero from extrapolation)
        with np.errstate(invalid='ignore', divide='ignore'):
            logrho_raw = np.where(rho_raw > 0, np.log10(rho_raw), np.nan)
            logs_raw = np.where(s_raw > 0, np.log10(s_raw), np.nan)

        # Store — grid is (n_P, n_T), axes (logP, logT), same as mg2sio4
        self.logpvals_pt = logp_cgs
        self.logtvals_pt = logt_1d
        self.logrho_pt = logrho_raw
        self.logs_pt = logs_raw

        # U is stored in LINEAR form (erg/g) — iron U can be negative
        # due to the DFT reference state, so log10 is not appropriate.
        self.u_pt = u_raw

        # RGI axes: (logP, logT) like AQUA / mg2sio4
        rgi_kw = dict(method='linear', bounds_error=False, fill_value=None)
        self.logrho_pt_rgi = RGI((logp_cgs, logt_1d), self.logrho_pt, **rgi_kw)
        self.logs_pt_rgi = RGI((logp_cgs, logt_1d), self.logs_pt, **rgi_kw)
        self.u_pt_rgi = RGI((logp_cgs, logt_1d), self.u_pt, **rgi_kw)

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

            smoothed_full = gaussian_filter(filled, sigma=[1.0, 1.0],
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
        if self.species in ('water', 'mg2sio4', 'iron'):
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
        if self.species == 'iron':
            raise AttributeError(
                "Iron U is stored in linear form (can be negative). "
                "Use get_u_pt() instead of get_logu_pt().")
        return self._interpolate_pt(self.logu_pt_rgi, lgp, lgt)

    def get_u_pt(self, lgp, lgt):
        """Internal energy in linear form (erg/g).

        Works for all species.  For iron the RGI stores linear U
        directly (allows negative values); for other species it
        exponentiates the log10 table.
        """
        if self.species == 'iron':
            return self._interpolate_pt(self.u_pt_rgi, lgp, lgt)
        return 10.0 ** self._interpolate_pt(self.logu_pt_rgi, lgp, lgt)

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
class z_eos_val_mixtures:
    """Volume Addition Law (VAL) mixer for up to four Z species.

    A deliberately simple heavy-element mixer.  Given a set of
    ``z_eos`` species (water, methane, ammonia, rock), it returns the
    metal-mixture density, entropy and internal energy as functions of
    (P, T, metal sub-fractions).

    Unlike ``val_mixtures`` (which mixes H-He-Z with both ideal and
    HG23 non-ideal corrections), this class:
      - mixes Z species only,
      - applies the Volume Addition Law for density,
      - mass-weights the per-species S and U,
      - optionally adds the *ideal* entropy of mixing for the
        standalone ``get_s_pt`` (there is no non-ideal information
        available for Z-Z mixtures),
      - fills the AQUA superionic NaN corner so that interpolation and
        inversion stay finite and well posed (see ``fill_z_nans``).

    ``val_mixtures`` delegates its metal mixing (``get_logrho_z``,
    ``get_s_z``, ``get_u_z``) to this class, which is why the same
    nested sub-fraction convention is used.

    Composition convention (same nested convention as ``val_mixtures``)
    ------------------------------------------------------------------
        _zm  : methane fraction within the metal budget
        _za  : ammonia fraction in the remainder after methane
        _zr  : rock fraction in the remainder after ammonia
        Physical mass fractions (within Z):
            f_water   = (1 - _zm) * (1 - _za) * (1 - _zr)
            f_methane = _zm * (1 - _za) * (1 - _zr)
            f_ammonia = _za * (1 - _zr)
            f_rock    = _zr

    For the canonical 50/50 water/rock mixture
    (``species_list=['water_revised', 'mg2sio4']``) call the methods
    with ``_zr = f_rock`` (and ``_zm = _za = 0``).

    Units (CGS, consistent with ``z_eos`` / ``val_mixtures``)
    --------------------------------------------------------
        logP   : log10(dyn/cm^2)
        logT   : log10(K)
        logrho : log10(g/cm^3)
        S      : erg/(g.K)   (linear; ``get_logt_sp`` takes kb/baryon)
        U      : erg/g       (linear)
    """

    # Molecular weights for ideal entropy of mixing
    _m_water   = 18.015
    _m_methane = 16.04
    _m_ammonia = 17.031
    _m_rock    = 140.6935   # Mg2SiO4 (forsterite)
    _m_iron    = 55.845

    # Minimum strictly-increasing step (dex of log10 S per grid cell)
    # imposed on filled corner cells so T(S,P) always brackets a root.
    _MONO_EPS = 1e-3

    def __init__(self, species_list=None, smooth_z=False, fill_z_nans=True):
        """
        Parameters
        ----------
        species_list : list of str or None
            Which Z species to load.  Default: all four metals
            ``['water_revised', 'methane', 'ammonia', 'mg2sio4']``
            (same default as ``val_mixtures``).  Recognised names:
              'water_revised' / 'aqua_revised'  -> revised AQUA water
              'water' / 'aqua' / 'aqua_original' -> original AQUA water
              'mg2sio4' / 'rock'                -> Mg2SiO4 forsterite
              'methane', 'ammonia', 'iron'      -> respective z_eos
            Each name maps to a canonical role key ('water', 'methane',
            'ammonia', 'mg2sio4', 'iron') used to index ``self.z``.
            For a simple water/rock mixture pass
            ``['water_revised', 'mg2sio4']``.
        smooth_z : bool
            Passed through to each underlying ``z_eos`` table loader.
        fill_z_nans : bool
            If True (default), build a NaN-free entropy interpolator for
            any species whose ``logs_pt`` table has missing cells.  The
            revised AQUA water table flags entropy as unavailable (raw
            sentinel = -1) in the high-P / low-T superionic-and-solid
            corner (logP >~ 12.9), which would otherwise poison the
            mixture entropy (and its T(S,P) inversion) with NaNs.  The
            gap is filled by 2-D ``np.interp`` along P then T, then a
            strict-monotonization pass so S increases in T across the
            corner (smooth along isotherms, single-valued for inversion).
            These filled values are *extrapolated and not validated* —
            they exist only to keep the mixer finite and well behaved in
            a region where planetary interiors are hot anyway.  The
            shared ``z_eos`` instances are never modified; the filled
            interpolator lives inside this mixer.  Set False to recover
            the raw (NaN-bearing) behaviour.
        """
        if species_list is None:
            species_list = ['water_revised', 'methane', 'ammonia', 'mg2sio4']
        self.fill_z_nans = fill_z_nans

        # Z EOS instances keyed by canonical role
        # ('water', 'methane', 'ammonia', 'mg2sio4', 'iron').
        self.z = {}
        for name in species_list:
            key, eos_obj = self._make_z_eos(name, smooth_z)
            self.z[key] = eos_obj

        if len(self.z) < 1:
            raise ValueError("z_eos_val_mixtures needs at least one species.")

        # Per-species log10(S) accessor f(lgp, lgt).  Uses a NaN-filled
        # interpolator where the raw table has gaps (and fill_z_nans is
        # on); otherwise falls through to the species' get_logs_pt.
        self._logs_fn = {}
        for key, eos_obj in self.z.items():
            filled = self._build_filled_logs(eos_obj) if fill_z_nans else None
            self._logs_fn[key] = (filled if filled is not None
                                  else eos_obj.get_logs_pt)

    # -----------------------------------------------------------------
    # construction helper
    # -----------------------------------------------------------------
    def _make_z_eos(self, name, smooth_z):
        """Map a user-facing name to a (canonical_key, z_eos) pair."""
        key = name.lower()
        if key in ('water_revised', 'aqua_revised'):
            return 'water', z_eos(species='water', smooth_z=smooth_z,
                                  aqua_version='revised')
        if key in ('water', 'aqua', 'aqua_original'):
            return 'water', z_eos(species='water', smooth_z=smooth_z,
                                  aqua_version='original')
        if key in ('mg2sio4', 'rock', 'forsterite'):
            return 'mg2sio4', z_eos(species='mg2sio4', smooth_z=smooth_z)
        if key == 'methane':
            return 'methane', z_eos(species='methane', smooth_z=smooth_z)
        if key == 'ammonia':
            return 'ammonia', z_eos(species='ammonia', smooth_z=smooth_z)
        if key == 'iron':
            return 'iron', z_eos(species='iron', smooth_z=smooth_z)
        raise ValueError(f"Unknown z_eos species name '{name}'. Use one of "
                         "'water_revised', 'water', 'mg2sio4', 'methane', "
                         "'ammonia', 'iron'.")

    # Molecular weight for each canonical role (ideal entropy of mixing)
    @property
    def _mu_by_key(self):
        return {'water': self._m_water, 'methane': self._m_methane,
                'ammonia': self._m_ammonia, 'mg2sio4': self._m_rock,
                'iron': self._m_iron}

    # -----------------------------------------------------------------
    # NaN-filled entropy interpolator (AQUA superionic corner)
    # -----------------------------------------------------------------
    def _fill_grid_nans_2d(self, grid, t_axis):
        """Fill NaN cells in a 2-D log10(S) grid (returns a copy).

        Pass 1 interpolates/extrapolates along the **pressure** axis;
        Pass 2 mops up any remainder along T.  Both use ``np.interp``,
        which clamps to the nearest finite endpoint.

        Pressure-first is deliberate.  In the AQUA superionic corner the
        missing-data boundary runs along pressure: at a fixed low T,
        entropy is tabulated up to logP ~ 12.8 and absent above.  Filling
        along P therefore extends each isotherm's last finite value
        flatly to higher P, which is (a) continuous across the boundary
        and (b) monotone along every isotherm.  Filling T-first instead
        clamps each P-row to its own low-T boundary; because those
        boundaries sit at very different temperatures (logT ~ 2.0 at
        logP=12.9 but ~3.3 at logP=13.5) with order-of-magnitude
        different S, that produces large non-monotone jumps along
        isotherms (up to ~2 dex) that break the S<->T inversion.

        After filling, a monotonization pass forces S to be *strictly*
        increasing in T within the *filled cells only* (real cells are
        never touched), by at least ``_MONO_EPS`` dex of log10(S) per
        grid cell.  The AQUA boundary entropy is itself non-monotone in T
        across the superionic transition, so the pressure-clamped corner
        inherits S(T) dips (and flats) that would give T(S,P) multiple
        roots or no bracket at all (the deep corner carries essentially
        no T information).  Pulling the filled cells down onto a strictly
        increasing profile — anchored at the real boundary above —
        guarantees a unique, finite, well-posed inversion in this
        otherwise unphysical region.  The imposed slope is tiny
        (~0.08 dex total across the block), so isotherms stay tame.
        """
        grid = np.asarray(grid, dtype=float)
        nan_mask = np.isnan(grid)
        out = grid.copy()
        p_axis = 1 - t_axis

        def _interp_line(line):
            bad = np.isnan(line)
            good = ~bad
            if bad.any() and good.sum() >= 2:
                line[bad] = np.interp(np.where(bad)[0],
                                      np.where(good)[0], line[good])
            return line

        # Pass 1: along P (clamps the high-P edge; continuous across the
        # P-aligned data boundary, monotone along isotherms)
        out = np.apply_along_axis(_interp_line, p_axis, out)
        # Pass 2: along T for any still-NaN line (safety net)
        if np.isnan(out).any():
            out = np.apply_along_axis(_interp_line, t_axis, out)

        # Pass 3: enforce S *strictly* increasing in T over the
        # originally-NaN cells, so T(S,P) is single-valued and always
        # brackets a root there.  Work in (P, T) view.
        work = out if t_axis == 1 else out.T          # view, shape (nP, nT)
        mask = nan_mask if t_axis == 1 else nan_mask.T
        for ip in range(work.shape[0]):
            line = work[ip]
            ml = mask[ip]
            for i in range(line.size - 2, -1, -1):    # high T -> low T
                if ml[i]:
                    cap = line[i + 1] - self._MONO_EPS
                    if line[i] > cap:
                        line[i] = cap
        return out

    def _build_filled_logs(self, eos_obj):
        """Build a NaN-free log10(S) accessor ``f(lgp, lgt)`` for one
        sub-EOS, or return None if its ``logs_pt`` grid has no NaNs.

        The new interpolator mirrors the sub-EOS's own axis convention
        (water/mg2sio4/iron store (logP, logT); methane/ammonia store
        (logT, logP)) and leaves the shared ``z_eos`` untouched.
        """
        grid = np.asarray(getattr(eos_obj, 'logs_pt'), dtype=float)
        if not np.isnan(grid).any():
            return None

        lp = eos_obj.logpvals_pt
        lt = eos_obj.logtvals_pt
        if eos_obj.species in ('water', 'mg2sio4', 'iron'):
            axes, t_axis = (lp, lt), 1          # grid is (n_p, n_t)
            reorder = lambda a, b: (a, b)       # query (lgp, lgt)
        else:
            axes, t_axis = (lt, lp), 0          # grid is (n_t, n_p)
            reorder = lambda a, b: (b, a)       # query (lgt, lgp)

        filled = self._fill_grid_nans_2d(grid, t_axis)
        rgi = RGI(axes, filled, method='linear',
                  bounds_error=False, fill_value=None)

        def _accessor(_lgp, _lgt):
            a, b = reorder(_lgp, _lgt)
            pts = np.column_stack([np.atleast_1d(a), np.atleast_1d(b)])
            res = rgi(pts)
            if np.isscalar(_lgp) and np.isscalar(_lgt):
                return res.item()
            return res

        return _accessor

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
        """Physical mass fractions **within** the metal budget Z.

        Same nested convention as ``val_mixtures._metal_fractions``.
        """
        f_water   = (1.0 - _zm) * (1.0 - _za) * (1.0 - _zr)
        f_methane = _zm * (1.0 - _za) * (1.0 - _zr)
        f_ammonia = _za * (1.0 - _zr)
        f_rock    = _zr
        return f_water, f_methane, f_ammonia, f_rock

    def _smix_ideal_z(self, f_w, f_m, f_a, f_r):
        """Ideal entropy of mixing  -Σ(x_i ln x_i) / q   [kb/baryon].

        Simplified, Z-only analogue of ``val_mixtures._smix_ideal``
        over the four metal species (water, methane, ammonia, rock).
        Returns kb/baryon; caller divides by ``erg_to_kbbar`` for
        erg/(g.K).
        """
        species = [(f_w, self._m_water), (f_m, self._m_methane),
                   (f_a, self._m_ammonia), (f_r, self._m_rock)]
        n_list = [np.where(np.asarray(f) > 0,
                           np.asarray(f, dtype=float) / mu, 0.0)
                  for f, mu in species]
        Ntot = sum(n_list)

        x_list = [n / Ntot for n in n_list]
        q = sum(mu * x for (_, mu), x in zip(species, x_list))

        s_id = -sum(self._guarded_xlogx(x) for x in x_list) / q
        return s_id

    # ordered (canonical_key, fraction-index) used by the metal loops
    _METAL_KEYS = ('water', 'methane', 'ammonia', 'mg2sio4')

    # =================================================================
    # metal mixing  (drop-in for val_mixtures.get_logrho_z / _s_z / _u_z)
    # =================================================================

    def get_logrho_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture density via VAL (log10 g/cm³)."""
        fracs = self._metal_fractions(_zm, _za, _zr)

        v_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)
        for key, f in zip(self._METAL_KEYS, fracs):
            if np.any(np.asarray(f) > 0) and key in self.z:
                rho_i = 10.0 ** self.z[key].get_logrho_pt(_lgp, _lgt)
                v_mix = v_mix + f / rho_i

        result = np.log10(1.0 / v_mix)
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return result.item()
        return result

    def _s_massweighted(self, _lgp, _lgt, fracs):
        """Mass-weighted metal entropy (erg/(g·K)), NaN-filled, NO mixing."""
        s_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)
        for key, f in zip(self._METAL_KEYS, fracs):
            if np.any(np.asarray(f) > 0) and key in self.z:
                s_mix = s_mix + f * 10.0 ** self._logs_fn[key](_lgp, _lgt)
        return s_mix

    def get_s_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture entropy, mass-weighted (erg/(g·K)).

        NOTE: does NOT include ideal entropy of mixing — that is added
        at the full H-He-Z level in ``val_mixtures.get_s_pt_val`` (and
        in this class's standalone ``get_s_pt``).  Uses the NaN-filled
        entropy accessor in the AQUA superionic corner.
        """
        s_mix = self._s_massweighted(_lgp, _lgt,
                                     self._metal_fractions(_zm, _za, _zr))
        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return s_mix.item()
        return s_mix

    def get_u_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture internal energy, mass-weighted (erg/g)."""
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)

        u_mix = np.zeros_like(np.atleast_1d(_lgp), dtype=float)
        for key, f in zip(self._METAL_KEYS, (f_w, f_m, f_a, f_r)):
            if np.any(np.asarray(f) > 0) and key in self.z:
                u_mix = u_mix + f * self.z[key].get_u_pt(_lgp, _lgt)
        if np.any(np.asarray(f_r) > 0) and 'iron' in self.z:
            u_mix = u_mix + f_r * self.z['iron'].get_u_pt(_lgp, _lgt)  # iron shares rock fraction for now

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return u_mix.item()
        return u_mix

    # =================================================================
    # standalone forward models (with ideal entropy of mixing)
    # =================================================================

    def get_logrho_pt(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Alias for ``get_logrho_z`` (density has no mixing term)."""
        return self.get_logrho_z(_lgp, _lgt, _zm, _za, _zr)

    def get_u_pt(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Alias for ``get_u_z`` (internal energy is mass-weighted)."""
        return self.get_u_z(_lgp, _lgt, _zm, _za, _zr)

    def get_s_pt(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Standalone Z-mixture entropy (erg/(g·K)).

        Mass-weighted per-species entropy **plus** the ideal entropy of
        mixing.  At an end-member (one fraction = 1) the mixing term
        vanishes and the result reduces to the pure species value.
        (``val_mixtures`` does not call this — it uses ``get_s_z`` and
        adds the full H-He-Z mixing itself.)
        """
        f_w, f_m, f_a, f_r = self._metal_fractions(_zm, _za, _zr)
        s_mix = self._s_massweighted(_lgp, _lgt, (f_w, f_m, f_a, f_r))

        # Ideal entropy of mixing (kb/baryon -> erg/(g.K))
        s_mix = s_mix + self._smix_ideal_z(f_w, f_m, f_a, f_r) / erg_to_kbbar

        if np.isscalar(_lgp) and np.isscalar(_lgt):
            return np.atleast_1d(s_mix).item()
        return s_mix

    # =================================================================
    # inversions  (reuse the self-contained z_eos._newton_1d_z solver)
    # =================================================================

    def _mu_mix(self, _zm, _za, _zr):
        """Mass-weighted molecular weight for ideal-gas initial guesses."""
        fracs = self._metal_fractions(_zm, _za, _zr)
        mus = (self._m_water, self._m_methane, self._m_ammonia, self._m_rock)
        return sum(float(np.mean(np.atleast_1d(f))) * mu
                   for f, mu in zip(fracs, mus))

    def _newton_solver(self):
        """The self-contained z_eos Newton/brentq solver (any species)."""
        return next(iter(self.z.values()))._newton_1d_z

    def get_logt_sp(self, _s_kb, _lgp, _zm=0.0, _za=0.0, _zr=0.0):
        """Temperature from (S, P, fractions) via Newton-Raphson.

        Inverts ``get_s_pt(P, T) = _s_kb`` for logT.  The target
        entropy ``_s_kb`` is in kb/baryon (consistent with
        ``z_eos.get_logt_sp``); it is converted internally to the
        erg/(g.K) used by ``get_s_pt``.
        """
        scalar = np.isscalar(_s_kb) and np.isscalar(_lgp)
        _s_kb = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgp = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _s_kb, _lgp = np.broadcast_arrays(_s_kb, _lgp)
        out = np.full_like(_s_kb, np.nan, dtype=float)

        lo_t, hi_t = 2.0, 7.0

        # Target entropy in erg/(g.K)
        s_target_cgs = _s_kb / erg_to_kbbar

        newton = self._newton_solver()
        ideal = ideal_eos.IdealEOS(self._mu_mix(_zm, _za, _zr))

        prev_sol = None
        for idx in np.ndindex(s_target_cgs.shape):
            s_i = float(s_target_cgs[idx])
            s_kb_i = float(_s_kb[idx])
            p_i = float(_lgp[idx])

            def err(lgt, _s=s_i, _p=p_i):
                return float(self.get_s_pt(_p, lgt, _zm, _za, _zr) - _s)

            if prev_sol is not None:
                guess = prev_sol
            else:
                guess = float(np.clip(ideal.get_t_sp(s_kb_i, p_i, 0.0),
                                      lo_t, hi_t))
            sol, ok = newton(err, guess, lo_t, hi_t)
            if np.isfinite(sol):
                out[idx] = sol
                prev_sol = sol

        return out.item() if scalar else out

    def get_logt_rhop(self, _lgrho, _lgp, _zm=0.0, _za=0.0, _zr=0.0):
        """Temperature from (rho, P, fractions) via Newton-Raphson.

        Inverts ``get_logrho_z(P, T) = _lgrho`` for logT.
        """
        scalar = np.isscalar(_lgrho) and np.isscalar(_lgp)
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgp = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _lgrho, _lgp = np.broadcast_arrays(_lgrho, _lgp)
        out = np.full_like(_lgrho, np.nan, dtype=float)

        lo_t, hi_t = 2.0, 7.0

        newton = self._newton_solver()
        ideal = ideal_eos.IdealEOS(self._mu_mix(_zm, _za, _zr))

        prev_sol = None
        for idx in np.ndindex(_lgrho.shape):
            rho_i = float(_lgrho[idx])
            p_i = float(_lgp[idx])

            def err(lgt, _rho=rho_i, _p=p_i):
                return float(self.get_logrho_z(_p, lgt, _zm, _za, _zr) - _rho)

            if prev_sol is not None:
                guess = prev_sol
            else:
                guess = float(np.clip(ideal.get_t_rhop(rho_i, p_i, 0.0),
                                      lo_t, hi_t))
            sol, ok = newton(err, guess, lo_t, hi_t)
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

        # Metal mixing is delegated to z_eos_val_mixtures, which loads the
        # Z EOS instances and adds NaN interpolation/extrapolation in the
        # AQUA superionic corner.  self.z is shared (same dict) so the
        # metal-mixing methods below read the same species objects.
        self.zmetal = z_eos_val_mixtures(species_list=species_list,
                                         smooth_z=smooth_z, fill_z_nans=True)
        self.z = self.zmetal.z

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
        """Metal-mixture density via VAL (log10 g/cm³).

        Delegated to ``z_eos_val_mixtures`` (which adds AQUA NaN-corner
        interpolation/extrapolation).
        """
        return self.zmetal.get_logrho_z(_lgp, _lgt, _zm, _za, _zr)

    def get_s_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture entropy, mass-weighted (erg/(g·K)).

        NOTE: does NOT include ideal entropy of mixing — that is added
        at the full H-He-Z level in get_s_pt_val.  Delegated to
        ``z_eos_val_mixtures`` (NaN-filled in the AQUA superionic corner).
        """
        return self.zmetal.get_s_z(_lgp, _lgt, _zm, _za, _zr)

    def get_u_z(self, _lgp, _lgt, _zm=0.0, _za=0.0, _zr=0.0):
        """Metal-mixture internal energy, mass-weighted (erg/g).

        Delegated to ``z_eos_val_mixtures``.
        """
        return self.zmetal.get_u_z(_lgp, _lgt, _zm, _za, _zr)

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


# =====================================================================
# Module-level worker scaffolding for parallel inversion-table builds
#
# multiprocessing on macOS uses spawn semantics: each worker re-imports
# this module and reconstructs the eos instance from kwargs.  The worker
# instance is held in a module global (_WORKER_EOS) so subsequent task
# dispatches re-use the same RGI tables instead of rebuilding them.
# =====================================================================

_WORKER_EOS = None


def _worker_init(init_kwargs, build_kind):
    """Pool initializer.  Constructs one ``hhe_z_mixtures`` per worker.

    Called once when the worker process starts.  ``build_kind`` selects
    which tables the worker should auto-load:
      - 'sp', 'rhot', 'rhop': only the P-T forward table is needed.
      - 'srho': both P-T and rho-T tables are needed (rho-T inversion
        is read inside the residual at every Newton iteration).

    The kwargs override the parent's pt_tab/inv_tab/srho_tab flags so
    the worker loads exactly the tables it needs (no more, no less).
    """
    global _WORKER_EOS
    kwargs = dict(init_kwargs)
    if build_kind in ('sp', 'rhot', 'rhop'):
        kwargs['pt_tab'] = True
        kwargs['inv_tab'] = False
        kwargs['srho_tab'] = False
    elif build_kind == 'srho':
        kwargs['pt_tab'] = True
        kwargs['inv_tab'] = True
        kwargs['srho_tab'] = False
    else:
        raise ValueError(f"Unknown build_kind: {build_kind!r}")
    _WORKER_EOS = hhe_z_mixtures(**kwargs)


def _worker_sp_yrow(args):
    """Run ``_build_sp_yrow`` on the worker's eos instance."""
    yp, zvals, _zm, _za, _zr, svals, logp = args
    return _WORKER_EOS._build_sp_yrow(
        float(yp), zvals, _zm, _za, _zr, svals, logp)


def _worker_rhot_yrow(args):
    """Run ``_build_rhot_yrow`` on the worker's eos instance."""
    (yp, zvals, _zm, _za, _zr,
     logrho, logt, lgp_lo, lgp_hi) = args
    return _WORKER_EOS._build_rhot_yrow(
        float(yp), zvals, _zm, _za, _zr,
        logrho, logt, lgp_lo, lgp_hi)


def _worker_rhop_yrow(args):
    """Run ``_build_rhop_yrow`` on the worker's eos instance."""
    yp, zvals, _zm, _za, _zr, logrho, logp = args
    return _WORKER_EOS._build_rhop_yrow(
        float(yp), zvals, _zm, _za, _zr, logrho, logp)


def _worker_srho_yrow(args):
    """Run ``_build_srho_yrow`` on the worker's eos instance."""
    (yp, zvals, _zm, _za, _zr,
     svals, logrho, lo_abs, hi_abs) = args
    return _WORKER_EOS._build_srho_yrow(
        float(yp), zvals, _zm, _za, _zr,
        svals, logrho, lo_abs, hi_abs)


_WORKER_DISPATCH = {
    'sp':   _worker_sp_yrow,
    'rhot': _worker_rhot_yrow,
    'rhop': _worker_rhop_yrow,
    'srho': _worker_srho_yrow,
}

_BUILD_DESC = {
    'sp':   "Inverting P,T -> S,P (parallel)",
    'rhot': "Inverting P,T -> rho,T (parallel)",
    'rhop': "Inverting P,T -> rho,P (parallel)",
    'srho': "Inverting P,T -> S,rho (parallel)",
}


class hhe_z_mixtures():
    """H-He-Z EOS with pre-computed inversion tables.

    Wraps ``val_mixtures`` (smoothed H-He + Z species via VAL) and
    serves the four basis inversions (S-P, ρ-T, ρ-P, S-ρ) used by
    ORCHARD's hydrostatic, transport, and evolution solvers.

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

    5. **Inversion tables** — Pre-built rectangular tables on the
       physical (S, P, Y', Z), (ρ, T, Y', Z), (ρ, P, Y', Z), and
       (S, ρ, Y', Z) grids store the inverse maps (logT, logP, etc.)
       and are queried via ``RegularGridInterpolator``.  NaN cells
       are repaired via interpolation when the tables are built.

    6. **Query** — Public methods (``get_logt_sp`` etc.) try the RGI
       table first.  When a table is unavailable or the query falls
       outside the table, they fall back to per-point Newton-Raphson
       with a brentq safety net (``_newton_1d`` / ``_newton_2d``).
    """

    # Table naming convention: {hhe_eos}/{hhe_eos}_{z_eos}_<basis>_*.npz
    # Auto-discovered relative to CURR_DIR (eos/ directory).
    _TABLE_BASES = {
        'pt':   '{hhe}_{z}_pt_square.npz',
        'sp':   '{hhe}_{z}_sp_square.npz',
        'rhot': '{hhe}_{z}_rhot_square.npz',
        'rhop': '{hhe}_{z}_rhop_square.npz',
        'srho': '{hhe}_{z}_srho_square.npz',
    }

    def __init__(self, hhe_eos_name='cd', hg=True,
                 smooth_hhe=False, smooth_z=False,
                 mu_h_vary=False,
                 species_list=None,
                 z_eos='aqua_revised',
                 pt_tab=True,
                 inv_tab=True,
                 srho_tab=False,
                 y_prime=True,
                 yprime_clip=False,
                 logp_range=(6.0, 14.0), logp_step=0.05,
                 logt_range=(1.3, 6.0),
                 logrho_range=(-8.0, 2.0), logrho_step=0.05,
                 interp_method='linear',
                 table_suffix='',
                 f_rock=0.0,
                 rock_interp=None):
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
            (S-P, ρ-T, ρ-P) for fast inversions and derivatives.
            Set False to use on-the-fly root-finding.
        srho_tab : bool
            If True, also load the pre-computed S-ρ table when
            ``inv_tab=True``.  Default False — the S-ρ inversion
            uses the 1-D decomposition via the ρ-T or S-P tables
            instead of the pre-computed 2-D S-ρ table.
        yprime_clip : bool
            If True, ``_to_yprime`` clips the converted Y' to [0, 1]
            (the tabulated Y' domain).  Protects high-Z cells where
            Y' = Y/(1-Z) blows past the table edge (e.g. Y=0.06 at
            Z=0.95 gives Y'=1.2 and silent RGI extrapolation today).
            Default False = bit-identical to the historical behavior;
            production-range queries (Y' well inside [0, 1]) are
            unaffected either way.
        logp_range, logp_step, logt_range, logrho_range, logrho_step :
            Grid bounds and steps for table-build mode.
        interp_method : str
            Interpolation method for inversion-table RGIs.
            'linear' (default) is C⁰; 'cubic' is C² but slower.
            The PT forward-model table always uses 'linear'.
        table_suffix : str
            Optional tag appended to the canonical table filename so
            multiple variants of the same (hhe_eos, z_eos) tables can
            coexist on disk without overwriting each other. Empty (the
            default) preserves the original canonical path
            ``{hhe}_{z}_{basis}_square.npz``; a non-empty value
            ``'highz'`` (say) loads/saves
            ``{hhe}_{z}_{basis}_square_highz.npz`` instead. Pair with
            the ``--suffix`` flag of ``eos_inversions.py`` when building
            tables from the CLI.
        f_rock : float
            Fixed rock (mg2sio4) mass fraction WITHIN the metal budget Z
            (the nested sub-fraction ``_zr``, with ``_zm = _za = 0``).
            When > 0 this (a) ensures the mg2sio4 EOS is loaded so the
            metal mixture actually contains rock, and (b) selects the
            rock-fraction-tagged table variant by extending
            ``table_suffix`` with ``frock{f_rock:.2f}`` — matching the
            filenames written by ``eos_inversions.py --f_rock``.  So
            ``f_rock=0.5`` with ``z_eos='aqua_revised'`` loads
            ``cd_aqua_revised_{basis}_square_frock0.50.npz``.  Default 0
            (pure-water metal, canonical ``_square.npz`` tables).
        rock_interp : bool or None
            Whether to 3-point interpolate across the f_rock = 0.0, 0.5,
            1.0 precomputed table sets.  Default None = AUTO-DETECT: it is
            switched on iff ``f_rock`` is not one of the table fractions
            {0.0, 0.5, 1.0}.  So f_rock=0.25 -> interpolation; f_rock=0.5
            -> the single ``frock0.50`` table (no interpolation needed).
            When on, three fixed-composition sub-instances are built
            (loading the ``_square``, ``frock0.50`` and ``frock1.00`` sets)
            and every EOS quantity is returned by piecewise-linear
            interpolation in the per-call rock fraction (the 5th
            positional ``_frock`` / ``_zr``); derivatives inherit it
            automatically.  The interpolating instance holds no tables of
            its own.  Pass an explicit bool to override the auto-detection
            (e.g. False for the internal sub-instances).
        """

        # --- 3-point rock-fraction interpolation across precomputed sets -
        # Auto-detect: interpolation is needed iff the requested rock
        # fraction is not one of the precomputed table fractions.  When on,
        # this instance holds no tables of its own; the three fixed-
        # composition sub-instances (built at the end of __init__) do, and
        # every base query interpolates among them.
        _ROCK_TABLE_FRACS = (0.0, 0.5, 1.0)
        if rock_interp is None:
            rock_interp = not any(abs(float(f_rock) - g) < 1e-9
                                  for g in _ROCK_TABLE_FRACS)
        self.rock_interp = bool(rock_interp)
        _sub_pt_tab, _sub_inv_tab, _sub_srho_tab = pt_tab, inv_tab, srho_tab
        if self.rock_interp:
            pt_tab = inv_tab = srho_tab = False
            f_rock = 0.0   # the main instance carries no single rock fraction

        # --- Fixed rock mass fraction within Z (nested _zr) -------------
        # Resolve before species_list / table_suffix are consumed below.
        self.f_rock = float(f_rock)
        if self.f_rock > 0.0:
            # Make sure the rock EOS is present in the metal mixture.
            if species_list is not None:
                _rock_names = {'mg2sio4', 'rock', 'forsterite'}
                if not any(str(s).lower() in _rock_names
                           for s in species_list):
                    species_list = list(species_list) + ['mg2sio4']
            # Select the rock-fraction-tagged tables (frock tag first,
            # like eos_inversions.py).
            _rock_tag = f'frock{self.f_rock:.2f}'
            _suff = str(table_suffix).strip('_')
            table_suffix = f'{_rock_tag}_{_suff}' if _suff else _rock_tag

        self.hhe_eos_name = hhe_eos_name
        self.z_eos_label = z_eos
        # Optional name tag for table variants (e.g. high-Z extension).
        # Normalize: strip leading/trailing underscores so callers can
        # pass either 'highz' or '_highz' and get the same filename.
        self.table_suffix = str(table_suffix).strip('_')
        self.pt_tab = pt_tab
        self.inv_tab = inv_tab
        self.srho_tab = srho_tab
        self.y_prime = y_prime
        self.yprime_clip = bool(yprime_clip)
        self._interp_method = interp_method

        # Store all init kwargs so worker processes can reconstruct
        # an equivalent instance (used by parallel build dispatchers).
        self._init_kwargs = dict(
            hhe_eos_name=hhe_eos_name, hg=hg,
            smooth_hhe=smooth_hhe, smooth_z=smooth_z,
            mu_h_vary=mu_h_vary,
            species_list=species_list,
            z_eos=z_eos,
            pt_tab=pt_tab, inv_tab=inv_tab, srho_tab=srho_tab,
            y_prime=y_prime,
            logp_range=logp_range, logp_step=logp_step,
            logt_range=logt_range,
            logrho_range=logrho_range, logrho_step=logrho_step,
            interp_method=interp_method,
            table_suffix=table_suffix,
        )

        # --- Forward-model mixer ---
        self.val = val_mixtures(
            hhe_eos_name=hhe_eos_name, hg=hg,
            smooth_hhe=smooth_hhe, smooth_z=smooth_z,
            mu_h_vary=mu_h_vary,
            species_list=species_list)

        # --- Grid parameters (used by table builders) ---
        self.logp_vals = np.arange(logp_range[0],
                                   logp_range[1] + logp_step * 0.1,
                                   logp_step)
        self.logt_min = logt_range[0]
        self.logt_max = logt_range[1]
        self.logrho_vals = np.arange(logrho_range[0],
                                      logrho_range[1] + logrho_step * 0.1,
                                      logrho_step)
        self.logt_vals = np.arange(logt_range[0], logt_range[1] + 0.01,
                                    logp_step)  # same step as logP

        # --- Pre-computed tables (None until loaded) ---
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
        self._svals_sp = None      # 1-D S grid for square SP tables
        self._svals_srho = None    # 1-D S grid for square S-rho tables

        # --- Auto-load tables based on pt_tab / inv_tab / srho_tab flags ---
        self._auto_load_tables()

        # --- Rock-fraction interpolation sub-instances (f_rock=0,0.5,1) ---
        if self.rock_interp:
            self._rock_fracs = (0.0, 0.5, 1.0)
            _sub_base = dict(
                hhe_eos_name=hhe_eos_name, hg=hg,
                smooth_hhe=smooth_hhe, smooth_z=smooth_z,
                mu_h_vary=mu_h_vary, z_eos=self.z_eos_label,
                interp_method=interp_method,
                logp_range=logp_range, logp_step=logp_step,
                logt_range=logt_range,
                logrho_range=logrho_range, logrho_step=logrho_step,
                pt_tab=_sub_pt_tab, inv_tab=_sub_inv_tab,
                srho_tab=_sub_srho_tab, y_prime=y_prime,
                yprime_clip=yprime_clip,
                rock_interp=False)
            self._rock_subs = [
                hhe_z_mixtures(species_list=['water_revised'],
                               f_rock=fr, **_sub_base)
                for fr in self._rock_fracs]

    # =================================================================
    # Rock-fraction interpolation helper
    # =================================================================

    def _interp_rock(self, frock, v0, v05, v1):
        """3-point piecewise-linear interpolation in rock fraction over
        the precomputed sets at f_rock = 0, 0.5, 1.0.

        Handles scalar or array values and tuple returns (e.g. the S-ρ
        inversion returns ``(logP, logT)``).  ``frock`` is clipped to
        [0, 1] and may be a scalar or a per-cell array broadcastable
        against the returned values.
        """
        if isinstance(v0, tuple):
            return tuple(self._interp_rock(frock, a, b, c)
                         for a, b, c in zip(v0, v05, v1))
        f = np.clip(np.asarray(frock, dtype=float), 0.0, 1.0)
        a0 = np.asarray(v0, dtype=float)
        a1 = np.asarray(v05, dtype=float)
        a2 = np.asarray(v1, dtype=float)
        lower = f <= 0.5
        t = np.where(lower, f / 0.5, (f - 0.5) / 0.5)
        lo = np.where(lower, a0, a1)
        hi = np.where(lower, a1, a2)
        out = lo + (hi - lo) * t
        if np.ndim(out) == 0:
            return float(out)
        return out

    # =================================================================
    # Auto-loading
    # =================================================================

    def _table_path(self, basis):
        """Return the expected on-disk file path for a given basis table.

        When ``self.table_suffix`` is non-empty, the suffix is inserted
        between the canonical stem and ``.npz``. This lets multiple
        variants of a table coexist (e.g. a high-Z, low-S extension
        built for sub-Neptune work) without overwriting the production
        ``_square.npz`` files.
        """
        fname = self._TABLE_BASES[basis].format(
            hhe=self.hhe_eos_name, z=self.z_eos_label)
        if self.table_suffix:
            stem, ext = os.path.splitext(fname)
            fname = f'{stem}_{self.table_suffix}{ext}'
        return os.path.join(CURR_DIR, self.hhe_eos_name, fname)

    def _auto_load_tables(self):
        """Try to load pre-computed tables from disk.

        Controlled by ``self.pt_tab`` (P-T basis table),
        ``self.inv_tab`` (inverted S-P, ρ-T, ρ-P tables), and
        ``self.srho_tab`` (S-ρ table, loaded only when True).  Each
        table is loaded from its canonical ``_square.npz`` path when
        the file is present.
        """
        if self.pt_tab:
            pt_path = self._table_path('pt')
            if os.path.isfile(pt_path):
                self.load_pt_table(pt_path)

        if self.inv_tab:
            for basis in ('sp', 'rhot', 'rhop'):
                path = self._table_path(basis)
                if os.path.isfile(path):
                    getattr(self, f'load_{basis}_table')(path)
            # S-ρ table only loaded when explicitly requested
            if self.srho_tab:
                path = self._table_path('srho')
                if os.path.isfile(path):
                    self.load_srho_table(path)

    # =================================================================
    # Y → Y' conversion
    # =================================================================

    def _to_yprime(self, _y, _z):
        """Convert absolute Y to Y' = Y/(1-Z) if y_prime=False.

        When self.y_prime=True, _y is already Y' and is returned as-is.
        When self.y_prime=False, _y is absolute Y and is divided by (1-Z).
        Works for scalar and array inputs.

        With ``yprime_clip=True`` (constructor kwarg) the result is
        clipped to the tabulated Y' domain [0, 1]: at high Z the
        division amplifies any Y noise (Y'=Y/(1-Z) is 20x Y at Z=0.95)
        and an off-domain Y' silently linear-extrapolates every RGI.
        Clip off (default) is bit-identical to the historical behavior.
        """
        if self.y_prime:
            return _y
        _z_arr = np.asarray(_z, dtype=float)
        _yp = np.asarray(_y, dtype=float) / (1.0 - _z_arr + 1e-6)
        if self.yprime_clip:
            _yp = np.clip(_yp, 0.0, 1.0)
        return _yp

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

        # --- Hampel pass 1: S along the P axis only ---
        # Targets the known CD21 H-He single-cell entropy spike (e.g.
        # 260x at logP=12.75, low T) from numerical artifacts in the
        # underlying tables that are 1-D in P.  This artifact is
        # persistent across (Y', Z), so it would not show up as an
        # outlier from the composition-axis perspective.
        if verbose:
            print("Running Hampel outlier filter on S along P axis ...")
        flat = s_pt.reshape(nP, -1)  # (nP, nT*nY*nZ)
        n_outliers = 0
        for j in range(flat.shape[1]):
            col = flat[:, j]
            cleaned, n_rep = hampel_filter_1d(col, window=7, n_sigma=3.0)
            if n_rep > 0:
                changed = (cleaned != col) & np.isfinite(col)
                flat[changed, j] = cleaned[changed]
                n_outliers += n_rep
        if n_outliers > 0 and verbose:
            print(f"Replaced {n_outliers} finite outlier cells "
                  f"in S (Hampel along P axis only)")

        # No Hampel along Y'/Z on the PT forward arrays.  The
        # composition-axis boundary cells (Y'=0/1, Z=0/1) carry real
        # logarithmic kinks in the mixing entropy that a window/MAD
        # test misclassifies as outliers, smearing the boundary into
        # the interior.  See diagnostic in eos_class history.

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
        """Save a P-T table to NPZ at the canonical auto-load path."""
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
        """S(P, T, Y', Z) — uses table RGI if loaded, else VAL.

        When ``rock_interp`` is on, interpolate among the f_rock=0,0.5,1
        sub-instances using ``_zr`` (rock fraction within Z).
        """
        if self.rock_interp:
            return self._interp_rock(_zr, *[s._s_pt(lgp, lgt, yp, z)
                                            for s in self._rock_subs])
        if self._s_pt_rgi is not None:
            return self._query_pt_rgi(self._s_pt_rgi, lgp, lgt, yp, z)
        return self.val.get_s_pt_val(lgp, lgt, yp, z, _zm, _za, _zr)

    def _logrho_pt(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0):
        """logrho(P, T, Y', Z) — uses table RGI if loaded, else VAL."""
        if self.rock_interp:
            return self._interp_rock(_zr, *[s._logrho_pt(lgp, lgt, yp, z)
                                            for s in self._rock_subs])
        if self._logrho_pt_rgi is not None:
            return self._query_pt_rgi(self._logrho_pt_rgi, lgp, lgt, yp, z)
        return self.val.get_logrho_pt_val(lgp, lgt, yp, z, _zm, _za, _zr)

    def _logu_pt(self, lgp, lgt, yp, z, _zm=0.0, _za=0.0, _zr=0.0):
        """logU(P, T, Y', Z) — uses table RGI if loaded, else VAL."""
        if self.rock_interp:
            return self._interp_rock(_zr, *[s._logu_pt(lgp, lgt, yp, z)
                                            for s in self._rock_subs])
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
        if self.rock_interp:
            return self._interp_rock(_frock, *[
                s.get_s_pt_tab(_lgp, _lgt, _y, _z, val=val, **kw)
                for s in self._rock_subs])
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
        if self.rock_interp:
            return self._interp_rock(_frock, *[
                s.get_logrho_pt_tab(_lgp, _lgt, _y, _z, val=val, **kw)
                for s in self._rock_subs])
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
        if self.rock_interp:
            return self._interp_rock(_frock, *[
                s.get_logu_pt_tab(_lgp, _lgt, _y, _z, val=val, **kw)
                for s in self._rock_subs])
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
    # Newton-Raphson solvers (1-D and 2-D)
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

    def _newton_1d_vec(self, residual_fn, guess, lo_abs, hi_abs,
                       max_iter=40, tol=1e-6, h=1e-3, max_step=1.0):
        """Vectorized Newton-Raphson with central-difference derivative.

        Solves ``f(x) = 0`` element-wise where ``f`` and ``x`` are
        arrays of any shape.  All operations broadcast — no Python
        loop over elements.  Used to invert the forward model on a
        whole (S, P) or (rho, T) etc. slab in one shot, replacing
        the per-cell Newton loop in the build_*_table methods.

        For batch inversion against the PT-RGI, this is typically
        50-200x faster than calling scalar ``_newton_1d`` per cell
        because each Newton iteration becomes one batched RGI
        evaluation rather than O(grid_size) scalar evaluations.

        Parameters
        ----------
        residual_fn : callable
            ``f(x_array) -> residual_array`` of the same shape as
            ``x_array``.  Must broadcast and return finite values
            where the inversion is well-posed; NaN elsewhere.
        guess : array_like
            Initial guess.
        lo_abs, hi_abs : float
            Hard bounds.  ``x`` is clipped after each Newton step.
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on |residual|.
        h : float
            Central-difference step for the derivative.
        max_step : float
            Cap on |Newton step| per iteration.

        Returns
        -------
        x : ndarray, same shape as guess
            Converged solution.  NaN where convergence failed.
        converged : bool ndarray
            True where ``|residual| < tol`` was reached.
        """
        x = np.clip(np.asarray(guess, dtype=float),
                    lo_abs, hi_abs).copy()
        # ``active`` = cells still being updated.  Once a cell hits
        # |f|<tol it is frozen; we still evaluate the residual on it
        # in subsequent iterations because the cost is dominated by
        # the batched RGI call (whose cost barely grows with the
        # subset size), but we don't update ``x`` on inactive cells.
        active = np.ones(x.shape, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break

            # Residual on the full slab (one batched RGI call)
            f = np.asarray(residual_fn(x), dtype=float)
            f_finite = np.isfinite(f)

            # Mark newly-converged cells
            ok = f_finite & (np.abs(f) < tol)
            active &= ~ok
            if not active.any():
                break

            # Central FD derivative — two more batched RGI calls
            x_p = np.clip(x + h, lo_abs, hi_abs)
            x_m = np.clip(x - h, lo_abs, hi_abs)
            f_p = np.asarray(residual_fn(x_p), dtype=float)
            f_m = np.asarray(residual_fn(x_m), dtype=float)
            denom = (x_p - x_m)
            with np.errstate(divide='ignore', invalid='ignore'):
                fp = (f_p - f_m) / denom
                step = np.where(
                    f_finite & np.isfinite(fp) & (np.abs(fp) > 1e-30),
                    f / fp, 0.0)
            step = np.clip(step, -max_step, max_step)

            x_new = np.clip(x - step, lo_abs, hi_abs)
            x = np.where(active, x_new, x)

        # Final Newton-pass check
        f = np.asarray(residual_fn(x), dtype=float)
        ok_after_newton = np.isfinite(f) & (np.abs(f) < tol)

        # --- Vectorized bisection fallback for cells where Newton
        # didn't converge (typically because the cold-start was far
        # from a steep-gradient root and Newton oscillated within the
        # max_step cap).  Bisection is unconditionally convergent
        # given a sign-changing bracket, fully batched (one RGI
        # evaluation per iteration regardless of failed-cell count).
        # Early-exit is in residual-space (|f(mid)| < tol everywhere
        # bracketed) so the convergence test matches what the build
        # actually cares about.  At steep gradients, an x-space exit
        # at |b-a|<tol can leave residuals 10-100x larger than tol,
        # which then fail the final tolerance check and get NaN'd.
        bad = ~ok_after_newton
        if bad.any():
            f_lo = np.asarray(residual_fn(
                np.where(bad, lo_abs, x)), dtype=float)
            f_hi = np.asarray(residual_fn(
                np.where(bad, hi_abs, x)), dtype=float)
            # Cells where the bracket [lo, hi] genuinely contains a
            # sign change are bisectable.
            bracketed = bad & np.isfinite(f_lo) & np.isfinite(f_hi) \
                        & (f_lo * f_hi < 0)
            if bracketed.any():
                a = np.where(bracketed, lo_abs, x)
                b = np.where(bracketed, hi_abs, x)
                fa = np.where(bracketed, f_lo, 0.0)
                # Cap at 60 iterations (|b-a| -> 5e-18 in x-space, way
                # past any meaningful gradient).  Early-exit when all
                # bracketed cells satisfy |f(mid)| < tol.
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    f_mid = np.asarray(residual_fn(mid), dtype=float)
                    valid = bracketed & np.isfinite(f_mid)
                    same_sign_as_a = valid & (fa * f_mid > 0)
                    a = np.where(same_sign_as_a, mid, a)
                    fa = np.where(same_sign_as_a, f_mid, fa)
                    b = np.where(valid & ~same_sign_as_a, mid, b)
                    # Residual-based early exit
                    if np.all(np.abs(f_mid[bracketed]) < tol):
                        break
                x = np.where(bracketed, 0.5 * (a + b), x)

        # Final pass — accept cells that satisfy the tolerance now.
        f = np.asarray(residual_fn(x), dtype=float)
        ok_final = np.isfinite(f) & (np.abs(f) < tol)
        # Cells that still don't converge become NaN (out-of-domain or
        # no sign change in [lo, hi]).
        x = np.where(ok_final, x, np.nan)
        return x, ok_final

    def get_logt_sp(self, _s_kb, _lgp, _yp, _z=0.0,
                    _frock=0.0, _zm=0.0, _za=0.0, _zr=None,
                    use_tab=True, **kw):
        """Temperature from (S, P) via Newton-Raphson on the forward model.

        Uses ``ideal_xy`` for initial guesses and the previous
        converged solution as the starting point when sweeping arrays
        (so isentropes naturally track the correct high-T branch).
        Falls back to adaptive brentq bracketed around Newton's last
        iterate, bounded by [1.5, 7.0] in logT.

        If a pre-computed S-P table has been loaded, the RGI lookup
        is used first; Newton/brentq only fills NaN entries.
        Set ``use_tab=False`` to force per-point Newton-Raphson
        even when an S-P table is loaded.

        Accepts ``_frock`` as a legacy alias for ``_zr`` (rock
        fraction within Z).

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
        use_tab : bool, optional
            Use pre-computed S-P table if available (default True).

        Returns
        -------
        logt : float or array
            log10 T [K].  NaN where S is outside the physical domain.
        """
        # Rock fraction within Z: 5th positional is _frock; an explicit
        # _zr keyword overrides; accept legacy _frock-in-kw too.
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_logt_sp(_s_kb, _lgp, _yp, _z, use_tab=use_tab, **kw)
                for s in self._rock_subs])
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
        if use_tab and self._logt_sp_rgi is not None:
            result = self._lookup_sp_table(_s_kb, _lgp, _yp, _z)
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
                      _frock=0.0, _zm=0.0, _za=0.0, _zr=None, **kw):
        """Density from (S, P) — calls get_logt_sp then forward model.

        Rock fraction within Z is the 5th positional ``_frock`` (``_zr``
        keyword overrides).
        """
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_logrho_sp(_s_kb, _lgp, _yp, _z, **kw)
                for s in self._rock_subs])
        # Y->Y' conversion bug fix (2026-08): get_logt_sp is the PUBLIC
        # method and converts internally -- passing an already-converted
        # Y' double-converted to Y/(1-Z)^2 (harmless at Z=0, +34% in T at
        # Z=0.3, catastrophic at high Z).  Pass the caller's raw Y to
        # get_logt_sp; convert once, separately, for the internal
        # _logrho_pt leaf (which expects Y').
        logt = self.get_logt_sp(_s_kb, _lgp, _yp, _z, _zm=_zm, _za=_za, _zr=_zr)
        _yp = self._to_yprime(_yp, _z)
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

    def _lookup_sp_table(self, _s_kb, _lgp, _yp, _z):
        """Query the (S, logP, Y', Z) S-P inversion RGI."""
        _s_kb = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgp  = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _yp_a = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a  = np.atleast_1d(np.asarray(_z, dtype=float))
        _s_kb, _lgp, _yp_a, _z_a = np.broadcast_arrays(
            _s_kb, _lgp, _yp_a, _z_a)
        pts = np.column_stack((_s_kb.ravel(), _lgp.ravel(),
                               _yp_a.ravel(), _z_a.ravel()))
        out = self._logt_sp_rgi(pts).reshape(_s_kb.shape)
        if out.size == 1:
            return out.item()
        return out

    def _load_sp_from_arrays(self, svals, logp, yvals, zvals, logt_sp):
        """Build the S-P RGI from raw arrays (square grid)."""
        self._svals_sp = svals
        self.logp_vals = logp
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        self._logt_sp_rgi = RGI((svals, logp, yvals, zvals),
                                 logt_sp, **rgi_kw)

    def load_sp_table(self, path):
        """Load a pre-computed S-P table from NPZ.

        logrho is not stored — it is computed on-the-fly from the
        forward model via ``get_logrho_sp``.
        """
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        self._load_sp_from_arrays(
            data['svals'], data['logpvals'],
            data['yvals'], data['zvals'],
            data['logt_sp'])

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

    @staticmethod
    def _hampel_nd(table, axes=None, window=5, n_sigma=3.0,
                   verbose=True):
        """Run Hampel outlier filter along multiple axes of an N-D table.

        For each axis in *axes*, the table is reshaped so that axis is
        the leading dimension, then ``hampel_filter_1d`` is applied to
        every 1-D column along that axis.  Axes with fewer than
        ``2*window + 1`` points are silently skipped.

        Parameters
        ----------
        table : ndarray
            N-D array to filter **in-place**.
        axes : list of int or None
            Axes to filter.  ``None`` means all axes.
        window : int
            Half-window for Hampel filter (full window = 2*window+1).
        n_sigma : float
            Number of MAD-scaled sigmas for outlier detection.
        verbose : bool
            Print per-axis outlier counts.

        Returns
        -------
        n_total : int
            Total number of replaced cells across all axes.
        """
        if axes is None:
            axes = list(range(table.ndim))
        n_total = 0
        for ax in axes:
            n_ax = table.shape[ax]
            if n_ax < 2 * window + 1:
                continue
            # moveaxis returns a view; reshape may copy
            view = np.moveaxis(table, ax, 0)
            orig_shape = view.shape
            flat = view.reshape(n_ax, -1).copy()
            n_out = 0
            for j in range(flat.shape[1]):
                col = flat[:, j]
                cleaned, n_rep = hampel_filter_1d(
                    col, window=window, n_sigma=n_sigma)
                if n_rep > 0:
                    changed = (cleaned != col) & np.isfinite(col)
                    flat[changed, j] = cleaned[changed]
                    n_out += n_rep
            # Write back
            np.moveaxis(table, ax, 0)[:] = flat.reshape(orig_shape)
            n_total += n_out
            if verbose and n_out > 0:
                print(f"  Axis {ax}: replaced {n_out} outlier cells")
        return n_total

    def _smooth_inverted_table(self, table, sigma=1.0,
                                hampel_window=5, hampel_n_sigma=3.0,
                                verbose=True):
        """Apply Hampel + light Gaussian smoothing to an inverted 4-D
        table (e.g. ``logt_sp``, ``logp_rhot``, ``logt_rhop``).

        Smoothing is applied **only** along the two physical axes
        (axes 0 and 1 — e.g. S and P for SP, ρ and T for ρT, etc.).
        The composition axes (Y' = axis 2, Z = axis 3) are deliberately
        left untouched: at the boundaries Y'=0/1 and Z=0/1 the physics
        is qualitatively different from the interior, and a Hampel /
        Gaussian pass along those axes smears the boundary inward and
        produces visible artifacts in plots near pure He / pure metal
        limits.

        Run this **after** ``_fill_table_nans``: Hampel handles
        isolated outliers from Newton convergence flicker, then a
        light σ=1-grid-cell Gaussian provides clean FD-derivative
        behavior for downstream code without distorting the table by
        more than ~1 grid cell of resolution.

        Parameters
        ----------
        table : ndarray, shape (n0, n1, nY, nZ)
            The 4-D inverted table.  Modified in-place by Hampel.
        sigma : float
            Gaussian sigma in grid cells along axes 0 and 1.
        hampel_window : int
            Half-window for the Hampel pass.
        hampel_n_sigma : float
            Outlier threshold (in MAD-sigmas) for the Hampel pass.
        verbose : bool
            Print Hampel replacement counts.

        Returns
        -------
        smoothed : ndarray
            New 4-D array with the same shape as ``table``.  NaN
            cells are preserved (the Gaussian is mask-aware).
        """
        # 1) Hampel along physical axes (in-place) — catches isolated
        #    Newton-flicker outliers before they get smeared by the
        #    Gaussian into broader bumps.
        n_replaced = self._hampel_nd(
            table, axes=(0, 1),
            window=hampel_window, n_sigma=hampel_n_sigma,
            verbose=verbose)
        if verbose and n_replaced > 0:
            print(f"  Hampel total: {n_replaced} outlier cells replaced")

        # 2) NaN-aware Gaussian along physical axes only.
        sigma_4d = [sigma, sigma, 0.0, 0.0]
        mask = np.isfinite(table)
        filled = np.where(mask, table, 0.0)
        weight = mask.astype(float)
        smoothed = gaussian_filter(filled, sigma=sigma_4d, mode='nearest')
        weight_s = gaussian_filter(weight, sigma=sigma_4d, mode='nearest')
        out = np.where(mask, smoothed / np.maximum(weight_s, 1e-10),
                       np.nan)
        if verbose:
            print(f"  Gaussian smoothing applied with sigma={sigma_4d}")
        return out

    # =================================================================
    # Table generation
    # =================================================================

    def _parallel_yrow_dispatch(self, kind, tasks, n_workers, verbose=True):
        """Dispatch a list of Y'-row build tasks across worker processes.

        Each worker process constructs its own ``hhe_z_mixtures`` instance
        once at startup (loading the relevant tables), then handles its
        assigned Y' rows.  Within a worker, the Z-axis warm-start chain
        is preserved exactly as in the serial path: each Y' row resets
        ``prev_sol`` to None, and the inner Z loop carries it forward.
        Y' rows are independent of one another, so the embarrassing-
        parallel split incurs no algorithmic change vs the serial build.

        Parameters
        ----------
        kind : str
            Build basis: 'sp', 'rhot', 'rhop', or 'srho'.  Selects the
            module-level worker function and the table-loading flags.
        tasks : list of tuple
            Per-Y'-row argument tuples.  Order is preserved in the
            returned list so the caller can stack rows along the Y' axis
            without reshuffling.
        n_workers : int
            Number of worker processes.  Capped internally at len(tasks).
        verbose : bool
            Print a progress bar.

        Returns
        -------
        rows : list of ndarray
            One slab per task, in the same order as ``tasks``.
        """
        import multiprocessing as mp

        worker_fn = _WORKER_DISPATCH[kind]
        n = max(1, min(int(n_workers), len(tasks)))

        if verbose:
            print(f"  Parallel build: {len(tasks)} Y' rows across "
                  f"{n} worker processes")

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n,
                      initializer=_worker_init,
                      initargs=(self._init_kwargs, kind)) as pool:
            rows = [None] * len(tasks)
            iterator = pool.imap(worker_fn, tasks)
            pbar = tqdm(total=len(tasks),
                        desc=_BUILD_DESC[kind],
                        disable=not verbose,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                   '[{elapsed}<{remaining}]')
            for i, slab in enumerate(iterator):
                rows[i] = slab
                pbar.update(1)
            pbar.close()
        return rows

    def _build_sp_yrow(self, yp, zvals, _zm, _za, _zr, svals, logp):
        """Build all Z slabs for ONE Y' row of the S-P table.

        Returns a (nS, nP, nZ) slab.  Inner Z loop preserves the
        warm-start chain (prev_sol from the previous Z is used as
        the initial guess for the next).  Used by both the serial
        and parallel build paths in ``build_sp_table``.
        """
        S_2d, P_2d = np.meshgrid(svals, logp, indexing='ij')
        nZ = len(zvals)
        out = np.full((len(svals), len(logp), nZ), np.nan, dtype=float)
        prev_sol = None
        for iz, zv in enumerate(zvals):
            if prev_sol is not None and np.all(np.isfinite(prev_sol)):
                guess = prev_sol
            else:
                guess = np.full_like(S_2d, 0.5 * (1.5 + 7.0))

            def residual(lgt_2d, _yp=yp, _zv=zv,
                         _zm_=_zm, _za_=_za, _zr_=_zr):
                s_test = self._s_pt(P_2d, lgt_2d, _yp, _zv,
                                    _zm_, _za_, _zr_) * erg_to_kbbar
                return s_test - S_2d

            sol, _ = self._newton_1d_vec(
                residual, guess, lo_abs=1.5, hi_abs=7.0)
            out[:, :, iz] = sol
            prev_sol = sol
        return out

    def build_sp_table(self, yvals, zvals,
                       _zm=0.0, _za=0.0, _zr=0.0,
                       s_lo=4.0, s_hi=12.0, s_step=0.1,
                       smooth_inverted=False,
                       n_workers=1,
                       verbose=True):
        """Build logT on a uniform (S, logP, Y', Z) grid.

        Parameters
        ----------
        yvals, zvals : array_like
            1-D Y' and Z grids.
        _zm, _za, _zr : float
            Fixed nested metal sub-fractions.
        s_lo, s_hi, s_step : float
            Entropy range and step in kb/baryon.
        smooth_inverted : bool
            If True, apply a Hampel + light Gaussian (σ=1 grid cell)
            pass along the S and logP axes after NaN-fill, and save
            to the ``*_smooth.npz`` variant of the auto-load path.
            Default False — saves to ``*_square.npz`` with no
            post-inversion smoothing.  Disjoint from ``smooth_hhe`` /
            ``smooth_z`` (those smooth the underlying H-He / Z
            component tables before VAL mixing).
        n_workers : int
            If > 1, dispatch Y' rows across this many worker
            processes via multiprocessing.  Capped at len(yvals).
            Default 1 (serial).  Each worker constructs its own
            ``hhe_z_mixtures`` instance once (loading the PT-RGI),
            then handles its assigned Y' rows; the warm-start chain
            within each Y' row is preserved.
        verbose : bool
            Print progress.

        Returns
        -------
        result : dict with keys svals, logpvals, yvals, zvals,
                 logt_sp (nS, nP, nY, nZ), logt_min, logt_max.
            Also loads the table into this instance.
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        svals = np.arange(s_lo, s_hi + s_step * 0.1, s_step)
        logp = self.logp_vals

        nS, nP, nY, nZ = len(svals), len(logp), len(yvals), len(zvals)

        if verbose:
            print(f"Building S-P square table: "
                  f"S=[{svals[0]:.2f}, {svals[-1]:.2f}] "
                  f"(dS={s_step:.3f}, {nS} pts), "
                  f"logP=[{logp[0]:.2f}, {logp[-1]:.2f}] "
                  f"(dlogP={logp[1]-logp[0]:.2f}, {nP} pts), "
                  f"logT=[{self.logt_min:.1f}, {self.logt_max:.1f}]")
            print(f"  Y' grid: {nY} pts [{yvals[0]:.3f} .. {yvals[-1]:.3f}], "
                  f"Z grid: {nZ} pts [{zvals[0]:.3f} .. {zvals[-1]:.3f}]")
            print(f"  Total cells: {nS}x{nP}x{nY}x{nZ} = "
                  f"{nS*nP*nY*nZ:,}")

        # Run inversions over Y' rows (serial or parallel)
        if int(n_workers) > 1 and nY > 1:
            tasks = [(float(yp), zvals, _zm, _za, _zr, svals, logp)
                     for yp in yvals]
            rows = self._parallel_yrow_dispatch(
                'sp', tasks, int(n_workers), verbose=verbose)
        else:
            pbar = tqdm(total=nY,
                         desc="Inverting P,T -> S,P (vectorized)",
                         disable=not verbose,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                    '[{elapsed}<{remaining}]')
            rows = []
            for yp in yvals:
                pbar.set_postfix_str(f"Y'={yp:.3f}")
                rows.append(self._build_sp_yrow(
                    float(yp), zvals, _zm, _za, _zr, svals, logp))
                pbar.update(1)
            pbar.close()

        # Stack rows into the (nS, nP, nY, nZ) table
        logt_sp = np.stack(rows, axis=2)

        # --- NaN filling ---
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

        # Hampel + Gaussian along (S, P) only when explicitly opted-in
        # via smooth_inverted=True.  Y' and Z axes are deliberately
        # left untouched: composition-axis boundary cells (Y'=0/1,
        # Z=0/1) carry qualitatively different physics from interior
        # cells and a window/MAD test along those axes smears the
        # boundary inward, producing visible artifacts near pure-He
        # and pure-metal limits.  Newton convergence failures already
        # show up as NaNs and were handled by _fill_table_nans above.
        if smooth_inverted:
            if verbose:
                print("Smoothing inverted table (Hampel + Gaussian "
                      "along S, P) ...")
            logt_sp = self._smooth_inverted_table(
                logt_sp, sigma=1.0, verbose=verbose)

        # --- Cast to float32 ---
        logt_sp_f32 = logt_sp.astype(np.float32)

        if verbose:
            mem_mb = logt_sp_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        result = {
            'svals':        svals.astype(np.float32),
            'logpvals':     logp,
            'yvals':        yvals,
            'zvals':        zvals,
            'logt_sp':      logt_sp_f32,
            'logt_min':     self.logt_min,
            'logt_max':     self.logt_max,
        }

        # Load into this instance
        self._load_sp_from_arrays(svals, logp, yvals, zvals, logt_sp_f32)

        if verbose:
            n_total = logt_sp.size
            n_good = np.isfinite(logt_sp).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan_before} were interpolated")

        return result

    def save_sp_table(self, result, path=None):
        """Save a table dict (from ``build_sp_table``) to NPZ at the
        canonical auto-load path."""
        if path is None:
            path = self._table_path('sp')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # rho-T inversion: P(rho, T, Y', Z)
    # =================================================================

    def get_logp_rhot(self, _lgrho, _lgt, _yp, _z=0.0,
                      _frock=0.0, _zm=0.0, _za=0.0, _zr=None,
                      use_tab=True, **kw):
        """Pressure from (rho, T) via root-finding or pre-computed table.

        Inverts rho(P, T, Y', Z) = 10^_lgrho to find logP.
        Rock fraction within Z is the 5th positional ``_frock`` (``_zr``
        keyword overrides).  Set ``use_tab=False`` to force per-point
        Newton-Raphson even when a ρ-T table is loaded.
        """
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_logp_rhot(_lgrho, _lgt, _yp, _z, use_tab=use_tab, **kw)
                for s in self._rock_subs])
        # Y->Y' conversion bug fix (2026-08): the table's axis 2 is Y',
        # but the fast path used to feed it the caller's ABSOLUTE Y (the
        # conversion below was unreachable with eos_tab=True) -- an error
        # of ~(1-Z) in the Y coordinate for every rho-T table query.
        # Convert before dispatching to either path.
        _yp = self._to_yprime(_yp, _z)
        # --- Fast path: pre-computed table ---
        if use_tab and self._logp_rhot_rgi is not None:
            return self._lookup_rhot_table(_lgrho, _lgt, _yp, _z)

        # --- Slow path: Newton-Raphson per-point ---
        scalar_input = np.isscalar(_lgrho) and np.isscalar(_lgt)
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgt   = np.atleast_1d(np.asarray(_lgt, dtype=float))
        _lgrho, _lgt = np.broadcast_arrays(_lgrho, _lgt)
        out = np.full_like(_lgrho, np.nan, dtype=float)

        lgp_lo, lgp_hi = 3, 16

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

    def _lookup_rhot_table(self, _lgrho, _lgt, _yp, _z):
        """Query the (logrho, logT, Y', Z) ρ-T inversion RGI."""
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgt   = np.atleast_1d(np.asarray(_lgt, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _lgrho, _lgt, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho, _lgt, _yp_a, _z_a)
        pts = np.column_stack((_lgrho.ravel(), _lgt.ravel(),
                               _yp_a.ravel(), _z_a.ravel()))
        out = self._logp_rhot_rgi(pts).reshape(_lgrho.shape)
        if out.size == 1:
            return out.item()
        return out

    def load_rhot_table(self, path):
        """Load a pre-computed ρ-T → P table from NPZ."""
        data = np.load(path)
        inv_rgi_kw = dict(method=self._interp_method, bounds_error=False,
                          fill_value=None)
        logt = data['logtvals']
        yv = data['yvals']
        zv = data['zvals']
        self.logt_vals = logt
        logrhovals = data['logrhovals']
        self._logp_rhot_rgi = RGI(
            (logrhovals, logt, yv, zv), data['logp_rhot'], **inv_rgi_kw)

    def _build_rhot_yrow(self, yp, zvals, _zm, _za, _zr,
                          logrho, logt, lgp_lo, lgp_hi):
        """Build all Z slabs for ONE Y' row of the rho-T table.

        Returns a (nR, nT, nZ) slab.  Inner Z loop preserves the
        warm-start chain.  Used by both serial and parallel paths in
        ``build_rhot_table``.
        """
        R_2d, T_2d = np.meshgrid(logrho, logt, indexing='ij')
        nZ = len(zvals)
        out = np.full((len(logrho), len(logt), nZ), np.nan, dtype=float)
        prev_sol = None
        for iz, zv in enumerate(zvals):
            if prev_sol is not None and np.all(np.isfinite(prev_sol)):
                guess = prev_sol
            else:
                guess = np.full_like(R_2d, 0.5 * (lgp_lo + lgp_hi))

            def residual(lgp_2d, _yp=yp, _zv=zv,
                         _zm_=_zm, _za_=_za, _zr_=_zr):
                rho_test = self._logrho_pt(lgp_2d, T_2d, _yp, _zv,
                                            _zm_, _za_, _zr_)
                return rho_test - R_2d

            sol, _ = self._newton_1d_vec(
                residual, guess, lo_abs=lgp_lo, hi_abs=lgp_hi)
            out[:, :, iz] = sol
            prev_sol = sol
        return out

    def build_rhot_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         smooth_inverted=False,
                         n_workers=1,
                         verbose=True):
        """Build logP on a uniform (logrho, logT, Y', Z) grid.

        Parameters
        ----------
        yvals, zvals : array_like
            1-D Y' and Z grids.
        _zm, _za, _zr : float
            Fixed nested metal sub-fractions.
        smooth_inverted : bool
            If True, apply a Hampel + light Gaussian (σ=1 grid cell)
            pass along the logρ and logT axes after NaN-fill, and save
            to the ``*_smooth.npz`` variant.  Default False — saves to
            ``*_square.npz`` with no post-inversion smoothing.
        n_workers : int
            If > 1, dispatch Y' rows across this many worker processes.
            Default 1 (serial).
        verbose : bool
            Print progress.
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        logrho = self.logrho_vals
        logt = self.logt_vals
        nR, nT, nY, nZ = len(logrho), len(logt), len(yvals), len(zvals)
        lgp_lo = float(self.logp_vals[0])
        lgp_hi = float(self.logp_vals[-1])

        if verbose:
            print(f"Building rho-T square table: "
                  f"logrho=[{logrho[0]:.2f}, {logrho[-1]:.2f}] ({nR} pts), "
                  f"logT=[{logt[0]:.2f}, {logt[-1]:.2f}] ({nT} pts)")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {nR}x{nT}x{nY}x{nZ} = "
                  f"{nR*nT*nY*nZ:,}")

        # Vectorized Newton on the entire (nR, nT) slab per (Y', Z).
        # _newton_1d_vec applies a vectorized bisection fallback to
        # any cell where Newton oscillates.
        if int(n_workers) > 1 and nY > 1:
            tasks = [(float(yp), zvals, _zm, _za, _zr,
                      logrho, logt, lgp_lo, lgp_hi)
                     for yp in yvals]
            rows = self._parallel_yrow_dispatch(
                'rhot', tasks, int(n_workers), verbose=verbose)
        else:
            pbar = tqdm(total=nY,
                         desc="Inverting P,T -> rho,T (vectorized)",
                         disable=not verbose,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                    '[{elapsed}<{remaining}]')
            rows = []
            for yp in yvals:
                pbar.set_postfix_str(f"Y'={yp:.3f}")
                rows.append(self._build_rhot_yrow(
                    float(yp), zvals, _zm, _za, _zr,
                    logrho, logt, lgp_lo, lgp_hi))
                pbar.update(1)
            pbar.close()

        # Stack rows into the (nR, nT, nY, nZ) table
        logp_tab = np.stack(rows, axis=2)

        # --- NaN filling ---
        n_nan = np.isnan(logp_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logp_tab = self._fill_table_nans(logp_tab)

        # Hampel + Gaussian along (logρ, logT) only when
        # smooth_inverted=True.  See build_sp_table for the
        # composition-axis rationale.
        if smooth_inverted:
            if verbose:
                print("Smoothing inverted table (Hampel + Gaussian "
                      "along rho, T) ...")
            logp_tab = self._smooth_inverted_table(
                logp_tab, sigma=1.0, verbose=verbose)

        logp_f32 = logp_tab.astype(np.float32)

        if verbose:
            mem_mb = logp_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        result = {
            'logrhovals':   logrho,
            'logtvals':     logt,
            'yvals':        yvals,
            'zvals':        zvals,
            'logp_rhot':    logp_f32,
            'logt_min':     self.logt_min,
            'logt_max':     self.logt_max,
        }

        # Load into this instance
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        self._logp_rhot_rgi = RGI(
            (logrho, logt, yvals, zvals), logp_f32, **rgi_kw)

        if verbose:
            n_total = logp_tab.size
            n_good = np.isfinite(logp_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def save_rhot_table(self, result, path=None):
        """Save a ρ-T table dict to NPZ at the canonical auto-load path."""
        if path is None:
            path = self._table_path('rhot')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # ρ-P inversion: T(ρ, P, Y', Z)
    # =================================================================

    def _logt_rhop_noconv(self, _lgrho, _lgp, _yp, _z=0.0,
                          _zm=0.0, _za=0.0, _zr=0.0):
        """Core logT(ρ, P) inversion — assumes _yp is already Y'.

        Used internally by derivative methods where Y' conversion
        has already been applied by the caller.
        """
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

    def get_logt_rhop(self, _lgrho, _lgp, _yp, _z=0.0,
                      _frock=0.0, _zm=0.0, _za=0.0, _zr=None, **kw):
        """Temperature from (ρ, P) via 1-D root-finding or table.

        Inverts ρ(P, T, Y', Z) = 10^logrho to find logT.
        Rock fraction within Z is the 5th positional ``_frock``
        (``_zr`` keyword overrides).

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
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_logt_rhop(_lgrho, _lgp, _yp, _z, **kw)
                for s in self._rock_subs])
        _yp = self._to_yprime(_yp, _z)
        return self._logt_rhop_noconv(
            _lgrho, _lgp, _yp, _z, _zm, _za, _zr)

    def get_s_rhop(self, _lgrho, _lgp, _yp, _z=0.0,
                   _frock=0.0, _zm=0.0, _za=0.0, _zr=None, **kw):
        """Entropy from (ρ, P) via 1-D root-finding.

        Finds T such that ρ(P, T, Y', Z) = 10^logrho, then
        evaluates S(P, T, Y', Z).

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
        _zm, _za, _zr : float
            Nested metal sub-fractions.

        Returns
        -------
        s_kb : float or array
            Entropy in kb/baryon.  NaN where no solution.
        """
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_s_rhop(_lgrho, _lgp, _yp, _z, **kw)
                for s in self._rock_subs])
        logt = self.get_logt_rhop(
            _lgrho, _lgp, _yp, _z, _zm=_zm, _za=_za, _zr=_zr)
        _yp = self._to_yprime(_yp, _z)

        logt_arr = np.atleast_1d(logt)
        _lgp_arr = np.atleast_1d(_lgp)
        logt_arr, _lgp_arr = np.broadcast_arrays(logt_arr, _lgp_arr)

        out = np.full_like(logt_arr, np.nan, dtype=float)
        good = np.isfinite(logt_arr)
        if good.any():
            s_cgs = self._s_pt(
                _lgp_arr[good], logt_arr[good], _yp, _z,
                _zm, _za, _zr)
            out[good] = s_cgs * erg_to_kbbar

        if out.size == 1:
            return out.item()
        return out

    def _lookup_rhop_table(self, _lgrho, _lgp, _yp, _z):
        """Query the (logrho, logP, Y', Z) ρ-P inversion RGI."""
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _lgp   = np.atleast_1d(np.asarray(_lgp, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _lgrho, _lgp, _yp_a, _z_a = np.broadcast_arrays(
            _lgrho, _lgp, _yp_a, _z_a)
        pts = np.column_stack((_lgrho.ravel(), _lgp.ravel(),
                               _yp_a.ravel(), _z_a.ravel()))
        out = self._logt_rhop_rgi(pts).reshape(_lgrho.shape)
        if out.size == 1:
            return out.item()
        return out

    def load_rhop_table(self, path):
        """Load a ρ-P → T table from NPZ."""
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        logrhovals = data['logrhovals']
        logp = data['logpvals']
        yv = data['yvals']
        zv = data['zvals']
        self._logt_rhop_rgi = RGI(
            (logrhovals, logp, yv, zv), data['logt_rhop'], **rgi_kw)

    def _build_rhop_yrow(self, yp, zvals, _zm, _za, _zr, logrho, logp):
        """Build all Z slabs for ONE Y' row of the rho-P table.

        Returns a (nR, nP, nZ) slab.  Inner Z loop preserves the
        warm-start chain.  Used by both serial and parallel paths in
        ``build_rhop_table``.
        """
        R_2d, P_2d = np.meshgrid(logrho, logp, indexing='ij')
        nZ = len(zvals)
        out = np.full((len(logrho), len(logp), nZ), np.nan, dtype=float)
        prev_sol = None
        for iz, zv in enumerate(zvals):
            if prev_sol is not None and np.all(np.isfinite(prev_sol)):
                guess = prev_sol
            else:
                guess = np.full_like(R_2d, 0.5 * (1.5 + 7.0))

            def residual(lgt_2d, _yp=yp, _zv=zv,
                         _zm_=_zm, _za_=_za, _zr_=_zr):
                rho_test = self._logrho_pt(P_2d, lgt_2d, _yp, _zv,
                                            _zm_, _za_, _zr_)
                return rho_test - R_2d

            sol, _ = self._newton_1d_vec(
                residual, guess, lo_abs=1.5, hi_abs=7.0)
            out[:, :, iz] = sol
            prev_sol = sol
        return out

    def build_rhop_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         smooth_inverted=False,
                         n_workers=1,
                         verbose=True):
        """Build logT on a uniform (logrho, logP, Y', Z) grid.

        Parameters
        ----------
        yvals, zvals : array_like
            1-D Y' and Z grids.
        _zm, _za, _zr : float
            Fixed nested metal sub-fractions.
        smooth_inverted : bool
            If True, apply a Hampel + light Gaussian (σ=1 grid cell)
            pass along the logρ and logP axes after NaN-fill, and save
            to the ``*_smooth.npz`` variant.  Default False — saves to
            ``*_square.npz`` with no post-inversion smoothing.
        n_workers : int
            If > 1, dispatch Y' rows across this many worker processes.
            Default 1 (serial).
        verbose : bool
            Print progress.
        """
        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        logrho = self.logrho_vals
        logp = self.logp_vals
        nR, nP, nY, nZ = len(logrho), len(logp), len(yvals), len(zvals)

        if verbose:
            print(f"Building rho-P square table: "
                  f"logrho=[{logrho[0]:.2f}, {logrho[-1]:.2f}] ({nR} pts), "
                  f"logP=[{logp[0]:.2f}, {logp[-1]:.2f}] ({nP} pts)")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {nR}x{nP}x{nY}x{nZ} = "
                  f"{nR*nP*nY*nZ:,}")

        # Vectorized Newton on the entire (nR, nP) slab per (Y', Z).
        # _newton_1d_vec applies a vectorized bisection fallback to
        # any cell where Newton oscillates.
        if int(n_workers) > 1 and nY > 1:
            tasks = [(float(yp), zvals, _zm, _za, _zr, logrho, logp)
                     for yp in yvals]
            rows = self._parallel_yrow_dispatch(
                'rhop', tasks, int(n_workers), verbose=verbose)
        else:
            pbar = tqdm(total=nY,
                         desc="Inverting P,T -> rho,P (vectorized)",
                         disable=not verbose,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                    '[{elapsed}<{remaining}]')
            rows = []
            for yp in yvals:
                pbar.set_postfix_str(f"Y'={yp:.3f}")
                rows.append(self._build_rhop_yrow(
                    float(yp), zvals, _zm, _za, _zr, logrho, logp))
                pbar.update(1)
            pbar.close()

        # Stack rows into the (nR, nP, nY, nZ) table
        logt_tab = np.stack(rows, axis=2)

        # --- NaN filling ---
        n_nan = np.isnan(logt_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logt_tab = self._fill_table_nans(logt_tab)

        # Hampel + Gaussian along (logρ, logP) only when
        # smooth_inverted=True.  See build_sp_table for the
        # composition-axis rationale.
        if smooth_inverted:
            if verbose:
                print("Smoothing inverted table (Hampel + Gaussian "
                      "along rho, P) ...")
            logt_tab = self._smooth_inverted_table(
                logt_tab, sigma=1.0, verbose=verbose)

        logt_f32 = logt_tab.astype(np.float32)

        if verbose:
            mem_mb = logt_f32.nbytes / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32)")

        result = {
            'logrhovals':   logrho,
            'logpvals':     logp,
            'yvals':        yvals,
            'zvals':        zvals,
            'logt_rhop':    logt_f32,
            'logt_min':     self.logt_min,
            'logt_max':     self.logt_max,
        }

        # Load into this instance
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        self._logt_rhop_rgi = RGI(
            (logrho, logp, yvals, zvals), logt_f32, **rgi_kw)

        if verbose:
            n_total = logt_tab.size
            n_good = np.isfinite(logt_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def save_rhop_table(self, result, path=None):
        """Save a ρ-P table to NPZ at the canonical auto-load path."""
        if path is None:
            path = self._table_path('rhop')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # S-ρ inversion: P,T(S, ρ, Y', Z) — 2-D least-squares
    # =================================================================

    # -----------------------------------------------------------------
    # 1-D decompositions for S-ρ inversion
    # -----------------------------------------------------------------

    def _srho_via_rhot(self, s_target, rho_target, yp, z,
                       zm=0.0, za=0.0, zr=0.0,
                       prev_lgt=None, use_tab=True):
        """(S, ρ) → (logP, logT) via 1-D root-find in T.

        Uses the ρ-T inversion P(ρ, T) and forward model S(P, T).
        At fixed ρ, S(T) is monotonically increasing, so brentq
        is guaranteed to converge.
        """
        lgt_lo, lgt_hi = self.logt_min, self.logt_max

        def err_t(lgt):
            lgp = self.get_logp_rhot(
                rho_target, lgt, yp, z, _zm=zm, _za=za, _zr=zr,
                use_tab=use_tab)
            if not np.isfinite(lgp):
                return np.nan
            s_val = (self._s_pt(lgp, lgt, yp, z, zm, za, zr)
                     * erg_to_kbbar)
            return float(s_val - s_target)

        guess = (prev_lgt if prev_lgt is not None
                 else 0.5 * (lgt_lo + lgt_hi))
        sol_lgt, ok = self._newton_1d(err_t, guess, lgt_lo, lgt_hi)
        if not ok or not np.isfinite(sol_lgt):
            return np.nan, np.nan
        sol_lgp = self.get_logp_rhot(
            rho_target, sol_lgt, yp, z, _zm=zm, _za=za, _zr=zr,
            use_tab=use_tab)
        return sol_lgp, sol_lgt

    def _srho_via_sp(self, s_target, rho_target, yp, z,
                     zm=0.0, za=0.0, zr=0.0,
                     prev_lgp=None, use_tab=True):
        """(S, ρ) → (logP, logT) via 1-D root-find in P.

        Uses the S-P inversion T(S, P) and forward model ρ(P, T).
        Along an isentrope, ρ(P) is monotonically increasing, so
        brentq is guaranteed to converge.
        """
        lgp_lo, lgp_hi = self.logp_vals[0], self.logp_vals[-1]

        def err_p(lgp):
            lgt = self.get_logt_sp(
                s_target, lgp, yp, z, _zm=zm, _za=za, _zr=zr,
                use_tab=use_tab)
            if not np.isfinite(lgt):
                return np.nan
            lgrho = self._logrho_pt(
                lgp, lgt, yp, z, zm, za, zr)
            return float(lgrho - rho_target)

        guess = (prev_lgp if prev_lgp is not None
                 else 0.5 * (lgp_lo + lgp_hi))
        sol_lgp, ok = self._newton_1d(err_p, guess, lgp_lo, lgp_hi)
        if not ok or not np.isfinite(sol_lgp):
            return np.nan, np.nan
        sol_lgt = self.get_logt_sp(
            s_target, sol_lgp, yp, z, _zm=zm, _za=za, _zr=zr,
            use_tab=use_tab)
        return sol_lgp, sol_lgt

    def get_logp_logt_srho(self, _s_kb, _lgrho, _yp, _z=0.0,
                            _frock=0.0, _zm=0.0, _za=0.0, _zr=None,
                            basis='rhot', use_tab=True, **kw):
        """Pressure and temperature from (S, ρ).

        Rock fraction within Z is the 5th positional ``_frock`` (``_zr``
        keyword overrides).

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
        basis : str, optional
            Which 1-D decomposition to use (default ``'rhot'``).
            ``'rhot'``: root-find in T using the ρ-T inversion
            P(ρ, T) and forward S(P, T).
            ``'sp'``: root-find in P using the S-P inversion
            T(S, P) and forward ρ(P, T).
            Ignored when the pre-computed S-ρ table is loaded
            (``srho_tab=True``), which is used as a fast path.
        use_tab : bool, optional
            If True (default), use pre-computed tables for the
            chosen basis.  Raises ``RuntimeError`` if the required
            table has not been loaded.
            If False, solve per-point via Newton-Raphson (slower
            but does not require pre-computed tables).

        Returns
        -------
        logp, logt : float or array
            log10 P [dyn/cm²] and log10 T [K].
            NaN where the solver fails.
        """
        if '_frock' in kw:
            _zr = kw.pop('_frock')
        if _zr is None:
            _zr = _frock
        if self.rock_interp:
            return self._interp_rock(_zr, *[
                s.get_logp_logt_srho(_s_kb, _lgrho, _yp, _z,
                                     basis=basis, use_tab=use_tab, **kw)
                for s in self._rock_subs])
        _yp = self._to_yprime(_yp, _z)

        # Fast path: use the pre-computed S-ρ table if loaded
        if self._srho_rgi_p is not None:
            return self._lookup_srho_table(_s_kb, _lgrho, _yp, _z)

        use_rhot = (basis == 'rhot')

        # Validate table availability when use_tab=True
        if use_tab:
            if use_rhot and self._logp_rhot_rgi is None:
                raise RuntimeError(
                    "S-ρ inversion with basis='rhot' and use_tab=True "
                    "requires a pre-computed ρ-T table.  "
                    "Build or load one first:\n"
                    "  eos.build_rhot_table(yvals, zvals)   # or\n"
                    "  eos.load_rhot_table(path)\n"
                    "Or set use_tab=False to use per-point "
                    "Newton-Raphson.")
            if not use_rhot and self._logt_sp_rgi is None:
                raise RuntimeError(
                    "S-ρ inversion with basis='sp' and use_tab=True "
                    "requires a pre-computed S-P table.  "
                    "Build or load one first:\n"
                    "  eos.build_sp_table(yvals, zvals)   # or\n"
                    "  eos.load_sp_table(path)\n"
                    "Or set use_tab=False to use per-point "
                    "Newton-Raphson.")

        scalar = np.isscalar(_s_kb) and np.isscalar(_lgrho)
        _s_kb  = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _s_kb, _lgrho = np.broadcast_arrays(_s_kb, _lgrho)

        lgp_out = np.full_like(_s_kb, np.nan, dtype=float)
        lgt_out = np.full_like(_s_kb, np.nan, dtype=float)

        prev_val = None  # logT for rhot, logP for sp

        for idx in np.ndindex(_s_kb.shape):
            s_i   = float(_s_kb[idx])
            rho_i = float(_lgrho[idx])

            if use_rhot:
                lgp, lgt = self._srho_via_rhot(
                    s_i, rho_i, _yp, _z, _zm, _za, _zr,
                    prev_lgt=prev_val, use_tab=use_tab)
            else:
                lgp, lgt = self._srho_via_sp(
                    s_i, rho_i, _yp, _z, _zm, _za, _zr,
                    prev_lgp=prev_val, use_tab=use_tab)

            if np.isfinite(lgp) and np.isfinite(lgt):
                lgp_out[idx] = lgp
                lgt_out[idx] = lgt
                prev_val = lgt if use_rhot else lgp

        if scalar:
            return lgp_out.item(), lgt_out.item()
        return lgp_out, lgt_out

    def _lookup_srho_table(self, _s_kb, _lgrho, _yp, _z):
        """Query the (S, logrho, Y', Z) S-ρ inversion RGI tables."""
        _s_kb  = np.atleast_1d(np.asarray(_s_kb, dtype=float))
        _lgrho = np.atleast_1d(np.asarray(_lgrho, dtype=float))
        _yp_a  = np.atleast_1d(np.asarray(_yp, dtype=float))
        _z_a   = np.atleast_1d(np.asarray(_z, dtype=float))
        _s_kb, _lgrho, _yp_a, _z_a = np.broadcast_arrays(
            _s_kb, _lgrho, _yp_a, _z_a)
        pts = np.column_stack((_s_kb.ravel(), _lgrho.ravel(),
                               _yp_a.ravel(), _z_a.ravel()))
        lgp_out = self._srho_rgi_p(pts).reshape(_s_kb.shape)
        lgt_out = self._srho_rgi_t(pts).reshape(_s_kb.shape)
        if lgp_out.size == 1:
            return lgp_out.item(), lgt_out.item()
        return lgp_out, lgt_out

    def load_srho_table(self, path):
        """Load a pre-computed S-ρ table from NPZ."""
        data = np.load(path)
        self.logt_min = float(data['logt_min'])
        self.logt_max = float(data['logt_max'])
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        svals = data['svals']
        logrho = data['logrhovals']
        yv = data['yvals']
        zv = data['zvals']
        self._srho_rgi_p = RGI(
            (svals, logrho, yv, zv), data['logp_srho'], **rgi_kw)
        self._srho_rgi_t = RGI(
            (svals, logrho, yv, zv), data['logt_srho'], **rgi_kw)
        self._svals_srho = svals

    def _build_srho_yrow(self, yp, zvals, _zm, _za, _zr,
                          svals, logrho, lo_abs, hi_abs):
        """Build all Z slabs for ONE Y' row of the S-rho table.

        Returns a tuple (logp_slab, logt_slab), each (nS, nR, nZ).
        Inner Z loop preserves the warm-start chain in logT.  The
        residual reads P from the worker's own rho-T inversion table
        at every Newton iteration.  Used by both serial and parallel
        paths in ``build_srho_table``.
        """
        S_2d, R_2d = np.meshgrid(svals, logrho, indexing='ij')
        nZ = len(zvals)
        nS, nR = len(svals), len(logrho)
        out_p = np.full((nS, nR, nZ), np.nan, dtype=float)
        out_t = np.full((nS, nR, nZ), np.nan, dtype=float)
        prev_sol = None
        for iz, zv in enumerate(zvals):
            if prev_sol is not None and np.all(np.isfinite(prev_sol)):
                guess = prev_sol
            else:
                guess = np.full_like(S_2d, 0.5 * (lo_abs + hi_abs))

            def residual(lgt_2d, _yp=yp, _zv=zv,
                         _zm_=_zm, _za_=_za, _zr_=_zr):
                lgp = self.get_logp_rhot(
                    R_2d, lgt_2d, _yp, _zv, _zm=_zm_, _za=_za_, _zr=_zr_,
                    use_tab=True)
                s_test = self._s_pt(
                    lgp, lgt_2d, _yp, _zv,
                    _zm_, _za_, _zr_) * erg_to_kbbar
                return s_test - S_2d

            sol, _ = self._newton_1d_vec(
                residual, guess, lo_abs=lo_abs, hi_abs=hi_abs)

            # Recover P from the rho-T table at the converged T
            lgp_slab = self.get_logp_rhot(
                R_2d, sol, yp, zv, _zm=_zm, _za=_za, _zr=_zr,
                use_tab=True)

            bad_final = ~(np.isfinite(lgp_slab) & np.isfinite(sol))
            out_p[:, :, iz] = np.where(bad_final, np.nan, lgp_slab)
            out_t[:, :, iz] = np.where(bad_final, np.nan, sol)
            prev_sol = sol
        return out_p, out_t

    def build_srho_table(self, yvals, zvals,
                         _zm=0.0, _za=0.0, _zr=0.0,
                         s_lo=4.0, s_hi=12.0, s_step=0.1,
                         smooth_inverted=False,
                         n_workers=1,
                         verbose=True):
        """Build logP, logT on a uniform (S, logrho, Y', Z) grid.

        Decomposes the 2-D (S, ρ) → (P, T) inversion into a 1-D
        outer Newton solve in logT, with the inner step using the
        pre-computed ρ-T inversion table to recover P at fixed
        (ρ, T):

            residual(T) = S_PT(P=P_rhoT(ρ, T, Y', Z), T, Y', Z) - S_target

        The outer Newton iterates only over T; P is read from the
        ρ-T table at each iteration.  After convergence in T, the
        final P is read once more from the ρ-T table.

        Requires the pre-computed ρ-T inversion table (built first
        with ``build_rhot_table``) and the PT forward table.

        Parameters
        ----------
        yvals, zvals : array_like
            1-D Y' and Z grids.
        _zm, _za, _zr : float
            Fixed nested metal sub-fractions.
        s_lo, s_hi, s_step : float
            Entropy range and step in kb/baryon.
        smooth_inverted : bool
            If True, apply a Hampel + light Gaussian (σ=1 grid cell)
            pass along the S and logρ axes after NaN-fill, and save
            to the ``*_smooth.npz`` variant.  Default False — saves to
            ``*_square.npz`` with no post-inversion smoothing.  Both
            ``logp_srho`` and ``logt_srho`` are smoothed.
        verbose : bool
            Print progress.
        """
        if self._logp_rhot_rgi is None:
            raise RuntimeError(
                "build_srho_table requires a pre-computed rho-T table "
                "(load via inv_tab=True or call load_rhot_table).  "
                "Build it first:\n"
                "  python eos_inversions.py --basis rhot ...")
        if self._s_pt_rgi is None:
            raise RuntimeError(
                "build_srho_table requires a pre-computed P-T table "
                "(load via pt_tab=True or call load_pt_table).  "
                "Build it first:\n"
                "  python eos_inversions.py --basis pt ...")

        yvals = np.asarray(yvals, dtype=float)
        zvals = np.asarray(zvals, dtype=float)
        svals = np.arange(s_lo, s_hi + s_step * 0.1, s_step)
        logrho = self.logrho_vals
        nS, nR, nY, nZ = len(svals), len(logrho), len(yvals), len(zvals)

        if verbose:
            print(f"Building S-rho square table: "
                  f"S=[{svals[0]:.2f}, {svals[-1]:.2f}] ({nS} pts), "
                  f"logrho=[{logrho[0]:.2f}, {logrho[-1]:.2f}] ({nR} pts)")
            print(f"  Y' grid: {nY} pts, Z grid: {nZ} pts")
            print(f"  Total cells: {nS}x{nR}x{nY}x{nZ} = "
                  f"{nS*nR*nY*nZ:,}")
            print(f"  Decomposition: 1-D outer Newton in T using "
                  f"the rho-T table for the inner P(rho,T) step.")

        # Outer Newton bounds in logT
        lo_abs, hi_abs = 1.5, 7.0

        # Vectorized 1-D outer Newton in logT on the entire (nS, nR)
        # slab per (Y', Z).  Inner P(rho, T) lookup is array-aware via
        # the loaded rho-T RGI, so each vectorized Newton iteration is
        # a few batched RGI calls.  _newton_1d_vec applies a vectorized
        # bisection fallback to any cell where Newton oscillates.
        if int(n_workers) > 1 and nY > 1:
            tasks = [(float(yp), zvals, _zm, _za, _zr,
                      svals, logrho, lo_abs, hi_abs)
                     for yp in yvals]
            row_results = self._parallel_yrow_dispatch(
                'srho', tasks, int(n_workers), verbose=verbose)
        else:
            pbar = tqdm(total=nY,
                         desc="Inverting P,T -> S,rho (1-D via rho-T)",
                         disable=not verbose,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                    '[{elapsed}<{remaining}]')
            row_results = []
            for yp in yvals:
                pbar.set_postfix_str(f"Y'={yp:.3f}")
                row_results.append(self._build_srho_yrow(
                    float(yp), zvals, _zm, _za, _zr,
                    svals, logrho, lo_abs, hi_abs))
                pbar.update(1)
            pbar.close()

        # Each row_results entry is (logp_slab, logt_slab) of shape
        # (nS, nR, nZ).  Stack along the Y' axis (axis=2).
        logp_tab = np.stack([r[0] for r in row_results], axis=2)
        logt_tab = np.stack([r[1] for r in row_results], axis=2)

        # --- NaN filling ---
        n_nan = np.isnan(logp_tab).sum()
        if n_nan > 0:
            if verbose:
                print(f"Filling {n_nan} NaN cells by interpolation ...")
            logp_tab = self._fill_table_nans(logp_tab)
            logt_tab = self._fill_table_nans(logt_tab)

        # Hampel + Gaussian along (S, logρ) for both arrays only when
        # smooth_inverted=True.  See build_sp_table for the
        # composition-axis rationale.
        if smooth_inverted:
            if verbose:
                print("Smoothing inverted table (Hampel + Gaussian "
                      "along S, rho) ...")
            logp_tab = self._smooth_inverted_table(
                logp_tab, sigma=1.0, verbose=verbose)
            logt_tab = self._smooth_inverted_table(
                logt_tab, sigma=1.0, verbose=verbose)

        logp_f32 = logp_tab.astype(np.float32)
        logt_f32 = logt_tab.astype(np.float32)

        if verbose:
            mem_mb = (logp_f32.nbytes + logt_f32.nbytes) / 1e6
            print(f"Table size: {mem_mb:.1f} MB (float32, P+T)")

        result = {
            'svals':        svals.astype(np.float32),
            'logrhovals':   logrho,
            'yvals':        yvals,
            'zvals':        zvals,
            'logp_srho':    logp_f32,
            'logt_srho':    logt_f32,
            'logt_min':     self.logt_min,
            'logt_max':     self.logt_max,
        }

        # Load into this instance
        rgi_kw = dict(method=self._interp_method, bounds_error=False,
                      fill_value=None)
        self._srho_rgi_p = RGI(
            (svals, logrho, yvals, zvals), logp_f32, **rgi_kw)
        self._srho_rgi_t = RGI(
            (svals, logrho, yvals, zvals), logt_f32, **rgi_kw)
        self._svals_srho = svals

        if verbose:
            n_total = logp_tab.size
            n_good = np.isfinite(logp_tab).sum()
            print(f"Done. {n_good}/{n_total} cells finite "
                  f"({100*n_good/n_total:.1f}%), "
                  f"{n_nan} were interpolated")

        return result

    def save_srho_table(self, result, path=None):
        """Save an S-ρ table to NPZ at the canonical auto-load path."""
        if path is None:
            path = self._table_path('srho')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        print(f"Saved {path}")

    # =================================================================
    # Thermodynamic derivatives
    #
    # Each derivative is single-basis: the basis is implied by the
    # method-name suffix and dictates which forward / inversion call
    # the method uses for the central difference.  Public inversions
    # (get_logt_sp, get_logp_rhot, get_logt_rhop, get_logp_logt_srho)
    # already handle "use table when loaded, fall back to root-finding"
    # internally — derivatives never call val_mixtures or _newton_*
    # directly.  No method= switch, no thermodynamic-identity
    # alternative.
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

    # =================================================================
    # PT-basis derivatives
    # =================================================================

    @staticmethod
    def _fd_xpair(x, dx, lo=0.0, hi=1.0, dx_min=1e-3):
        """Composition finite-difference abscissae, robust at the edges.

        Returns (x_m, x_p, denom) with x_m/x_p clipped to [lo, hi] and
        denom = x_p - x_m.  The step is floored at dx_min so a shrunken
        step (e.g. adaptive_dx -> 1e-6 at Z = 0) cannot amplify
        table-interpolation noise by a vanishing denominator; near the
        edges the stencil degrades gracefully to a one-sided difference
        of full width instead of stepping outside the table domain.
        """
        x = np.asarray(x, dtype=float)
        dx_eff = np.maximum(np.asarray(dx, dtype=float), dx_min)
        x_m = np.clip(x - dx_eff, lo, hi)
        x_p = np.clip(x + dx_eff, lo, hi)
        denom = np.maximum(x_p - x_m, dx_min)
        return x_m, x_p, denom

    def get_dsdy_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dy=0.01, **kw):
        """dS/dY|_{P,T} (total Y) via FD on the P-T forward model.

        Differentiates w.r.t. TOTAL Y: the step is taken in total Y and
        get_s_pt_tab performs the single Y->Y' conversion per point.  Do
        NOT pre-convert here (the leaf converts) -- see get_dtds_sp note.
        """
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        s1 = self.get_s_pt_tab(_lgp, _lgt, y_m, _z, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, y_p, _z, _frock)
        return (s2 - s1) / dy2

    def get_dsdz_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dz=0.01, **kw):
        """dS/dZ|_{P,T} (total Y held fixed) via FD on the P-T forward model."""
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        s1 = self.get_s_pt_tab(_lgp, _lgt, _y, z_m, _frock)
        s2 = self.get_s_pt_tab(_lgp, _lgt, _y, z_p, _frock)
        return (s2 - s1) / dz2

    def get_dlogrho_dlogt_py(self, _lgp, _lgt, _y, _z, _frock=0.0,
                              dt=1e-2, **kw):
        """dlogρ/dlogT|_P (= -δ) via FD on the P-T forward model."""
        _y = self._to_yprime(_y, _z)
        r1 = self._logrho_pt(_lgp, _lgt - dt, _y, _z, _zr=_frock)
        r2 = self._logrho_pt(_lgp, _lgt + dt, _y, _z, _zr=_frock)
        return (r2 - r1) / (2 * dt)

    def get_cp_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, dt=1e-2, **kw):
        """C_P = dS/d(lnT)|_P  [erg/(g·K)] via FD on the P-T forward model."""
        _y = self._to_yprime(_y, _z)
        s1 = self._s_pt(_lgp, _lgt - dt, _y, _z, _zr=_frock)
        s2 = self._s_pt(_lgp, _lgt + dt, _y, _z, _zr=_frock)
        return (s2 - s1) / (2 * dt * log10_to_loge)

    # =================================================================
    # S-P-basis derivatives
    # =================================================================

    def get_nabla_ad(self, _s, _lgp, _y, _z, _frock=0.0, dp=1e-2, **kw):
        """∇_ad = dlogT/dlogP|_S via FD on the S-P inversion."""
        lgt1 = self.get_logt_sp(_s, _lgp - dp, _y, _z, _zr=_frock)
        lgt2 = self.get_logt_sp(_s, _lgp + dp, _y, _z, _zr=_frock)
        return (lgt2 - lgt1) / (2 * dp)

    def get_gamma1(self, _s, _lgp, _y, _z, _frock=0.0, dp=1e-2, **kw):
        """Γ₁ = dlogP/dlogρ|_S via FD on the S-P inversion + ρ(P,T)."""
        lgt1 = self.get_logt_sp(_s, _lgp - dp, _y, _z, _zr=_frock)
        lgt2 = self.get_logt_sp(_s, _lgp + dp, _y, _z, _zr=_frock)
        r1 = self.get_logrho_pt_tab(_lgp - dp, lgt1, _y, _z, _frock)
        r2 = self.get_logrho_pt_tab(_lgp + dp, lgt2, _y, _z, _frock)
        return (2 * dp) / (r2 - r1)

    def get_dlogrho_ds_py(self, _s, _lgp, _y, _z, _frock=0.0,
                           ds=0.1, **kw):
        """dlogρ/dS|_P (Brunt coefficient in dρ space).

        FD on the S-P inversion: T(S±dS, P) → ρ(P, T) → difference.
        """
        lgt1 = self.get_logt_sp(_s - ds, _lgp, _y, _z, _zr=_frock)
        lgt2 = self.get_logt_sp(_s + ds, _lgp, _y, _z, _zr=_frock)
        r1 = self.get_logrho_pt_tab(_lgp, lgt1, _y, _z, _frock)
        r2 = self.get_logrho_pt_tab(_lgp, lgt2, _y, _z, _frock)
        return (r2 - r1) * log10_to_loge / (2 * ds / erg_to_kbbar)

    def get_dtds_sp(self, _s, _lgp, _y, _z, _frock=0.0, ds=0.1, **kw):
        """dT/dS|_P [K·g·K/erg] via FD on the S-P inversion.

        NOTE: get_logt_sp() already calls self._to_yprime(_yp, _z) at
        line 2335, so we MUST NOT call it here as well -- doing so
        applies the (1-Z) division twice when self.y_prime=False, which
        sends the lookup to Y'/(1-Z) (e.g. Y'=0.275 becomes 5.5 at
        Z=0.95) and produces garbage extrapolation. Pass the caller's
        _y straight through to get_logt_sp and let it do the conversion.
        """
        t1 = 10.0 ** self.get_logt_sp(_s - ds, _lgp, _y, _z, _zr=_frock)
        t2 = 10.0 ** self.get_logt_sp(_s + ds, _lgp, _y, _z, _zr=_frock)
        return (t2 - t1) * erg_to_kbbar / (2 * ds)

    # =================================================================
    # ρ-T-basis derivatives
    # =================================================================

    def get_cv_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0, dt=1e-2, **kw):
        """C_V = dS/d(lnT)|_ρ  [erg/(g·K)].

        ρ-T basis: at each T±dT, find P via ρ-T inversion, then S(P,T).
        """
        p1 = self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _zr=_frock)
        p2 = self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _zr=_frock)
        s1 = self.get_s_pt_tab(p1, _lgt - dt, _y, _z, _frock)
        s2 = self.get_s_pt_tab(p2, _lgt + dt, _y, _z, _frock)
        return (s2 - s1) / (2 * dt * log10_to_loge)

    def get_dlogt_dy_rhop_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0,
                                dy=0.01, dt=0.1, **kw):
        """dlogT/dY|_{ρ,P} = χ_Y / χ_T via FD on the ρ-T inversion.

        The composition stencil is edge-clipped via _fd_xpair so it
        never queries Y < 0 or Y > 1 (one-sided at the edges).
        """
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        chi_y = (self.get_logp_rhot(_lgrho, _lgt, y_p, _z, _zr=_frock)
                 - self.get_logp_rhot(_lgrho, _lgt, y_m, _z, _zr=_frock)
                 ) * log10_to_loge / dy2
        chi_t = (self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _zr=_frock)
                 - self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _zr=_frock)
                 ) / (2 * dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(chi_t) < 1e-30, np.nan, chi_y / chi_t)

    def get_dlogt_dz_rhop_rhot(self, _lgrho, _lgt, _y, _z, _frock=0.0,
                                dz=0.01, dt=0.1, **kw):
        """dlogT/dZ|_{ρ,P} = χ_Z / χ_T via FD on the ρ-T inversion.

        The composition stencil is edge-clipped via _fd_xpair so it
        never queries Z < 0 or Z > 1 (one-sided at the edges).
        """
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        chi_z = (self.get_logp_rhot(_lgrho, _lgt, _y, z_p, _zr=_frock)
                 - self.get_logp_rhot(_lgrho, _lgt, _y, z_m, _zr=_frock)
                 ) * log10_to_loge / dz2
        chi_t = (self.get_logp_rhot(_lgrho, _lgt + dt, _y, _z, _zr=_frock)
                 - self.get_logp_rhot(_lgrho, _lgt - dt, _y, _z, _zr=_frock)
                 ) / (2 * dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(chi_t) < 1e-30, np.nan, chi_z / chi_t)

    def get_dpdt_rhot_rhoy(self, _lgrho, _lgt, _y, _z, _frock=0.0,
                            dT=0.1, **kw):
        """dP/dT|_{ρ,Y} [dyn/cm²/K] via FD on the ρ-T inversion."""
        T0 = 10.0 ** np.asarray(_lgt)
        T1 = T0 * (1 - dT)
        T2 = T0 * (1 + dT)
        P1 = 10.0 ** self.get_logp_rhot(_lgrho, np.log10(T1), _y, _z, _zr=_frock)
        P2 = 10.0 ** self.get_logp_rhot(_lgrho, np.log10(T2), _y, _z, _zr=_frock)
        return (P2 - P1) / (T2 - T1)

    # =================================================================
    # ρ-P-basis derivatives (Ledoux dS at fixed ρ, P)
    # =================================================================

    def get_dsdy_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1,
                            dy=0.01, **kw):
        """dS/dY|_{ρ,P} (Ledoux).  Takes (S, ρ) inputs.

        Inverts (S, ρ) → P first, then FD at fixed (ρ, P) by varying Y
        and using the ρ-P inversion to find T(ρ, P, Y±dY, Z).

        Differentiates w.r.t. TOTAL Y; the leaf inversions perform the
        single Y->Y' conversion, so we MUST NOT pre-convert here.
        """
        # dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kw)
        #dPdY|{S, rho, Y}:
        dpdy_srho = self.get_dpdy_srho(_s, _lgrho, _y, _z, _frock, dy=dy, **kw)

        #dSdY|{rho, P, Z} = -dPdY|{S, rho, Y} / dPdS|{rho, Y, Z}
        dsdy_rhopy = -dpdy_srho/dpds_rhoy_srho # triple product rule

        return dsdy_rhopy
    
    def get_dsdy_rhop(self, _lgrho, _lgp, _y, _z, _frock=0.0,
                            dy=0.01, **kw):
        """dS/dY|_{ρ,P} (Ledoux).  Takes (S, ρ) inputs.

        Inverts (S, ρ) → P first, then FD at fixed (ρ, P) by varying Y
        and using the ρ-P inversion to find T(ρ, P, Y±dY, Z).
        Differentiates w.r.t. TOTAL Y (leaves convert; do not pre-convert).
        """
        lgt_m = self.get_logt_rhop(_lgrho, _lgp, _y - dy, _z, _zr=_frock)
        lgt_p = self.get_logt_rhop(_lgrho, _lgp, _y + dy, _z, _zr=_frock)
        s_m = self.get_s_pt(_lgp, lgt_m, _y - dy, _z, _frock)
        s_p = self.get_s_pt(_lgp, lgt_p, _y + dy, _z, _frock)
        return (s_p - s_m) / (2 * dy)
    
    def get_dsdz_rhop_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1,
                            dz=0.01, **kw):
        """dS/dZ|_{ρ,P} (Ledoux).  Takes (S, ρ) inputs.

        Inverts (S, ρ) → P first, then FD at fixed (ρ, P) by varying Z
        and using the ρ-P inversion to find T(ρ, P, Y, Z±dZ).
        Total Y held fixed (leaves convert; do not pre-convert here).
        """
        # dPdS|{rho, Y, Z}:
        dpds_rhoy_srho = self.get_dpds_rhoy_srho(_s, _lgrho, _y, _z, _frock, ds=ds, **kw)
        #dPdZ|{S, rho, Z}:
        dpdz_srho = self.get_dpdz_srho(_s, _lgrho, _y, _z, _frock, dz=dz, **kw)
        #dSdZ|{rho, P, Y} = -dPdZ|{S, rho, Z} / dPdS|{rho, Y, Z}
        dsdz_rhopz = -dpdz_srho/dpds_rhoy_srho # triple product rule
        return dsdz_rhopz

    def get_dsdz_rhop(self, _lgrho, _lgp, _y, _z, _frock=0.0,
                            dz=0.01, **kw):
        """dS/dZ|_{ρ,P} (Ledoux).  Takes (S, ρ) inputs.

        Inverts (S, ρ) → P first, then FD at fixed (ρ, P) by varying Z
        and using the ρ-P inversion to find T(ρ, P, Y, Z±dZ).
        Total Y held fixed (leaves convert; do not pre-convert here).
        """
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        lgt_m = self.get_logt_rhop(_lgrho, _lgp, _y, z_m, _zr=_frock)
        lgt_p = self.get_logt_rhop(_lgrho, _lgp, _y, z_p, _zr=_frock)
        s_m = self.get_s_pt_tab(_lgp, lgt_m, _y, z_m, _frock)
        s_p = self.get_s_pt_tab(_lgp, lgt_p, _y, z_p, _frock)
        return (s_p - s_m) / dz2

    # =================================================================
    # S-ρ-basis derivatives
    # =================================================================

    def get_dpds_rhoy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, ds=0.1, **kw):
        """dP/dS|_{ρ,Y,Z} via FD on the S-ρ inversion."""
        lgp_m, _ = self.get_logp_logt_srho(_s - ds, _lgrho, _y, _z, _zr=_frock)
        lgp_p, _ = self.get_logp_logt_srho(_s + ds, _lgrho, _y, _z, _zr=_frock)
        return (10.0 ** lgp_p - 10.0 ** lgp_m) / (2 * ds / erg_to_kbbar)
    
    def get_dpdy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.01, **kw):
        """dP/dY|_{S,ρ} (total Y) via FD on the S-ρ inversion."""
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        lgp_m, _ = self.get_logp_logt_srho(_s, _lgrho, y_m, _z, _zr=_frock)
        lgp_p, _ = self.get_logp_logt_srho(_s, _lgrho, y_p, _z, _zr=_frock)
        return (10.0 ** lgp_p - 10.0 ** lgp_m) / dy2

    def get_dpdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.01, **kw):
        """dP/dZ|_{S,ρ} via FD on the S-ρ inversion."""
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        lgp_m, _ = self.get_logp_logt_srho(_s, _lgrho, _y, z_m, _zr=_frock)
        lgp_p, _ = self.get_logp_logt_srho(_s, _lgrho, _y, z_p, _zr=_frock)
        return (10.0 ** lgp_p - 10.0 ** lgp_m) / dz2

    def get_dtdy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.01, **kw):
        """dT/dY|_{S,ρ} (total Y) via FD on the S-ρ inversion."""
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        _, lgt_m = self.get_logp_logt_srho(_s, _lgrho, y_m, _z, _zr=_frock)
        _, lgt_p = self.get_logp_logt_srho(_s, _lgrho, y_p, _z, _zr=_frock)
        return (10.0 ** lgt_p - 10.0 ** lgt_m) / dy2

    def get_dtdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.01, **kw):
        """dT/dZ|_{S,ρ} via FD on the S-ρ inversion."""
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        _, lgt_m = self.get_logp_logt_srho(_s, _lgrho, _y, z_m, _zr=_frock)
        _, lgt_p = self.get_logp_logt_srho(_s, _lgrho, _y, z_p, _zr=_frock)
        return (10.0 ** lgt_p - 10.0 ** lgt_m) / dz2

    def get_dudy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.01, **kw):
        """dU/dY|_{S,ρ} (total Y) via FD on the S-ρ inversion + U(P,T).

        Total Y; the leaves convert.  get_logu_pt_tab takes total Y and
        does the single conversion, so we do NOT pre-convert here.
        """
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        p_m, t_m = self.get_logp_logt_srho(_s, _lgrho, y_m, _z, _zr=_frock)
        p_p, t_p = self.get_logp_logt_srho(_s, _lgrho, y_p, _z, _zr=_frock)
        u_m = 10.0 ** self.get_logu_pt_tab(p_m, t_m, y_m, _z, _frock)
        u_p = 10.0 ** self.get_logu_pt_tab(p_p, t_p, y_p, _z, _frock)
        return (u_p - u_m) / dy2

    def get_dudz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.01, **kw):
        """dU/dZ|_{S,ρ} via FD on the S-ρ inversion + U(P,T)."""
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        p_m, t_m = self.get_logp_logt_srho(_s, _lgrho, _y, z_m, _zr=_frock)
        p_p, t_p = self.get_logp_logt_srho(_s, _lgrho, _y, z_p, _zr=_frock)
        u_m = 10.0 ** self.get_logu_pt_tab(p_m, t_m, _y, z_m, _frock)
        u_p = 10.0 ** self.get_logu_pt_tab(p_p, t_p, _y, z_p, _frock)
        return (u_p - u_m) / dz2

    # =================================================================
    # Forward-model aliases (legacy ORCHARD interface)
    # =================================================================

    def adaptive_dx(self, x, dx0=0.01):
        """Alias for _adaptive_dx (old mixtures interface)."""
        return self._adaptive_dx(x, dx0)

    def get_s_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """S(P, T, Y, Z).  Delegates to get_s_pt_tab (forwards _frock)."""
        return self.get_s_pt_tab(_lgp, _lgt, _y, _z, _frock=_frock, **kw)

    def get_logrho_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """log10 ρ(P, T, Y, Z).  Delegates to get_logrho_pt_tab (forwards _frock)."""
        return self.get_logrho_pt_tab(_lgp, _lgt, _y, _z, _frock=_frock, **kw)

    def get_logu_pt(self, _lgp, _lgt, _y, _z, _frock=0.0, **kw):
        """log10 U(P, T, Y, Z).  Delegates to get_logu_pt_tab (forwards _frock)."""
        return self.get_logu_pt_tab(_lgp, _lgt, _y, _z, _frock=_frock, **kw)


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
        """Finite-difference step for composition derivatives.

        Historically this shrank dx to x near the domain edges, which
        collapsed the step to 0 at Z = 0 (or Y = 0) and made the FD
        derivatives divide table-interpolation noise by a vanishing
        denominator. The FD methods now clip their stencils to [0, 1]
        internally (see _fd_xpair), so the step no longer needs to
        shrink at the edges -- it only needs a finite floor.
        """
        x = np.asarray(x_profile, dtype=float)
        dx = np.full_like(x, initial_dx)
        dx = np.minimum(dx, np.maximum(x, tolerance))
        dx = np.minimum(dx, np.maximum(1.0 - x, tolerance))
        return np.maximum(dx, tolerance)

    @staticmethod
    def _fd_xpair(x, dx, lo=0.0, hi=1.0, dx_min=1e-3):
        """Composition finite-difference abscissae, robust at the edges.

        Returns (x_m, x_p, denom) with x_m/x_p clipped to [lo, hi] and
        denom = x_p - x_m. The step is floored at dx_min so a shrunken
        step cannot amplify table-interpolation noise by a vanishing
        denominator; near the edges the stencil degrades gracefully to
        a one-sided difference of full width instead of stepping
        outside the table domain (e.g. negative Z).
        """
        x = np.asarray(x, dtype=float)
        dx_eff = np.maximum(np.asarray(dx, dtype=float), dx_min)
        x_m = np.clip(x - dx_eff, lo, hi)
        x_p = np.clip(x + dx_eff, lo, hi)
        denom = np.maximum(x_p - x_m, dx_min)
        return x_m, x_p, denom


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
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        p1 = 10**self.get_logp_srho(_s, _lgrho, y_m, _z, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, y_p, _z, **kwargs)

        return (p2 - p1) / dy2


    def get_dpdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        p1 = 10**self.get_logp_srho(_s, _lgrho, _y, z_m, _frock, **kwargs)
        p2 = 10**self.get_logp_srho(_s, _lgrho, _y, z_p, _frock, **kwargs)

        return (p2 - p1) / dz2

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
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        u1 = 10**self.get_logu_srho(_s, _lgrho, y_m, _z, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, y_p, _z, _frock, **kwargs)

        return (u2 - u1)/dy2

    def get_dudz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_p_guess=None, arr_t_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_p_guess': arr_p_guess, 'arr_t_guess': arr_p_guess, 'method': method, 'tab':tab}
        dz = _z*0.1 if dz is None else dz
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        u1 = 10**self.get_logu_srho(_s, _lgrho, _y, z_m, _frock, **kwargs)
        u2 = 10**self.get_logu_srho(_s, _lgrho, _y, z_p, _frock, **kwargs)

        return (u2 - u1)/dz2

    ########### Conductive Flux Terms ###########

    def get_dtdy_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dy=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        y_m, y_p, dy2 = self._fd_xpair(_y, dy)
        t1 = 10**self.get_logt_srho(_s, _lgrho, y_m, _z, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, y_p, _z, _frock, **kwargs)

        return (t2 - t1)/dy2

    def get_dtdz_srho(self, _s, _lgrho, _y, _z, _frock=0.0, dz=0.1, ideal_guess=True, arr_guess=None, method='newton_brentq', tab=True):
        kwargs = {'ideal_guess': ideal_guess, 'arr_guess': arr_guess, 'method': method, 'tab':tab}
        z_m, z_p, dz2 = self._fd_xpair(_z, dz)
        t1 = 10**self.get_logt_srho(_s, _lgrho, _y, z_m, _frock, **kwargs)
        t2 = 10**self.get_logt_srho(_s, _lgrho, _y, z_p, _frock, **kwargs)

        return (t2 - t1)/dz2

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
