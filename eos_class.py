import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
import const
import pdb
import pandas as pd
from tqdm import tqdm
from numba import njit
from eos import ideal_eos, metals_eos, ice_eos
from scipy.optimize import root, newton, brentq, brenth, minimize
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from astropy.constants import k_B
from astropy.constants import u as amu
from astropy import units as u

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ideal_xy = ideal_eos.IdealHHeMix()

mh = 1
mhe = 4.0026

ices = ice_eos.ice_eos() # for water, methane, ammonia, rock, and iron mixture

##### useful unit conversions #####

mp = amu.to('g') # grams
kb = k_B.to('erg/K') # ergs/K
erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/mp)
MJ_to_kbbar = (u.MJ/u.Kelvin/u.kg).to(k_B/amu)
dyn_to_bar = (u.dyne/(u.cm)**2).to('bar')
erg_to_MJ = (u.erg/u.Kelvin/u.gram).to(u.MJ/u.Kelvin/u.kg)
MJ_to_erg = (u.MJ/u.kg).to('erg/g')

log10_to_loge = np.log(10)

class hhe:
    def __init__(self, hhe_eos):
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


class mixtures(hhe):
    def __init__(self, hhe_eos,
                    z_eos,
                    zmix_eos1 = 'aqua',
                    zmix_eos2 = 'methane',
                    zmix_eos3 = 'ammonia',
                    zmix_eos4 = 'ppv2',
                    zmix_eos5 = 'iron2',
                    hg=True,
                    y_prime=False,
                    interp_method='linear',
                    new_z_mix=False,
                    rhot_sp_inv = False,
                    srho_rhop_inv = False
                    ):

        super().__init__(hhe_eos=hhe_eos)

        self.y_prime = y_prime
        self.hg = hg
        self.z_eos = z_eos
        self.new_z_mix = new_z_mix

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
        m_methane = 17.031
        m_ammonia = 16.04
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

        s_xy = np.zeros_like(_lgp)

        s_x = 10 ** self.get_s_h(_lgp, _lgt)
        s_y = 10 ** self.get_s_he(_lgp, _lgt)

        s_xy = s_x * (1 - _y_prime) + s_y * _y_prime

        if self.z_eos == 'mixture':
            s_z = metals_eos.get_s_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=self.f_ppv, f_fe=self.f_fe,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
        elif self.z_eos == 'ice_mixture':
            s_z = ices.get_s_pt_val(_lgp, _lgt, _zm, _za)
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _z, _zm=_zm, _za=_za, _zr=0.0, _zfe=0.0) / erg_to_kbbar

        elif self.z_eos == 'ice_rock_mixture':
            s_z_ice = ices.get_s_pt_val(_lgp, _lgt, _zm, _za)
            s_z_rock = metals_eos.get_s_pt_tab(_lgp, _lgt, eos='ppv2')

            s_z = (1 - _zr) * s_z_ice + _zr * s_z_rock
            smix_xyz_ideal = self.get_smix_ideal(_y_prime, _z, _zm=_zm, _za=_za, _zr=_zr, _zfe=0.0) / erg_to_kbbar

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

        rho_h = 10 ** self.get_logrho_h(_lgp, _lgt)
        rho_he = 10 ** self.get_logrho_he(_lgp, _lgt)

        if self.z_eos == 'mixture':
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=self.f_ppv, f_fe=self.f_fe,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
        elif self.z_eos == 'ice_mixture':
            rho_z = 10 ** ices.get_logrho_pt_val(_lgp, _lgt, _zm, _za)
        elif self.z_eos == 'ice_rock_mixture':
            rho_z_ice = 10 ** ices.get_logrho_pt_val(_lgp, _lgt, _zm, _za)
            rho_z_rock = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos='ppv2')

            rho_z = 1 / ((1 - _zr) / rho_z_ice + _zr / rho_z_rock)

        elif self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
            self.z_eos = 'aqua'
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)
        else:
            rho_z = 10 ** metals_eos.get_rho_pt_tab(_lgp, _lgt, eos=self.z_eos)

        mixture_density = (1 - _y_prime) * (1 - _z) / rho_h + _y_prime * (1 - _z) / rho_he + vmix * (1 - _z) + _z / rho_z

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

        u_h = 10 ** self.get_logu_h(_lgp, _lgt)
        u_he = 10 ** self.get_logu_he(_lgp, _lgt)
        if self.z_eos == 'mixture':
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos, f_ppv=self.f_ppv, f_fe=self.f_fe,
                                            z_eos1=self.zmix_eos1, z_eos2=self.zmix_eos2, z_eos3=self.zmix_eos3)
        elif self.z_eos == 'ice_mixture':
            u_z = ices.get_u_pt_val(_lgp, _lgt, _zm, _za)

        elif self.z_eos == 'ice_rock_mixture':
            u_z_ice = ices.get_u_pt_val(_lgp, _lgt, _zm, _za)
            u_z_rock = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos='ppv2')

            u_z = (1 - _zr) * u_z_ice + _zr * u_z_rock

        elif self.z_eos == 'aqua_smooth' or self.z_eos == 'aqua_smooth2':
            self.z_eos = 'aqua'
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)
        else:
            u_z = 10 ** metals_eos.get_u_pt_tab(_lgp, _lgt, eos=self.z_eos)

        mixture_energy = (
            u_h * (1 - _y_prime) * (1 - _z)
            + u_he * _y_prime * (1 - _z)
            + umix * (1 - _z)
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
                 new_z_mix: bool = True
                    ):

        super().__init__(hhe_eos=hhe_eos, z_eos=z_eos, hg=hg, y_prime=y_prime, interp_method=interp_method, new_z_mix=new_z_mix)

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
            _za = _z / (1 - _zr+1e-6)

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
