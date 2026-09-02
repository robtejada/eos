"""Combined CH4 / NH3 ices EOS, read from the fitted Helmholtz tables.

    This module is a consumer layer.  It performs no fitting and no inversion:
    everything it serves was produced by `ch4_nh3_helmholtz.py` (the joint
    Helmholtz free-energy fit) and tabulated by `ch4_nh3_surface.py` in two
    bases.  Four `.npz` products per version live in `methane_ammonia/`:

        helmholtz_table_{species}_{ver}.npz   (rho, T) basis -> p, u, s
        helmholtz_pt_{species}_{ver}.npz      (P, T)   basis -> rho, u, s
        helmholtz_fit_{species}_{ver}.npz     spline coefficients + s_offset

    Reading the tables rather than evaluating the B-spline surface is roughly
    three orders of magnitude faster and is what every other tabulated EOS in
    this package does.

    Units are CGS throughout the public interface, linear rather than log:

        P    dyn/cm^2        rho  g/cm^3        T  K
        u    erg/g           s    erg/(g K)

    log10 adapters are layered on top for the modules that speak that dialect
    (`aqua_eos.py`, `ice_eos.py`, and `eos_class.z_eos`).

    Mixtures use the volume addition law (the linear mixing rule) in the P-T
    basis, parameterised by the ammonia mass fraction f_a.  See `ice_eos.py`
    for the three-component version this is built to extend.

    Authors: Roberto Tejada Arevalo
"""

import json
import os
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(CURR_DIR, 'methane_ammonia')

SPECIES = ('methane', 'ammonia')
TABLE_VERSION = 'v2'

# Molar masses [g/mol], used only for the optional ideal entropy of mixing.
MOLAR_MASS = {'methane': 16.043, 'ammonia': 17.031}

erg_to_kbbar = float((u.erg / u.Kelvin / u.gram).to(k_B / amu))
R_GAS = float((k_B / amu).to('erg/(K*g)').value)   # erg/(g K) per (g/mol)

# The repo standard: linear interpolation, no bounds error, extrapolate past
# the edge.  Matches aqua_eos.py, ch4.py, mg2sio4_aneos_eos.py.
_RGI_KW = dict(method='linear', bounds_error=False, fill_value=None)

# Below this the log10 of a positive quantity is meaningless; guards a caller
# passing 0 or a negative P/T/rho rather than silently returning nan.
_FLOOR = 1e-300

_mp = amu.to('g') # grams
_kb = k_B.to('erg/K') # ergs/K
_ERG_TO_KBBAR = (u.erg/u.Kelvin/u.gram).to(k_B/_mp)


@dataclass(frozen=True)
class Domain:
    """Extent of the tabulated surface, in linear CGS."""
    rho_min: float
    rho_max: float
    T_min: float
    T_max: float
    P_min: float
    P_max: float


# ---------------------------------------------------------------------------
# Shape and unit helpers (mirror aqua_revised_core_eos.py / mg2sio4_aneos_eos.py)
# ---------------------------------------------------------------------------
def _broadcast(a, b):
    scalar = np.isscalar(a) and np.isscalar(b)
    a_arr = np.array(a, ndmin=1, dtype=float)
    b_arr = np.array(b, ndmin=1, dtype=float)
    if a_arr.shape != b_arr.shape:
        a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
    return scalar, a_arr, b_arr


def _broadcast3(a, b, c):
    scalar = np.isscalar(a) and np.isscalar(b) and np.isscalar(c)
    arrs = [np.array(x, ndmin=1, dtype=float) for x in (a, b, c)]
    if not (arrs[0].shape == arrs[1].shape == arrs[2].shape):
        arrs = list(np.broadcast_arrays(*arrs))
    return scalar, arrs[0], arrs[1], arrs[2]


def _maybe_scalar(scalar, vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.reshape(-1)[0]) if scalar else vals


def _interp(rgi, x_arr, y_arr):
    pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
    return rgi(pts).reshape(x_arr.shape)


def _log10(x):
    return np.log10(np.clip(np.asarray(x, dtype=float), _FLOOR, None))


def _guarded_xlogx(x):
    """x*ln(x) with the x -> 0 limit taken as 0, elementwise.

    Mirrors ice_eos.guarded_log, but vectorised: the number fraction of an
    absent component is 0 and contributes nothing to the mixing entropy, while
    np.log(0) would poison the whole array.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0):
        raise ValueError('number fraction went negative.')
    out = np.zeros_like(x)
    nz = x > 0.0
    out[nz] = x[nz] * np.log(x[nz])
    return out


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def rhot_table_path(species, version=TABLE_VERSION):
    return os.path.join(CACHE_DIR, f'helmholtz_table_{species}_{version}.npz')


def pt_table_path(species, version=TABLE_VERSION):
    return os.path.join(CACHE_DIR, f'helmholtz_pt_{species}_{version}.npz')


def fit_cache_path(species, version=TABLE_VERSION):
    return os.path.join(CACHE_DIR, f'helmholtz_fit_{species}_{version}.npz')


# ---------------------------------------------------------------------------
# Single species
# ---------------------------------------------------------------------------
class HelmholtzSpeciesEOS:
    """One species (methane or ammonia) in both the (rho,T) and (P,T) bases.

    Every field of both tables is strictly positive over the whole domain, so
    the interpolants are built on log10 of each quantity rather than on the
    quantity itself.  That is not cosmetic.  Pressure spans ten decades, and
    log10(P) against (log10 rho, log10 T) is close to bilinear where P itself is
    strongly curved; interpolating the log both improves accuracy and makes a
    negative interpolated pressure structurally impossible.

    The two bases do not cover the same region.  The (P,T) table extends down
    to 1 bar, and its low-P / high-T corner reaches densities near 1e-5 g/cm^3,
    two decades below the (rho,T) table's floor of 1e-3.  About 11 percent of
    the (P,T) grid lies there.  Calling `get_p_rhot` at such a density
    extrapolates: RGI continues with the last cell's slope, Gamma = 0.986
    instead of the ideal-gas value of exactly 1, which costs roughly 0.03 dex
    in P after two decades.  That is well behaved but it is not table data, and
    `in_domain_rhot` is the way to test for it.

    Parameters
    ----------
    species : {'methane', 'ammonia'}
    version : str
        Table version tag, e.g. 'v2'.
    rhot, pt : bool
        Load the (rho,T) and (P,T) tables respectively.  Both default True;
        skipping one halves the load time and memory if only one basis is used.
    """

    def __init__(self, species, version=TABLE_VERSION, rhot=True, pt=True):
        if species not in SPECIES:
            raise ValueError(f'species must be one of {SPECIES}, got {species!r}')
        if not (rhot or pt):
            raise ValueError('at least one of rhot= or pt= must be True')

        self.species = species
        self.version = str(version)
        self.molar_mass = MOLAR_MASS[species]

        self._has_rhot = False
        self._has_pt = False

        if rhot:
            self._load_rhot_table(rhot_table_path(species, self.version))
        if pt:
            self._load_pt_table(pt_table_path(species, self.version))

        self.s_offset = self._load_s_offset(fit_cache_path(species, self.version))
        self.domain = self._build_domain()

    def __repr__(self):
        bases = [b for b, ok in (('rho-T', self._has_rhot), ('P-T', self._has_pt)) if ok]
        return (f'<HelmholtzSpeciesEOS {self.species} {self.version} '
                f'[{", ".join(bases)}]>')

    # -- loading ----------------------------------------------------------
    def _load_rhot_table(self, path):
        """(rho,T) basis.  Native axis order is [i_rho, j_T]."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'missing {os.path.basename(path)}; build it with\n'
                f'    python eos/ch4_nh3_surface.py tabulate')
        d = np.load(path, allow_pickle=False)

        self.logrho_vals = np.asarray(d['logrho'], dtype=float)
        self.logT_vals_rhot = np.asarray(d['logT'], dtype=float)
        axes = (self.logrho_vals, self.logT_vals_rhot)

        self._logp_rgi_rhot = RGI(axes, _log10(d['p_cgs']), **_RGI_KW)
        self._logu_rgi_rhot = RGI(axes, _log10(d['u_cgs']), **_RGI_KW)
        self._logs_rgi_rhot = RGI(axes, _log10(d['s_cgs']), **_RGI_KW)

        self.meta_rhot = json.loads(str(d['meta'][0]))
        self._has_rhot = True

    def _load_pt_table(self, path):
        """(P,T) basis.  Native axis order is [i_T, j_P] -- note it differs from
        the (rho,T) table's [i_rho, j_T].  Kept native; no transpose."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'missing {os.path.basename(path)}; build it with\n'
                f'    python eos/ch4_nh3_surface.py pt')
        d = np.load(path, allow_pickle=False)

        self.logT_vals_pt = np.asarray(d['logT'], dtype=float)
        self.logP_vals = np.asarray(d['logP'], dtype=float)
        axes = (self.logT_vals_pt, self.logP_vals)

        # logrho is already stored as log10; u and s are stored linear.
        self._logrho_rgi_pt = RGI(axes, np.asarray(d['logrho'], dtype=float), **_RGI_KW)
        self._logu_rgi_pt = RGI(axes, _log10(d['u']), **_RGI_KW)
        self._logs_rgi_pt = RGI(axes, _log10(d['s']), **_RGI_KW)

        # `supported` marks cells inside the DFT density range; elsewhere the
        # surface is a constrained extrapolation.  Nearest-neighbour, since a
        # linear blend of a boolean mask means nothing.
        self._supported_rgi = RGI(axes, np.asarray(d['supported'], dtype=float),
                                  method='nearest', bounds_error=False, fill_value=0.0)

        self.meta_pt = json.loads(str(d['meta'][0]))
        self._has_pt = True

    def _load_s_offset(self, path):
        """The post-fit third-law entropy gauge, in erg/(g K).

        A constant added to s so that the tabulated entropy stays positive over
        the whole domain (the consumers store log10 s).  It cancels from every
        derivative, so it changes no thermodynamics, but it dominates the
        absolute value: see `CH4_NH3_EOS.gauge_report`.
        """
        if not os.path.exists(path):
            return 0.0
        z = np.load(path, allow_pickle=False)
        return float(z['s_offset'][0]) if 's_offset' in z.files else 0.0

    def _build_domain(self):
        rho_min = rho_max = T_min = T_max = P_min = P_max = np.nan
        if self._has_rhot:
            rho_min, rho_max = 10.0 ** self.logrho_vals[[0, -1]]
            T_min, T_max = 10.0 ** self.logT_vals_rhot[[0, -1]]
        if self._has_pt:
            P_min, P_max = 10.0 ** self.logP_vals[[0, -1]]
            if not self._has_rhot:
                T_min, T_max = 10.0 ** self.logT_vals_pt[[0, -1]]
        return Domain(float(rho_min), float(rho_max), float(T_min), float(T_max),
                      float(P_min), float(P_max))

    def _require(self, basis):
        if basis == 'rhot' and not self._has_rhot:
            raise RuntimeError(f'{self.species}: (rho,T) table not loaded')
        if basis == 'pt' and not self._has_pt:
            raise RuntimeError(f'{self.species}: (P,T) table not loaded')

    # -- rho-T basis, linear in and out -----------------------------------
    def get_p_rhot(self, _rho, _T):
        """P(rho, T) [dyn/cm^2] from rho [g/cm^3] and T [K]."""
        self._require('rhot')
        scalar, rho_arr, T_arr = _broadcast(_rho, _T)
        logp = _interp(self._logp_rgi_rhot, _log10(rho_arr), _log10(T_arr))
        return _maybe_scalar(scalar, 10.0 ** logp)

    def get_u_rhot(self, _rho, _T):
        """U(rho, T) [erg/g]."""
        self._require('rhot')
        scalar, rho_arr, T_arr = _broadcast(_rho, _T)
        logu = _interp(self._logu_rgi_rhot, _log10(rho_arr), _log10(T_arr))
        return _maybe_scalar(scalar, 10.0 ** logu)

    def get_s_rhot(self, _rho, _T):
        """S(rho, T) [erg/(g K)], including the gauge offset `s_offset`."""
        self._require('rhot')
        scalar, rho_arr, T_arr = _broadcast(_rho, _T)
        logs = _interp(self._logs_rgi_rhot, _log10(rho_arr), _log10(T_arr))
        return _maybe_scalar(scalar, 10.0 ** logs)

    # -- P-T basis, linear in and out -------------------------------------
    def get_rho_pt(self, P, T):
        """rho(P, T) [g/cm^3] from P [dyn/cm^2] and T [K]."""
        self._require('pt')
        scalar, P_arr, T_arr = _broadcast(P, T)
        logrho = _interp(self._logrho_rgi_pt, _log10(T_arr), _log10(P_arr))
        return _maybe_scalar(scalar, 10.0 ** logrho)

    def get_u_pt(self, P, T):
        """U(P, T) [erg/g]."""
        self._require('pt')
        scalar, P_arr, T_arr = _broadcast(P, T)
        logu = _interp(self._logu_rgi_pt, _log10(T_arr), _log10(P_arr))
        return _maybe_scalar(scalar, 10.0 ** logu)

    def get_s_pt(self, P, T):
        """S(P, T) [erg/(g K)], including the gauge offset `s_offset`."""
        self._require('pt')
        scalar, P_arr, T_arr = _broadcast(P, T)
        logs = _interp(self._logs_rgi_pt, _log10(T_arr), _log10(P_arr))
        return _maybe_scalar(scalar, 10.0 ** logs)

    # -- log10 adapters ----------------------------------------------------
    # Argument and return conventions here match aqua_eos.py / ice_eos.py:
    # log10 in, log10 out for rho, P and u; linear for s.
    def get_logp_rhot(self, _lgrho, _lgt):
        """log10 P [dyn/cm^2] from log10 rho and log10 T."""
        self._require('rhot')
        scalar, lgr, lgt = _broadcast(_lgrho, _lgt)
        return _maybe_scalar(scalar, _interp(self._logp_rgi_rhot, lgr, lgt))

    def get_logu_rhot(self, _lgrho, _lgt):
        self._require('rhot')
        scalar, lgr, lgt = _broadcast(_lgrho, _lgt)
        return _maybe_scalar(scalar, _interp(self._logu_rgi_rhot, lgr, lgt))

    def get_s_rhot_log(self, _lgrho, _lgt):
        """Linear S [erg/(g K)] from log10 rho and log10 T."""
        self._require('rhot')
        scalar, lgr, lgt = _broadcast(_lgrho, _lgt)
        return _maybe_scalar(scalar, 10.0 ** _interp(self._logs_rgi_rhot, lgr, lgt))

    def get_logrho_pt(self, _lgp, _lgt):
        """log10 rho [g/cm^3] from log10 P and log10 T."""
        self._require('pt')
        scalar, lgp, lgt = _broadcast(_lgp, _lgt)
        return _maybe_scalar(scalar, _interp(self._logrho_rgi_pt, lgt, lgp))

    def get_logu_pt(self, _lgp, _lgt):
        """log10 U [erg/g] from log10 P and log10 T."""
        self._require('pt')
        scalar, lgp, lgt = _broadcast(_lgp, _lgt)
        return _maybe_scalar(scalar, _interp(self._logu_rgi_pt, lgt, lgp))

    def get_s_pt_log(self, _lgp, _lgt):
        """Linear S [erg/(g K)] from log10 P and log10 T.

        Matches the signature and return units of aqua_eos.get_s_pt_tab.
        """
        self._require('pt')
        scalar, lgp, lgt = _broadcast(_lgp, _lgt)
        return _maybe_scalar(scalar, 10.0 ** _interp(self._logs_rgi_pt, lgt, lgp))

    # -- domain queries ----------------------------------------------------
    def supported_pt(self, P, T):
        """True where (P,T) falls inside the DFT density range of the fit.

        Roughly a third of the tabulated (P,T) grid; outside it the surface is
        a physically constrained extrapolation rather than data.
        """
        self._require('pt')
        scalar, P_arr, T_arr = _broadcast(P, T)
        vals = _interp(self._supported_rgi, _log10(T_arr), _log10(P_arr)) > 0.5
        return bool(vals.reshape(-1)[0]) if scalar else vals

    def in_domain_pt(self, P, T):
        """True where (P,T) is inside the table, i.e. not extrapolated."""
        self._require('pt')
        scalar, P_arr, T_arr = _broadcast(P, T)
        d = self.domain
        vals = ((P_arr >= d.P_min) & (P_arr <= d.P_max)
                & (T_arr >= d.T_min) & (T_arr <= d.T_max))
        return bool(vals.reshape(-1)[0]) if scalar else vals

    def in_domain_rhot(self, _rho, _T):
        """True where (rho,T) is inside the table, i.e. not extrapolated."""
        self._require('rhot')
        scalar, rho_arr, T_arr = _broadcast(_rho, _T)
        d = self.domain
        vals = ((rho_arr >= d.rho_min) & (rho_arr <= d.rho_max)
                & (T_arr >= d.T_min) & (T_arr <= d.T_max))
        return bool(vals.reshape(-1)[0]) if scalar else vals


# ---------------------------------------------------------------------------
# Two-component mixture
# ---------------------------------------------------------------------------
class CH4_NH3_EOS:
    """Methane and ammonia, pure and mixed.

    The two pure species are reachable as `.methane` and `.ammonia`, each a
    `HelmholtzSpeciesEOS`.  Mixtures follow the volume addition law (linear
    mixing rule) in the P-T basis, parameterised by the ammonia mass fraction
    f_a, with the methane fraction taken as 1 - f_a:

        1/rho_mix = f_a/rho_NH3(P,T) + (1 - f_a)/rho_CH4(P,T)
        u_mix     = f_a*u_NH3 + (1 - f_a)*u_CH4
        s_mix     = f_a*s_NH3 + (1 - f_a)*s_CH4   [+ ideal mixing, optional]

    Mixtures are P-T only.  Volume addition closes at fixed pressure, which is
    the shared intensive variable; a (rho,T) mixture would require solving
    1/rho = sum_i X_i/rho_i(P,T) for P, and is deliberately not provided here.

    Entropy caveat
    --------------
    Each species carries an arbitrary third-law gauge constant (`s_offset`,
    about 5e8 erg/(g K) for both), which is 84-85 percent of the tabulated
    entropy at typical interior conditions and which AQUA water does not share.
    Mixture entropy differences in P or T are physical; the absolute value is
    not, and is not comparable across species.  Call `gauge_report()` for the
    numbers.
    """

    def __init__(self, version=TABLE_VERSION, rhot=True, pt=True):
        self.version = str(version)
        self.methane = HelmholtzSpeciesEOS('methane', self.version, rhot=rhot, pt=pt)
        self.ammonia = HelmholtzSpeciesEOS('ammonia', self.version, rhot=rhot, pt=pt)

    def __repr__(self):
        return f'<CH4_NH3_EOS {self.version} [methane, ammonia]>'

    @property
    def domain(self):
        """Intersection of the two species' domains."""
        a, m = self.ammonia.domain, self.methane.domain
        return Domain(max(a.rho_min, m.rho_min), min(a.rho_max, m.rho_max),
                      max(a.T_min, m.T_min), min(a.T_max, m.T_max),
                      max(a.P_min, m.P_min), min(a.P_max, m.P_max))

    @staticmethod
    def _fractions(f_a):
        """(f_ammonia, f_methane) with f_a validated as a mass fraction."""
        f_a = np.asarray(f_a, dtype=float)
        if np.any(f_a < 0.0) or np.any(f_a > 1.0):
            raise ValueError('f_a (ammonia mass fraction) must lie in [0, 1]')
        return f_a, 1.0 - f_a

    # -- mixture getters, linear in and out -------------------------------
    def get_rho_pt_mix(self, P, T, f_a):
        """Mixture density [g/cm^3] by volume addition.

        Parameters
        ----------
        P : float or array_like    pressure [dyn/cm^2]
        T : float or array_like    temperature [K]
        f_a : float or array_like  ammonia mass fraction, in [0, 1]
        """
        scalar, P_arr, T_arr, fa_arr = _broadcast3(P, T, f_a)
        f_am, f_me = self._fractions(fa_arr)

        v_am = 1.0 / np.asarray(self.ammonia.get_rho_pt(P_arr, T_arr), dtype=float)
        v_me = 1.0 / np.asarray(self.methane.get_rho_pt(P_arr, T_arr), dtype=float)

        rho_mix = 1.0 / (f_am * v_am + f_me * v_me)
        return _maybe_scalar(scalar, rho_mix)

    def get_u_pt_mix(self, P, T, f_a):
        """Mixture specific internal energy [erg/g], mass weighted."""
        scalar, P_arr, T_arr, fa_arr = _broadcast3(P, T, f_a)
        f_am, f_me = self._fractions(fa_arr)

        u_am = np.asarray(self.ammonia.get_u_pt(P_arr, T_arr), dtype=float)
        u_me = np.asarray(self.methane.get_u_pt(P_arr, T_arr), dtype=float)

        return _maybe_scalar(scalar, f_am * u_am + f_me * u_me)

    def get_s_pt_mix(self, P, T, f_a, ideal_mixing=False):
        """Mixture specific entropy [erg/(g K)], mass weighted.

        `ideal_mixing` defaults False, matching ice_eos.get_s_pt_val, which
        likewise returns the intrinsic (unmixed) entropy.  Setting it True adds
        the ideal entropy of mixing computed from number fractions:

            s_id = -(R/M_bar) * sum_i x_i ln x_i,    1/M_bar = sum_i X_i/M_i

        Note that the sum of the two intrinsic terms carries
        f_a*s_offset_NH3 + (1-f_a)*s_offset_CH4 of pure gauge.  Differences in
        P or T are unaffected; the absolute value is arbitrary.
        """
        scalar, P_arr, T_arr, fa_arr = _broadcast3(P, T, f_a)
        f_am, f_me = self._fractions(fa_arr)

        s_am = np.asarray(self.ammonia.get_s_pt(P_arr, T_arr), dtype=float)
        s_me = np.asarray(self.methane.get_s_pt(P_arr, T_arr), dtype=float)
        s_mix = f_am * s_am + f_me * s_me

        if ideal_mixing:
            s_mix = s_mix + self.get_s_ideal_mix(fa_arr)

        return _maybe_scalar(scalar, s_mix)

    def get_s_ideal_mix(self, f_a):
        """Ideal entropy of mixing [erg/(g K)] for an ammonia mass fraction f_a."""
        scalar = np.isscalar(f_a)
        f_am, f_me = self._fractions(np.atleast_1d(f_a))

        n_am = f_am / MOLAR_MASS['ammonia']     # moles per gram of mixture
        n_me = f_me / MOLAR_MASS['methane']
        n_tot = n_am + n_me

        x_am = np.divide(n_am, n_tot, out=np.zeros_like(n_am), where=n_tot > 0)
        x_me = np.divide(n_me, n_tot, out=np.zeros_like(n_me), where=n_tot > 0)

        # -R * sum x_i ln x_i  per mole, times moles per gram, gives erg/(g K).
        s_id = -R_GAS * n_tot * (_guarded_xlogx(x_am) + _guarded_xlogx(x_me))
        return _maybe_scalar(scalar, s_id)

    # -- log10 adapters, shaped for eos_class / ice_comb_eos --------------
    # These mirror ice_eos.get_{logrho,u,s}_pt_val: log10 P and log10 T in,
    # log10 rho out for density, linear erg/g and erg/(g K) for u and s.
    def get_logrho_pt_val(self, _lgp, _lgt, _f_a):
        """log10 of the mixture density, from log10 P and log10 T."""
        scalar, lgp, lgt, fa = _broadcast3(_lgp, _lgt, _f_a)
        rho = self.get_rho_pt_mix(10.0 ** lgp, 10.0 ** lgt, fa)
        return _maybe_scalar(scalar, np.log10(np.asarray(rho, dtype=float)))

    def get_u_pt_val(self, _lgp, _lgt, _f_a):
        """Mixture specific internal energy [erg/g], from log10 P and log10 T."""
        scalar, lgp, lgt, fa = _broadcast3(_lgp, _lgt, _f_a)
        return _maybe_scalar(scalar, self.get_u_pt_mix(10.0 ** lgp, 10.0 ** lgt, fa))

    def get_s_pt_val(self, _lgp, _lgt, _f_a, ideal_mixing=False):
        """Mixture specific entropy [erg/(g K)], from log10 P and log10 T."""
        scalar, lgp, lgt, fa = _broadcast3(_lgp, _lgt, _f_a)
        return _maybe_scalar(scalar, self.get_s_pt_mix(10.0 ** lgp, 10.0 ** lgt, fa,
                                                       ideal_mixing=ideal_mixing))

    # -- diagnostics -------------------------------------------------------
    def gauge_report(self, P=1e12, T=2000.0, verbose=True):
        """Report each species' entropy gauge and its share of the total.

        The offsets are arbitrary constants fixed after the fit so that the
        tabulated entropy stays positive.  They cancel from every derivative
        but dominate the absolute value, and AQUA water carries no equivalent
        term.  Mass-weighting entropies across the three is therefore fine for
        derivatives and meaningless in absolute terms until the gauges are
        reconciled.
        """
        rows = {}
        for name in SPECIES:
            e = getattr(self, name)
            s_tot = float(e.get_s_pt(P, T))
            rows[name] = dict(s_offset=e.s_offset, s_total=s_tot,
                              share=e.s_offset / s_tot if s_tot else np.nan)
        if verbose:
            print(f'entropy gauge at P = {P:.3g} dyn/cm^2, T = {T:.4g} K')
            print(f'  {"species":<10s} {"s_offset":>12s} {"s_total":>12s} {"gauge share":>12s}')
            for name, r in rows.items():
                print(f'  {name:<10s} {r["s_offset"]:12.4e} {r["s_total"]:12.4e}'
                      f' {100 * r["share"]:11.1f}%')
            print('  AQUA water carries no offset; absolute entropies are not'
                  ' comparable across species.')
        return rows


# ---------------------------------------------------------------------------
# Module-level singleton, built lazily so importing this file is cheap
# ---------------------------------------------------------------------------
_DEFAULT = None


def default_eos(version=TABLE_VERSION):
    """Shared CH4_NH3_EOS instance, loaded on first use."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.version != str(version):
        _DEFAULT = CH4_NH3_EOS(version=version)
    return _DEFAULT
