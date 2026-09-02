"""Combined water + methane + ammonia ices EOS (ternary volume addition law).

    This module is the ices analogue of the silicate combination modules
    (`mg2sio4_aneos_eos.py`, `mgsio3_comb_eos.py`): one class that serves a
    three-component mixture from previously built pure-species surfaces.  It
    performs no fitting and no inversion of its own.

    Components and their sources:

        water    `aqua_eos.py` with AQUA_VERSION='revised'
                 (Haldemann et al. 2020; revised P-T table of Cano Amoros et al.)
        methane  `ch4_nh3_eos.py` -> helmholtz_pt_methane_{ver}.npz
        ammonia  `ch4_nh3_eos.py` -> helmholtz_pt_ammonia_{ver}.npz

    Composition convention: WATER IS THE PRIMARY SUBSTANCE, exactly as hydrogen
    is for hydrogen-helium mixtures.  The two arguments are the direct mass
    fractions of the minor species,

        Z_m = methane mass fraction,   Z_a = ammonia mass fraction,
        f_w = 1 - Z_m - Z_a            (water carries the remainder),

    NOT the nested convention of the legacy `ice_eos.py`, where
    f_water = (1-z_m)(1-z_a) and f_methane = z_m(1-z_a).  The static helper
    `nested_to_direct` converts between the two.
    The module functions `cno_to_mass_fractions` / `mass_fractions_to_cno`
    convert between these fractions and the C:N:O atom-number ratios used to
    specify ice mixtures (solar 4:1:7 -> Z_m = 0.310, Z_a = 0.082).

    Mixing rules in the P-T basis, where pressure and temperature are the shared
    intensive variables:

        1/rho_mix = f_w/rho_w + Z_m/rho_m + Z_a/rho_a     (volume addition law)
        u_mix     = f_w*u_w + Z_m*u_m + Z_a*u_a
        s_mix     = f_w*s_w + Z_m*s_m + Z_a*s_a  [+ ideal mixing, optional]

    Entropy gauge.  The CH4/NH3 tables carry a post-fit offset of 100.50 R_s per
    species, added only so downstream log10(s) storage stays defined; measured at
    1 bar and 500 K, subtracting it returns both species to the third-law scale
    (207.1 and 212.6 J/mol/K against the JANAF standard values), where the
    revised AQUA water table already sits (206.5 J/mol/K at the same state).
    Mixing entropies across species is only meaningful on a common absolute
    scale, so this module REMOVES the offsets by default (s_gauge='thirdlaw').
    Doing so exposes the region the offsets were hiding: the CH4/NH3 entropy is
    negative, hence wrong, in the cold dense corner, and AQUA's own entropy grid
    had a sentinel-NaN corner of its own, now filled at load time with
    extrapolated placeholder values (see aqua_eos._fill_s_pt_nans) so square
    sweeps and inverters never meet a NaN.  `entropy_valid_pt` reports where the
    mixture entropy can be trusted, and the filled corner stays flagged; density
    and internal energy are unaffected anywhere.

    Units are CGS and linear on the primary interface (P dyn/cm^2, T K,
    rho g/cm^3, u erg/g, s erg/(g K)); log10 adapters `get_*_pt_val` follow the
    `ice_eos.py` / `eos_class.z_eos` dialect.

    Authors: Roberto Tejada Arevalo
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

from eos import aqua_eos
from eos.ch4_nh3_eos import (CH4_NH3_EOS, Domain, MOLAR_MASS, R_GAS,
                             TABLE_VERSION, _broadcast, _guarded_xlogx,
                             _interp, _log10, _maybe_scalar, erg_to_kbbar)

# Molar mass of water [g/mol]; MOLAR_MASS carries methane and ammonia.
M_WATER = 18.015

# The revised AQUA P-T grid (measured at import of aqua_eos):
#   logP in [0, 15.602] dyn/cm^2, logT in [2.0, 4.770] K.
# Its entropy grid carried NaN cells (the revised table's s = -1 sentinels)
# over roughly logP in [12.89, 15.60] x logT in [2.0, 3.88] -- water's own
# cold dense corner.  aqua_eos now FILLS those cells with extrapolated values
# at load time so square-table builds and inverters never meet a NaN; the
# filled cells are recorded in aqua_eos.svals_pt_filled_mask and remain
# UNTRUSTED here (see entropy_valid_pt).  rho and u were NaN-free throughout.
WATER_LOGT_MIN = float(aqua_eos.logtvals.min())
WATER_LOGT_MAX = float(aqua_eos.logtvals.max())
WATER_LOGP_MIN = float(aqua_eos.logpvals_pt.min())
WATER_LOGP_MAX = float(aqua_eos.logpvals_pt.max())


def _broadcast4(a, b, c, d):
    scalar = all(np.isscalar(x) for x in (a, b, c, d))
    arrs = [np.array(x, ndmin=1, dtype=float) for x in (a, b, c, d)]
    if len({x.shape for x in arrs}) > 1:
        arrs = list(np.broadcast_arrays(*arrs))
    return (scalar, *arrs)


# ---------------------------------------------------------------------------
# C:N:O number ratios <-> ice mass fractions
# ---------------------------------------------------------------------------
# The "C:N:O" convention of the Uranus/Neptune models (Tejada Arevalo 2025;
# solar 4:1:7) is a ratio of ATOM NUMBERS, not of masses.  With every carbon
# atom bound in CH4, every nitrogen in NH3 and every oxygen in H2O, the
# molecule numbers equal the atom numbers, so the mass fractions follow from
# the molar masses alone:
#
#     Z_i = n_i M_i / sum_j n_j M_j ,   i in {CH4, NH3, H2O}.
#
# Any partitioning of carbon into CO/CO2 or of nitrogen into N2 lies outside
# this convention.  The same rule is used by eos_pt_calc.number_to_mass_fraction
# for the legacy ice_mixture tables; this copy returns the DIRECT
# (water-primary) fractions this module consumes.

_CNO_MOLAR = np.array([MOLAR_MASS['methane'], MOLAR_MASS['ammonia'], M_WATER])

# Round-off tolerance on mass fractions: sums may overshoot 1 by this much,
# and a water remainder 1 - Z_m - Z_a below it is an exact zero, not a trace
# of water (a water-free 5:1:0 leaves 1.1e-16 behind in floating point).
_FRAC_TOL = 1e-12


def _water_remainder(Z_m, Z_a):
    """1 - Z_m - Z_a with sub-tolerance residuals snapped to exactly zero."""
    f_w = 1.0 - Z_m - Z_a
    return np.where(f_w < _FRAC_TOL, 0.0, np.minimum(f_w, 1.0))


def parse_cno(cno):
    """Normalise a C:N:O specification to three floats (n_C, n_N, n_O).

    Accepts '4:1:7', '4/1/7', '4 1 7', '4,1,7', the three-digit shorthand
    '417' used by eos_pt_calc --cno_ratio, or any 3-sequence of numbers.
    """
    if isinstance(cno, str):
        txt = cno.strip()
        for sep in (':', '/', ',', ';'):
            txt = txt.replace(sep, ' ')
        parts = txt.split()
        if len(parts) == 1 and len(parts[0]) == 3 and parts[0].isdigit():
            parts = list(parts[0])          # '417' -> ['4', '1', '7']
        if len(parts) != 3:
            raise ValueError(f"cannot parse C:N:O ratio {cno!r}; "
                             "use e.g. '4:1:7' or '417'")
        vals = [float(p) for p in parts]
    else:
        vals = list(np.asarray(cno, dtype=float).reshape(-1))
        if len(vals) != 3:
            raise ValueError('a C:N:O ratio needs exactly three numbers')
    return tuple(vals)


def cno_to_mass_fractions(n_C, n_N=None, n_O=None):
    """C:N:O atom-number ratio -> direct mass fractions (Z_m, Z_a, f_w).

    Call as cno_to_mass_fractions(4, 1, 7), cno_to_mass_fractions('4:1:7')
    or cno_to_mass_fractions((4, 1, 7)).  Only the ratio matters, so 4:1:7
    and 8:2:14 give the same result.  Array inputs broadcast elementwise so
    a grid of ratios converts in one call; scalar input returns floats.

    Returns
    -------
    Z_m, Z_a, f_w : methane, ammonia and water mass fractions, summing to 1,
        in the water-primary convention of ICES_COMB_EOS (pass Z_m, Z_a
        straight to its getters; f_w is the remainder).
    """
    if n_N is None and n_O is None:
        n_C, n_N, n_O = parse_cno(n_C)
    elif n_N is None or n_O is None:
        raise ValueError('give all three of n_C, n_N, n_O, or a single C:N:O spec')
    scalar = all(np.isscalar(x) for x in (n_C, n_N, n_O))
    n = [np.array(x, ndmin=1, dtype=float) for x in (n_C, n_N, n_O)]
    if len({x.shape for x in n}) > 1:
        n = list(np.broadcast_arrays(*n))
    n_C, n_N, n_O = n
    if not all(np.all(np.isfinite(x)) for x in n):
        raise ValueError('C:N:O numbers must be finite')
    if np.any(n_C < 0) or np.any(n_N < 0) or np.any(n_O < 0):
        raise ValueError('C:N:O numbers must be non-negative')
    m_C = n_C * _CNO_MOLAR[0]
    m_N = n_N * _CNO_MOLAR[1]
    m_O = n_O * _CNO_MOLAR[2]
    tot = m_C + m_N + m_O
    if np.any(tot <= 0.0):
        raise ValueError('at least one of C, N, O must be positive')
    Z_m, Z_a, f_w = m_C / tot, m_N / tot, m_O / tot
    if scalar:
        return float(Z_m[0]), float(Z_a[0]), float(f_w[0])
    return Z_m, Z_a, f_w


def mass_fractions_to_cno(Z_m, Z_a, normalize='min'):
    """Inverse of cno_to_mass_fractions: (Z_m, Z_a) -> (n_C, n_N, n_O).

    Water is the remainder 1 - Z_m - Z_a, with residuals below _FRAC_TOL
    treated as exactly zero so that water-free mixtures invert cleanly.
    `normalize` sets the scale of the returned numbers: 'min' divides by the
    smallest entry above _FRAC_TOL of the largest (so solar comes back as
    4:1:7), 'N' or 'O' set that element to 1, and None returns raw moles per
    gram of mixture.
    """
    scalar = np.isscalar(Z_m) and np.isscalar(Z_a)
    Z_m = np.array(Z_m, ndmin=1, dtype=float)
    Z_a = np.array(Z_a, ndmin=1, dtype=float)
    if Z_m.shape != Z_a.shape:
        Z_m, Z_a = np.broadcast_arrays(Z_m, Z_a)
    if np.any(~np.isfinite(Z_m)) or np.any(~np.isfinite(Z_a)):
        raise ValueError('mass fractions must be finite')
    if np.any(Z_m < 0) or np.any(Z_a < 0) or np.any(Z_m + Z_a > 1.0 + _FRAC_TOL):
        raise ValueError('need Z_m, Z_a >= 0 and Z_m + Z_a <= 1')
    f_w = _water_remainder(Z_m, Z_a)
    n = np.stack([Z_m, Z_a, f_w], axis=0) / _CNO_MOLAR.reshape(3, *([1] * Z_m.ndim))
    if normalize == 'min':
        # Relative threshold: a component below _FRAC_TOL of the largest is
        # absent, not the reference (otherwise round-off sets the scale).
        pos = np.where(n > _FRAC_TOL * n.max(axis=0), n, np.inf)
        scale = pos.min(axis=0)
    elif normalize in ('N', 'n'):
        scale = n[1]
    elif normalize in ('O', 'o'):
        scale = n[2]
    elif normalize is None:
        scale = np.ones_like(n[0])
    else:
        raise ValueError("normalize must be 'min', 'N', 'O' or None")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('cannot normalise: the chosen reference element is absent')
    n = n / scale
    if scalar:
        return float(n[0][0]), float(n[1][0]), float(n[2][0])
    return n[0], n[1], n[2]


def cno_label(n_C, n_N=None, n_O=None):
    """'4:1:7'-style string, integers shown without a decimal point."""
    if n_N is None and n_O is None:
        n_C, n_N, n_O = parse_cno(n_C)
    elif n_N is None or n_O is None:
        raise ValueError('give all three of n_C, n_N, n_O, or a single C:N:O spec')

    def _fmt(x):
        x = float(x)
        return f'{int(round(x))}' if abs(x - round(x)) < 1e-9 else f'{x:g}'
    return ':'.join(_fmt(x) for x in (n_C, n_N, n_O))


class ICES_COMB_EOS:
    """Water-methane-ammonia mixture over the pure-species P-T tables.

    Parameters
    ----------
    version : str
        CH4/NH3 table version tag (default the current TABLE_VERSION).
    s_gauge : {'thirdlaw', 'table'}
        'thirdlaw' (default) removes each species' s_offset so all three
        components share the absolute third-law scale; this is the only gauge
        in which the mass-weighted mixture entropy means anything.  'table'
        keeps the offsets as stored, matching what `ch4_nh3_eos` reports for
        the pure species; use it only for comparisons against those tables.
    """

    def __init__(self, version=TABLE_VERSION, s_gauge='thirdlaw'):
        if s_gauge not in ('thirdlaw', 'table'):
            raise ValueError("s_gauge must be 'thirdlaw' or 'table'")
        self.s_gauge = s_gauge
        self.version = str(version)

        # Pure species: methane and ammonia from the Helmholtz tables (P-T
        # basis only; the rho-T basis plays no role in P-T mixing), water from
        # the module-level revised-AQUA interpolants.
        self.ch4_nh3 = CH4_NH3_EOS(version=self.version, rhot=False, pt=True)
        self.methane = self.ch4_nh3.methane
        self.ammonia = self.ch4_nh3.ammonia
        self.water = aqua_eos

        # Where the water entropy is an invented fill rather than table data:
        # nearest-neighbour lookup of aqua_eos's filled-cell mask, so
        # entropy_valid_pt keeps flagging the region even though the values
        # there are now finite.
        fill = getattr(aqua_eos, 'svals_pt_filled_mask', None)
        if fill is not None and np.any(fill):
            self._water_fill_rgi = RGI(
                (aqua_eos.logpvals_pt, aqua_eos.logtvals), fill.astype(float),
                method='nearest', bounds_error=False, fill_value=0.0)
        else:
            self._water_fill_rgi = None

        # Joint in-table domain: the intersection of the three tables.  The
        # binding edges are the CH4/NH3 pressure span (AQUA is wider on both
        # sides) and AQUA's 100 K temperature floor (CH4/NH3 reach 50 K).
        m = self.methane.domain
        self.domain = Domain(
            rho_min=np.nan, rho_max=np.nan,
            T_min=max(m.T_min, 10.0 ** WATER_LOGT_MIN),
            T_max=min(m.T_max, 10.0 ** WATER_LOGT_MAX),
            P_min=max(m.P_min, 10.0 ** WATER_LOGP_MIN),
            P_max=min(m.P_max, 10.0 ** WATER_LOGP_MAX),
        )

    def __repr__(self):
        return (f'<ICES_COMB_EOS water+CH4+NH3 {self.version} '
                f's_gauge={self.s_gauge}>')

    # ------------------------------------------------------------------
    # Composition handling
    # ------------------------------------------------------------------
    @staticmethod
    def _fractions(Z_m, Z_a):
        """(f_w, Z_m, Z_a) with the water-primary convention validated.

        Non-finite fractions are REJECTED, not propagated.  A NaN passes every
        ordering comparison (NaN < 0 and NaN + Z > 1 are both False) and would
        then be converted to an exact zero weight by `_weighted`'s f > 0 gate,
        so without this check a NaN composition returns finite, plausible,
        wrong values (an un-renormalized single-component density, or
        rho = inf with u = s = 0) that pass downstream isfinite validators
        instead of tripping them.
        """
        Z_m = np.asarray(Z_m, dtype=float)
        Z_a = np.asarray(Z_a, dtype=float)
        if not (np.all(np.isfinite(Z_m)) and np.all(np.isfinite(Z_a))):
            raise ValueError('mass fractions Z_m, Z_a must be finite')
        if np.any(Z_m < 0.0) or np.any(Z_a < 0.0):
            raise ValueError('mass fractions Z_m, Z_a must be non-negative')
        if np.any(Z_m + Z_a > 1.0 + _FRAC_TOL):
            raise ValueError('Z_m + Z_a must not exceed 1 (water is the remainder)')
        # Snapped remainder: a water-free mixture must not carry a 1e-16
        # trace of water into the weights or the composition-aware mask.
        f_w = _water_remainder(Z_m, Z_a)
        return f_w, Z_m, Z_a

    @staticmethod
    def nested_to_direct(z_m_nested, z_a_nested):
        """Convert legacy `ice_eos.py` nested fractions to direct ones.

        ice_eos uses f_water = (1-z_m)(1-z_a), f_methane = z_m(1-z_a),
        f_ammonia = z_a.  The direct (water-primary) fractions are therefore
        Z_m = z_m(1-z_a) and Z_a = z_a.
        """
        z_m_nested = np.asarray(z_m_nested, dtype=float)
        z_a_nested = np.asarray(z_a_nested, dtype=float)
        return z_m_nested * (1.0 - z_a_nested), z_a_nested

    # ------------------------------------------------------------------
    # Pure-component fields at (P, T), linear CGS
    # ------------------------------------------------------------------
    # The aqua_eos getters build their query points as np.array([lgp, lgt]).T,
    # which is correct for 1-D vectors but for 2-D grids reverses ALL axes and
    # silently evaluates at transposed coordinates.  Every water call therefore
    # goes through _water_call, which ravels the inputs and restores the shape.
    @staticmethod
    def _water_call(fn, lgp, lgt):
        lgp = np.asarray(lgp, dtype=float)
        lgt = np.asarray(lgt, dtype=float)
        vals = np.asarray(fn(lgp.ravel(), lgt.ravel()), dtype=float)
        return vals.reshape(lgp.shape)

    def _water_rho(self, lgp, lgt):
        return 10.0 ** self._water_call(self.water.get_logrho_pt_tab, lgp, lgt)

    def _water_u(self, lgp, lgt):
        return 10.0 ** self._water_call(self.water.get_logu_pt_tab, lgp, lgt)

    def _water_s(self, lgp, lgt):
        return self._water_call(self.water.get_s_pt_tab, lgp, lgt)

    def _species_s(self, species, P, T):
        """Entropy of one CH4/NH3 species in the active gauge."""
        e = getattr(self, species)
        s = np.asarray(e.get_s_pt(P, T), dtype=float)
        if self.s_gauge == 'thirdlaw':
            s = s - e.s_offset
        return s

    @staticmethod
    def _weighted(f_w, w_term, Z_m, m_term, Z_a, a_term):
        """Sum of weighted terms with exact-zero weights contributing exactly 0.

        A component with zero mass fraction must not poison the sum through a
        NaN in its table (water's entropy grid has a NaN corner), and the pure
        end members must reproduce the pure tables bit for bit.
        """
        out = np.zeros(np.broadcast(f_w, w_term).shape)
        for f, term in ((f_w, w_term), (Z_m, m_term), (Z_a, a_term)):
            out = out + np.where(f > 0.0, f * term, 0.0)
        return out

    # ------------------------------------------------------------------
    # Mixture getters, linear in and out
    # ------------------------------------------------------------------
    def get_rho_pt(self, P, T, Z_m, Z_a):
        """Mixture density [g/cm^3] by the volume addition law.

        Parameters
        ----------
        P : float or array_like     pressure [dyn/cm^2]
        T : float or array_like     temperature [K]
        Z_m : float or array_like   methane mass fraction
        Z_a : float or array_like   ammonia mass fraction
            (water mass fraction is 1 - Z_m - Z_a)
        """
        scalar, P_arr, T_arr, Zm_arr, Za_arr = _broadcast4(P, T, Z_m, Z_a)
        f_w, Zm_arr, Za_arr = self._fractions(Zm_arr, Za_arr)
        lgp, lgt = _log10(P_arr), _log10(T_arr)

        v_w = 1.0 / self._water_rho(lgp, lgt)
        v_m = 1.0 / np.asarray(self.methane.get_rho_pt(P_arr, T_arr), dtype=float)
        v_a = 1.0 / np.asarray(self.ammonia.get_rho_pt(P_arr, T_arr), dtype=float)

        v_mix = self._weighted(f_w, v_w, Zm_arr, v_m, Za_arr, v_a)
        return _maybe_scalar(scalar, 1.0 / v_mix)

    def get_u_pt(self, P, T, Z_m, Z_a):
        """Mixture specific internal energy [erg/g], mass weighted.

        Note that each component's energy zero is its own: water's from AQUA,
        methane's and ammonia's from the T = 0 molecular ground state (their
        a_2 = 0 gauge).  Energy DIFFERENCES in P or T are physical; the
        absolute mixture value shifts by a composition-dependent constant.
        """
        scalar, P_arr, T_arr, Zm_arr, Za_arr = _broadcast4(P, T, Z_m, Z_a)
        f_w, Zm_arr, Za_arr = self._fractions(Zm_arr, Za_arr)
        lgp, lgt = _log10(P_arr), _log10(T_arr)

        u_w = self._water_u(lgp, lgt)
        u_m = np.asarray(self.methane.get_u_pt(P_arr, T_arr), dtype=float)
        u_a = np.asarray(self.ammonia.get_u_pt(P_arr, T_arr), dtype=float)

        return _maybe_scalar(scalar, self._weighted(f_w, u_w, Zm_arr, u_m, Za_arr, u_a))

    def get_s_pt(self, P, T, Z_m, Z_a, ideal_mixing=False):
        """Mixture specific entropy [erg/(g K)], mass weighted.

        In the default 'thirdlaw' gauge all three components sit on the
        absolute third-law scale, so the weighted sum is meaningful; see the
        module docstring for the two regions where it is not trustworthy and
        `entropy_valid_pt` for the mask.  `ideal_mixing` adds the ideal entropy
        of mixing from number fractions; it defaults off, matching the legacy
        `ice_eos.get_s_pt_val`.
        """
        scalar, P_arr, T_arr, Zm_arr, Za_arr = _broadcast4(P, T, Z_m, Z_a)
        f_w, Zm_arr, Za_arr = self._fractions(Zm_arr, Za_arr)

        s_mix = self._s_mix_arrays(P_arr, T_arr, f_w, Zm_arr, Za_arr)
        if ideal_mixing:
            s_mix = s_mix + self.get_s_ideal_mix(Zm_arr, Za_arr)
        return _maybe_scalar(scalar, s_mix)

    def get_s_ideal_mix(self, Z_m, Z_a):
        """Ideal entropy of mixing [erg/(g K)] for the ternary composition."""
        scalar = np.isscalar(Z_m) and np.isscalar(Z_a)
        # Broadcast BEFORE the mole arithmetic: with unequal shapes, n_tot
        # broadcasts while the per-component n does not, and the np.divide
        # below would then refuse its out= operand.
        Zm_b, Za_b = np.broadcast_arrays(np.atleast_1d(np.asarray(Z_m, float)),
                                         np.atleast_1d(np.asarray(Z_a, float)))
        f_w, Zm_arr, Za_arr = self._fractions(Zm_b, Za_b)

        n_w = f_w / M_WATER                        # moles per gram of mixture
        n_m = Zm_arr / MOLAR_MASS['methane']
        n_a = Za_arr / MOLAR_MASS['ammonia']
        n_tot = n_w + n_m + n_a

        s_id = np.zeros_like(n_tot)
        for n in (n_w, n_m, n_a):
            x = np.divide(n, n_tot, out=np.zeros_like(n), where=n_tot > 0)
            s_id = s_id - R_GAS * n_tot * _guarded_xlogx(x)
        return _maybe_scalar(scalar, s_id)

    # ------------------------------------------------------------------
    # log10 adapters (ice_eos.py / eos_class.z_eos dialect)
    # ------------------------------------------------------------------
    def get_logrho_pt_val(self, _lgp, _lgt, _zm, _za):
        """log10 of the mixture density from log10 P and log10 T.

        NOTE: _zm and _za here are DIRECT mass fractions (water primary), not
        the nested fractions of `ice_eos.get_logrho_pt_val`.  Convert legacy
        callers through `nested_to_direct` before wiring this in.
        """
        scalar = all(np.isscalar(x) for x in (_lgp, _lgt, _zm, _za))
        rho = self.get_rho_pt(10.0 ** np.asarray(_lgp, dtype=float),
                              10.0 ** np.asarray(_lgt, dtype=float), _zm, _za)
        return _maybe_scalar(scalar, np.log10(np.asarray(rho, dtype=float)))

    def get_u_pt_val(self, _lgp, _lgt, _zm, _za):
        """Mixture internal energy [erg/g] from log10 P and log10 T."""
        scalar = all(np.isscalar(x) for x in (_lgp, _lgt, _zm, _za))
        u = self.get_u_pt(10.0 ** np.asarray(_lgp, dtype=float),
                          10.0 ** np.asarray(_lgt, dtype=float), _zm, _za)
        return _maybe_scalar(scalar, u)

    def get_s_pt_val(self, _lgp, _lgt, _zm, _za, ideal_mixing=False):
        """Mixture entropy [erg/(g K)] from log10 P and log10 T."""
        scalar = all(np.isscalar(x) for x in (_lgp, _lgt, _zm, _za))
        s = self.get_s_pt(10.0 ** np.asarray(_lgp, dtype=float),
                          10.0 ** np.asarray(_lgt, dtype=float), _zm, _za,
                          ideal_mixing=ideal_mixing)
        return _maybe_scalar(scalar, s)

    # ------------------------------------------------------------------
    # (S, P) -> T inversion
    # ------------------------------------------------------------------
    # Entropy enters in k_B per baryon (the evolution code's unit), while every
    # S(P,T) getter above returns erg/(g K); the conversion factor is
    # erg_to_kbbar = 1.2027e-8.  All root finding below is done in the cgs
    # value so that the third-law entropy, which is legitimately NEGATIVE in
    # the CH4/NH3 cold-dense corner, needs no logarithm.
    #
    # s(T) at fixed P is strictly monotone increasing for every composition on
    # the joint domain (measured: zero cells with ds/dT < 0 on a 161 x 801
    # sweep for the six reference mixtures), so each (S, P) isobar has at most
    # one root, and a bracket whose ends straddle S contains exactly one.  The
    # solver is therefore a vectorised safeguarded Newton (Numerical Recipes'
    # rtsafe): every element carries its own bracket, a Newton step is taken
    # in y = ln T when it lands inside the bracket and the local slope is
    # positive, and a bisection step otherwise.  Entropy calls are made on the
    # still-active elements only, so a million-point (S, P, Z_m, Z_a) grid
    # costs a few dozen vectorised table look-ups rather than a Python loop
    # over scipy root finders.

    _SP_STATUS = {0: 'converged', 1: 'S below s(P, T_min)',
                  2: 'S above s(P, T_max)', 3: 'not converged',
                  4: 'non-finite input', 5: 'bracket collapsed on a jump',
                  6: 'P outside the table domain'}

    # Dilute reference state for the ideal-gas first guess (1 bar; every
    # component is a gas at 500-1000 K there).
    _GUESS_P0, _GUESS_T0, _GUESS_T1 = 1.0e6, 500.0, 1000.0

    def _s_mix_arrays(self, P_arr, T_arr, f_w, Zm_arr, Za_arr):
        """Mixture entropy [erg/(g K)] on validated, equally shaped arrays.

        The composition arrays must already have passed `_fractions`; this is
        the inner loop of the inversion and repeats no checks.
        """
        lgp, lgt = _log10(P_arr), _log10(T_arr)
        s_w = self._water_s(lgp, lgt)
        s_m = self._species_s('methane', P_arr, T_arr)
        s_a = self._species_s('ammonia', P_arr, T_arr)
        return self._weighted(f_w, s_w, Zm_arr, s_m, Za_arr, s_a)

    def _ideal_guess_coeffs(self):
        """Per-species (s0, c_p, R/M) at the dilute reference state, in the
        active gauge, measured from the tables once and cached."""
        cached = getattr(self, '_ideal_coeffs', None)
        if cached is not None:
            return cached
        P0, T0, T1 = self._GUESS_P0, self._GUESS_T0, self._GUESS_T1
        lgp0 = np.log10(P0)
        s0, cp, rm = {}, {}, {}
        for sp in ('water', 'methane', 'ammonia'):
            if sp == 'water':
                sA = float(np.ravel(self._water_s(lgp0, np.log10(T0)))[0])
                sB = float(np.ravel(self._water_s(lgp0, np.log10(T1)))[0])
                M = M_WATER
            else:
                sA = float(np.ravel(self._species_s(sp, P0, T0))[0])
                sB = float(np.ravel(self._species_s(sp, P0, T1))[0])
                M = MOLAR_MASS[sp]
            s0[sp] = sA
            cp[sp] = (sB - sA) / np.log(T1 / T0)      # effective c_p [erg/g/K]
            rm[sp] = R_GAS / M                        # R per gram
        self._ideal_coeffs = (s0, cp, rm)
        return self._ideal_coeffs

    def _ideal_t_guess(self, S_cgs, P_arr, f_w, Zm_arr, Za_arr):
        """Ideal-gas T(S, P): s = s0 + c_p ln(T/T0) - (R/M) ln(P/P0), with the
        three species mass weighted.  Returns ln T, unclipped."""
        s0, cp, rm = self._ideal_guess_coeffs()
        w = {'water': f_w, 'methane': Zm_arr, 'ammonia': Za_arr}
        s0_mix = sum(w[k] * s0[k] for k in w)
        cp_mix = sum(w[k] * cp[k] for k in w)
        rm_mix = sum(w[k] * rm[k] for k in w)
        lnP = np.log(np.where(P_arr > 0.0, P_arr, np.nan) / self._GUESS_P0)
        return np.log(self._GUESS_T0) + (S_cgs - s0_mix + rm_mix * lnP) / cp_mix

    def _solve_sp_flat(self, S_cgs, P_arr, f_w, Zm_arr, Za_arr, y0, y_min,
                       y_max, method, maxiter, tol_cgs, xtol, h):
        """Vectorised safeguarded Newton for ln T on flat arrays.

        Returns T, status, n_iter, n_newton, n_bisect, residual (cgs).
        """
        n = S_cgs.size
        T = np.full(n, np.nan)
        status = np.full(n, 3, dtype=np.int8)
        n_iter = np.zeros(n, dtype=np.int32)
        n_newton = np.zeros(n, dtype=np.int32)
        n_bisect = np.zeros(n, dtype=np.int32)
        resid = np.full(n, np.nan)

        good = (np.isfinite(S_cgs) & np.isfinite(P_arr) & (P_arr > 0.0)
                & np.isfinite(y0))
        status[~good] = 4
        # The single-root premise (s monotone in T on every isobar) is only
        # established inside the joint table domain; beyond it the RGIs
        # extrapolate and the CH4/NH3 isobars turn non-monotone above P_max.
        out_p = good & ((P_arr < self.domain.P_min) | (P_arr > self.domain.P_max))
        status[out_p] = 6
        good &= ~out_p

        def s_at(y, idx):
            return self._s_mix_arrays(P_arr[idx], np.exp(y), f_w[idx],
                                      Zm_arr[idx], Za_arr[idx])

        # Bracket the whole domain first.  Two vectorised look-ups settle
        # every out-of-range element and give each survivor a bracket with
        # known signs at both ends, so no later step can wander.
        y_lo = np.full(n, y_min)
        y_hi = np.full(n, y_max)
        idx = np.flatnonzero(good)
        g_lo = s_at(y_lo[idx], idx) - S_cgs[idx]
        g_hi = s_at(y_hi[idx], idx) - S_cgs[idx]
        hit_lo = np.abs(g_lo) <= tol_cgs
        hit_hi = np.abs(g_hi) <= tol_cgs
        below = (g_lo > tol_cgs) & ~hit_lo
        above = (g_hi < -tol_cgs) & ~hit_hi
        for hit, y_end, g_end in ((hit_lo, y_min, g_lo), (hit_hi, y_max, g_hi)):
            k = idx[hit]
            T[k], status[k], resid[k] = np.exp(y_end), 0, g_end[hit]
        status[idx[below]] = 1
        status[idx[above]] = 2
        active = np.zeros(n, dtype=bool)
        active[idx[~(hit_lo | hit_hi | below | above)]] = True

        # Start from the caller's / ideal guess where it lies inside the
        # bracket; elsewhere (the ideal guess leaves the table for ~40% of
        # the domain) use the regula-falsi point of the two endpoint
        # residuals already in hand, which costs nothing and beats an
        # endpoint start.
        g_lo_all = np.full(n, np.nan)
        g_hi_all = np.full(n, np.nan)
        g_lo_all[idx], g_hi_all[idx] = g_lo, g_hi
        with np.errstate(divide='ignore', invalid='ignore'):
            y_rf = y_min + (y_max - y_min) * (-g_lo_all) / (g_hi_all - g_lo_all)
        y_rf = np.clip(y_rf, y_min, y_max)
        inside = (y0 > y_min) & (y0 < y_max)
        y = np.where(active, np.where(inside, y0, y_rf), np.nan)
        # step sizes for rtsafe's guard: the Newton step is compared with
        # half the step taken TWO iterations earlier (comparing with the
        # last step rejects the first Newton step after every bisection)
        dy_old = np.full(n, y_max - y_min)
        dy_old2 = dy_old.copy()
        for _ in range(maxiter):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            yi = y[idx]
            g = s_at(yi, idx) - S_cgs[idx]
            n_iter[idx] += 1
            # tighten the bracket from the sign of the residual
            neg = g < 0.0
            y_lo[idx[neg]] = yi[neg]
            y_hi[idx[~neg]] = yi[~neg]
            done = np.abs(g) <= tol_cgs
            narrow = ~done & ((y_hi[idx] - y_lo[idx]) <= xtol)
            # The interpolants are continuous, so a bracket that has shrunk
            # to xtol locates the root; a residual still above tol_s there
            # is the evaluation floor of s (~1e-14 k_B/baryon inside the
            # table, more when extrapolating).  Only a residual far above
            # tol_s, which no continuous s(T) can produce, is flagged.
            collapsed = narrow & (np.abs(g) > 1.0e3 * tol_cgs)
            done |= narrow & ~collapsed
            for sel, code in ((done, 0), (collapsed, 5)):
                k = idx[sel]
                T[k], status[k], resid[k] = np.exp(yi[sel]), code, g[sel]
                active[k] = False
            rem = ~(done | collapsed)
            if not np.any(rem):
                continue
            k = idx[rem]
            yr, gr = yi[rem], g[rem]
            lo, hi = y_lo[k], y_hi[k]
            if method == 'bisect':
                y[k] = 0.5 * (lo + hi)
                n_bisect[k] += 1
                continue
            # One-sided difference in ln T taken TOWARD the root (backward
            # when s > S, forward when s < S).  The interpolants are
            # piecewise linear in ln T with cells of 0.01-0.02, so with h
            # far inside a cell this is the exact slope of the piece that
            # contains the root once the iterate is within h of it; a
            # difference away from the root can straddle a knot (the water
            # liquid-vapour step is one) and stall Newton on the wrong slope.
            sgn = np.where(gr > 0.0, -1.0, 1.0)
            gp = sgn * (s_at(yr + sgn * h, k) - (gr + S_cgs[k])) / h
            with np.errstate(divide='ignore', invalid='ignore'):
                dy = -gr / gp
            y_new = yr + dy
            ok = np.isfinite(y_new) & (gp > 0.0) & (y_new > lo) & (y_new < hi)
            if method == 'newton':
                # plain Newton, clipped to the domain: for testing the basin
                y_new = np.where(ok, y_new, np.clip(y_new, lo, hi))
                y_new = np.where(np.isfinite(y_new), y_new, 0.5 * (lo + hi))
                n_newton[k] += 1
            else:
                # rtsafe's second guard: a Newton step that does not at least
                # halve the step before last is not converging; bisect instead
                ok &= np.abs(dy) <= 0.5 * dy_old2[k]
                y_new = np.where(ok, y_new, 0.5 * (lo + hi))
                n_newton[k[ok]] += 1
                n_bisect[k[~ok]] += 1
            dy_old2[k] = dy_old[k]
            dy_old[k] = np.abs(y_new - yr)
            y[k] = y_new
        return T, status, n_iter, n_newton, n_bisect, resid

    def get_t_sp_inv(self, S, P, Z_m, Z_a, *, s_units='kbbar', ideal_mixing=False,
                     T_guess=None, ideal_guess=True, warm_axis=None,
                     bounds_T=None, method='newton_bisect', maxiter=60,
                     tol_s=1.0e-12, xtol=1.0e-13, h=1.0e-6,
                     return_diagnostics=False):
        """Temperature T(S, P, Z_m, Z_a) [K] by inverting the mixture entropy.

        Parameters
        ----------
        S : float or array_like
            Specific entropy.  In k_B PER BARYON by default (`s_units=
            'kbbar'`), the unit the evolution code carries; pass
            `s_units='cgs'` for erg/(g K), the unit `get_s_pt` returns.
        P : float or array_like        pressure [dyn/cm^2]
        Z_m, Z_a : float or array_like methane and ammonia mass fractions
            (direct fractions; water is the remainder).  All four inputs
            broadcast together.
        ideal_mixing : bool
            Whether S includes the ideal entropy of mixing; must match the
            flag used when S was produced (default False, as in `get_s_pt`).
        T_guess : array_like, optional
            Starting temperatures [K], broadcastable to the inputs; used
            where finite, e.g. the previous model's T profile.
        ideal_guess : bool
            Without `T_guess`, start from the ideal-gas inversion measured
            from the tables at 1 bar (True) or from the geometric midpoint of
            the temperature bounds (False).
        warm_axis : int, optional
            March along this axis of the broadcast inputs, seeding each slice
            with the previous slice's solution (an isentrope integrated in P,
            or a table built one S row at a time).  None solves all elements
            at once from the initial guess.
        bounds_T : (float, float), optional
            Temperature bracket [K]; defaults to the joint table domain,
            [100, 30000] K.  Water-free mixtures may pass (50, 30000).
        method : {'newton_bisect', 'newton', 'bisect'}
            'newton_bisect' (default) is Newton in ln T safeguarded by a
            per-element bracket, bisected whenever a Newton step leaves it
            or fails to halve the previous step (Numerical Recipes'
            rtsafe); 'bisect' is pure bisection on the same bracket;
            'newton' is unsafeguarded Newton clipped to the bounds (for
            tests of the Newton basin only).
        maxiter, tol_s, xtol, h : float
            Iteration cap; residual tolerance |s(T) - S| in k_B/baryon;
            bracket width tolerance in ln T; finite-difference step in ln T
            (taken toward the root; keep it far below the 0.01 table cell).
        return_diagnostics : bool
            Also return a dict with per-element 'status' (codes in
            `_SP_STATUS`), 'converged', 'n_iter', 'n_newton', 'n_bisect'
            and 'residual' (k_B/baryon).

        Returns
        -------
        T : float or ndarray
            Temperature [K]; NaN where S lies outside the isobar's entropy
            range, where P lies outside the joint table domain
            [`domain.P_min`, `domain.P_max`] (the (P, T) getters extrapolate
            there, the inversion does not), where the inputs are non-finite,
            or (never observed with the default `maxiter`) where the
            iteration cap was hit.  Scalar inputs give a float.

        Notes
        -----
        The entropy is evaluated in the active gauge (`s_gauge`) without any
        validity gating: an S that maps into the masked cold-dense corner is
        inverted like any other.  Test `entropy_valid_pt(P, T, Z_m, Z_a)` on
        the result if that matters.  Inside that corner the AQUA fill drives
        the water entropy along the load-time fill: S at or below ~1e-12
        k_B/baryon returns T_min, a larger S inverts the invented fill to an
        interior T that carries no information.  Mask the result with
        `entropy_valid_pt(P, T, Z_m, Z_a)` before storing it.  In the
        trusted region the tolerance maps to at most ~1e-9 in ln T.
        Unsafeguarded Newton (`method='newton'`) fails on about 10% of
        trusted-region points, at the AQUA phase steps near 1000 K; the
        default never has.

        Status 5 (bracket collapsed to `xtol` with a residual far above
        `tol_s`) carries a FINITE T: T is finite iff status is 0 or 5.  With
        the defaults it has not been observed; a tiny `tol_s` produces it.
        Every element of Z_m, Z_a must satisfy Z_m + Z_a <= 1 (ValueError
        otherwise, for the whole call): mask a rectangular composition grid
        to the feasible triangle before calling.  `s_units` also accepts
        'kb/baryon' and 'erg/g/K', case-insensitively.
        """
        su = str(s_units).lower()
        if su in ('kbbar', 'kb/baryon'):
            in_kb = True
        elif su in ('cgs', 'erg/g/k'):
            in_kb = False
        else:
            raise ValueError("s_units must be 'kbbar' or 'cgs'")
        if method not in ('newton_bisect', 'newton', 'bisect'):
            raise ValueError("method must be 'newton_bisect', 'newton' or 'bisect'")
        scalar, S_arr, P_arr, Zm_arr, Za_arr = _broadcast4(S, P, Z_m, Z_a)
        shape = S_arr.shape
        f_w, Zm_arr, Za_arr = self._fractions(Zm_arr, Za_arr)
        with np.errstate(over='ignore', invalid='ignore'):
            S_cgs = S_arr / erg_to_kbbar if in_kb else S_arr.copy()
            if ideal_mixing:
                S_cgs = S_cgs - self.get_s_ideal_mix(Zm_arr, Za_arr)
        tol_cgs = tol_s / erg_to_kbbar

        T_lo, T_hi = bounds_T if bounds_T is not None else (self.domain.T_min,
                                                            self.domain.T_max)
        if not (0.0 < T_lo < T_hi):
            raise ValueError('bounds_T must satisfy 0 < T_lo < T_hi')
        y_min, y_max = np.log(T_lo), np.log(T_hi)

        # the caller's guess (finite and positive) wins everywhere it is
        # given, then the warm seed, then the ideal guess
        if T_guess is not None:
            Tg = np.broadcast_to(np.asarray(T_guess, float), shape).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                y_user = np.where(Tg > 0.0, np.log(Tg), np.nan)
        else:
            y_user = np.full(shape, np.nan)
        if ideal_guess:
            y_id = self._ideal_t_guess(S_cgs, P_arr, f_w, Zm_arr, Za_arr)
        else:
            y_id = np.full(shape, 0.5 * (y_min + y_max))
        y0 = np.where(np.isfinite(y_user), y_user, y_id)

        flat = (lambda a: np.ascontiguousarray(a).reshape(-1))
        args = (S_cgs, P_arr, f_w, Zm_arr, Za_arr, y0, y_user)
        if warm_axis is not None and S_arr.ndim > 0:
            nd = S_arr.ndim
            if not -nd <= int(warm_axis) < nd:
                raise ValueError(f'warm_axis {warm_axis} out of range for '
                                 f'{nd}-D inputs')
        if warm_axis is None or S_arr.ndim == 0 or S_arr.size == 0:
            out = self._solve_sp_flat(*(flat(a) for a in args[:6]), y_min, y_max,
                                      method, maxiter, tol_cgs, xtol, h)
            out = [o.reshape(shape) for o in out]
        else:
            ax = int(warm_axis) % S_arr.ndim
            moved = [np.moveaxis(a, ax, 0) for a in args]
            pieces = [[] for _ in range(6)]
            y_prev = None
            for k in range(moved[0].shape[0]):
                sl = [m[k] for m in moved]
                y_start = sl[5]
                if y_prev is not None:
                    y_start = np.where(np.isfinite(sl[6]), sl[6],
                                       np.where(np.isfinite(y_prev), y_prev, y_start))
                sub_shape = sl[0].shape
                res = self._solve_sp_flat(*(flat(a) for a in sl[:5]),
                                          flat(y_start), y_min, y_max, method,
                                          maxiter, tol_cgs, xtol, h)
                res = [r.reshape(sub_shape) for r in res]
                with np.errstate(divide='ignore', invalid='ignore'):
                    y_prev = np.log(res[0])
                for p, r in zip(pieces, res):
                    p.append(r)
            out = [np.moveaxis(np.stack(p, axis=0), 0, ax) for p in pieces]
        T_out, status, n_iter, n_newton, n_bisect, resid = out

        T_ret = _maybe_scalar(scalar, T_out)
        if not return_diagnostics:
            return T_ret
        diag = dict(status=status, converged=(status == 0),
                    n_iter=n_iter, n_newton=n_newton, n_bisect=n_bisect,
                    residual=resid * erg_to_kbbar,
                    status_names=dict(self._SP_STATUS))
        if scalar:
            for key in ('status', 'converged', 'n_iter', 'n_newton',
                        'n_bisect', 'residual'):
                diag[key] = diag[key].reshape(-1)[0].item()
        return T_ret, diag

    # case alias only: iron_gonzalez_eos spells it get_T_sp_inv, but its
    # positional arguments (bracket, P in GPa) differ from these
    get_T_sp_inv = get_t_sp_inv

    def get_t_sp(self, S, P, Z_m, Z_a, s_units='kbbar', tab=False, **kwargs):
        """T(S, P, Z_m, Z_a) [K].  `tab=True` will read the 4-D (S, P, Z_m,
        Z_a) table once it exists; until then only the inversion is
        available.  Extra keywords go to `get_t_sp_inv`.

        This class speaks the eos_class.z_eos dialect (P in dyn/cm^2,
        composition positionals); it is NOT a drop-in `mantle_eos` for
        hydrostatic.py, which expects P in GPa and no composition arguments.
        """
        if tab:
            raise NotImplementedError('no (S, P, Z_m, Z_a) table yet; '
                                      'use tab=False (inversion)')
        return self.get_t_sp_inv(S, P, Z_m, Z_a, s_units=s_units, **kwargs)

    def _pt_on_finite_t(self, fn, P, T, Z_m, Z_a):
        """Evaluate a (P,T) getter where T is finite; NaN elsewhere."""
        scalar, P_arr, T_arr, Zm_arr, Za_arr = _broadcast4(P, T, Z_m, Z_a)
        ok = np.isfinite(T_arr)
        out = np.full(T_arr.shape, np.nan)
        if np.any(ok):
            out[ok] = np.asarray(fn(P_arr[ok], T_arr[ok], Zm_arr[ok], Za_arr[ok]),
                                 dtype=float)
        return _maybe_scalar(scalar, out)

    def _sp_getter(self, fn, S, P, Z_m, Z_a, s_units, kwargs):
        """fn(P, T(S, P)); with return_diagnostics=True returns (value, diag)."""
        res = self.get_t_sp(S, P, Z_m, Z_a, s_units=s_units, **kwargs)
        T, diag = res if isinstance(res, tuple) else (res, None)
        val = self._pt_on_finite_t(fn, P, T, Z_m, Z_a)
        return val if diag is None else (val, diag)

    def get_rho_sp(self, S, P, Z_m, Z_a, s_units='kbbar', **kwargs):
        """Density [g/cm^3] on the (S, P) basis: rho(P, T(S, P)).  Extra
        keywords go to `get_t_sp_inv`; `return_diagnostics=True` gives
        (rho, diag)."""
        return self._sp_getter(self.get_rho_pt, S, P, Z_m, Z_a, s_units, kwargs)

    def get_u_sp(self, S, P, Z_m, Z_a, s_units='kbbar', **kwargs):
        """Specific internal energy [erg/g] on the (S, P) basis:
        u(P, T(S, P)).  Keywords as for `get_rho_sp`."""
        return self._sp_getter(self.get_u_pt, S, P, Z_m, Z_a, s_units, kwargs)

    # log10 adapters on the (S, P) basis: S in k_B/baryon, log10 P in
    # dyn/cm^2 (the eos_class.z_eos get_logt_sp(_s_kb, _lgp, ...) dialect)
    @staticmethod
    def _pow10(_lg):
        return 10.0 ** float(_lg) if np.isscalar(_lg) else 10.0 ** np.asarray(_lg, float)

    # All three adapters: S in k_B/baryon (the unit is pinned: no `s_units`),
    # log10 P in dyn/cm^2, Z_m and Z_a are DIRECT mass fractions with water
    # the remainder (legacy nested callers: convert with `nested_to_direct`).
    # Extra keywords go to `get_t_sp_inv`.
    @staticmethod
    def _pin_kbbar(kwargs):
        if str(kwargs.pop('s_units', 'kbbar')).lower() not in ('kbbar', 'kb/baryon'):
            raise ValueError('the *_sp_val adapters take S in k_B/baryon only')
        return kwargs

    def get_logt_sp_val(self, _s_kb, _lgp, _zm, _za, **kwargs):
        """log10 T(S, P, Z_m, Z_a).  S in k_B/baryon, log10 P in dyn/cm^2,
        DIRECT mass fractions (see the note above)."""
        kwargs = self._pin_kbbar(kwargs)
        T = self.get_t_sp(_s_kb, self._pow10(_lgp), _zm, _za, **kwargs)
        return np.log10(T) if np.isscalar(T) else np.log10(np.asarray(T))

    def get_logrho_sp_val(self, _s_kb, _lgp, _zm, _za, **kwargs):
        """log10 rho(S, P, Z_m, Z_a) [g/cm^3].  S in k_B/baryon, log10 P in
        dyn/cm^2, DIRECT mass fractions (see the note above)."""
        kwargs = self._pin_kbbar(kwargs)
        rho = self.get_rho_sp(_s_kb, self._pow10(_lgp), _zm, _za, **kwargs)
        return np.log10(rho) if np.isscalar(rho) else np.log10(np.asarray(rho))

    def get_u_sp_val(self, _s_kb, _lgp, _zm, _za, **kwargs):
        """u(S, P, Z_m, Z_a) [erg/g], linear.  S in k_B/baryon, log10 P in
        dyn/cm^2, DIRECT mass fractions (see the note above)."""
        kwargs = self._pin_kbbar(kwargs)
        return self.get_u_sp(_s_kb, self._pow10(_lgp), _zm, _za, **kwargs)

    # ------------------------------------------------------------------
    # Validity and diagnostics
    # ------------------------------------------------------------------
    def in_domain_pt(self, P, T):
        """True where (P,T) lies inside all three source tables."""
        scalar, P_arr, T_arr = _broadcast(P, T)
        d = self.domain
        ok = ((P_arr >= d.P_min) & (P_arr <= d.P_max)
              & (T_arr >= d.T_min) & (T_arr <= d.T_max))
        return bool(ok.reshape(-1)[0]) if scalar else ok

    def entropy_valid_pt(self, P, T, Z_m=None, Z_a=None):
        """True where the mixture entropy is trustworthy.

        Without Z_m, Z_a the test is composition blind and conservative: it
        masks any point where ANY of the three components is untrusted.  With
        Z_m, Z_a given, only components actually present (fraction > 0) are
        tested, so pure water is not masked where methane alone misbehaves.

        Three conditions are combined:
        (1) (P,T) lies inside the joint table domain (`in_domain_pt`) -- the
            RGIs extrapolate silently outside it, and below AQUA's 100 K floor
            the linearly extrapolated water entropy goes negative;
        (2) water's entropy there is table data, not the load-time NaN fill:
            the revised-AQUA grid's s = -1 sentinel corner (roughly logP in
            [12.9, 15.6] x logT in [2.0, 3.9]) is filled with extrapolated
            values so square sweeps never break, but those cells are invented
            and stay masked here;
        (3) the CH4/NH3 surfaces' third-law entropy is positive there (it is
            negative, hence wrong, over about 17% / 14% of their (P,T) grid,
            at logP >~ 10.7 and logT <~ 4.0).
        Density and internal energy carry masks for none of (2)-(3), but (1)
        applies to them as well: outside the joint domain they are
        extrapolations, not table data.
        """
        if (Z_m is None) != (Z_a is None):
            raise ValueError('give both Z_m and Z_a, or neither')
        if Z_m is None:
            scalar, P_arr, T_arr = _broadcast(P, T)
            present = {'water': True, 'methane': True, 'ammonia': True}
        else:
            scalar, P_arr, T_arr, Z_m, Z_a = _broadcast4(P, T, Z_m, Z_a)
            f_w, Z_m, Z_a = self._fractions(Z_m, Z_a)
            present = {'water': f_w > 0.0, 'methane': Z_m > 0.0,
                       'ammonia': Z_a > 0.0}
        lgp, lgt = _log10(P_arr), _log10(T_arr)
        d = self.domain
        ok = ((P_arr >= d.P_min) & (P_arr <= d.P_max)
              & (T_arr >= d.T_min) & (T_arr <= d.T_max))
        bad_w = ~np.isfinite(self._water_s(lgp, lgt))
        if self._water_fill_rgi is not None:
            bad_w |= _interp(self._water_fill_rgi, lgp, lgt) > 0.5
        ok &= ~(bad_w & present['water'])
        for sp in ('methane', 'ammonia'):
            e = getattr(self, sp)
            s_raw = np.asarray(e.get_s_pt(P_arr, T_arr), dtype=float) - e.s_offset
            ok &= ~((s_raw <= 0.0) & present[sp])
        return bool(ok.reshape(-1)[0]) if scalar else ok

    def gauge_report(self, P=1.0e6, T=500.0, verbose=True):
        """Entropy of all three components at a common dilute-gas state.

        With s_gauge='thirdlaw' the three values should agree with the JANAF
        gas-phase standard entropies and hence sit within a few percent of one
        another per mole; with 'table' the CH4/NH3 rows are larger by their
        respective offsets (about 100.5 R_s each).
        """
        rows = {}
        lgp, lgt = np.log10(P), np.log10(T)
        rows['water'] = dict(s=float(np.atleast_1d(self._water_s(lgp, lgt))[0]),
                             M=M_WATER, s_offset=0.0)
        for sp in ('methane', 'ammonia'):
            e = getattr(self, sp)
            rows[sp] = dict(s=float(np.atleast_1d(self._species_s(sp, P, T))[0]),
                            M=MOLAR_MASS[sp], s_offset=e.s_offset)
        if verbose:
            print(f'component entropies at P = {P:.3g} dyn/cm^2, T = {T:.4g} K'
                  f'  (s_gauge = {self.s_gauge})')
            print(f'  {"component":<10s} {"s [erg/g/K]":>14s} {"s [J/mol/K]":>13s}'
                  f' {"offset removed":>15s}')
            for name, r in rows.items():
                removed = (r['s_offset'] if self.s_gauge == 'thirdlaw'
                           and r['s_offset'] else 0.0)
                print(f'  {name:<10s} {r["s"]:14.5e} {r["s"] * r["M"] * 1e-7:13.2f}'
                      f' {removed:15.4e}')
        return rows


# ---------------------------------------------------------------------------
# Module-level singleton, built lazily so importing this file is cheap
# ---------------------------------------------------------------------------
_DEFAULT = None


def default_eos(version=TABLE_VERSION, s_gauge='thirdlaw'):
    """Shared ICES_COMB_EOS instance, loaded on first use."""
    global _DEFAULT
    if (_DEFAULT is None or _DEFAULT.version != str(version)
            or _DEFAULT.s_gauge != s_gauge):
        _DEFAULT = ICES_COMB_EOS(version=version, s_gauge=s_gauge)
    return _DEFAULT
