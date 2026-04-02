'''
every function should:

- accept scalars or vectors as args (but not python lists)
- same naming convention as SCvH (eventually)
- argument order is: two of (S, logrho, logP, logT) [in that order], Y, and
  others

Proposed naming convention:
Getters for getting q1 as a function of q2, q3):
    f"get_{q2}_{q2}{q3}(q2, q3, Y))"
Getters for partials of q1 WRT q2 leaving q3q4 constant as functions of q5q6:
    f"d{q1}d{q2}_{q3}{q4}_{q5}{q6}(q5, q6, Y)"
Ordering on quantities follows the ordering above
'''
import numpy as np
from scipy.optimize import brenth, minimize, root_scalar, root
from astropy import units as u
from astropy.constants import k_B, m_p
from astropy.constants import u as amu
from scipy.interpolate import RegularGridInterpolator as RGI

"""
    This file provides access to the ideal EOS and accepts
    the same format as the other non-ideal EOSes. 
    
    This file does not use precomputed tables and it can be
    used to compare to the other EOSes. The ideal EOS is also 
    used to provided initial guesses to the inversion optimization
    functions used in all other EOS files.

    Authors: Yubo Su, Roberto Tejada Arevalo
    
"""

erg_to_kbbar = (u.erg/u.Kelvin/u.gram).to(k_B/amu)

kB = 1.380649e-16          # erg/K  (Boltzmann)
mp = amu.to('g').value     # g      (atomic mass unit, matching eos_class convention)
h_planck = 6.62607015e-27  # erg·s  (Planck, exact since 2019 SI)

# Sackur-Tetrode base constants (per particle, in units of kB).
# S_particle/kB = _C0_XX + f(T, P or rho, m)
# All three are derived from the same fundamental constants and are
# guaranteed mutually consistent via the ideal-gas law.
#
# S(P,T):   s_p = _C0_PT   + 5/2 ln T - ln P     + 3/2 ln m
# S(rho,T): s_p = _C0_RHOT + 3/2 ln T - ln rho   + 5/2 ln m
# S(rho,P): s_p = _C0_RHOP + 3/2 ln P + 4   ln m - 5/2 ln rho
_C0_PT = (2.5
          + 2.5 * np.log(kB)
          + 1.5 * np.log(2 * np.pi)
          + 1.5 * np.log(mp)
          - 3.0 * np.log(h_planck))

# Derive from _C0_PT via ideal gas:  P = rho kB T / (m mp)
_C0_RHOT = _C0_PT + np.log(mp) - np.log(kB)
_C0_RHOP = _C0_PT + 2.5 * np.log(mp) - 2.5 * np.log(kB)

U_UNIT = kB / mp  # must use same mass as the ideal gas law (proton mass)
Rideal = kB / mp   # R per baryon in erg/(g·K); same as U_UNIT

class IdealEOS(object):
    """
    ideal eos with proton mass m
    """
    def __init__(self, m):
        super(IdealEOS, self).__init__()
        self.m = m

    ## S getters — all return S in erg/(g·K)
    def get_s_pt(self, logp, logt, _y):
        p = 10**logp
        t = 10**logt
        s_kbbar = (_C0_PT + 2.5 * np.log(t) - np.log(p)
                   + 1.5 * np.log(self.m)) / self.m
        return s_kbbar / erg_to_kbbar

    def get_s_rhot(self, logrho, logt, _y):
        rho = 10**logrho
        t = 10**logt
        s_kbbar = (_C0_RHOT + 1.5 * np.log(t) - np.log(rho)
                   + 2.5 * np.log(self.m)) / self.m
        return s_kbbar / erg_to_kbbar

    def get_s_rhop(self, logrho, logp, _y):
        p = 10**logp
        rho = 10**logrho
        s_kbbar = (_C0_RHOP + 1.5 * np.log(p) + 4.0 * np.log(self.m)
                   - 2.5 * np.log(rho)) / self.m
        return s_kbbar / erg_to_kbbar

    ## rho getters
    def get_rho_pt(self, logp, logt, _y):
        p = 10**logp
        t = 10**logt
        return np.log10(p * self.m * mp / (kB * t))

    def get_rho_sp(self, s, logp, _y):
        """Invert S(rho,P) for rho.  s in kB/baryon."""
        p = 10**logp
        rho = np.exp((_C0_RHOP + 1.5 * np.log(p)
                       + 4.0 * np.log(self.m)
                       - s * self.m) / 2.5)
        return np.log10(rho)

    ## P getters
    def get_p_rhot(self, logrho, logt, _y):
        rho = 10**logrho
        t = 10**logt
        return np.log10(rho * kB * t / (self.m * mp))

    def get_p_srho(self, s, logrho, _y):
        """Invert S(rho,P) for P.  s in kB/baryon."""
        rho = 10**logrho
        logp_nat = (s * self.m - _C0_RHOP
                    - 4.0 * np.log(self.m)
                    + 2.5 * np.log(rho)) / 1.5
        return np.log10(np.exp(logp_nat))

    ## T getters
    def get_t_rhop(self, logrho, logp, _y):
        p = 10**logp
        rho = 10**logrho
        return np.log10(p * self.m * mp / (rho * kB))

    def get_t_sp(self, s, logp, _y):
        """Invert S(P,T) for T.  s in kB/baryon."""
        p = 10**logp
        logt_nat = (s * self.m - _C0_PT
                    + np.log(p) - 1.5 * np.log(self.m)) / 2.5
        return np.log10(np.exp(logt_nat))

    def get_t_srho(self, s, logrho, _y):
        """Invert S(rho,T) for T.  s in kB/baryon."""
        rho = 10**logrho
        logt_nat = (s * self.m - _C0_RHOT
                    + np.log(rho) - 2.5 * np.log(self.m)) / 1.5
        return np.log10(np.exp(logt_nat))

    ## U getters
    def get_u_pt(self, logp, logt, _y):
        return np.log10(U_UNIT * 3/2 * 10**logt / self.m)

    def get_u_srho(self, s, logrho, _y):
        logp, logt = self.get_pt_srho(s, logrho, _y)
        return self.get_u_pt(logp, logt, _y)

    ## combined getters
    def get_sp_rhot(self, logrho, logt, _y):
        return self.get_s_rhot(logrho, logt), self.get_p_rhot(logrho, logt)

    def get_rhot_sp(self, s, logp, _y):
        return self.get_rho_sp(s, logp, _y), self.get_t_sp(s, logp, _y)

    def get_pt_srho(self, s, logrho, _y):
        return self.get_p_srho(s, logrho, _y), self.get_t_srho(s, logrho, _y)

    ## analytic derivatives
    def get_chirho_sp(self, s, _logp, _y):
        # idiomatic (?) way of returning the same type as s
        return 0 * s + 1

    def get_grad_ad(self, s, _logp, _y):
        # nabla_ad = d ln T / d ln P |_S = (gamma-1)/gamma = 2/5
        return 0 * s + 2/5

    ## misc
    def get_c_p(self, s, _logp, _y):
        return 0 * s + 5/2 * Rideal / self.m
    def get_c_v(self, s, _logp, _y):
        return 0 * s + 3/2 * Rideal / self.m

# =====================================================================
# Predefined molecular species — NIST CCCBDB harmonic frequencies
# θ = (hc/kB) × ω,  with hc/kB = 1.4388 cm·K
# H₂:  B = 60.853 cm⁻¹ → θ_rot = 87.6 K;  ω_e = 4401 cm⁻¹ → θ_vib = 6332 K
# H₂O: A,B,C = 27.877, 14.512, 9.285 cm⁻¹;  ω = 1649, 3832, 3943 cm⁻¹
#       → θ_vib = (2373, 5514, 5674) K
# =====================================================================
SPECIES = {
    'H2':     dict(m=2.016,  geometry='linear',    sigma=2,
                   theta_rot=87.6, theta_vib=(6332.,)),
    'He':     dict(m=4.0026, geometry='monatomic'),
    'H2O':    dict(m=18.015, geometry='nonlinear', sigma=2,
                   theta_rot=(40.1, 20.9, 13.4), theta_vib=(2373., 5514., 5674.)),
    'Fe':     dict(m=55.845, geometry='monatomic'),
    'MgSiO3': dict(m=100.39, geometry='monatomic'),
}


class MolecularIdealEOS(IdealEOS):
    """
    Ideal gas EOS with translational, rotational, and vibrational
    degrees of freedom.

    Rotational: classical high-T limit (valid for T >> theta_rot).
    Vibrational: quantum harmonic oscillator (Einstein model).

    For monatomic species this reduces exactly to IdealEOS.
    """

    def __init__(self, m, geometry='monatomic', sigma=1,
                 theta_rot=None, theta_vib=None):
        super().__init__(m)
        self.geometry = geometry
        self.sigma = sigma

        # Rotational characteristic temperatures
        if geometry == 'monatomic':
            self._n_rot_dof = 0
            self._theta_rot = None
        elif geometry == 'linear':
            self._n_rot_dof = 2
            self._theta_rot = float(theta_rot) if theta_rot is not None else 1.0
        elif geometry == 'nonlinear':
            self._n_rot_dof = 3
            self._theta_rot = tuple(theta_rot) if theta_rot is not None else (1., 1., 1.)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")

        # Vibrational characteristic temperatures
        if theta_vib is None or len(theta_vib) == 0:
            self._theta_vib = ()
        else:
            self._theta_vib = tuple(theta_vib)

    @classmethod
    def from_species(cls, name):
        """Construct from a predefined species name (e.g. 'H2', 'H2O')."""
        return cls(**SPECIES[name])

    # -----------------------------------------------------------------
    # Private helpers — per-particle contributions in units of kB
    # All accept scalar or array T (linear temperature in K).
    # -----------------------------------------------------------------

    def _s_rot_per_particle(self, T):
        """Rotational entropy / kB per particle (classical limit)."""
        if self._n_rot_dof == 0:
            return 0.0
        elif self._n_rot_dof == 2:  # linear
            return np.log(T / (self.sigma * self._theta_rot)) + 1.0
        else:  # nonlinear
            thA, thB, thC = self._theta_rot
            q_rot = (np.sqrt(np.pi) / self.sigma) * np.sqrt(
                T**3 / (thA * thB * thC))
            return np.log(q_rot) + 1.5

    def _s_vib_per_particle(self, T):
        """Vibrational entropy / kB per particle (Einstein model)."""
        if len(self._theta_vib) == 0:
            return 0.0
        s = 0.0
        for th in self._theta_vib:
            x = th / T
            # Clamp large x to avoid overflow
            safe = np.where(x < 500, x, 500.0)
            ex = np.exp(safe)
            contrib = safe / (ex - 1.0) - np.log1p(-1.0 / ex)
            s = s + np.where(x < 500, contrib, 0.0)
        return s

    def _u_rot_per_particle(self, T):
        """Rotational energy / kB per particle (in K)."""
        if self._n_rot_dof == 0:
            return 0.0
        return self._n_rot_dof / 2.0 * T

    def _u_vib_per_particle(self, T):
        """Vibrational energy / kB per particle (in K)."""
        if len(self._theta_vib) == 0:
            return 0.0
        u = 0.0
        for th in self._theta_vib:
            x = th / T
            safe = np.where(x < 500, x, 500.0)
            u = u + np.where(x < 500, th / (np.exp(safe) - 1.0), 0.0)
        return u

    def _cv_rot_per_gram(self):
        """Rotational c_v in erg/(g·K) — T-independent in classical limit."""
        return self._n_rot_dof / 2.0 * Rideal / self.m

    def _cv_vib_per_gram(self, T):
        """Vibrational c_v in erg/(g·K)."""
        if len(self._theta_vib) == 0:
            return 0.0
        cv = 0.0
        for th in self._theta_vib:
            x = th / T
            safe = np.where(x < 500, x, 500.0)
            ex = np.exp(safe)
            contrib = safe**2 * ex / (ex - 1.0)**2
            cv = cv + np.where(x < 500, contrib, 0.0)
        return cv * kB / (self.m * mp)

    # -----------------------------------------------------------------
    # Overridden public methods
    # -----------------------------------------------------------------

    def get_s_pt(self, logp, logt, _y):
        """S(P,T) with rot+vib. Returns erg/(g·K)."""
        s_trans = super().get_s_pt(logp, logt, _y)
        T = 10.0**logt
        s_rot = self._s_rot_per_particle(T) * kB / (self.m * mp)
        s_vib = self._s_vib_per_particle(T) * kB / (self.m * mp)
        return s_trans + s_rot + s_vib

    def get_s_rhot(self, logrho, logt, _y):
        """S(rho,T) with rot+vib. Returns erg/(g·K)."""
        s_trans = super().get_s_rhot(logrho, logt, _y)
        T = 10.0**logt
        s_rot = self._s_rot_per_particle(T) * kB / (self.m * mp)
        s_vib = self._s_vib_per_particle(T) * kB / (self.m * mp)
        return s_trans + s_rot + s_vib

    def get_s_rhop(self, logrho, logp, _y):
        """S(rho,P) with rot+vib. Returns erg/(g·K)."""
        # S_rot and S_vib depend on T, so compute T from ideal gas law
        logt = self.get_t_rhop(logrho, logp, _y)
        return self.get_s_pt(logp, logt, _y)

    def get_u_pt(self, logp, logt, _y):
        """Internal energy with rot+vib. Returns log10(U) in erg/g."""
        T = 10.0**logt
        u_trans = 1.5 * U_UNIT * T / self.m
        u_rot = self._u_rot_per_particle(T) * kB / (self.m * mp)
        u_vib = self._u_vib_per_particle(T) * kB / (self.m * mp)
        return np.log10(u_trans + u_rot + u_vib)

    def get_c_v(self, s, _logp, _y):
        """c_v with rot+vib — now T-dependent. Returns erg/(g·K)."""
        logt = self.get_t_sp(s, _logp, _y)
        T = 10.0**logt
        cv_trans = 1.5 * Rideal / self.m
        cv_rot = self._cv_rot_per_gram()
        cv_vib = self._cv_vib_per_gram(T)
        return cv_trans + cv_rot + cv_vib

    def get_c_p(self, s, _logp, _y):
        """c_p = c_v + R/m for ideal gas. Returns erg/(g·K)."""
        return self.get_c_v(s, _logp, _y) + Rideal / self.m

    def get_grad_ad(self, s, _logp, _y):
        """Adiabatic gradient: nabla_ad = 1 - c_v/c_p."""
        cv = self.get_c_v(s, _logp, _y)
        cp = cv + Rideal / self.m
        return 1.0 - cv / cp

    # -----------------------------------------------------------------
    # Newton-based inversions
    # -----------------------------------------------------------------

    def _newton_logt_sp(self, s_kbbar, logp, _y, tol=1e-12, maxiter=30):
        """Invert S(P,T) for logT via Newton iteration. s in kB/baryon."""
        logt = IdealEOS.get_t_sp(self, s_kbbar, logp, _y)
        s_target = s_kbbar / erg_to_kbbar  # to erg/(g·K)
        for _ in range(maxiter):
            s_curr = self.get_s_pt(logp, logt, _y)
            resid = s_curr - s_target
            rel = np.abs(resid) / (np.abs(s_target) + 1e-30)
            if np.all(rel < tol):
                break
            T = 10.0**logt
            # dS/dlogT at const P: translational + rotational + vibrational
            dsdt = (2.5 + self._n_rot_dof / 2.0) * kB / (self.m * mp) * np.log(10)
            dsdt = dsdt + self._cv_vib_per_gram(T) * np.log(10)
            logt = logt - resid / dsdt
        return logt

    def _newton_logt_srho(self, s_kbbar, logrho, _y, tol=1e-12, maxiter=30):
        """Invert S(rho,T) for logT via Newton iteration. s in kB/baryon."""
        logt = IdealEOS.get_t_srho(self, s_kbbar, logrho, _y)
        s_target = s_kbbar / erg_to_kbbar
        for _ in range(maxiter):
            s_curr = self.get_s_rhot(logrho, logt, _y)
            resid = s_curr - s_target
            rel = np.abs(resid) / (np.abs(s_target) + 1e-30)
            if np.all(rel < tol):
                break
            T = 10.0**logt
            # dS/dlogT at const rho: (3/2 + n_rot/2) + cv_vib
            dsdt = (1.5 + self._n_rot_dof / 2.0) * kB / (self.m * mp) * np.log(10)
            dsdt = dsdt + self._cv_vib_per_gram(T) * np.log(10)
            logt = logt - resid / dsdt
        return logt

    def get_t_sp(self, s, logp, _y):
        """Invert S(P,T) for T. s in kB/baryon. Returns log10(T)."""
        if self._n_rot_dof == 0 and len(self._theta_vib) == 0:
            return super().get_t_sp(s, logp, _y)
        return self._newton_logt_sp(s, logp, _y)

    def get_t_srho(self, s, logrho, _y):
        """Invert S(rho,T) for T. s in kB/baryon. Returns log10(T)."""
        if self._n_rot_dof == 0 and len(self._theta_vib) == 0:
            return super().get_t_srho(s, logrho, _y)
        return self._newton_logt_srho(s, logrho, _y)

    def get_p_srho(self, s, logrho, _y):
        """Invert S(rho,P) for P. s in kB/baryon. Returns log10(P)."""
        if self._n_rot_dof == 0 and len(self._theta_vib) == 0:
            return super().get_p_srho(s, logrho, _y)
        # Get T from S(rho,T) inversion, then P from ideal gas law
        logt = self._newton_logt_srho(s, logrho, _y)
        return self.get_p_rhot(logrho, logt, _y)

    def get_rho_sp(self, s, logp, _y):
        """Invert S(rho,P) for rho. s in kB/baryon. Returns log10(rho)."""
        if self._n_rot_dof == 0 and len(self._theta_vib) == 0:
            return super().get_rho_sp(s, logp, _y)
        # Get T from S(P,T) inversion, then rho from ideal gas law
        logt = self._newton_logt_sp(s, logp, _y)
        return self.get_rho_pt(logp, logt, _y)


def get_number_fracs(y, m_h, m_he):
    # vector-compatible expressions
    f_h = (1 - y) / m_h
    f_he = y / m_he
    f_tot = f_h + f_he
    return f_h / f_tot, f_he / f_tot
def get_smix(y, m_h, m_he):
    f_h, f_he = get_number_fracs(y, m_h, m_he)
    # entropy of mixing in units of kB / baryon
    smix = -(f_h * np.log(f_h) + f_he * np.log(f_he)) / (
        f_h * m_h + f_he * m_he)
    return smix

TBOUNDS = (-100, 100)
PBOUNDS = (-100, 100)
class IdealHHeMix(object):
    """
    ideal eos with proton mass m
    """
    def __init__(self, m_h=2.016, m_he=4.0026, eos_h=None, eos_he=None):
        super(IdealHHeMix, self).__init__()
        self.m_h = m_h
        self.m_he = m_he
        self.eos_h = eos_h if eos_h is not None else IdealEOS(m_h)
        self.eos_he = eos_he if eos_he is not None else IdealEOS(m_he)

    ## S getters — all return S in erg/(g·K)
    def get_s_pt(self, logp, logt, y):
        # Component entropies already return erg/(g·K)
        smix_kbbar = get_smix(y, self.m_h, self.m_he)
        return (
            (1 - y) * self.eos_h.get_s_pt(logp, logt, y)
            + y * self.eos_he.get_s_pt(logp, logt, y)
            + smix_kbbar / erg_to_kbbar)

    def get_s_rhot(self, logrho, logt, y):
        logp = self.get_p_rhot(logrho, logt, y)
        return self.get_s_pt(logp, logt, y)

    def get_s_rhop(self, logrho, logp, y):
        logt = self.get_t_rhop(logrho, logp, y)
        return self.get_s_pt(logp, logt, y)

    ## rho getters
    def get_rho_pt(self, logp, logt, y):
        return np.log10(1 / (
            (1 - y) / 10**self.eos_h.get_rho_pt(logp, logt, y)
            + y / 10**self.eos_he.get_rho_pt(logp, logt, y)))

    def get_rho_sp(self, s, logp, y):
        """s in kB/baryon."""
        if not np.isscalar(y):
            rets = [self.get_rho_sp(*el)
                    for el in zip(s, logp, y)]
            return np.array(rets)
        s_erg = s / erg_to_kbbar  # convert target to erg/(g·K)
        def obj(logt):
            return self.get_s_pt(logp, logt, y) / s_erg - 1
        opt_logt = brenth(obj, *TBOUNDS)
        return self.get_rho_pt(logp, opt_logt, y)

    ## P getters
    def get_p_rhot(self, logrho, logt, y):
        if not np.isscalar(y):
            rets = [self.get_p_rhot(*el)
                    for el in zip(logrho, logt, y)]
            return np.array(rets)
        def obj(logp):
            return self.get_rho_pt(logp, logt, y) / logrho - 1
        return brenth(obj, *PBOUNDS)

    def get_p_srho(self, s, logrho, y):
        return self.get_pt_srho(s, logrho, y)[:,0]

    ## T getters
    def get_t_rhop(self, logrho, logp, y):
        if not np.isscalar(y):
            rets = [self.get_t_rhop(*el)
                    for el in zip(logrho, logp, y)]
            return np.array(rets)
        # def obj(logt):
        #     return self.get_rho_pt(logp, logt, y) / logrho - 1
        # return brenth(obj, *TBOUNDS)

        def obj(logt):
            return self.get_rho_pt(logp, logt, y) / logrho - 1
        return root_scalar(obj, method='brenth', bracket=TBOUNDS).root

    def get_t_sp(self, s, logp, y):
        """s in kB/baryon."""
        if not np.isscalar(y):
            rets = [self.get_t_sp(*el)
                    for el in zip(s, logp, y)]
            return np.array(rets)
        s_erg = s / erg_to_kbbar
        def obj(logt):
            return self.get_s_pt(logp, logt, y) / s_erg - 1
        return root_scalar(obj, method='brenth', bracket=TBOUNDS).root

    def get_t_srho(self, s, logrho, y):
        if not np.isscalar(s):
            return self.get_pt_srho(s, logrho, y)[:,1]
        else:
            return self.get_pt_srho(s, logrho, y)[1]

    ## U getters
    def get_u_pt(self, logp, logt, y):
        # Linear (arithmetic) mixing of specific energies, not log mixing
        u_h = 10**self.eos_h.get_u_pt(logp, logt, y)
        u_he = 10**self.eos_he.get_u_pt(logp, logt, y)
        return np.log10((1 - y) * u_h + y * u_he)

    def get_u_srho(self, s, logrho, y):
        logp, logt = self.get_pt_srho(s, logrho, y)
        return self.get_u_pt(logp, logt, y)

    ## combined getters
    def get_sp_rhot(self, logrho, logt, _y):
        return self.get_s_rhot(logrho, logt), self.get_p_rhot(logrho, logt)

    def get_rhot_sp(self, s, logp, _y):
        return self.get_rho_sp(s, logp, _y), self.get_t_sp(s, logp, _y)

    def get_pt_srho(self, s, logrho, y):
        """2D inversion. s in kB/baryon."""
        if not np.isscalar(s):
            return np.array([self.get_pt_srho(*el)
                             for el in zip(s, logrho, y)])
        s_erg = s / erg_to_kbbar
        def opt(v):
            logp, logt = v
            return (
                (self.get_s_pt(logp, logt, y) / (s_erg + 1e-15) - 1)**2
                + (self.get_rho_pt(logp, logt, y) / (logrho + 1e-15) - 1)**2
            )
        sol = minimize(opt, [8, 3],
                       bounds=(PBOUNDS, TBOUNDS),
                       method='nelder-mead')
        return sol.x

    ## analytic derivatives
    def get_chirho_sp(self, s, _logp, _y):
        # idiomatic (?) way of returning the same type as s
        return 0 * s + 1

    def get_grad_ad(self, s, _logp, _y):
        # nabla_ad = (gamma-1)/gamma = 2/5
        return 0 * s + 2/5

    ## misc
    def get_c_p(self, s, _logp, y):
        # c_p per gram = (5/2) R / mu_eff, with mu_eff = 1/[(1-y)/m_h + y/m_he]
        return 0 * s + 5/2 * Rideal * ((1 - y) / self.m_h + y / self.m_he)
    def get_c_v(self, s, _logp, y):
        return 0 * s + 3/2 * Rideal * ((1 - y) / self.m_h + y / self.m_he)

class EOSFiniteDs(object):
    """
    wraps an eos and attaches finite difference functions

    any unknown function names call through to eos!
    """
    def __init__(self, eos):
        super(EOSFiniteDs, self).__init__()
        self.eos = eos
        self.d = 1e-3

    def __getattr__(self, attr):
        '''
        https://stackoverflow.com/questions/57091503/catch-all-method-in-class-that-passes-all-unknown-functions-to-instance-in-class
        '''
        return getattr(self.eos, attr)


    def get_chirho_sp(self, s, logp, y):
        logrho = self.eos.get_rho_sp(s, logp, y)
        logt = self.eos.get_t_sp(s, logp, y)
        return (
            self.eos.get_p_rhot(logrho * (1 + self.d), logt)
            - self.eos.get_p_rhot(logrho, logt)
        ) / (logrho * self.d)
    def get_grad_ad(self, s, logp, y):
        logt1 = self.eos.get_t_sp(s, logp, y)
        logt2 = self.eos.get_t_sp(s, logp * (1 + self.d), y)
        return (logt2 - logt1) / (logp * self.d)

    ## dQ/dY
    def get_dsdy_rhop_pt(self, logp, logt, y):
        logrho = self.eos.get_rho_pt(logp, logt, y)
        logt2 = self.eos.get_t_rhop(logrho, logp, y * (1 + self.d))
        s1 = self.eos.get_s_pt(logp, logt, y)
        s2 = self.eos.get_s_pt(logp, logt2, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_dsdy_rhop(self, logrho, logp, y):
        logt1 = self.eos.get_t_rhop(logrho, logp, y)
        logt2 = self.eos.get_t_rhop(logrho, logp, y * (1 + self.d))
        s1 = self.eos.get_s_pt(logp, logt1, y)
        s2 = self.eos.get_s_pt(logp, logt2, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_dsdy_pt(self, logp, logt, y):
        s1 = self.eos.get_s_pt(logp, logt, y)
        s2 = self.eos.get_s_pt(logp, logt, y * (1 + self.d))
        return (s2 - s1) / (y * self.d)
    def get_drhody_pt(self, logp, logt, y):
        rho1 = self.eos.get_rho_pt(logp, logt, y)
        rho2 = self.eos.get_rho_pt(logp, logt, y * (1 + self.d))
        return (rho2 - rho1) / (y * self.d)
    def get_dtdy_sp(self, s, logp, y):
        t1 = self.eos.get_t_sp(s, logp, y)
        t2 = self.eos.get_t_sp(s, logp, y * (1 + self.d))
        return (t2 - t1) / (y * self.d)
    def get_drhody_sp(self, s, logp, y):
        rho1 = self.eos.get_rho_sp(s, logp, y)
        rho2 = self.eos.get_rho_sp(s, logp, y * (1 + self.d))
        return (rho2 - rho1) / (y * self.d)

    def get_dudy_srho(self, s, logrho, y):
        logp1, logt1 = self.eos.get_pt_srho(s, logrho, y)
        logp2, logt2 = self.eos.get_pt_srho(s, logrho, y * (1 + self.d))
        u1 = self.eos.get_u_pt(logp1, logt1, y)
        u2 = self.eos.get_u_pt(logp2, logt2, y * (1 + self.d))
        return (u2 - u1) / (y * self.d)

    def get_duds_rhoy_srho(self, s, rho, y, ds=0.001):
        S1 = s/erg_to_kbbar
        S2 = S1*(1+ds)
        U0 = 10**self.get_u_srho(S1*erg_to_kbbar, rho, y)
        U1 = 10**self.get_u_srho(S2*erg_to_kbbar, rho, y)
        return (U1 - U0)/(S1*ds)

    def get_dudrho_sy_srho(self, s, rho, y, drho=0.1):
        R1 = 10**rho
        R2 = R1*(1+drho)
        #rho1 = np.log10((10**rho)*(1+drho))
        U0 = 10**self.get_u_srho(s, np.log10(R1), y)
        U1 = 10**self.get_u_srho(s, np.log10(R2), y)
        #return (U1 - U0)/(R1*drho)
        return (U1 - U0)/((1/R1) - (1/R2))

    def get_dtdy_srho(self, s, logrho, y):
        _, logt1 = self.eos.get_pt_srho(s, logrho, y)
        _, logt2 = self.eos.get_pt_srho(s, logrho, y * (1 + self.d))
        return (logt2 - logt1) / (y * self.d)

def test_ideal_eos(verbose=True, plot_dir=None):
    """
    Self-consistency and thermodynamic-consistency tests for the ideal EOS.

    Checks:
      1. Entropy self-consistency: S(P,T) == S(rho,T) == S(rho,P)
      2. Round-trip inversions: T(S,P) -> S(P,T), etc.
      3. First law at constant P:  dU/dT|_P = T dS/dT|_P + (P/rho^2) drho/dT|_P
      4. First law at constant T:  dU/dP|_T = T dS/dP|_T + (P/rho^2) drho/dP|_T
      5. Maxwell relation (from G): (P/rho^2) drho/dT|_P = T dS/dP|_T

    Returns True if all tests pass.
    """
    import os

    passed = True
    species = [('H2', 2.016), ('He', 4.0026), ('H2O', 18.0),
               ('Fe', 56.0), ('MgSiO3', 100.39)]

    # ---------- 1. Entropy self-consistency ----------
    if verbose:
        print('=== 1. Entropy self-consistency ===')
    for name, m in species:
        eos_i = IdealEOS(m)
        for lp, lt in [(11, 3), (12, 4), (10, 3.5)]:
            s_pt = eos_i.get_s_pt(lp, lt, 0)
            lr = eos_i.get_rho_pt(lp, lt, 0)
            s_rt = eos_i.get_s_rhot(lr, lt, 0)
            s_rp = eos_i.get_s_rhop(lr, lp, 0)
            err = max(abs(s_rt / s_pt - 1), abs(s_rp / s_pt - 1))
            if err > 1e-10:
                passed = False
                if verbose:
                    print(f'  FAIL {name} (logP={lp}, logT={lt}): rel err = {err:.2e}')
        if verbose:
            print(f'  {name}: OK')

    # ---------- 2. Round-trip inversions ----------
    if verbose:
        print('=== 2. Round-trip inversions ===')
    for name, m in species:
        eos_i = IdealEOS(m)
        for lp, lt in [(11, 3), (12, 4), (10, 3.5)]:
            s_erg = eos_i.get_s_pt(lp, lt, 0)
            s_kb = float(s_erg * erg_to_kbbar)
            lr = float(eos_i.get_rho_pt(lp, lt, 0))
            errs = [
                abs(float(eos_i.get_t_sp(s_kb, lp, 0)) - lt),
                abs(float(eos_i.get_t_srho(s_kb, lr, 0)) - lt),
                abs(float(eos_i.get_p_srho(s_kb, lr, 0)) - lp),
                abs(float(eos_i.get_rho_sp(s_kb, lp, 0)) - lr),
            ]
            if max(errs) > 1e-6:
                passed = False
                if verbose:
                    print(f'  FAIL {name} (logP={lp}, logT={lt}): max err = {max(errs):.2e}')
        if verbose:
            print(f'  {name}: OK')

    # ---------- 3-5. First law & Maxwell (finite-difference) ----------
    if verbose:
        print('=== 3-5. Thermodynamic consistency (first law & Maxwell) ===')

    h = 1e-5  # finite-difference step (fractional)
    for name, m in species:
        eos_i = IdealEOS(m)
        max_fl_p = 0.0
        max_fl_t = 0.0
        max_maxwell = 0.0
        for lp in [10.0, 11.0, 12.0, 13.0]:
            for lt in [3.0, 3.5, 4.0, 4.5]:
                P = 10**lp
                T = 10**lt

                # Centered finite differences
                s_p = eos_i.get_s_pt(lp, lt + h, 0)
                s_m = eos_i.get_s_pt(lp, lt - h, 0)
                dSdlogT_P = (s_p - s_m) / (2 * h)  # erg/(g K) per unit logT

                s_p2 = eos_i.get_s_pt(lp + h, lt, 0)
                s_m2 = eos_i.get_s_pt(lp - h, lt, 0)
                dSdlogP_T = (s_p2 - s_m2) / (2 * h)

                rho_p = 10**eos_i.get_rho_pt(lp, lt + h, 0)
                rho_m = 10**eos_i.get_rho_pt(lp, lt - h, 0)
                rho_0 = 10**eos_i.get_rho_pt(lp, lt, 0)
                drhodlogT_P = (rho_p - rho_m) / (2 * h)

                rho_p2 = 10**eos_i.get_rho_pt(lp + h, lt, 0)
                rho_m2 = 10**eos_i.get_rho_pt(lp - h, lt, 0)
                drhodlogP_T = (rho_p2 - rho_m2) / (2 * h)

                u_p = 10**eos_i.get_u_pt(lp, lt + h, 0)
                u_m = 10**eos_i.get_u_pt(lp, lt - h, 0)
                u_0 = 10**eos_i.get_u_pt(lp, lt, 0)
                dUdlogT_P = (u_p - u_m) / (2 * h)

                u_p2 = 10**eos_i.get_u_pt(lp + h, lt, 0)
                u_m2 = 10**eos_i.get_u_pt(lp - h, lt, 0)
                dUdlogP_T = (u_p2 - u_m2) / (2 * h)

                # First law at const P: dU/dlogT = T dS/dlogT + (P/rho^2) drho/dlogT
                lhs_p = dUdlogT_P
                rhs_p = T * dSdlogT_P + (P / rho_0**2) * drhodlogT_P
                if abs(lhs_p) > 1e-30:
                    fl_p = abs(lhs_p - rhs_p) / abs(lhs_p)
                    max_fl_p = max(max_fl_p, fl_p)

                # First law at const T: dU/dlogP = T dS/dlogP + (P/rho^2) drho/dlogP
                lhs_t = dUdlogP_T
                rhs_t = T * dSdlogP_T + (P / rho_0**2) * drhodlogP_T
                if abs(lhs_t) > 1e-30:
                    fl_t = abs(lhs_t - rhs_t) / abs(lhs_t)
                    max_fl_t = max(max_fl_t, fl_t)

                # Maxwell: (P/rho^2) drho/dlogT|_P = T dS/dlogP|_T
                lhs_mx = (P / rho_0**2) * drhodlogT_P
                rhs_mx = T * dSdlogP_T
                if abs(lhs_mx) > 1e-30:
                    mx = abs(lhs_mx - rhs_mx) / abs(lhs_mx)
                    max_maxwell = max(max_maxwell, mx)

        ok = max_fl_p < 1e-6 and max_fl_t < 1e-6 and max_maxwell < 1e-6
        if not ok:
            passed = False
        if verbose:
            status = 'OK' if ok else 'FAIL'
            print(f'  {name}: 1stLaw(P)={max_fl_p:.2e}  1stLaw(T)={max_fl_t:.2e}  Maxwell={max_maxwell:.2e}  [{status}]')

    # ---------- IdealHHeMix tests ----------
    if verbose:
        print('=== 6. IdealHHeMix thermodynamic consistency ===')
    hhe = IdealHHeMix()
    for Y in [0.0, 0.25, 0.50, 1.0]:
        max_fl_p = 0.0
        max_maxwell = 0.0
        for lp in [10.0, 11.0, 12.0]:
            for lt in [3.0, 3.5, 4.0]:
                P = 10**lp
                T = 10**lt

                s_p = hhe.get_s_pt(lp, lt + h, Y)
                s_m = hhe.get_s_pt(lp, lt - h, Y)
                dSdlogT_P = (s_p - s_m) / (2 * h)

                s_p2 = hhe.get_s_pt(lp + h, lt, Y)
                s_m2 = hhe.get_s_pt(lp - h, lt, Y)
                dSdlogP_T = (s_p2 - s_m2) / (2 * h)

                rho_0 = 10**hhe.get_rho_pt(lp, lt, Y)
                rho_p = 10**hhe.get_rho_pt(lp, lt + h, Y)
                rho_m = 10**hhe.get_rho_pt(lp, lt - h, Y)
                drhodlogT_P = (rho_p - rho_m) / (2 * h)

                u_0 = 10**hhe.get_u_pt(lp, lt, Y)
                u_p = 10**hhe.get_u_pt(lp, lt + h, Y)
                u_m = 10**hhe.get_u_pt(lp, lt - h, Y)
                dUdlogT_P = (u_p - u_m) / (2 * h)

                lhs_p = dUdlogT_P
                rhs_p = T * dSdlogT_P + (P / rho_0**2) * drhodlogT_P
                if abs(lhs_p) > 1e-30:
                    max_fl_p = max(max_fl_p, abs(lhs_p - rhs_p) / abs(lhs_p))

                lhs_mx = (P / rho_0**2) * drhodlogT_P
                rhs_mx = T * dSdlogP_T
                if abs(lhs_mx) > 1e-30:
                    max_maxwell = max(max_maxwell, abs(lhs_mx - rhs_mx) / abs(lhs_mx))

        ok = max_fl_p < 1e-4 and max_maxwell < 1e-4
        if not ok:
            passed = False
        if verbose:
            status = 'OK' if ok else 'FAIL'
            print(f'  Y={Y:.2f}: 1stLaw(P)={max_fl_p:.2e}  Maxwell={max_maxwell:.2e}  [{status}]')

    if verbose:
        print()
        print('ALL PASSED' if passed else 'SOME TESTS FAILED')
    return passed


def plot_ideal_eos(plot_dir):
    """
    Plot S(P,T) heatmap and T(S,P) isentropes for the ideal EOS.
    Saves to plot_dir.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    species = [('H$_2$', 2.016), ('He', 4.0026), ('H$_2$O', 18.0)]

    # --- Figure 1: S(P,T) heatmap for each species ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    logp_grid = np.linspace(8, 14, 100)
    logt_grid = np.linspace(2.5, 5.5, 100)
    LP, LT = np.meshgrid(logp_grid, logt_grid)

    for ax, (name, m) in zip(axes, species):
        eos_i = IdealEOS(m)
        S = eos_i.get_s_pt(LP, LT, 0) * erg_to_kbbar  # to kB/baryon for display
        im = ax.pcolormesh(logp_grid, logt_grid, S, shading='auto', cmap='viridis')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('S [kB/baryon]')
        ax.set_xlabel('log P [dyn/cm$^2$]')
        ax.set_ylabel('log T [K]')
        ax.set_title(f'Ideal {name} (m={m})')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_eos_s_pt.png'), dpi=150)
    plt.close(fig)

    # --- Figure 2: T(S,P) isentropes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    logp_line = np.linspace(8, 14, 200)

    for ax, (name, m) in zip(axes, species):
        eos_i = IdealEOS(m)
        s_values = np.linspace(0.5, 5.0, 10)  # kB/baryon
        for s_kb in s_values:
            logt_line = np.array([float(eos_i.get_t_sp(s_kb, lp, 0)) for lp in logp_line])
            ax.plot(logp_line, logt_line, label=f'S={s_kb:.1f}')
        ax.set_xlabel('log P [dyn/cm$^2$]')
        ax.set_ylabel('log T [K]')
        ax.set_title(f'Ideal {name} isentropes')
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_eos_isentropes.png'), dpi=150)
    plt.close(fig)

    # --- Figure 3: HHe mixture S(P,T) for different Y ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hhe = IdealHHeMix()
    Y_values = [0.0, 0.25, 0.50]

    for ax, Y in zip(axes, Y_values):
        S = hhe.get_s_pt(LP, LT, Y) * erg_to_kbbar
        im = ax.pcolormesh(logp_grid, logt_grid, S, shading='auto', cmap='viridis')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('S [kB/baryon]')
        ax.set_xlabel('log P [dyn/cm$^2$]')
        ax.set_ylabel('log T [K]')
        ax.set_title(f'Ideal H-He mix (Y={Y:.2f})')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_hhe_s_pt.png'), dpi=150)
    plt.close(fig)

    # --- Figure 4: HHe isentropes for Y=0.25 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    Y = 0.25
    s_values = np.linspace(3, 12, 8)  # kB/baryon
    for s_kb in s_values:
        logt_line = np.array([float(hhe.get_t_sp(s_kb, lp, Y)) for lp in logp_line])
        ax.plot(logp_line, logt_line, label=f'S={s_kb:.1f}')
    ax.set_xlabel('log P [dyn/cm$^2$]')
    ax.set_ylabel('log T [K]')
    ax.set_title(f'Ideal H-He isentropes (Y={Y})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_hhe_isentropes.png'), dpi=150)
    plt.close(fig)

    print(f'Plots saved to {plot_dir}/')


def test_molecular_ideal_eos(verbose=True):
    """
    Benchmark tests for MolecularIdealEOS.

    1. Monatomic equivalence
    2. Entropy self-consistency (PT == rhoT == rhoP)
    3. Round-trip inversions
    4. Thermodynamic consistency (first law + Maxwell)
    5. Classical equipartition limits (high T)
    6. Low-T freeze-out (no vibrational contribution)
    7. Known values (H₂ c_p at 300 K)
    8. ∇_ad behavior
    """
    passed = True

    # ---------- 1. Monatomic equivalence ----------
    if verbose:
        print('=== Molecular 1. Monatomic equivalence ===')
    mono = IdealEOS(m=55.845)
    mol_mono = MolecularIdealEOS(m=55.845, geometry='monatomic')
    for lp, lt in [(11, 3), (12, 4), (10, 3.5)]:
        s1 = mono.get_s_pt(lp, lt, 0)
        s2 = mol_mono.get_s_pt(lp, lt, 0)
        err = abs(s1 - s2) / (abs(s1) + 1e-30)
        if err > 1e-14:
            passed = False
            if verbose:
                print(f'  FAIL get_s_pt: {err:.2e}')

        u1 = mono.get_u_pt(lp, lt, 0)
        u2 = mol_mono.get_u_pt(lp, lt, 0)
        err = abs(u1 - u2) / (abs(u1) + 1e-30)
        if err > 1e-14:
            passed = False
            if verbose:
                print(f'  FAIL get_u_pt: {err:.2e}')

        s_kb = float(s1 * erg_to_kbbar)
        t1 = float(mono.get_t_sp(s_kb, lp, 0))
        t2 = float(mol_mono.get_t_sp(s_kb, lp, 0))
        err = abs(t1 - t2)
        if err > 1e-10:
            passed = False
            if verbose:
                print(f'  FAIL get_t_sp: {err:.2e}')
    if verbose:
        print('  OK')

    # ---------- 2. Entropy self-consistency ----------
    if verbose:
        print('=== Molecular 2. Entropy self-consistency ===')
    for name in ['H2', 'H2O']:
        eos_m = MolecularIdealEOS.from_species(name)
        for lp, lt in [(11, 3), (12, 4), (10, 3.5), (9, 4.5)]:
            s_pt = eos_m.get_s_pt(lp, lt, 0)
            lr = eos_m.get_rho_pt(lp, lt, 0)
            s_rt = eos_m.get_s_rhot(lr, lt, 0)
            s_rp = eos_m.get_s_rhop(lr, lp, 0)
            err = max(abs(s_rt / s_pt - 1), abs(s_rp / s_pt - 1))
            if err > 1e-10:
                passed = False
                if verbose:
                    print(f'  FAIL {name} (logP={lp}, logT={lt}): {err:.2e}')
        if verbose:
            print(f'  {name}: OK')

    # ---------- 3. Round-trip inversions ----------
    if verbose:
        print('=== Molecular 3. Round-trip inversions ===')
    for name in ['H2', 'H2O']:
        eos_m = MolecularIdealEOS.from_species(name)
        for lp, lt in [(11, 3), (12, 4), (10, 3.5), (9, 4.5)]:
            s_erg = eos_m.get_s_pt(lp, lt, 0)
            s_kb = float(s_erg * erg_to_kbbar)
            lr = float(eos_m.get_rho_pt(lp, lt, 0))
            errs = [
                abs(float(eos_m.get_t_sp(s_kb, lp, 0)) - lt),
                abs(float(eos_m.get_t_srho(s_kb, lr, 0)) - lt),
                abs(float(eos_m.get_p_srho(s_kb, lr, 0)) - lp),
                abs(float(eos_m.get_rho_sp(s_kb, lp, 0)) - lr),
            ]
            if max(errs) > 1e-6:
                passed = False
                if verbose:
                    print(f'  FAIL {name} (logP={lp}, logT={lt}): max err = {max(errs):.2e}')
        if verbose:
            print(f'  {name}: OK')

    # ---------- 4. Thermodynamic consistency ----------
    if verbose:
        print('=== Molecular 4. Thermodynamic consistency (1st law + Maxwell) ===')
    h = 1e-5
    for name in ['H2', 'H2O']:
        eos_m = MolecularIdealEOS.from_species(name)
        max_fl_p = 0.0
        max_maxwell = 0.0
        for lp in [10.0, 11.0, 12.0, 13.0]:
            for lt in [3.0, 3.5, 4.0, 4.5]:
                P = 10**lp
                T = 10**lt
                s_p = eos_m.get_s_pt(lp, lt + h, 0)
                s_m = eos_m.get_s_pt(lp, lt - h, 0)
                dSdlogT_P = (s_p - s_m) / (2 * h)

                s_p2 = eos_m.get_s_pt(lp + h, lt, 0)
                s_m2 = eos_m.get_s_pt(lp - h, lt, 0)
                dSdlogP_T = (s_p2 - s_m2) / (2 * h)

                rho_0 = 10**eos_m.get_rho_pt(lp, lt, 0)
                rho_p = 10**eos_m.get_rho_pt(lp, lt + h, 0)
                rho_m = 10**eos_m.get_rho_pt(lp, lt - h, 0)
                drhodlogT_P = (rho_p - rho_m) / (2 * h)

                u_p = 10**eos_m.get_u_pt(lp, lt + h, 0)
                u_m = 10**eos_m.get_u_pt(lp, lt - h, 0)
                dUdlogT_P = (u_p - u_m) / (2 * h)

                # First law at const P: dU/dlogT = T dS/dlogT + (P/rho^2) drho/dlogT
                lhs_p = dUdlogT_P
                rhs_p = T * dSdlogT_P + (P / rho_0**2) * drhodlogT_P
                if abs(lhs_p) > 1e-30:
                    max_fl_p = max(max_fl_p, abs(lhs_p - rhs_p) / abs(lhs_p))

                # Maxwell: (P/rho^2) drho/dlogT|_P = T dS/dlogP|_T
                lhs_mx = (P / rho_0**2) * drhodlogT_P
                rhs_mx = T * dSdlogP_T
                if abs(lhs_mx) > 1e-30:
                    max_maxwell = max(max_maxwell, abs(lhs_mx - rhs_mx) / abs(lhs_mx))

        ok = max_fl_p < 1e-6 and max_maxwell < 1e-6
        if not ok:
            passed = False
        if verbose:
            status = 'OK' if ok else 'FAIL'
            print(f'  {name}: 1stLaw(P)={max_fl_p:.2e}  Maxwell={max_maxwell:.2e}  [{status}]')

    # ---------- 5. Classical equipartition limits ----------
    if verbose:
        print('=== Molecular 5. Classical equipartition limits (high T) ===')

    # At T >> theta_vib, all modes active
    lt_high = 5.5  # T = 316,000 K >> 6332 K
    lp_ref = 12.0

    h2 = MolecularIdealEOS.from_species('H2')
    h2o = MolecularIdealEOS.from_species('H2O')

    # Get s in kB/baryon for c_v evaluation
    s_h2 = float(h2.get_s_pt(lp_ref, lt_high, 0) * erg_to_kbbar)
    s_h2o = float(h2o.get_s_pt(lp_ref, lt_high, 0) * erg_to_kbbar)

    cv_h2 = float(h2.get_c_v(s_h2, lp_ref, 0))
    cv_h2o = float(h2o.get_c_v(s_h2o, lp_ref, 0))

    # Expected: H2 → 7/2 R/m, H2O → 6 R/m
    cv_h2_expected = 3.5 * Rideal / 2.016
    cv_h2o_expected = 6.0 * Rideal / 18.015

    err_h2 = abs(cv_h2 / cv_h2_expected - 1)
    err_h2o = abs(cv_h2o / cv_h2o_expected - 1)

    ok = err_h2 < 0.01 and err_h2o < 0.01
    if not ok:
        passed = False
    if verbose:
        print(f'  H2:  cv = {cv_h2:.4e}, expected 7/2 R/m = {cv_h2_expected:.4e}, err = {err_h2:.2e}  [{"OK" if err_h2 < 0.01 else "FAIL"}]')
        print(f'  H2O: cv = {cv_h2o:.4e}, expected   6 R/m = {cv_h2o_expected:.4e}, err = {err_h2o:.2e}  [{"OK" if err_h2o < 0.01 else "FAIL"}]')

    # ---------- 6. Low-T freeze-out ----------
    if verbose:
        print('=== Molecular 6. Low-T vibrational freeze-out ===')

    lt_low = 2.0  # T = 100 K << 2373 K (lowest vib mode of H2O)
    s_h2_low = float(h2.get_s_pt(lp_ref, lt_low, 0) * erg_to_kbbar)
    s_h2o_low = float(h2o.get_s_pt(lp_ref, lt_low, 0) * erg_to_kbbar)

    cv_h2_low = float(h2.get_c_v(s_h2_low, lp_ref, 0))
    cv_h2o_low = float(h2o.get_c_v(s_h2o_low, lp_ref, 0))

    # Expected: H2 → 5/2 R/m (trans + rot), H2O → 3 R/m (trans + rot)
    cv_h2_low_exp = 2.5 * Rideal / 2.016
    cv_h2o_low_exp = 3.0 * Rideal / 18.015

    err_h2_low = abs(cv_h2_low / cv_h2_low_exp - 1)
    err_h2o_low = abs(cv_h2o_low / cv_h2o_low_exp - 1)

    ok = err_h2_low < 0.01 and err_h2o_low < 0.01
    if not ok:
        passed = False
    if verbose:
        print(f'  H2:  cv = {cv_h2_low:.4e}, expected 5/2 R/m = {cv_h2_low_exp:.4e}, err = {err_h2_low:.2e}  [{"OK" if err_h2_low < 0.01 else "FAIL"}]')
        print(f'  H2O: cv = {cv_h2o_low:.4e}, expected   3 R/m = {cv_h2o_low_exp:.4e}, err = {err_h2o_low:.2e}  [{"OK" if err_h2o_low < 0.01 else "FAIL"}]')

    # ---------- 7. Known values ----------
    if verbose:
        print('=== Molecular 7. Known values (H₂ c_p at 300K) ===')
    # At 300K: rot active, vib frozen. c_p = 7/2 R/m
    T_300 = 300.0
    lt_300 = np.log10(T_300)
    lp_1atm = np.log10(1.01325e6)  # 1 atm in dyn/cm²

    s_300 = float(h2.get_s_pt(lp_1atm, lt_300, 0) * erg_to_kbbar)
    cp_h2_300 = float(h2.get_c_p(s_300, lp_1atm, 0))
    # Convert to J/(mol·K) for comparison: erg/(g·K) × m_mol × 1e-7
    cp_jmol = cp_h2_300 * 2.016 * 1e-7
    # NIST value: 28.85 J/(mol·K)
    err_cp = abs(cp_jmol / 28.85 - 1)

    ok = err_cp < 0.02
    if not ok:
        passed = False
    if verbose:
        print(f'  H2 c_p at 300K: {cp_jmol:.2f} J/(mol·K), NIST ≈ 28.85, err = {err_cp:.2e}  [{"OK" if ok else "FAIL"}]')

    # ---------- 8. ∇_ad behavior ----------
    if verbose:
        print('=== Molecular 8. Adiabatic gradient ===')

    # Low T: monatomic limit 2/5
    grad_low = float(h2.get_grad_ad(s_h2_low, lp_ref, 0))
    # At low T, H2 has trans(3/2) + rot(1) active → cv = 5/2 R/m → ∇_ad = R/(cv+R) = 1/3.5 = 2/7
    # Wait: trans + rot for diatomic = 5/2 → cv/R = 5/2 → ∇_ad = 1/(1 + 5/2) = 2/7
    expected_low = 2.0 / 7.0
    err_low = abs(grad_low / expected_low - 1)

    # High T (all vib active): cv = 7/2 R → ∇_ad = 2/9
    grad_high = float(h2.get_grad_ad(s_h2, lp_ref, 0))
    expected_high = 2.0 / 9.0
    err_high = abs(grad_high / expected_high - 1)

    ok = err_low < 0.01 and err_high < 0.01
    if not ok:
        passed = False
    if verbose:
        print(f'  H2 low-T ∇_ad = {grad_low:.4f}, expected 2/7 = {expected_low:.4f}, err = {err_low:.2e}  [{"OK" if err_low < 0.01 else "FAIL"}]')
        print(f'  H2 high-T ∇_ad = {grad_high:.4f}, expected 2/9 = {expected_high:.4f}, err = {err_high:.2e}  [{"OK" if err_high < 0.01 else "FAIL"}]')

    if verbose:
        print()
        print('ALL PASSED' if passed else 'SOME TESTS FAILED')
    return passed


def plot_molecular_comparison(plot_dir):
    """
    Comparison plots: monatomic vs molecular ideal EOS.
    Saves to plot_dir.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    logp_grid = np.linspace(8, 14, 100)
    logt_grid = np.linspace(2.0, 5.5, 100)
    LP, LT = np.meshgrid(logp_grid, logt_grid)

    # --- Plot A: S(P,T) heatmap comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, name in enumerate(['H2', 'H2O']):
        sp = SPECIES[name]
        eos_mono = IdealEOS(sp['m'])
        eos_mol = MolecularIdealEOS.from_species(name)

        S_mono = eos_mono.get_s_pt(LP, LT, 0) * erg_to_kbbar
        S_mol = eos_mol.get_s_pt(LP, LT, 0) * erg_to_kbbar
        S_diff = S_mol - S_mono

        im0 = axes[0, col].pcolormesh(logp_grid, logt_grid, S_mono,
                                        shading='auto', cmap='viridis')
        fig.colorbar(im0, ax=axes[0, col])
        axes[0, col].set_title(f'{name} monatomic S [kB/bar]')
        axes[0, col].set_xlabel('log P')
        axes[0, col].set_ylabel('log T')

        im1 = axes[1, col].pcolormesh(logp_grid, logt_grid, S_mol,
                                        shading='auto', cmap='viridis')
        fig.colorbar(im1, ax=axes[1, col])
        axes[1, col].set_title(f'{name} molecular S [kB/bar]')
        axes[1, col].set_xlabel('log P')
        axes[1, col].set_ylabel('log T')

    # Difference panels in column 2
    for row, name in enumerate(['H2', 'H2O']):
        sp = SPECIES[name]
        eos_mono = IdealEOS(sp['m'])
        eos_mol = MolecularIdealEOS.from_species(name)
        S_diff = (eos_mol.get_s_pt(LP, LT, 0) - eos_mono.get_s_pt(LP, LT, 0)) * erg_to_kbbar
        im = axes[row, 2].pcolormesh(logp_grid, logt_grid, S_diff,
                                      shading='auto', cmap='RdBu_r')
        fig.colorbar(im, ax=axes[row, 2])
        axes[row, 2].set_title(f'{name} ΔS (mol − mono) [kB/bar]')
        axes[row, 2].set_xlabel('log P')
        axes[row, 2].set_ylabel('log T')

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_molecular_vs_monatomic_s_pt.png'), dpi=150)
    plt.close(fig)

    # --- Plot B: Isentrope comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    logp_line = np.linspace(8, 14, 200)

    for ax, name in zip(axes, ['H2', 'H2O']):
        sp = SPECIES[name]
        eos_mono = IdealEOS(sp['m'])
        eos_mol = MolecularIdealEOS.from_species(name)

        s_values = np.linspace(2, 8, 7) if name == 'H2' else np.linspace(0.5, 4, 7)
        for s_kb in s_values:
            lt_mono = np.array([float(eos_mono.get_t_sp(s_kb, lp, 0)) for lp in logp_line])
            lt_mol = np.array([float(eos_mol.get_t_sp(s_kb, lp, 0)) for lp in logp_line])
            ax.plot(logp_line, lt_mono, '--', color='gray', alpha=0.6)
            ax.plot(logp_line, lt_mol, '-', label=f'S={s_kb:.1f}')

        ax.set_xlabel('log P [dyn/cm²]')
        ax.set_ylabel('log T [K]')
        ax.set_title(f'{name}: molecular (solid) vs monatomic (dashed)')
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_molecular_vs_monatomic_isentropes.png'), dpi=150)
    plt.close(fig)

    # --- Plot C: c_v(T) and ∇_ad(T) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    T_arr = np.logspace(2, 5.5, 500)
    logT_arr = np.log10(T_arr)
    lp_fixed = 12.0

    for col, name in enumerate(['H2', 'H2O']):
        eos_mol = MolecularIdealEOS.from_species(name)
        m = eos_mol.m

        cv_arr = np.zeros_like(T_arr)
        grad_arr = np.zeros_like(T_arr)
        for i, lt in enumerate(logT_arr):
            s_kb = float(eos_mol.get_s_pt(lp_fixed, lt, 0) * erg_to_kbbar)
            cv_arr[i] = float(eos_mol.get_c_v(s_kb, lp_fixed, 0))
            grad_arr[i] = float(eos_mol.get_grad_ad(s_kb, lp_fixed, 0))

        # Normalize c_v by R/m
        cv_norm = cv_arr / (Rideal / m)

        # c_v panel
        ax = axes[0, col]
        ax.semilogx(T_arr, cv_norm, 'b-', lw=2)
        if name == 'H2':
            ax.axhline(2.5, ls='--', color='gray', label='trans+rot (5/2)')
            ax.axhline(3.5, ls='--', color='red', label='all active (7/2)')
        else:
            ax.axhline(3.0, ls='--', color='gray', label='trans+rot (3)')
            ax.axhline(6.0, ls='--', color='red', label='all active (6)')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('c_v / (R/m)')
        ax.set_title(f'{name} heat capacity')
        ax.legend()

        # ∇_ad panel
        ax = axes[1, col]
        ax.semilogx(T_arr, grad_arr, 'b-', lw=2)
        ax.axhline(0.4, ls='--', color='gray', label='monatomic (2/5)')
        if name == 'H2':
            ax.axhline(2/7, ls='--', color='orange', label='trans+rot (2/7)')
            ax.axhline(2/9, ls='--', color='red', label='all active (2/9)')
        else:
            ax.axhline(1/3, ls='--', color='orange', label='trans+rot (1/3)')
            ax.axhline(2/13, ls='--', color='red', label='all active (2/13)')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('∇_ad')
        ax.set_title(f'{name} adiabatic gradient')
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'ideal_molecular_cv_grad.png'), dpi=150)
    plt.close(fig)

    print(f'Molecular comparison plots saved to {plot_dir}/')


if __name__ == '__main__':
    import os
    PLOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'plots')
    test_ideal_eos(verbose=True)
    test_molecular_ideal_eos(verbose=True)
    plot_ideal_eos(PLOT_DIR)
    plot_molecular_comparison(PLOT_DIR)
