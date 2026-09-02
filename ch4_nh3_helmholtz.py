"""
Thermodynamically consistent Helmholtz free-energy EOS for CH4 and NH3.

A single free energy F(rho,T) is carried per species and P, U, S, c_v are obtained
by differentiating it, so the Maxwell relations and path independence hold by
construction rather than by tuning.

    F(rho,T) = R_s * T * [ alpha0(delta,tau) + alpha_exc(delta,tau) ]     [erg/g]

with delta = rho/rho_red, tau = T_red/T.

`alpha0` is the ANALYTIC ideal-gas term taken from the reference equations of state
(Setzmann & Wagner 1991 for CH4, Gao et al. 2023 for NH3).  It carries the ln(delta)
singularity that no polynomial can represent, is exact as rho -> 0, and carries the
physical reference state.

`alpha_exc` is a single fitted surface

    alpha_exc = delta * A(y) + delta**2 * Psi(x, y)

in scaled coordinates x = a_r(ln rho - ln rho_lo) - 1, y = a_t(ln T - ln T_lo) - 1.
The delta**1 block makes the ideal-gas limit exact by construction (A *is* the reduced
second virial coefficient); the delta**2 prefactor absorbs the high-density growth so
Psi stays nearly flat.

A and Psi are expanded in tensor-product B-splines, NOT in a global polynomial basis
-- see the comment above `_bspline_knots` for the measurement that forced that choice.

There is no blend seam anywhere -- one analytic object over the whole domain.

Why not a weighted blend of the two source free energies:  for F = w F_ref + (1-w) F_DFT
the pressure picks up rho (dw/dln rho) (F_ref - F_DFT).  Because
d(F_ref - F_DFT)/dln rho = (P_ref - P_DFT)/rho, a ramp of width W accumulates
|Delta F| ~ eta P W / rho while max|dw/dln rho| ~ 2/W, so the spurious term is ~eta P
INDEPENDENT of the ramp width.  Measured eta = 1 - P_ref/P_DFT is 0.09-0.44 (CH4) and
0.04-0.35 (NH3), and Setzmann's pressure turns negative at rho = 1.4 g/cm^3, so no
choice of ramp makes a blend viable.

Units are CGS throughout: rho [g/cm^3], T [K], P [dyn/cm^2], F and U [erg/g],
S and c_v [erg/(g K)].

Authors: Roberto Tejada Arevalo
"""
import numpy as np
from scipy.interpolate import BSpline as _BSpline

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8.314462618e7          # erg/(mol K)

# Standard molar entropies of the IDEAL GAS at 298.15 K, 100 kPa (JANAF / CODATA).
# These set the third-law entropy gauge; see `IdealTerm.regauge_entropy`.
S_STANDARD = {'methane': 186.25, 'ammonia': 192.77}     # J/(mol K)

# Domain-safety entropy headroom, in units of R_s, added on top of the third-law
# gauge.  THIS IS A GAUGE CHOICE, NOT PHYSICS, and it is recorded rather than hidden.
#
# WHY IT IS NEEDED.  ds/dT|_rho = c_v/T > 0, so entropy falls as temperature falls.
# Anchoring s(298.15 K, 1 bar) to the JANAF standard molar entropy leaves NO headroom
# at the cold dense corner: the DFT-accurate v1 fit was pinned at exactly the s floor
# (+0.020 R_s) at rho = 30, T = 150 K within its own box, and reaches -3.28 R_s at
# 50 K.  Measured requirement, identical for both species: 3.78 R_s.  Without this,
# "s > 0 at 50 K" and "reproduce the DFT-MD pressures" are JOINTLY INFEASIBLE, and
# the solver resolves the conflict by throwing away dense-region pressure -- measured
# as the DFT error going from 3.6% to 56%.
#
# WHY IT IS LEGITIMATE.  a1 is pure gauge: it enters S as -R_s*a1 and cancels exactly
# out of U, P and c_v (see IdealTerm's class docstring).  Only entropy DIFFERENCES are
# physical.  The "S > 0" requirement is numerical -- eos_pt_calc.py:379 does an
# unguarded np.log10(u_mix) and the CH4/NH3 consumer at eos_class.py:416 does
# np.where(s > 0, np.log10(s), nan) -- not thermodynamic.  This is the same thing the
# legacy code did with its hand-tuned +1.03e8 (nh3.py:181) and the notebook
# prototype's S_OFFSET = 2.0e8, but sized from a measurement and reported.
#
# SUPERSEDED.  A uniform gauge shift does NOT relieve the conflict: measured, the fit
# simply spends the extra headroom on more compression and re-pins at the floor
# (min s stayed at exactly 0.020 R_s with +4 R_s applied).  The working fix is to take
# `s` out of the fitted constraint set entirely and choose the gauge AFTER the fit --
# see `solve_s_offset`.  This constant is kept at 0.0 rather than deleted so the
# reasoning survives.
#
# CONSEQUENCE TO CARRY FORWARD: s is offset from the JANAF third-law value by
# `eos.s_offset`.  Any mixture that adds entropies across species
# (eos/ice_eos.py:258) inherits the mass-weighted sum of the per-species offsets.
S_HEADROOM_RS = 0.0   # superseded by solve_s_offset(); kept at 0, see note below


# ---------------------------------------------------------------------------
# Ideal-gas term
# ---------------------------------------------------------------------------
class IdealTerm:
    """
    Dimensionless ideal-gas Helmholtz energy in the Span-Wagner form

        alpha0 = ln(delta) + a1 + a2*tau + c*ln(tau)
                 + sum_k v_k * ln(1 - exp(-theta_k * tau))

    Both reference equations share this form, so one implementation covers CH4
    (Setzmann & Wagner Eq. 5.2) and NH3 (Gao et al.).

    The two constants are pure gauge and decouple exactly:

      * `a2` enters U as the constant R_s*T_red*a2 and cancels out of S, P and c_v.
      * `a1` enters S as -R_s*a1 and cancels out of U, P and c_v.

    so both can be reset without touching any physics.
    """

    def __init__(self, a1, a2, c, v, theta, R_s, T_red, rho_red, name=''):
        self.a1, self.a2, self.c = float(a1), float(a2), float(c)
        self.v = np.asarray(v, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.R_s, self.T_red, self.rho_red = R_s, T_red, rho_red
        self.name = name

    # -- the Planck-Einstein sum and its tau-derivatives ---------------------
    def _planck(self, tau):
        x = np.outer(np.atleast_1d(tau), self.theta)          # (n, k)
        return np.log1p(-np.exp(-x)) @ self.v

    def _planck_tau(self, tau):
        x = np.outer(np.atleast_1d(tau), self.theta)
        return np.expm1(x) ** -1.0 @ (self.v * self.theta)

    def _planck_tautau(self, tau):
        x = np.outer(np.atleast_1d(tau), self.theta)
        ex = np.exp(x)
        return -(ex / np.expm1(x) ** 2) @ (self.v * self.theta ** 2)

    # -- alpha0 and its tau-derivatives -------------------------------------
    def alpha(self, delta, tau):
        delta, tau = np.atleast_1d(delta), np.atleast_1d(tau)
        return (np.log(delta) + self.a1 + self.a2 * tau
                + self.c * np.log(tau) + self._planck(tau))

    def alpha_tau(self, tau):
        tau = np.atleast_1d(tau)
        return self.a2 + self.c / tau + self._planck_tau(tau)

    def alpha_tautau(self, tau):
        tau = np.atleast_1d(tau)
        return -self.c / tau ** 2 + self._planck_tautau(tau)

    # -- the D = d/dln(rho) and E = d/dln(T) operators -----------------------
    # Only ln(delta) depends on delta, so D alpha0 = 1 exactly and D^2 = D E = 0.
    def D(self, delta, tau):
        return np.ones_like(np.atleast_1d(delta), dtype=float)

    def DD(self, delta, tau):
        return np.zeros_like(np.atleast_1d(delta), dtype=float)

    def DE(self, delta, tau):
        return np.zeros_like(np.atleast_1d(delta), dtype=float)

    def E(self, delta, tau):
        """E alpha0 = -tau * d(alpha0)/d(tau)."""
        tau = np.atleast_1d(tau)
        return -tau * self.alpha_tau(tau)

    def EE(self, delta, tau):
        """E^2 alpha0 = tau*alpha_tau + tau^2*alpha_tautau."""
        tau = np.atleast_1d(tau)
        return tau * self.alpha_tau(tau) + tau ** 2 * self.alpha_tautau(tau)

    # -- gauge fixing --------------------------------------------------------
    def regauge_energy(self):
        """
        Set a2 = 0 so u_ideal(T -> 0) = 0, i.e. energies are measured from the T=0
        molecular ground state.  Every Planck-Einstein term is positive and c > 0,
        so this makes U_ideal > 0 for all T > 0 -- the U > 0 requirement becomes
        structural rather than an offset that has to be tuned.
        """
        shift = self.R_s * self.T_red * self.a2      # erg/g removed from U
        self.a2 = 0.0
        return shift

    def regauge_entropy(self, S_std_J_per_mol_K, M_g_per_mol,
                        T0=298.15, P0=1.0e6):
        """
        Set a1 so that s_ideal(T0, P0) equals the third-law standard molar entropy.
        P0 defaults to 100 kPa = 1e6 dyn/cm^2, which is exactly the state S° is
        tabulated at, so this is an identity rather than an approximation.

        Returns the applied shift in erg/(g K).
        """
        rho0 = P0 / (self.R_s * T0)                  # ideal-gas density at (T0, P0)
        s_now = self.s(rho0, T0)[0]
        s_target = S_std_J_per_mol_K * 1.0e7 / M_g_per_mol
        shift = s_target - s_now
        self.a1 -= shift / self.R_s                  # s contains -R_s*a1
        return shift

    # -- ideal-gas thermodynamics (used for the constraint floors) -----------
    def _reduced(self, rho, T):
        return (np.atleast_1d(rho) / self.rho_red,
                self.T_red / np.atleast_1d(T))

    def s(self, rho, T):
        d, tau = self._reduced(rho, T)
        return self.R_s * (tau * self.alpha_tau(tau) - self.alpha(d, tau))

    def u(self, rho, T):
        d, tau = self._reduced(rho, T)
        return self.R_s * np.atleast_1d(T) * tau * self.alpha_tau(tau)

    def cv(self, rho, T):
        d, tau = self._reduced(rho, T)
        return -self.R_s * tau ** 2 * self.alpha_tautau(tau)

    def p(self, rho, T):
        rho = np.atleast_1d(rho)
        return rho * self.R_s * np.atleast_1d(T)     # D alpha0 = 1  =>  Z = 1


# ---------------------------------------------------------------------------
# Species definitions
# ---------------------------------------------------------------------------
def _make_ideal(species):
    """Build the IdealTerm for a species, with both gauges applied."""
    if species == 'methane':
        # Setzmann & Wagner (1991) Eq. 5.2
        M = 16.043
        R_s = 518.2705 * 1.0e4                       # erg/(g K)
        a = np.array([9.91243972, -6.33270087, 3.0016,
                      0.008449, 4.6942, 3.4865, 1.6572, 1.4115])
        theta = np.array([3.40043240, 10.26951575, 20.43932747,
                          29.93744884, 79.13351945])
        it = IdealTerm(a1=a[0], a2=a[1], c=a[2], v=a[3:], theta=theta,
                       R_s=R_s, T_red=190.564, rho_red=0.16266, name='methane')
    elif species == 'ammonia':
        # Gao et al. (2023)
        M = 17.03052
        R_s = R_UNIVERSAL / M
        Tc = 405.56
        u_k = np.array([1646.0, 3965.0, 7231.0])     # K
        it = IdealTerm(a1=-6.59406093943886, a2=5.60101151987913, c=4.0 - 1.0,
                       v=np.array([2.224, 3.148, 0.9579]), theta=u_k / Tc,
                       R_s=R_s, T_red=Tc, rho_red=13.696 * M / 1000.0,
                       name='ammonia')
    else:
        raise ValueError(f'unknown species: {species!r}')

    it.M = M
    it.gauge_energy_shift = it.regauge_energy()
    it.gauge_entropy_shift = it.regauge_entropy(S_STANDARD[species], M)
    # domain-safety headroom on top of the third-law gauge (see S_HEADROOM_RS)
    it.a1 -= S_HEADROOM_RS                     # s contains -R_s*a1, so this ADDS to s
    it.gauge_headroom_shift = S_HEADROOM_RS * it.R_s
    return it


IDEAL = {s: _make_ideal(s) for s in ('methane', 'ammonia')}


# ---------------------------------------------------------------------------
# B-spline basis for the excess term
# ---------------------------------------------------------------------------
# WHY B-SPLINES AND NOT A GLOBAL POLYNOMIAL.  A tensor-product Chebyshev basis was
# tried first and fails STRUCTURALLY here, for a reason worth recording so it is
# not retried.  The two data sources occupy two disjoint patches of the fit box --
# reference EOS at rho <= 1.2 g/cm^3 and T <= 625 K, DFT-MD at rho >= 0.5 and
# T >= 1000 K -- separated and surrounded by large data-free regions (the
# cold-dense corner, the hot-dilute wedge, and everything above the DFT ceiling).
#
# Every global basis function is nonzero over the entire box, so the coefficients
# that fit the data are the SAME coefficients that set the behaviour where there is
# none.  Measured consequences: the unconstrained Chebyshev fit reproduced the DFT
# pressure to 0.59% but reached Z = -2.3e6 and violated the stability constraints on
# 6-16% of the domain; imposing those constraints then drove the DFT pressure error
# to 59-82%, because the constraints and the data were competing for one shared set
# of 99 coefficients.  Neither regularisation nor a better solver can resolve that --
# it is a property of the basis.
#
# A B-spline of order k is nonzero over only k+1 knot intervals.  A coefficient
# sitting in the data-free cold-dense corner can therefore be pulled to the ideal
# gas without perturbing the coefficients that fit the DFT.  Everything downstream
# (`_blocks`, `_exc_terms`, `_design_G`, `design`) consumes the basis only through
# the (V, D1, D2) triple, so the swap is local to these two functions.
#
# Order k=4 (quartic) is the default: F is then C^3, so P, U and S are C^2 and the
# second derivatives that expose seams -- Gamma = dlnP/dlnrho and c_v -- are still
# C^1.  Cubic would leave visible kinks in exactly those diagnostics.

def _bspline_knots(n_basis, k):
    """Clamped uniform knot vector on [-1, 1] carrying `n_basis` basis functions."""
    if n_basis < k + 1:
        raise ValueError(f'need n_basis >= k+1 = {k + 1}, got {n_basis}')
    breaks = np.linspace(-1.0, 1.0, n_basis - k + 1)
    return np.concatenate([np.full(k, -1.0), breaks, np.full(k, 1.0)])


def _bspl(z, knots, k):
    """B_i(z), dB_i/dz, d2B_i/dz2 -- each (n, n_basis)."""
    n_basis = len(knots) - k - 1
    V = np.empty((len(z), n_basis))
    D1 = np.empty_like(V)
    D2 = np.empty_like(V)
    e = np.zeros(n_basis)
    for i in range(n_basis):
        e[i] = 1.0
        spl = _BSpline(knots, e, k, extrapolate=True)
        V[:, i] = spl(z)
        D1[:, i] = spl(z, nu=1)
        D2[:, i] = spl(z, nu=2)
        e[i] = 0.0
    return V, D1, D2


def _gamma_floor(rho, lo=0.02, hi=0.90, rho_ramp=0.6, width=None):
    """
    Density-dependent floor on Gamma = (dlnP/dlnrho)_T.

    `hi` MUST BE STRICTLY LESS THAN 1.  An ideal gas has Gamma == 1 exactly, and
    `fit()` asserts that the ideal gas (theta = 0) is STRICTLY feasible for every
    constraint round -- `_solve_qp_interior` is a primal log-barrier method and
    requires an interior warm start.  A floor of 1.0 puts theta = 0 exactly on the
    barrier and the assert fires; anything above 1.0 makes the problem infeasible
    at the warm start.  0.90 still caps the inversion's conditioning at
    |dln rho| <= 1.11 |dln P|, against ~3000 on the unconstrained fit.

    The floor must RAMP OFF at low density.  Near a critical point Gamma genuinely
    approaches zero -- that is what critical opalescence is -- and the measured
    true Gamma from the reference EOS reaches 0.24 (CH4) / 0.11 (NH3) there.
    Measured minima, which set the ramp:

        rho bin        CH4 true Gamma    NH3 true Gamma
        0.05 - 0.30        0.000             0.000        <- critical region
        0.30 - 0.50        2.42              0.002        <- NH3 still subcritical
        0.50 - 0.80        3.83              3.06
        DFT range          1.57              1.38         (rho 0.5 - 5.25)

    So 0.90 is inactive on every real data point above rho ~ 0.5, and the ramp
    centre of 0.6 g/cm^3 (~3.7 rho_c for CH4, ~2.6 rho_c for NH3) keeps it clear
    of the critical region entirely.
    """
    if width is None:
        width = np.log(3.0)
    assert hi < 1.0, 'gamma floor must be < 1: the ideal gas has Gamma == 1 exactly'
    s = 0.5 * (1.0 + np.tanh(np.log(np.asarray(rho, float) / rho_ramp) / width))
    return lo + (hi - lo) * s


def _diff2(n):
    """Second-difference operator, (n-2, n).  Rows of [1, -2, 1]."""
    D = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


class HelmholtzEOS:
    """
    F(rho,T) = R_s T [ alpha0 + delta*A(y) + delta^2*Psi(x,y) ].

    A and Psi are tensor-product B-splines of order `k` with `n_t` (and `n_r x n_t`)
    coefficients on clamped uniform knots in the scaled coordinates x, y.

    With alpha_exc identically zero this reduces to the re-gauged ideal gas, which
    is a strictly feasible point of every stability constraint -- so the fit has a
    free warm start and no phase-1 problem, in every cutting-plane round.
    """

    def __init__(self, species, rho_lo=3e-6, rho_hi=30.0, T_lo=150.0, T_hi=30000.0,
                 n_r=20, n_t=12, k=4, coef=None):
        self.species = species
        self.ideal = IDEAL[species]
        self.R_s = self.ideal.R_s
        self.rho_red, self.T_red = self.ideal.rho_red, self.ideal.T_red

        self.lr0, self.lr1 = np.log(rho_lo), np.log(rho_hi)
        self.lt0, self.lt1 = np.log(T_lo), np.log(T_hi)
        self.a_r = 2.0 / (self.lr1 - self.lr0)
        self.a_t = 2.0 / (self.lt1 - self.lt0)

        self.n_r, self.n_t, self.k = n_r, n_t, k
        self.kx = _bspline_knots(n_r, k)
        self.ky = _bspline_knots(n_t, k)
        self.n_A = n_t
        self.n_Psi = n_r * n_t
        self.n_coef = self.n_A + self.n_Psi
        self.coef = np.zeros(self.n_coef) if coef is None else np.asarray(coef, float)
        # Post-fit entropy gauge.  See `solve_s_offset`: s > 0 is a DOWNSTREAM
        # numerical requirement, not a thermodynamic one, so it is imposed on the
        # output rather than as a constraint competing with the data.
        self.s_offset = 0.0

    # -- coordinates ---------------------------------------------------------
    def _xy(self, rho, T):
        """
        Scaled spline coordinates, CLAMPED to [-1, 1].

        A B-spline extrapolated past its last knot continues as the edge polynomial
        piece, which is far tamer than the cosh(n arccosh|x|) blow-up of a global
        polynomial but still not something to rely on.  Clamping gives a constant
        extension instead, and it is safe here because the analytic delta and
        delta^2 prefactors -- not the spline -- carry the rho -> 0 behaviour: below
        rho_lo the excess vanishes linearly in delta whatever the frozen spline
        value is, so the ideal-gas limit stays exact (|Z-1| < 1e-5 at rho = 1e-6).
        """
        x = np.clip(self.a_r * (np.log(rho) - self.lr0) - 1.0, -1.0, 1.0)
        y = np.clip(self.a_t * (np.log(T) - self.lt0) - 1.0, -1.0, 1.0)
        return x, y

    def _blocks(self, rho, T):
        x, y = self._xy(np.atleast_1d(rho), np.atleast_1d(T))
        return _bspl(x, self.kx, self.k) + _bspl(y, self.ky, self.k)

    @property
    def _A(self):
        return self.coef[:self.n_A]

    @property
    def _Psi(self):
        return self.coef[self.n_A:].reshape(self.n_r, self.n_t)

    # -- the excess term and its D / E derivatives ---------------------------
    # alpha_exc = delta*A(y) + delta^2*Psi(x,y)
    #   D(delta A)      = delta A                D^2(delta A)   = delta A
    #   D(delta^2 Psi)  = delta^2 (2 Psi + a_r Psi_x)
    #   D^2(delta^2Psi) = delta^2 (4 Psi + 4 a_r Psi_x + a_r^2 Psi_xx)
    #   E(delta A)      = delta a_t A_y          E^2(delta A)   = delta a_t^2 A_yy
    #   E(delta^2 Psi)  = delta^2 a_t Psi_y      E^2            = delta^2 a_t^2 Psi_yy
    #   DE(delta A)     = delta a_t A_y
    #   DE(delta^2 Psi) = delta^2 (2 a_t Psi_y + a_r a_t Psi_xy)
    def _exc_terms(self, rho, T):
        Vx, dVx, d2Vx, Vy, dVy, d2Vy = self._blocks(rho, T)
        d = np.atleast_1d(rho) / self.rho_red
        A, P = self._A, self._Psi
        e = dict(
            A=Vy @ A, Ay=dVy @ A, Ayy=d2Vy @ A,
            P=np.einsum('ki,ij,kj->k', Vx, P, Vy),
            Px=np.einsum('ki,ij,kj->k', dVx, P, Vy),
            Pxx=np.einsum('ki,ij,kj->k', d2Vx, P, Vy),
            Py=np.einsum('ki,ij,kj->k', Vx, P, dVy),
            Pyy=np.einsum('ki,ij,kj->k', Vx, P, d2Vy),
            Pxy=np.einsum('ki,ij,kj->k', dVx, P, dVy),
        )
        e['d'], e['d2'] = d, d ** 2
        return e

    def _G(self, rho, T):
        """Return (G, D G, D^2 G, E G, E^2 G, D E G) for G = alpha0 + alpha_exc."""
        d = np.atleast_1d(rho) / self.rho_red
        tau = self.T_red / np.atleast_1d(T)
        it = self.ideal
        e = self._exc_terms(rho, T)
        ar, at = self.a_r, self.a_t

        G = it.alpha(d, tau) + e['d'] * e['A'] + e['d2'] * e['P']
        DG = it.D(d, tau) + e['d'] * e['A'] + e['d2'] * (2 * e['P'] + ar * e['Px'])
        DDG = it.DD(d, tau) + e['d'] * e['A'] + e['d2'] * (
            4 * e['P'] + 4 * ar * e['Px'] + ar ** 2 * e['Pxx'])
        EG = it.E(d, tau) + e['d'] * at * e['Ay'] + e['d2'] * at * e['Py']
        EEG = it.EE(d, tau) + e['d'] * at ** 2 * e['Ayy'] + e['d2'] * at ** 2 * e['Pyy']
        DEG = it.DE(d, tau) + e['d'] * at * e['Ay'] + e['d2'] * (
            2 * at * e['Py'] + ar * at * e['Pxy'])
        return G, DG, DDG, EG, EEG, DEG

    # -- public thermodynamic surface ---------------------------------------
    # NOTE on the ideal term: D alpha0 = 1 exactly (it comes from the ln(delta)).
    # The textbook forms P = rho R_s T (1 + delta*alpha_r_delta) and
    # c_v/R = -tau^2(alpha0_tautau + alpha_r_tautau) are written with the RESIDUAL
    # only, so their leading "1" IS this D alpha0.  Since G here carries alpha0,
    # that 1 must not be added again -- doing so doubles the pressure.
    def free_energy(self, rho, T):
        """F = R_s T G, MINUS T*s_offset so the entropy gauge stays consistent.

        S = -(dF/dT)_rho, so shifting s by +s_offset REQUIRES shifting F by
        -T*s_offset.  Both P and U are provably untouched by this term: it is
        rho-independent, so P = rho^2 (dF/drho)_T does not see it, and in
        U = F + T S the two contributions cancel exactly.  Only F itself changes --
        which is why the Legendre identity |u - (F + Ts)| is the test that catches
        a mismatch here, and it did (1.28e+02 before this line was added).
        """
        T = np.atleast_1d(T)
        G, *_ = self._G(rho, T)
        return self.R_s * T * G - T * self.s_offset

    def p(self, rho, T):
        """P = rho^2 (dF/drho)_T = rho R_s T * D G."""
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        _, DG, *_ = self._G(rho, T)
        return rho * self.R_s * T * DG

    def s(self, rho, T):
        """S = -(dF/dT)_rho = -R_s (G + E G), plus the post-fit gauge offset."""
        G, _, _, EG, _, _ = self._G(rho, T)
        return -self.R_s * (G + EG) + self.s_offset

    def u(self, rho, T):
        """U = F + T S = -R_s T * E G."""
        T = np.atleast_1d(T)
        _, _, _, EG, _, _ = self._G(rho, T)
        return -self.R_s * T * EG

    def cv(self, rho, T):
        """c_v = (dU/dT)_rho = -R_s (E G + E^2 G)."""
        _, _, _, EG, EEG, _ = self._G(rho, T)
        return -self.R_s * (EG + EEG)

    def dpdrho(self, rho, T):
        """(dP/drho)_T = R_s T (D G + D^2 G)."""
        T = np.atleast_1d(T)
        _, DG, DDG, _, _, _ = self._G(rho, T)
        return self.R_s * T * (DG + DDG)

    def dpdT(self, rho, T):
        """(dP/dT)_rho = R_s rho (D G + D E G)."""
        rho = np.atleast_1d(rho)
        _, DG, _, _, _, DEG = self._G(rho, T)
        return self.R_s * rho * (DG + DEG)

    def Z(self, rho, T):
        """Compressibility factor P/(rho R_s T) = D G; -> 1 as rho -> 0."""
        _, DG, *_ = self._G(rho, T)
        return DG

    def gamma(self, rho, T):
        """Gamma = (dlnP/dlnrho)_T = (D G + D^2 G) / D G.

        Note this is EXACT and needs no division by P: from
        P = rho R_s T (DG) and dP/drho = R_s T (DG + DDG),

            Gamma = rho/P * dP/drho = (DG + DDG)/DG.

        Gamma governs the conditioning of the (P,T) -> rho inversion, whose
        Newton step is -f/Gamma.  It is therefore a first-class diagnostic of
        this surface, not a derived convenience.
        """
        _, DG, DDG, *_ = self._G(rho, T)
        return (DG + DDG) / DG

    def p_and_gamma(self, rho, T):
        """(P, Gamma) from a SINGLE _G evaluation.

        `p` and `gamma` each rebuild the whole B-spline basis via `_bspl`, which
        loops in Python over the basis functions, so calling them separately
        doubles the cost of every Newton iteration (measured ~10 -> ~5 us/point).
        The inversion calls this on every iteration, so the fusion is worth having.
        """
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        _, DG, DDG, *_ = self._G(rho, T)
        return rho * self.R_s * T * DG, (DG + DDG) / DG

    def ln_p_and_gamma(self, rho, T):
        """(ln P, Gamma) -- the residual and derivative the inversion actually needs.

        ln P is assembled as ln(DG) + ln(rho) + ln(R_s T) rather than log(P), so the
        product rho*R_s*T*DG (which reaches 6e14 over this domain) is never formed.
        """
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        _, DG, DDG, *_ = self._G(rho, T)
        with np.errstate(divide='ignore', invalid='ignore'):
            lnp = np.log(DG) + np.log(rho) + np.log(self.R_s * T)
        return lnp, (DG + DDG) / DG

    # ======================================================================
    # Design matrices.  Every quantity is AFFINE in the coefficients:
    #     q(rho,T) = q_ideal(rho,T) + M_q(rho,T) @ theta
    # which is what makes both the objective and the constraint set convex.
    # ======================================================================
    def _design_G(self, rho, T):
        """
        Return {name: (ideal_part, coef_matrix)} for G and its D/E derivatives.
        coef_matrix has shape (n_points, n_coef).
        """
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        Vx, dVx, d2Vx, Vy, dVy, d2Vy = self._blocks(rho, T)
        d = (rho / self.rho_red)[:, None]
        d2 = d ** 2
        tau = self.T_red / T
        it, ar, at = self.ideal, self.a_r, self.a_t
        kron = lambda Ax, Ay: np.einsum('ki,kj->kij', Ax, Ay).reshape(len(rho), -1)

        # A-block columns use only the y-basis; Psi-block columns are the kron.
        out = {}
        out['G'] = (it.alpha(rho / self.rho_red, tau),
                    np.hstack([d * Vy, d2 * kron(Vx, Vy)]))
        out['DG'] = (it.D(rho, tau),
                     np.hstack([d * Vy,
                                d2 * (2 * kron(Vx, Vy) + ar * kron(dVx, Vy))]))
        out['DDG'] = (it.DD(rho, tau),
                      np.hstack([d * Vy,
                                 d2 * (4 * kron(Vx, Vy) + 4 * ar * kron(dVx, Vy)
                                       + ar ** 2 * kron(d2Vx, Vy))]))
        out['EG'] = (it.E(rho, tau),
                     np.hstack([d * at * dVy, d2 * at * kron(Vx, dVy)]))
        out['EEG'] = (it.EE(rho, tau),
                      np.hstack([d * at ** 2 * d2Vy, d2 * at ** 2 * kron(Vx, d2Vy)]))
        out['DEG'] = (it.DE(rho, tau),
                      np.hstack([d * at * dVy,
                                 d2 * (2 * at * kron(Vx, dVy)
                                       + ar * at * kron(dVx, dVy))]))
        return out

    def design(self, rho, T, quantity):
        """
        (ideal_part, coef_matrix) for one physical quantity, in CGS.
        `quantity` is one of p, s, u, cv, dpdrho, dpdT, Z.
        """
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        g = self._design_G(rho, T)
        Rs = self.R_s
        if quantity == 'p':
            c = (rho * Rs * T)[:, None]
            return rho * Rs * T * g['DG'][0], c * g['DG'][1]
        if quantity == 'Z':
            return g['DG'][0], g['DG'][1]
        if quantity == 's':
            return -Rs * (g['G'][0] + g['EG'][0]), -Rs * (g['G'][1] + g['EG'][1])
        if quantity == 'u':
            c = (Rs * T)[:, None]
            return -Rs * T * g['EG'][0], -c * g['EG'][1]
        if quantity == 'cv':
            return -Rs * (g['EG'][0] + g['EEG'][0]), -Rs * (g['EG'][1] + g['EEG'][1])
        if quantity == 'dpdrho':
            c = (Rs * T)[:, None]
            return Rs * T * (g['DG'][0] + g['DDG'][0]), c * (g['DG'][1] + g['DDG'][1])
        if quantity == 'dpdT':
            c = (Rs * rho)[:, None]
            return Rs * rho * (g['DG'][0] + g['DEG'][0]), c * (g['DG'][1] + g['DEG'][1])
        raise ValueError(f'unknown quantity {quantity!r}')


# ===========================================================================
# Fitting
# ===========================================================================
import os as _os
import pandas as _pd
from scipy.optimize import minimize as _minimize

_HERE = _os.path.dirname(_os.path.realpath(__file__))

# Reference-EOS validity limits (from the module docstrings of the source
# implementations).  P_MAX is 1000 MPa in dyn/cm^2.
REF_LIMITS = {
    'methane': dict(T_lo=95.0, T_hi=625.0, P_max=1.0e10),
    # NH3 is subcritical below Tc = 405.56 K and the Gao EOS then has a genuine
    # van der Waals loop (verified at 400 K over rho in [0.153, 0.313]).  Start
    # pseudo-data at 1.02*Tc so the loop is never in the training set.
    'ammonia': dict(T_lo=413.7, T_hi=725.0, P_max=1.0e10),
}

# The offset conventionally added to the DFT internal energy.  It is absorbed by
# the free Delta_u parameter, so its value only sets the starting point.
DFT_U_OFFSET = {'methane': 145.0, 'ammonia': 125.0}   # kJ/g


def load_dft(species):
    """Raw Bethkenhagen+2017 points: rho [g/cm^3], T [K], P [dyn/cm^2], u [erg/g]."""
    fn = _os.path.join(_HERE, 'methane_ammonia', f'DFT_EOS_{species}.dat')
    df = _pd.read_csv(fn, sep=r'\s+', comment='#', header=None,
                      skip_blank_lines=True, names=['rho', 'T', 'p', 'u']).dropna()
    return (df['rho'].values, df['T'].values, df['p'].values * 1e10,
            (df['u'].values + DFT_U_OFFSET[species]) * 1e10)


def _reference_module(species):
    if species == 'methane':
        from eos import ch4_setzmann_eos as m
    else:
        from eos import nh3_gao_eos as m
    return m


def _reference_residual(species, rho, T):
    """
    alpha^r and its (D alpha^r, tau*alpha^r_tau) from the reference EOS.

    Only the RESIDUAL is taken from the reference; the ideal part always comes from
    our own re-gauged `IdealTerm`.  That guarantees the two ideal contributions are
    identical by construction rather than by agreement, and it sidesteps the fact
    that the reference modules' `_alpha_ideal_tau` is written for scalars only
    (it does `u_k * tau / Tc` with u_k of shape (3,), which will not broadcast).
    """
    it = IDEAL[species]
    d = np.atleast_1d(rho) / it.rho_red
    tau = it.T_red / np.atleast_1d(T)
    if species == 'methane':
        from eos import ch4_setzmann_eos as m
        ar = m._phi_residual(d, tau)
        ar_d = m._phi_r_delta(d, tau)
        ar_t = m._phi_r_tau(d, tau)
    else:
        from eos import nh3_gao_eos as m
        ar = m._alpha_residual(d, tau)
        ar_d = m._alpha_r_delta(d, tau)
        ar_t = m._alpha_r_tau(d, tau)
    return (np.asarray(ar, float), d * np.asarray(ar_d, float),
            tau * np.asarray(ar_t, float))


def reference_pseudodata(species, n_rho=48, n_T=22, rho_max=1.2):
    """
    P, U, S from the reference EOS on a grid strictly inside its validity domain,
    expressed in OUR gauge.  Points with P <= 0 or (dP/drho)_T <= 0 are rejected --
    that filter is what keeps any van der Waals structure out of the training set.
    """
    it = IDEAL[species]
    lim = REF_LIMITS[species]
    rho = np.exp(np.linspace(np.log(3e-6), np.log(rho_max), n_rho))
    T = np.exp(np.linspace(np.log(lim['T_lo']), np.log(lim['T_hi']), n_T))
    R, Tg = np.meshgrid(rho, T)
    r, t = R.ravel(), Tg.ravel()

    with np.errstate(all='ignore'):
        ar, Dar, tau_ar_t = _reference_residual(species, r, t)
        # ideal + residual, both in our gauge
        p = r * it.R_s * t * (1.0 + Dar)
        u = it.u(r, t) + it.R_s * t * tau_ar_t
        s = it.s(r, t) + it.R_s * (tau_ar_t - ar)
        h = 1e-5
        _, Dp, _ = _reference_residual(species, r * (1 + h), t)
        _, Dm, _ = _reference_residual(species, r * (1 - h), t)
        pp = r * (1 + h) * it.R_s * t * (1.0 + Dp)
        pm = r * (1 - h) * it.R_s * t * (1.0 + Dm)
        dpdr = (pp - pm) / (2 * h * r)

    good = (np.isfinite(p) & np.isfinite(u) & np.isfinite(s)
            & (p > 0) & (dpdr > 0) & (p <= lim['P_max']))
    return r[good], t[good], p[good], u[good], s[good]


def lowT_pseudodata(species, T_lo=50.0, T_knee=1000.0, n_T=6):
    """
    P-only cold anchor BELOW the DFT temperature floor, derived from the DFT itself.

    WHY THIS IS NEEDED.  There is no training data below 200 K (CH4) / 413.7 K (NH3)
    at any density, and none below 1000 K above rho = 0.7.  Lowering `T_lo` alone
    therefore does not extend the model -- it only enlarges the region where nothing
    constrains it, and there the roughness penalty continues the surface LINEARLY in
    y = a_t ln T, whereas the physical excess behaves like tau = T_red/T.

    WHAT THE DATA SAY.  Least-squares of P = P_cold(rho) + b(rho)*T on each DFT
    isochore over T in [1000, 6000] K is linear to a few percent, and the
    extrapolated intercept is 70-100% of P(1000 K):

        CH4  rho=1.00  P_cold/P(1e3) = 0.963      NH3  rho=1.00  0.700
             rho=2.50                  0.976           rho=3.50  0.966
             rho=4.50                  0.980           rho=5.25  0.987

    i.e. in the dense regime P is nearly T-INDEPENDENT below 1000 K, because it is
    dominated by the cold (degeneracy + repulsion) curve.  The clamped v1 fit instead
    gave P(10, 50 K)/P(10, 150 K) = 1/3 -- a factor-three error, and the reason the
    cold high-pressure corner was unreachable at all.

    This has exactly the epistemic status of `highdensity_pseudodata`: a smooth
    physically-motivated continuation of the data, carried at a comparable weight,
    not a measurement.
    """
    r, t, p, _ = load_dft(species)
    rr = np.unique(r)
    out_r, out_t, out_p = [], [], []
    for r0 in rr:
        m = np.isclose(r, r0) & (t >= 1000.0) & (t <= 6000.0)
        if m.sum() < 3:
            continue
        b, p_cold = np.polyfit(t[m], p[m], 1)          # P = p_cold + b*T
        # the anchor is only meaningful if the intercept is physical
        if not np.isfinite(p_cold) or p_cold <= 0:
            continue
        p_knee = p_cold + b * T_knee
        for Tq in np.linspace(T_lo, T_knee, n_T, endpoint=False):
            out_r.append(r0)
            out_t.append(Tq)
            out_p.append(p_cold + (p_knee - p_cold) * Tq / T_knee)
    return (np.array(out_r), np.array(out_t), np.array(out_p))


def highdensity_pseudodata(species, n_rho=8, rho_max=30.0):
    """
    P-only continuation above the DFT ceiling, from a Gamma = dlnP/dlnrho
    extrapolation that relaxes toward the free-electron value 5/3.  Carries a
    deliberately weak weight -- it only steers the extrapolation, and given P
    everywhere plus U on the DFT grid the Maxwell relation fixes U.
    """
    r, t, p, _ = load_dft(species)
    rr, TT = np.unique(r), np.unique(t)
    out_r, out_t, out_p = [], [], []
    for T0 in TT:
        m = t == T0
        if m.sum() < 3:
            continue
        rs, ps = r[m], p[m]
        o = np.argsort(rs)
        rs, ps = rs[o], ps[o]
        G0 = np.polyfit(np.log(rs[-3:]), np.log(ps[-3:]), 1)[0]
        r_top, p_top = rs[-1], ps[-1]
        for rq in np.exp(np.linspace(np.log(r_top * 1.15), np.log(rho_max), n_rho)):
            # integral of Gamma(rho') dln rho' with Gamma -> 5/3 as rho grows
            L = np.log(rq / r_top)
            G = 5.0 / 3.0 + (G0 - 5.0 / 3.0) * np.sqrt(r_top / rq)
            out_r.append(rq)
            out_t.append(T0)
            out_p.append(p_top * np.exp(0.5 * (G0 + G) * L))
    return (np.array(out_r), np.array(out_t), np.array(out_p))


def solve_s_offset(eos, rho_lo=None, rho_hi=None, T_lo=None, T_hi=None,
                   n=400, margin_Rs=0.5):
    """
    Choose the additive entropy gauge so s > 0 over the whole evaluation domain.

    Entropy is physical only up to an additive constant (a1 is pure gauge and cancels
    exactly out of U, P and c_v), so this changes no thermodynamics whatsoever -- it
    only moves the zero.  Doing it AFTER the fit means the requirement costs the fit
    nothing, whereas imposing it DURING the fit makes it compete with the data for
    coefficients and demonstrably wrecks the dense-region pressure.

    Returns the offset in erg/(g K) and also sets `eos.s_offset`.
    """
    rho_lo = np.exp(eos.lr0) if rho_lo is None else rho_lo
    rho_hi = np.exp(eos.lr1) if rho_hi is None else rho_hi
    T_lo = np.exp(eos.lt0) if T_lo is None else T_lo
    T_hi = np.exp(eos.lt1) if T_hi is None else T_hi
    R, T = np.meshgrid(np.exp(np.linspace(np.log(rho_lo), np.log(rho_hi), n)),
                       np.exp(np.linspace(np.log(T_lo), np.log(T_hi), n)))
    eos.s_offset = 0.0
    s_min = float(eos.s(R.ravel(), T.ravel()).min())
    eos.s_offset = max(0.0, margin_Rs * eos.R_s - s_min)
    return eos.s_offset


def fit(species, n_r=20, n_t=13, k=4, n_per_knot=4,
        rho_lo=1e-3, rho_hi=60.0, T_lo=50.0, T_hi=30000.0,
        # hi_p is a smooth PHYSICAL MODEL (Gamma -> 5/3), not a noisy measurement:
        # its implied Gamma is 1.89-2.44, monotone, and joins the DFT top smoothly.
        # Its sigma is a genuine two-sided trade-off, measured on CH4:
        #     sigma  weight   audit   DFT P median   non-positive cells (500^2)
        #      0.30   0.15    DIRTY      4.09%            62   <- ringing at rho~12
        #      0.20   0.30    CLEAN      3.37%            32
        #      0.10   0.50    CLEAN      3.55%             0   <- chosen
        #      0.05   0.50    CLEAN      4.15%             0
        #      0.03   1.00    DIRTY      6.59%             0   <- model overrides DFT
        # Too loose and the fit rings between its points -- delta^2 ~ 5400 at rho = 12
        # turns a tiny coefficient wiggle into a huge (dP/drho)_T swing.  Too tight and
        # a Gamma-extrapolation MODEL starts overriding real DFT data at rho = 4.5.
        sig=dict(ref_p=0.002, ref_u=0.02, ref_s=0.02, dft_p=0.02, dft_u=0.3,
                 hi_p=0.10, lowT_p=0.10),
        wt=dict(ref=1.0, dft=1.0, hi=0.5, lowT=0.3),
        # floors are now multiples of the strictly-positive scales above:
        # Z [-], dpdrho [R_s T], dpdT [R_s rho], cv [R_s], u [R_s T], s [R_s]
        floors=dict(Z=0.05, dpdrho=0.30, dpdT=0.02, cv=0.5, u=0.02, s=-100.0,
                    gamma=0.90),
        lam=0.1, ridge=1e-5, audit_per_knot=24, max_cuts=20, verbose=True):
    """
    Fit alpha_exc for one species by convex QP.

    Objective: sum over data classes of (w_k / N_k) * ||r_k / sigma_k||^2, with
    dimensionless residuals and each class divided by its own point count so a
    class cannot buy influence by being densely sampled.

    Constraints: Z, (dP/drho)_T, (dP/dT)_rho and c_v are held above small
    positive multiples of their ideal-gas values.  Relative floors are scale
    free, and every row is normalised by its own magnitude -- unnormalised rows
    (~1e12 against an O(1) objective) make SLSQP report "incompatible" and quit.

    theta = 0 (the re-gauged ideal gas) is strictly feasible for every
    constraint, so the warm start is free and there is no phase-1 problem.
    """
    eos = HelmholtzEOS(species, rho_lo, rho_hi, T_lo, T_hi, n_r, n_t, k)
    n_A, n_s = eos.n_A, eos.n_coef
    n_th = n_s + 1                      # + free Delta_u for the DFT energy zero
    Rs = eos.R_s

    rows, rhs, tags = [], [], []

    def add(M, y, w, tag):
        """Append rows scaled by sqrt(w / N) so each class is count-normalised."""
        if len(y) == 0:
            return
        f = np.sqrt(w / len(y))
        rows.append(M * f)
        rhs.append(np.asarray(y, float) * f)
        tags.append((tag, len(y)))

    # ---- class 1: reference pseudo-data (P, U, S) -------------------------
    r1, t1, p1, u1, s1 = reference_pseudodata(species)
    i, M = eos.design(r1, t1, 'p')
    add(np.hstack([M / p1[:, None], np.zeros((len(r1), 1))]) / sig['ref_p'],
        (1.0 - i / p1) / sig['ref_p'], wt['ref'], 'ref_P')
    i, M = eos.design(r1, t1, 'u')
    sc = (Rs * t1)[:, None]
    add(np.hstack([M / sc, np.zeros((len(r1), 1))]) / sig['ref_u'],
        ((u1 - i) / (Rs * t1)) / sig['ref_u'], wt['ref'], 'ref_U')
    i, M = eos.design(r1, t1, 's')
    add(np.hstack([M / Rs, np.zeros((len(r1), 1))]) / sig['ref_s'],
        ((s1 - i) / Rs) / sig['ref_s'], wt['ref'], 'ref_S')

    # ---- class 2: DFT-MD, scattered (no gridding, no differencing) --------
    r2, t2, p2, u2 = load_dft(species)
    i, M = eos.design(r2, t2, 'p')
    add(np.hstack([M / p2[:, None], np.zeros((len(r2), 1))]) / sig['dft_p'],
        (1.0 - i / p2) / sig['dft_p'], wt['dft'], 'dft_P')
    i, M = eos.design(r2, t2, 'u')
    sc = (Rs * t2)[:, None]
    col = np.full((len(r2), 1), -1.0) / sc      # d r_u / d Delta_u
    add(np.hstack([M / sc, col]) / sig['dft_u'],
        ((u2 - i) / (Rs * t2)) / sig['dft_u'], wt['dft'], 'dft_U')

    # ---- class 3: high-density P continuation (weak) ----------------------
    r3, t3, p3 = highdensity_pseudodata(species, rho_max=rho_hi)
    i, M = eos.design(r3, t3, 'p')
    add(np.hstack([M / p3[:, None], np.zeros((len(r3), 1))]) / sig['hi_p'],
        (1.0 - i / p3) / sig['hi_p'], wt['hi'], 'hi_P')

    # ---- class 4: cold anchor below the DFT temperature floor -------------
    # Without this, lowering T_lo to 50 K merely enlarges the unconstrained region.
    # See lowT_pseudodata for the measurement that motivates it.
    r4, t4, p4 = lowT_pseudodata(species, T_lo=T_lo)
    if len(r4):
        i, M = eos.design(r4, t4, 'p')
        add(np.hstack([M / p4[:, None], np.zeros((len(r4), 1))]) / sig['lowT_p'],
            (1.0 - i / p4) / sig['lowT_p'], wt['lowT'], 'lowT_P')

    A = np.vstack(rows)
    b = np.concatenate(rhs)

    # ---- column scaling ---------------------------------------------------
    # delta^2 spans ~14 decades over the box, so the raw design matrix has
    # cond ~ 5e16 and the optimal coefficients are O(1e6).  Solving in scaled
    # coordinates z = cs * theta makes the columns unit-norm, which is what lets
    # the roughness penalty and the barrier both act on an O(1) quantity.  Without
    # this the penalty term (~lam * |theta|^2 ~ 1e12) buries the O(1) data term.
    cs = np.linalg.norm(A, axis=0)
    cs[cs <= 0] = 1.0
    As = A / cs

    # ---- constraints ------------------------------------------------------
    # Log-UNIFORM collocation, several nodes per knot interval.  The old
    # Chebyshev-Gauss-Lobatto set was right for a global polynomial, whose
    # excursions are largest at the edges of the box, but it is wrong for a
    # B-spline: a spline's excursions are uniform in the scaled coordinate and
    # local to a knot interval, and CGL nodes thin out precisely in the middle of
    # the box, where they then let violations slip through between points.  Tie
    # the node count to the knots so refining the basis refines the collocation.
    n_cr = n_per_knot * (n_r - k) + 1
    n_ct = n_per_knot * (n_t - k) + 1
    rc = np.exp(np.linspace(np.log(rho_lo), np.log(rho_hi), n_cr))
    tc = np.exp(np.linspace(np.log(T_lo), np.log(T_hi), n_ct))
    RC, TC = np.meshgrid(rc, tc)
    rq, tq = RC.ravel(), TC.ravel()

    # Every quantity the user requires to be positive is constrained.  Each row is
    # normalised by a STRICTLY POSITIVE natural scale -- never by the ideal-gas value
    # of the quantity itself, because s_ideal contains -R_s*ln(delta) and turns
    # negative at high density; dividing an inequality by a negative number flips it,
    # silently converting the s > 0 floor into an upper bound.
    # NOTE on floors['dpdT']: like the Gamma floor, this is a floor in a WEAK scale
    # (rho*R_s), and setting it too high has a non-obvious consequence.  It bounds
    # chi_T = T dpdT/P from below by floors['dpdT']*rho*R_s*T/P, and grad_ad =
    # B chi_T/(c_v + B chi_T^2) with B = P/(rho T chi_rho) peaks at 0.5*sqrt(B/c_v),
    # which at rho=10, T=50 is 13.2.  With dpdT >= 0.02 the induced chi_T floor is
    # 2.07e-3 while the physics needs < 1.4e-3, so grad_ad was pinned ABOVE 1 on 8%
    # of the table.  A cold dense fluid genuinely has chi_T -> 0; the floor must let
    # it.
    _QSCALE = {'Z': lambda r, t: np.ones_like(r),
               'dpdrho': lambda r, t: Rs * t,
               'dpdT': lambda r, t: Rs * r,
               'cv': lambda r, t: np.full_like(r, Rs),
               'u': lambda r, t: Rs * t,
               's': lambda r, t: np.full_like(r, Rs)}

    def gamma_rows(r, t):
        """
        Rows enforcing  Gamma = (dlnP/dlnrho)_T >= gamma_floor(rho).

        WHY THIS EXISTS.  `dpdrho >= 0.30 * R_s * T` sounds like it controls
        compressibility, but R_s*T is a negligible scale against P at high
        density, so that floor permits a near-PLATEAU in P(rho).  Measured on the
        v1 fit: Gamma collapsed to 3.3e-4 (CH4) / 9.5e-4 (NH3) at rho ~ 11 g/cm^3,
        with P flat from rho = 8 to 24 (22 -> 28 Mbar).  That makes the (P,T) ->
        rho inversion catastrophically ill-conditioned: dlogrho/dlogP = 1/Gamma
        reaches 3000, so the tabulated logrho is a step function and interpolating
        it is meaningless.  Measured cost: the interpolation round trip in that
        pressure band was 27-33x worse than the adjacent band.

        Gamma is a RATIO of two affine functions, but the CONSTRAINT is affine:

            Gamma >= g   <=>   rho*(dP/drho) - g*P >= 0

        so the QP stays convex and the existing cutting-plane machinery applies.

        The row is normalised by rho*R_s*T, which is strictly positive -- the same
        discipline as _QSCALE above, and for the same reason.
        """
        g = _gamma_floor(r, hi=floors['gamma'])
        i_d, M_d = eos.design(r, t, 'dpdrho')
        i_p, M_p = eos.design(r, t, 'p')
        scale = (r * Rs * t)[:, None]
        G = np.hstack([(r[:, None] * M_d - g[:, None] * M_p) / scale,
                       np.zeros((len(r), 1))])
        h = (g * i_p - r * i_d) / scale[:, 0]
        return G, h

    def gamma_violates(r, t):
        """Audit predicate for the Gamma floor, in the same form as the others."""
        return (r * eos.dpdrho(r, t)
                - _gamma_floor(r, hi=floors['gamma']) * eos.p(r, t)) < 0.0

    # 's' is deliberately NOT constrained during the fit.  Entropy is defined only
    # up to an additive constant, and the s > 0 requirement is numerical (consumers
    # take log10 of it), so it belongs on the OUTPUT gauge, not in the objective's
    # feasible set.  Measured cost of getting this wrong: with s >= 0.02 R_s enforced
    # and T_lo = 50 K, the constraint is active at the cold dense corner and, because
    # one F serves the whole surface, it caps dense-region pressure -- the DFT error
    # went from 3.6% to 56%, and a +4 R_s gauge shift did NOT relieve it (the fit
    # simply spends the headroom on more compression and re-pins at the floor).
    # `s` IS constrained, but with a permissive floor (a large negative multiple of
    # R_s), not s > 0.  The two failure modes this threads between, both measured:
    #   floors['s'] = +0.02  -> active at the cold dense corner; because one F serves
    #                           the whole surface it caps dense-region pressure and the
    #                           DFT error goes 3.6% -> 56%.  A +4 R_s gauge shift does
    #                           NOT help: the fit spends it and re-pins at the floor.
    #   no s constraint      -> DFT error 1.13%, but s runs away to -38,436 R_s at
    #                           rho = 60 (no data there), so the post-fit gauge needed
    #                           to restore s > 0 is so large that S becomes 99.95%
    #                           constant and loses all dynamic range.
    # A permissive floor bounds the runaway without competing with the data; the
    # post-fit gauge (`solve_s_offset`) then supplies s > 0 for free.  Calibrated:
    #     floor        DFT P median   s_offset      S dynamic range
    #     +0.02 R_s       56.3 %         --              --
    #      -20  R_s       23.1 %       20.5 R_s        99.5 %
    #     -100  R_s        2.6 %      100.5 R_s        99.7 %   <- chosen
    #      none            1.1 %    38437   R_s         0.05 %  <- S unusable
    _FIT_CONSTRAINED = tuple(_QSCALE)

    def constraint_rows(r, t, quantities=_FIT_CONSTRAINED):
        """Rows of G, h enforcing  q(r,t)/scale >= floor  for each quantity,
        plus the Gamma floor."""
        Gr, Hr = [], []
        for q in quantities:
            scale = _QSCALE[q](r, t)
            assert scale.min() > 0, f'constraint scale for {q} must be strictly positive'
            i, M = eos.design(r, t, q)
            Gr.append(np.hstack([M, np.zeros((len(r), 1))]) / scale[:, None])
            Hr.append(floors[q] - i / scale)
        Gg, hg = gamma_rows(r, t)
        Gr.append(Gg)
        Hr.append(hg)
        return np.vstack(Gr), np.concatenate(Hr)

    Gc, hc = constraint_rows(rq, tq)

    # ---- regularisation ---------------------------------------------------
    # TWO penalties, both quadratic in theta (so the problem stays a QP) and both
    # acting in UNSCALED theta space, where the B-spline coefficients are close to
    # local values of A and Psi and are therefore all of comparable size:
    #
    #   roughness (lam)  -- squared second differences between NEIGHBOURING
    #     coefficients.  Its null space is the bilinear functions, so where there
    #     is no data the surface continues linearly out of the nearest data
    #     instead of oscillating.
    #   Tikhonov (ridge) -- shrinks alpha_exc toward zero, i.e. toward the ideal
    #     gas.  In a LOCAL basis this is automatically data-adaptive and replaces
    #     the explicit geometric "distance from the nearest data point" prior the
    #     Chebyshev version needed: a coefficient the data pins down resists
    #     shrinkage, one in a data-free region does not, and shrinking the latter
    #     costs the former nothing because their supports do not overlap.
    #
    # Both are calibrated against the DATA quadratic form evaluated at the
    # unconstrained solution, so lam and ridge are dimensionless fractions of the
    # data misfit.  The earlier trace(Hs)/trace(Pen) normalisation is wrong once
    # the penalty is mapped into scaled coordinates: Pen/(cs cs') has a few
    # enormous entries wherever cs is small, they dominate its trace, and lam
    # collapses to zero -- measured as lam = 0, 1e-6 and 1e-3 giving bit-identical
    # fits.
    Dr, Dt = _diff2(n_r), _diff2(n_t)
    Rx = np.kron(Dr, np.eye(n_t))       # Psi is flattened row-major: idx = i*n_t + j
    Ry = np.kron(np.eye(n_r), Dt)
    Pen = np.zeros((n_th, n_th))
    Pen[:n_A, :n_A] = Dt.T @ Dt
    Pen[n_A:n_s, n_A:n_s] = Rx.T @ Rx + Ry.T @ Ry
    Tik = np.eye(n_th)
    Tik[-1, -1] = 0.0                   # Delta_u is physical (~ -5e11 erg/g); never shrink it

    # ---- assemble the QP in SCALED coordinates z = cs * theta --------------
    # theta = z / cs, so a penalty theta' Pen theta becomes z' (Pen / cs cs') z.
    H0 = As.T @ As
    gs = As.T @ b
    z0 = np.linalg.lstsq(H0 + 1e-14 * np.trace(H0) / n_th * np.eye(n_th), gs,
                         rcond=None)[0]
    dat0 = float(z0 @ H0 @ z0)
    Pen_s, Tik_s = Pen / np.outer(cs, cs), Tik / np.outer(cs, cs)
    Hs = H0.copy()
    if lam:
        Hs += lam * dat0 / max(float(z0 @ Pen_s @ z0), 1e-300) * Pen_s
    if ridge:
        Hs += ridge * dat0 / n_th * Tik_s
    Hs += 1e-12 * np.trace(H0) / n_th * np.eye(n_th)     # guard against exact singularity
    # theta_j = z_j / cs_j, so (G theta)_i = sum_j (G_ij / cs_j) z_j.
    # Note an assert at z = 0 cannot catch getting this backwards, since G @ 0 = 0
    # for either scaling -- the error only shows up once the solver moves.
    Gs = Gc / cs

    # ---- solve, with a cutting-plane loop ---------------------------------
    # Collocation alone cannot certify these constraints.  (dP/drho)_T carries a
    # delta^2 * a_r^2 * Psi_xx term whose prefactor reaches ~7400 at rho = 14, so
    # the quantity swings by O(1e5) in units of R_s T between neighbouring knots
    # and a violation can hide between nodes even at 4 nodes per knot interval --
    # measured as ~1% of a 200x200 audit grid violating (dP/drho)_T at rho = 10-18,
    # the extrapolation region above the DFT ceiling, while every collocation node
    # was satisfied.  Refining the collocation grid globally is the wrong lever:
    # the Hessian assembly is linear in the row count, and 8 nodes per interval
    # already means ~50k rows for violations confined to ~1% of the domain.
    #
    # Instead: solve, audit on a dense grid, append only the points that actually
    # violate, and re-solve.  This is a cutting-plane method for what is really a
    # semi-infinite program, and it converges in a handful of rounds because the
    # violations are spatially clustered.  Each round restarts from theta = 0
    # rather than warm-starting from the previous z: the previous solution
    # violates precisely the rows just added, so it is infeasible for the new
    # problem, whereas the ideal gas is strictly feasible for every round.
    # The audit grid must RESOLVE THE KNOTS, not merely be "dense": a spline with
    # more knots wiggles on a finer scale, so a fixed 220x220 grid that certifies
    # n_r = 20 silently misses violations at n_r = 28 and the loop then reports
    # "clean" on a solution that is not.  Tie the audit resolution to the knot
    # count, and stagger the grid by a different fraction of a cell each round so
    # the union of rounds samples between the previous rounds' nodes as well.
    n_ar = audit_per_knot * (n_r - k) + 1
    n_at = audit_per_knot * (n_t - k) + 1

    def audit_grid(shift):
        la = np.linspace(np.log(rho_lo), np.log(rho_hi), n_ar)
        lb = np.linspace(np.log(T_lo), np.log(T_hi), n_at)
        la = np.clip(la + shift * (la[1] - la[0]), la[0], la[-1])
        lb = np.clip(lb + shift * (lb[1] - lb[0]), lb[0], lb[-1])
        RA, TA = np.meshgrid(np.exp(la), np.exp(lb))
        return RA.ravel(), TA.ravel()

    Gs, hcur = Gc / cs, hc
    n_added, rounds, clean = 0, 0, False
    for rounds in range(1, max_cuts + 1):
        z0 = np.zeros(n_th)
        assert (Gs @ z0 - hcur).min() > 0, 'ideal gas should be strictly feasible'
        z, info = _solve_qp_interior(Hs, gs, Gs, hcur, z0)
        eos.coef, eos.delta_u_dft = (z / cs)[:n_s], (z / cs)[n_s]

        raq, taq = audit_grid(0.5 * (rounds % 2) + 0.17 * (rounds % 3))
        bad = np.zeros(len(raq), bool)
        for q in _FIT_CONSTRAINED:
            bad |= getattr(eos, q)(raq, taq) / _QSCALE[q](raq, taq) < floors[q]
        # The Gamma floor MUST be audited too.  Adding its constraint rows without
        # adding it here would let the loop report CLEAN on a surface that violates
        # it between collocation nodes -- exactly the failure the dilation below
        # was written to prevent, and the same class of error as certifying
        # positivity on a grid too coarse to resolve the violation.
        bad |= gamma_violates(raq, taq)
        if not bad.any():
            clean = True
            break

        # DILATE the violation set before cutting.  Constraining only the points
        # that violate is whack-a-mole: the spline satisfies them and dips again
        # just BETWEEN them on the next round, so the loop adds thousands of points
        # and still exits dirty (measured: 12 rounds, 2629 points, 3 violations
        # left).  Growing each violation into its 3x3 neighbourhood gives the
        # constraint spatial margin instead of a set of isolated pin-pricks, which
        # is what makes the loop actually terminate.
        B = bad.reshape(n_at, n_ar)
        D = B.copy()
        for ax in (0, 1):
            D |= np.roll(B, 1, axis=ax) | np.roll(B, -1, axis=ax)
        bad = D.ravel()
        Gv, hv = constraint_rows(raq[bad], taq[bad])
        Gs = np.vstack([Gs, Gv / cs])
        hcur = np.concatenate([hcur, hv])
        n_added += int(bad.sum())
    th = z / cs
    eos.audit_clean = clean

    eos.coef = th[:n_s]
    eos.delta_u_dft = th[n_s]
    eos.cond = float(np.linalg.cond(As))
    eos.fit_info = info
    eos.cut_rounds, eos.cut_points = rounds, n_added
    solve_s_offset(eos)
    eos.n_violations = int(((Gs @ z - hcur) < -1e-9).sum())
    if verbose:
        print(f'  [{species}] classes: ' + ', '.join(f'{t}={n}' for t, n in tags))
        print(f'  [{species}] {n_th} params, {len(b)} data rows, {len(hc)} constraints'
              f' (+{n_added} cut in {rounds} rounds,'
              f' audit {"CLEAN" if clean else "NOT CLEAN"}) -> residual violations'
              f' {eos.n_violations}, Delta_u = {eos.delta_u_dft:+.4e} erg/g,'
              f' max|coef| = {np.abs(eos.coef).max():.3e}')
    return eos


def _solve_qp_interior(H, g, G, h, x0, mu0=1.0, mu_min=1e-12, tol=1e-10,
                       max_newton=200):
    """
    Minimise 0.5 x'Hx - g'x subject to Gx >= h, by a primal log-barrier method.

        phi_mu(x) = 0.5 x'Hx - g'x - mu * sum(log(Gx - h))
        grad      = Hx - g - mu G' s          s_i = 1/(G_i x - h_i)
        hess      = H + mu G' diag(s^2) G

    SLSQP is not usable at this size -- its dense LSQ subproblem gives up with
    "More than 3*n iterations" once the constraint count reaches ~2000 -- and
    trust-constr is ~60x slower for the same answer.  This is ~40 lines, needs no
    new dependency, and relies on x0 being STRICTLY feasible, which the re-gauged
    ideal gas guarantees.
    """
    x = x0.astype(float).copy()
    n_steps = 0
    mu = mu0
    while mu > mu_min:
        for _ in range(max_newton):
            r = G @ x - h
            if r.min() <= 0:
                break
            s = 1.0 / r
            grad = H @ x - g - mu * (G.T @ s)
            Hess = H + mu * (G.T * s ** 2) @ G
            try:
                dx = np.linalg.solve(Hess, -grad)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(Hess, -grad, rcond=None)[0]
            # backtrack, keeping strict feasibility and decreasing the barrier
            f0 = 0.5 * x @ H @ x - g @ x - mu * np.log(r).sum()
            step = 1.0
            for _ in range(60):
                xt = x + step * dx
                rt = G @ xt - h
                if rt.min() > 0:
                    ft = 0.5 * xt @ H @ xt - g @ xt - mu * np.log(rt).sum()
                    if ft <= f0 - 1e-4 * step * (grad @ dx) * -1e-12 or ft < f0:
                        break
                step *= 0.5
            else:
                break
            x = x + step * dx
            n_steps += 1
            if np.abs(step * dx).max() < tol * max(1.0, np.abs(x).max()):
                break
        mu *= 0.1
    return x, dict(newton_steps=n_steps, min_slack=float((G @ x - h).min()))
