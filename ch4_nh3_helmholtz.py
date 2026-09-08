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
        # e^x/(e^x - 1)^2 written as e^-x/(1 - e^-x)^2: identical, but it
        # cannot overflow (x = theta*tau reaches 750 at T = 20 K, where the
        # naive form returns inf/inf = nan)
        x = np.outer(np.atleast_1d(tau), self.theta)
        emx = np.exp(-x)
        return -(emx / (1.0 - emx) ** 2) @ (self.v * self.theta ** 2)

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


def _bspl(z, knots, k, d1=None, d2=None):
    """B_i(z), and its first and second derivatives WITH RESPECT TO THE RAW
    (linear) coordinate -- each (n, n_basis).

    `z` is the softly saturated coordinate (see `_soft_clip`) and `d1`, `d2` its
    first and second derivatives with respect to the raw linear coordinate
    x_raw = a_r (ln rho - ln rho_lo) - 1.  By the chain rule
        dB/dx_raw   = B'(z) d1,
        d2B/dx_raw2 = B''(z) d1^2 + B'(z) d2,
    so every downstream formula written for a linear coordinate stays valid.
    Inside the linear part of the box d1 = 1 and d2 = 0 and nothing changes;
    outside, the coordinate saturates and the derivatives go smoothly to zero,
    so P = rho^2 (dF/drho)_T and S = -(dF/dT)_rho hold everywhere AND P stays
    continuous.  (v2 kept the derivative rows under a hard clamp, so P was not
    the derivative of F outside the box -- off by 86% at rho = 80 g/cm^3; a
    hard mask instead makes P jump at the box edge, which the (P,T) inversion
    then cannot bracket.)
    """
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
    if d1 is not None:
        d1 = np.asarray(d1, dtype=float)[:, None]
        d2 = np.asarray(d2, dtype=float)[:, None]
        D2 = D2 * d1 ** 2 + D1 * d2
        D1 = D1 * d1
    return V, D1, D2


# Reach (in scaled-coordinate units) of the soft saturation OUTSIDE the box:
# the coordinate is exactly linear for |x_raw| <= 1 (so nothing inside the box
# changes) and saturates as tanh to +-(1 + w) beyond, C^2 at the edge.  The
# spline is then evaluated at most w past its last knot, as the edge
# polynomial piece.  (A saturation zone INSIDE the box, tried first, put a
# curvature ~1/w into the last 5% of the box and produced c_v spikes of 40 R_s
# at 50 K.)
SOFT_CLIP_W = 0.05


def _soft_clip(z, w=SOFT_CLIP_W):
    """Softly saturate a raw scaled coordinate beyond +-1.

    Returns (zs, dzs/dz, d2zs/dz2).  Identity for |z| <= 1; beyond,
    zs = +-1 + w tanh((z -+ 1)/w), which has the same value, slope (1) and
    curvature (0) as the linear part at the edge, and tends to +-(1 + w).
    """
    z = np.asarray(z, dtype=float)
    zs = z.copy()
    d1 = np.ones_like(z)
    d2 = np.zeros_like(z)
    for sign in (1.0, -1.0):
        m = sign * z > 1.0
        if np.any(m):
            u = (z[m] - sign) / w
            t = np.tanh(u)
            zs[m] = sign + w * t
            d1[m] = 1.0 - t ** 2
            d2[m] = -2.0 * t * (1.0 - t ** 2) / w
    return zs, d1, d2


# Knot stretching in density.  With 20 uniform knots the interval is 0.69 in
# ln rho, and the reference ceiling (0.53 / 0.69 g/cm^3), the cold-curve onset
# and the DFT floor (0.6 / 0.5 g/cm^3) all fall inside ONE interval, where P
# rises by a factor 3 over 13% in density.  A monotone warp of the linear
# coordinate, with slope 1 + WARP_B exp(-(x - x_c)^2 / 2 sigma^2), triples the
# knot density around the junction at no cost in coefficients; the far regions
# (dilute gas, cold-curve-dominated dense fluid) are smooth and tolerate the
# ~1.4x coarser spacing.  The warp enters the derivatives through the same
# chain rule as the saturation (see `_bspl`).
WARP_B = 2.0
WARP_SIGMA = 0.15


def _warp(x_lin, x_c, B=WARP_B, sigma=WARP_SIGMA):
    """x_raw(x_lin) with x_raw(-1) = -1, x_raw(1) = 1, and its two derivatives."""
    from scipy.special import erf
    x = np.asarray(x_lin, dtype=float)
    g = lambda t: t + B * sigma * np.sqrt(np.pi / 2.0) * erf((t - x_c) / (np.sqrt(2.0) * sigma))
    g0, g1 = g(-1.0), g(1.0)
    a = 2.0 / (g1 - g0)
    xr = a * (g(x) - g0) - 1.0
    d1 = a * (1.0 + B * np.exp(-(x - x_c) ** 2 / (2.0 * sigma ** 2)))
    d2 = a * (-B * (x - x_c) / sigma ** 2 * np.exp(-(x - x_c) ** 2 / (2.0 * sigma ** 2)))
    return xr, d1, d2


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
                 n_r=20, n_t=12, k=4, coef=None, cold=None, warp_rho=None):
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
        # Post-fit energy gauge, the same idea for u (see `solve_u_offset`).  A
        # constant in F shifts F and U only; S, P, c_v and every derivative are
        # untouched.  v2 instead imposed u >= 0.02 R_s T during the fit, which
        # (through the c_v floor along each isochore) made the reference liquid's
        # NEGATIVE internal energy -- the cohesive energy of a bound fluid measured
        # from the T = 0 molecular gas -- infeasible, at a cost of 28-147% in P on
        # those points.
        self.u_offset = 0.0
        # Structural cold curve (v3): a FIXED, T-independent u_cold(rho) added to F,
        # so that P carries P_cold(rho) = rho^2 u_cold' and U carries u_cold,
        # while S, c_v and (dP/dT)_rho are untouched.  None reproduces the v2 form.
        self.cold = cold
        # knot-stretching centre in density (see `_warp`); None = uniform knots
        self.warp_rho = warp_rho
        self.warp_xc = (None if warp_rho is None
                        else self.a_r * (np.log(warp_rho) - self.lr0) - 1.0)

    # -- coordinates ---------------------------------------------------------
    def _xy(self, rho, T):
        """
        Scaled spline coordinates, softly saturated to (-1, 1), with the
        derivatives of the saturation map (see `_soft_clip`, `_bspl`).

        A B-spline extrapolated past its last knot continues as the edge polynomial
        piece, which is far tamer than the cosh(n arccosh|x|) blow-up of a global
        polynomial but still not something to rely on.  Clamping gives a constant
        extension instead, and it is safe here because the analytic delta and
        delta^2 prefactors -- not the spline -- carry the rho -> 0 behaviour: below
        rho_lo the excess vanishes linearly in delta whatever the frozen spline
        value is, so the ideal-gas limit stays exact (|Z-1| < 1e-5 at rho = 1e-6).

        The saturation map's derivatives are what keep the thermodynamics exact
        outside the box: the chain rule is applied with them in `_bspl`, so the
        derivatives of F with respect to ln rho and ln T are the true ones
        everywhere, and P is continuous across the edge (a hard clamp with masked
        derivative rows made P jump there by ~0.07% at rho_lo).
        """
        xl = self.a_r * (np.log(rho) - self.lr0) - 1.0
        yr = self.a_t * (np.log(T) - self.lt0) - 1.0
        if self.warp_xc is None:
            xr, w1, w2 = xl, np.ones_like(xl), np.zeros_like(xl)
        else:
            xr, w1, w2 = _warp(xl, self.warp_xc)
        x, s1, s2 = _soft_clip(xr)
        # compose: z(x_raw(x_lin)); dz/dx_lin = s1 w1, d2z/dx_lin2 = s2 w1^2 + s1 w2
        dx, d2x = s1 * w1, s2 * w1 ** 2 + s1 * w2
        y, dy, d2y = _soft_clip(yr)
        return x, y, (dx, d2x), (dy, d2y)

    def _blocks(self, rho, T):
        x, y, (dx, d2x), (dy, d2y) = self._xy(np.atleast_1d(rho), np.atleast_1d(T))
        return _bspl(x, self.kx, self.k, dx, d2x) + _bspl(y, self.ky, self.k, dy, d2y)

    # -- the cold curve's contribution to G = F/(R_s T) and its D/E images ----
    # With F = R_s T G_spline + u_cold(rho), the extra term in G is
    # g_c = u_cold/(R_s T).  Since d/dln rho of u_cold is P_cold/rho and
    # d/dln T of 1/T is -1/T:
    #   D g_c  = P_cold/(rho R_s T)          E g_c  = -u_cold/(R_s T)
    #   D^2g_c = (rho P_cold' - P_cold)/(rho R_s T)   E^2 g_c = +u_cold/(R_s T)
    #   D E g_c = -P_cold/(rho R_s T)
    # so P gains P_cold, (dP/drho)_T gains P_cold', U gains u_cold, and S, c_v,
    # (dP/dT)_rho gain nothing.  All six enter as constant vectors, so every
    # quantity stays affine in the fitted coefficients.
    def _cold_terms(self, rho, T):
        rho, T = np.atleast_1d(rho), np.atleast_1d(T)
        if self.cold is None:
            z = np.zeros_like(rho, dtype=float)
            return z, z, z, z, z, z
        u_c, p_c, dp_c = self.cold.u(rho), self.cold.p(rho), self.cold.dpdrho(rho)
        rt = rho * self.R_s * T
        G = u_c / (self.R_s * T)
        DG = p_c / rt
        DDG = (rho * dp_c - p_c) / rt
        EG = -G
        EEG = G
        DEG = -DG
        return G, DG, DDG, EG, EEG, DEG

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

        cG, cDG, cDDG, cEG, cEEG, cDEG = self._cold_terms(rho, T)
        G = it.alpha(d, tau) + e['d'] * e['A'] + e['d2'] * e['P'] + cG
        DG = it.D(d, tau) + e['d'] * e['A'] + e['d2'] * (2 * e['P'] + ar * e['Px']) + cDG
        DDG = it.DD(d, tau) + e['d'] * e['A'] + e['d2'] * (
            4 * e['P'] + 4 * ar * e['Px'] + ar ** 2 * e['Pxx']) + cDDG
        EG = it.E(d, tau) + e['d'] * at * e['Ay'] + e['d2'] * at * e['Py'] + cEG
        EEG = (it.EE(d, tau) + e['d'] * at ** 2 * e['Ayy'] + e['d2'] * at ** 2 * e['Pyy']
               + cEEG)
        DEG = it.DE(d, tau) + e['d'] * at * e['Ay'] + e['d2'] * (
            2 * at * e['Py'] + ar * at * e['Pxy']) + cDEG
        return G, DG, DDG, EG, EEG, DEG

    # -- public thermodynamic surface ---------------------------------------
    # NOTE on the ideal term: D alpha0 = 1 exactly (it comes from the ln(delta)).
    # The textbook forms P = rho R_s T (1 + delta*alpha_r_delta) and
    # c_v/R = -tau^2(alpha0_tautau + alpha_r_tautau) are written with the RESIDUAL
    # only, so their leading "1" IS this D alpha0.  Since G here carries alpha0,
    # that 1 must not be added again -- doing so doubles the pressure.
    def free_energy(self, rho, T):
        """F = R_s T G - T*s_offset + u_offset.

        G already carries u_cold/(R_s T), so R_s T G includes the cold curve.
        S = -(dF/dT)_rho, so shifting s by +s_offset REQUIRES shifting F by
        -T*s_offset.  Both P and U are provably untouched by that term: it is
        rho-independent, so P = rho^2 (dF/drho)_T does not see it, and in
        U = F + T S the two contributions cancel exactly.  Only F itself changes --
        which is why the Legendre identity |u - (F + Ts)| is the test that catches
        a mismatch here, and it did (1.28e+02 before this line was added).
        The energy gauge u_offset is a plain constant: it shifts F and U alike
        and nothing else.
        """
        T = np.atleast_1d(T)
        G, *_ = self._G(rho, T)
        return self.R_s * T * G - T * self.s_offset + self.u_offset

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
        """U = F + T S = -R_s T * E G, plus the post-fit energy gauge."""
        T = np.atleast_1d(T)
        _, _, _, EG, _, _ = self._G(rho, T)
        return -self.R_s * T * EG + self.u_offset

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

    # -- coexistence curve of the fitted surface (see `eos_saturation`) ------
    def saturation(self, T):
        """(P_sat, rho_l_star, rho_v, T_c) of the fit's own coexistence curve
        (`eos_saturation`, `_sat_interp`): P_sat = 0 and rho_v = nan where the
        vapour branch is below the box's lowest pressure; nan/0 for T >= T_c."""
        if getattr(self, 'sat', None) is None:
            eos_saturation(self)
        return _sat_interp(self.sat, T)

    def in_two_phase(self, rho, T):
        """Inside the fit's own coexistence region (rho_v < rho < rho_l_star, T < T_c)."""
        rho = np.asarray(rho, dtype=float)
        T = np.asarray(T, dtype=float)
        P, rl, rv, T_c = self.saturation(T)
        rv = np.where(np.isfinite(rv), rv, 0.0)
        return (T < T_c) & (rho > rv) & (rho < np.where(np.isfinite(rl), rl, 0.0))

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
        # the cold curve is a fixed part of every "ideal" (coefficient-free) vector
        cG, cDG, cDDG, cEG, cEEG, cDEG = self._cold_terms(rho, T)

        # A-block columns use only the y-basis; Psi-block columns are the kron.
        out = {}
        out['G'] = (it.alpha(rho / self.rho_red, tau) + cG,
                    np.hstack([d * Vy, d2 * kron(Vx, Vy)]))
        out['DG'] = (it.D(rho, tau) + cDG,
                     np.hstack([d * Vy,
                                d2 * (2 * kron(Vx, Vy) + ar * kron(dVx, Vy))]))
        out['DDG'] = (it.DD(rho, tau) + cDDG,
                      np.hstack([d * Vy,
                                 d2 * (4 * kron(Vx, Vy) + 4 * ar * kron(dVx, Vy)
                                       + ar ** 2 * kron(d2Vx, Vy))]))
        out['EG'] = (it.E(rho, tau) + cEG,
                     np.hstack([d * at * dVy, d2 * at * kron(Vx, dVy)]))
        out['EEG'] = (it.EE(rho, tau) + cEEG,
                      np.hstack([d * at ** 2 * d2Vy, d2 * at ** 2 * kron(Vx, d2Vy)]))
        out['DEG'] = (it.DE(rho, tau) + cDEG,
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
from scipy.integrate import cumulative_simpson as _cumsimp, quad as _quad
from scipy.interpolate import make_interp_spline as _mkspl
from scipy.optimize import least_squares as _lsq

_HERE = _os.path.dirname(_os.path.realpath(__file__))

# Reference-EOS validity limits (from the module docstrings of the source
# implementations).  P_MAX is 1000 MPa in dyn/cm^2.  For methane the fluid
# region is additionally bounded by the melting line and by the vapour-liquid
# dome (both from the SW91 auxiliary equations in ch4_setzmann_eos); see
# `reference_pseudodata`.
REF_LIMITS = {
    'methane': dict(T_lo=95.0, T_hi=625.0, P_max=1.0e10),
    # NH3 is subcritical below Tc = 405.56 K and the Gao EOS then has a genuine
    # van der Waals loop (verified at 400 K over rho in [0.153, 0.313]).  v2
    # started the pseudo-data at 1.02 Tc so the loop was never in the training
    # set -- which left a data-free hole from 50 K to 414 K below 0.7 g/cm^3
    # where only the constraints shaped the surface (and their reach extended
    # into the reference/DFT seam).  v3 keeps the liquid and vapour branches
    # down to 200 K (triple point 195.49 K) and replaces the loop by the
    # equilibrium two-phase construction (`dome_pseudodata`).
    'ammonia': dict(T_lo=200.0, T_hi=725.0, P_max=1.0e10),
}
TRIPLE_T = {'methane': 90.6941, 'ammonia': 195.49}     # K

# The offset conventionally added to the DFT internal energy.  It is absorbed by
# the free Delta_u parameter, so its value only sets the starting point.
DFT_U_OFFSET = {'methane': 145.0, 'ammonia': 125.0}   # kJ/g

# ---------------------------------------------------------------------------
# Intramolecular vibrations, for the classical -> quantum correction of the
# DFT-MD internal energies.
# ---------------------------------------------------------------------------
# Bethkenhagen et al. (2017) propagate the ions classically and removed the
# nuclear-quantum correction from their data set for consistency, so every
# vibrational mode carries k_B T of thermal energy in their u, whereas the
# reference equations (and our ideal term) carry the Planck-Einstein energy.
# At 1000 K the difference is 6-7 R_s T for CH4: the DFT heat capacity is
# 3 k_B per atom (15 R_s CH4, 12 R_s NH3) against 7-8 R_s quantum, and the v2
# fit bridged the jump by dipping c_v BELOW the ideal-gas value in the
# 625-1000 K gap.  Fundamental frequencies (cm^-1) and degeneracies from the
# NIST Chemistry WebBook (Shimanouchi 1972).
VIB_MODES = {
    'methane': ((2917.0, 1), (1534.0, 2), (3019.0, 3), (1306.0, 3)),
    'ammonia': ((3337.0, 1), (950.0, 1), (3444.0, 2), (1627.0, 2)),
}
N_ATOMS = {'methane': 5, 'ammonia': 4}
CM_TO_K = 1.438776877          # h c / k_B  [K cm]


def vib_thetas(species):
    """(theta_i [K], g_i) for the intramolecular modes."""
    th = np.array([nu * CM_TO_K for nu, _ in VIB_MODES[species]])
    g = np.array([float(g) for _, g in VIB_MODES[species]])
    return th, g


def e_intra(species, T):
    """Quantum thermal energy of the internal modes, no zero point [erg/g]."""
    th, g = vib_thetas(species)
    R_s = IDEAL[species].R_s
    x = th / np.atleast_1d(T)[:, None]
    return R_s * (g * th / np.expm1(x)).sum(axis=1)


def cv_intra(species, T):
    """Quantum heat capacity of the internal modes [erg/(g K)]."""
    th, g = vib_thetas(species)
    R_s = IDEAL[species].R_s
    x = th / np.atleast_1d(T)[:, None]
    with np.errstate(over='ignore', invalid='ignore'):
        f = np.where(x < 500.0, x ** 2 * np.exp(x) / np.expm1(x) ** 2, 0.0)
    return R_s * (g * f).sum(axis=1)


def dft_quantum_correction(species, T):
    """u_quantum - u_classical for the internal modes [erg/g].

    Per mode: theta/(e^{theta/T} - 1) - T, i.e. the Planck-Einstein energy
    without zero point minus the classical k_B T.  It tends to -ZPE as
    T -> infinity; that constant is absorbed by the free Delta_u, so what the
    fit sees is a correction that RAISES the low-T DFT energies relative to
    the high-T ones (+6.4 R_s T at 1000 K for CH4 once the constant is
    removed) and lowers the DFT heat capacity to the quantum value.  Above
    ~3000 K the molecules dissociate and the intact-molecule correction is
    approximate; that is carried in `dft_sigma_u`, not by fading the
    correction, since a fade would add its own dE/dT artefact to c_v.
    """
    T = np.atleast_1d(T).astype(float)
    n_int = float(sum(g for _, g in VIB_MODES[species]))
    R_s = IDEAL[species].R_s
    return e_intra(species, T) - R_s * n_int * T


def load_dft(species, quantum=True):
    """Bethkenhagen+2017 points: rho [g/cm^3], T [K], P [dyn/cm^2], u [erg/g].

    With `quantum=True` (the v3 default) the internal energies carry the
    classical -> quantum correction of `dft_quantum_correction`; the
    pressures are never modified.
    """
    fn = _os.path.join(_HERE, 'methane_ammonia', f'DFT_EOS_{species}.dat')
    df = _pd.read_csv(fn, sep=r'\s+', comment='#', header=None,
                      skip_blank_lines=True, names=['rho', 'T', 'p', 'u']).dropna()
    r, t = df['rho'].values, df['T'].values
    u = (df['u'].values + DFT_U_OFFSET[species]) * 1e10
    if quantum:
        u = u + dft_quantum_correction(species, t)
    return r, t, df['p'].values * 1e10, u


def dft_sigma_u(T, sig_lo=0.3, sig_hi=1.0, T0=2500.0, T1=5000.0):
    """sigma of the DFT internal-energy residual in units of R_s T, ramping
    from `sig_lo` below T0 to `sig_hi` above T1 (log-linear) because the
    intact-molecule quantum correction is wrong by up to ~1 R_s T where the
    fluid is dissociated (CH4 -> C + 2 H2 changes the zero-point energy by
    7300 K)."""
    T = np.atleast_1d(T).astype(float)
    f = np.clip(np.log(T / T0) / np.log(T1 / T0), 0.0, 1.0)
    return sig_lo + (sig_hi - sig_lo) * f


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


_SAT_CACHE = {}


def _maxwell_construction(pz, mu, rho_c, T, rho_lo=None, rho_hi=None, n=6000):
    """
    Vapour-liquid equilibrium of ONE isotherm of any Helmholtz surface, by the
    equal-chemical-potential construction (Maxwell): find rho'' < rho' with
    P(rho'') = P(rho') = P_sat and mu(rho'') = mu(rho').  `pz(rho)` must return
    P/(rho_c R_s T) and `mu(rho)` mu/(R_s T) up to a T-only constant (for the
    reference residual, ln delta + a^r + delta a^r_delta; for the fitted
    surface, (F + P/rho)/(R_s T)).  The isotherm is scanned on n log-spaced
    densities; the vapour spinodal is the first descent and the liquid root is
    the LAST crossing of P = P_sat from the dense side -- inside the dome the
    reference residual is a meaningless continuation (SW91's Eq. 5.3 dips to
    P = -9e4 rho_c R_s T at 0.13 g/cm^3 and T_t) with several extra crossings
    that must not be mistaken for the liquid.

    Returns dict(P, rho_l, rho_v, rho_sv, rho_sl, ok): P in units of
    rho_c R_s T.  `ok` is False when the isotherm has no loop or when P_sat
    lies below the lowest scanned pressure (then only rho_sl, the liquid
    spinodal, is meaningful).
    """
    from scipy.optimize import brentq
    lo = 1e-8 * rho_c if rho_lo is None else rho_lo
    hi = 3.5 * rho_c if rho_hi is None else rho_hi
    rr = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    with np.errstate(all='ignore'):
        P = pz(rr)
    dP = np.diff(P)
    out = dict(P=np.nan, rho_l=np.nan, rho_v=np.nan, rho_sv=np.nan, rho_sl=np.nan, ok=False)
    if not (dP < 0).any():
        return out
    i_max = int(np.argmax(dP < 0))                  # first descent: vapour spinodal
    j_min = int(np.flatnonzero(dP < 0)[-1]) + 1     # last descent ends: liquid spinodal
    out['rho_sv'], out['rho_sl'] = rr[i_max], rr[j_min]
    p_hi = P[i_max]
    p_floor = max(P[0], P[i_max:].min()) * (1.0 + 1e-9)
    p_ceil = p_hi * (1.0 - 1e-9)
    if not (p_ceil > p_floor):
        return out
    idx = np.arange(len(P))

    def roots(ps):
        rv = brentq(lambda r: pz(np.array([r]))[0] - ps, rr[0], rr[i_max],
                    xtol=1e-14, rtol=1e-13)
        j = idx[(P < ps) & (idx > i_max)][-1]
        rl = brentq(lambda r: pz(np.array([r]))[0] - ps, rr[j], rr[j + 1],
                    xtol=1e-14, rtol=1e-13)
        return rl, rv

    def dmu(ps):
        rl, rv = roots(ps)
        return mu(np.array([rl]))[0] - mu(np.array([rv]))[0]

    try:
        f_lo, f_hi = dmu(p_floor), dmu(p_ceil)
    except (ValueError, IndexError):
        return out
    if not (f_lo > 0 > f_hi):
        return out                                   # P_sat below the scan floor
    ps = brentq(dmu, p_floor, p_ceil, xtol=1e-16, rtol=1e-12)
    rl, rv = roots(ps)
    out.update(P=ps, rho_l=rl, rho_v=rv, ok=True)
    return out


def _saturation_at(species, T):
    """Saturation state of the REFERENCE EOS at one T < T_c (see
    `_maxwell_construction`).  Returns (P_sat [dyn/cm^2], rho_l, rho_v)."""
    it = IDEAL[species]
    rho_c = it.rho_red

    def pz(rho):
        _, Dar, _ = _reference_residual(species, np.atleast_1d(rho), T)
        return (np.atleast_1d(rho) / rho_c) * (1.0 + Dar)

    def mu(rho):
        ar, Dar, _ = _reference_residual(species, np.atleast_1d(rho), T)
        return np.log(np.atleast_1d(rho) / rho_c) + ar + Dar

    m = _maxwell_construction(pz, mu, rho_c, T)
    if not m['ok']:
        raise ValueError(f'no vapour-liquid equilibrium found for {species} at T = {T}')
    return (m['P'] * rho_c * it.R_s * T, m['rho_l'], m['rho_v'],
            m['rho_sv'], m['rho_sl'])


def saturation_states(species, T):
    """
    Saturation pressure and coexisting densities of the reference EOS on
    T in [T_t, T_c), tabulated once per species on 80 temperatures and
    interpolated (log P_sat, rho', rho'' against ln T).  Values outside the
    range are clipped to the end points.  For methane the construction
    reproduces the Setzmann & Wagner ancillary equations to < 0.1% in rho'
    and rho'' and < 0.2% in p_s over the whole range (checked in
    `validation/ch4_nh3_helmholtz_checks.py`).
    """
    it = IDEAL[species]
    T_c, T_t = it.T_red, TRIPLE_T[species]
    if species not in _SAT_CACHE:
        Tg = T_c - (T_c - T_t) * np.linspace(0.0, 1.0, 80) ** 1.5
        Tg = np.clip(Tg, T_t, 0.9995 * T_c)[::-1]
        out = np.array([_saturation_at(species, float(t)) for t in Tg])
        _SAT_CACHE[species] = (np.log(Tg), np.log(out[:, 0]), out[:, 1], out[:, 2],
                               out[:, 3], out[:, 4])
    lt, lp, rl, rv = _SAT_CACHE[species][:4]
    x = np.log(np.clip(np.asarray(T, dtype=float), np.exp(lt[0]), np.exp(lt[-1])))
    return np.exp(np.interp(x, lt, lp)), np.interp(x, lt, rl), np.interp(x, lt, rv)


def spinodal_states(species, T):
    """
    Vapour and liquid spinodal densities of the reference EOS, (rho_sv, rho_sl),
    interpolated on the same table as `saturation_states`.

    The spinodals, not the binodals, bound the region where an analytic
    Helmholtz surface MUST be unstable.  Between the binodal and the spinodal
    the fluid is metastable but locally stable: (dP/drho)_T > 0 and c_v > 0
    still hold, and on the vapour side P > 0 as well.  v3 up to 2026-09-07
    freed the whole binodal span instead, which at low temperature spans nearly
    the entire dilute range (NH3 at 200 K: 8.9e-5 to 0.73 g/cm^3), so the cold
    dilute fluid carried no stability constraint at all and the fitted Z fell
    to -30 there.
    """
    saturation_states(species, np.atleast_1d(T)[:1])       # ensure the cache
    lt = _SAT_CACHE[species][0]
    rsv, rsl = _SAT_CACHE[species][4], _SAT_CACHE[species][5]
    x = np.log(np.clip(np.asarray(T, dtype=float), np.exp(lt[0]), np.exp(lt[-1])))
    return np.interp(x, lt, rsv), np.interp(x, lt, rsl)


def in_two_phase(species, rho, T):
    """True inside the vapour-liquid dome of the reference EOS: T_t <= T < T_c
    and rho''(T) < rho < rho'(T).

    BELOW THE TRIPLE POINT this returns False everywhere.  There the
    equilibrium state is solid plus vapour, which this equation of state does
    not model at all; what it delivers instead is the metastable fluid
    continuation, and that continuation is a single stable phase which must
    carry the ordinary stability constraints.  Treating the sub-triple-point
    fluid as two-phase (v3 up to 2026-09-07) left it unconstrained over five
    decades of density and let P, u and c_v go negative there."""
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    T_c, T_t = IDEAL[species].T_red, TRIPLE_T[species]
    _, rl, rv = saturation_states(species, np.minimum(T, 0.9995 * T_c))
    return (T < T_c) & (T >= T_t) & (rho > rv) & (rho < rl)


def dome_pseudodata(species, n_T=14, n_rho=9, T_top=0.985):
    """
    Equilibrium two-phase pseudo-data inside the vapour-liquid dome: on each
    of n_T temperatures between the triple point and T_top T_c, n_rho
    densities log-spaced strictly inside (rho''(T), rho'(T)) carry
        P = P_sat(T),   u = x u'' + (1 - x) u',   s = x s'' + (1 - x) s',
    with the vapour mass fraction x from the lever rule
    1/rho = x/rho'' + (1 - x)/rho'.  This is the Maxwell construction of the
    reference EOS: F is linear in 1/rho across the dome (its convex hull), so
    (dP/drho)_T = 0 and the fitted surface is stable but not stiff there.
    v2 had no data at all in the dome and the fit's own van der Waals loop
    (P < 0 at 0.5 g/cm^3, 200-300 K for NH3) was removed by the constraints
    alone, whose reach extended into the reference/DFT seam.
    """
    it = IDEAL[species]
    T_c, T_t = it.T_red, TRIPLE_T[species]
    Tq = np.exp(np.linspace(np.log(T_t), np.log(T_top * T_c), n_T))
    ps, rl, rv = saturation_states(species, Tq)
    rows = []
    for T0, p0, r_l, r_v in zip(Tq, ps, rl, rv):
        ar, Dar, tau_ar_t = _reference_residual(species, np.array([r_l, r_v]), T0)
        u_lv = it.u(np.array([r_l, r_v]), T0) + it.R_s * T0 * tau_ar_t
        s_lv = it.s(np.array([r_l, r_v]), T0) + it.R_s * (tau_ar_t - ar)
        rq = np.exp(np.linspace(np.log(r_v), np.log(r_l), n_rho + 2))[1:-1]
        x = (1.0 / rq - 1.0 / r_l) / (1.0 / r_v - 1.0 / r_l)
        for r0, x0 in zip(rq, x):
            rows.append((r0, T0, p0, x0 * u_lv[1] + (1 - x0) * u_lv[0],
                         x0 * s_lv[1] + (1 - x0) * s_lv[0], x0))
    a = np.array(rows, dtype=float)
    return dict(rho=a[:, 0], T=a[:, 1], p=a[:, 2], u=a[:, 3], s=a[:, 4], x=a[:, 5])


def _sat_interp(sat, T):
    """(P_sat, rho_l_star, rho_v, T_c) from a coexistence table `sat` (as built by
    `eos_saturation`) at temperatures T: P_sat and rho'' interpolated in ln T
    where the vapour branch exists (0 / nan elsewhere and for T >= T_c); the
    liquid bracket rho_l* from the NEAREST tabulated isotherm, one per cent
    denser (it jumps where the loop changes shape, and an interpolated value
    can sit inside a wiggle with P < 0)."""
    T = np.asarray(T, dtype=float)
    lt = np.log(sat['T'])
    x = np.log(np.clip(T, sat['T'][0], sat['T'][-1]))
    ok = np.isfinite(sat['P'])
    P = np.zeros(T.shape)
    rv = np.full(T.shape, np.nan)
    if ok.any():
        live = (T >= sat['T'][ok][0]) & (T < sat['T_c'])
        P = np.where(live, np.exp(np.interp(x, lt[ok], np.log(sat['P'][ok]))), 0.0)
        rv = np.where(live, np.exp(np.interp(x, lt[ok], np.log(sat['rho_v'][ok]))), np.nan)
    j = np.clip(np.rint(np.interp(x, lt, np.arange(lt.size))).astype(int), 0, lt.size - 1)
    rl = np.where(T < sat['T_c'], 1.01 * sat['rho_l_star'][j], np.nan)
    return P, rl, rv, sat['T_c']


def two_phase_region(species, rho, T, dilate_v=0.9, dilate_l=1.05, crit_band=0.03,
                     prior_sat=None):
    """
    The region where NO stability constraint is imposed and no certification
    is claimed: the reference EOS's vapour-liquid dome, dilated by `dilate_v`
    on the vapour side and `dilate_l` on the liquid side (the fitted surface's
    own loop reaches slightly beyond the reference coexistence curve), plus a
    band |T/T_c - 1| < crit_band, 0.4 < rho/rho_c < 2.5 around the critical
    point where Gamma -> 0 and no floor can be met.  Inside it the fitted
    surface is an analytic continuation with a van der Waals loop -- exactly
    as the reference equations themselves behave there -- and the (P,T)
    inversion resolves it by branch selection on the fit's own coexistence
    curve (`eos_saturation`, `ch4_nh3_invert.march_pt`).  Imposing even bare
    positivity inside the dome deformed the surface into the reference/DFT
    seam (NH3 reference energies 7 R_s T off, DFT pressures at 1000 K 60%
    off); with the region free the same fit reproduces the reference to
    0.2 R_s T and the DFT to 6-20%.
    """
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    it = IDEAL[species]
    T_c, rho_c, T_t = it.T_red, it.rho_red, TRIPLE_T[species]
    Tq = np.minimum(T, 0.9995 * T_c)
    _, rl, rv = saturation_states(species, Tq)
    rsv, rsl = spinodal_states(species, Tq)
    # Vapour side: free only ABOVE the vapour spinodal.  Between rho'' and
    # rho_sv the supersaturated vapour is metastable but locally stable, it is
    # where the cold dilute fluid lives, and the reference equations describe
    # it well (Z from 0.34 to 0.99, (dP/drho)_T > 0 throughout, measured).
    # Liquid side: free out to rho', because the stretched liquid between
    # rho_sl and rho' is under genuine tension (reference Z down to -2.6) and
    # a positivity floor there would be wrong.
    # Above T_t only: below the triple point there is no coexistence to respect.
    dome = ((T < T_c) & (T >= T_t)
            & (rho > dilate_v * rsv) & (rho < dilate_l * rl))
    band = (np.abs(T / T_c - 1.0) < crit_band) & (rho > 0.4 * rho_c) & (rho < 2.5 * rho_c)
    if prior_sat is not None:
        # second pass: only what BOTH the reference dome and a previous fit's
        # own loop call two-phase is left unconstrained, so that every state
        # the tables can contain (outside the fit's loop) is certified
        _, rl_f, rv_f, T_cf = _sat_interp(prior_sat, T)
        rv_f = np.where(np.isfinite(rv_f), rv_f, 0.0)
        rl_f = np.where(np.isfinite(rl_f), rl_f, 0.0)
        own = (T < T_cf) & (rho > 0.7 * rv_f) & (rho < rl_f / 1.01)
        dome = dome & own
    return dome | band


def eos_saturation(eos, n_T=120, T_lo=None, verbose=False):
    """
    Coexistence curve of the FITTED surface: its critical temperature (the
    highest T with a loop, by bisection between 0.9 and 1.15 T_c,ref), and on
    n_T temperatures below it P_sat, rho', rho'' by the Maxwell construction
    applied to the fit's own P and mu = F + P/rho.  Where P_sat falls below
    the box's lowest pressure P(rho_lo, T) (cold liquids: NH3 below ~230 K)
    the vapour branch is outside the table and only the liquid bracket
    rho_l* -- the last density from the dense side at which P = P(rho_lo, T)
    -- is stored.  Stored on `eos.sat` and in the cache; used by the (P,T)
    inversion to stay on one monotone branch, and reported against the
    reference coexistence curve as a validation of the fit in the dome.
    """
    it = IDEAL[eos.species]
    rho_c, R_s = it.rho_red, it.R_s
    r_lo, r_hi = float(np.exp(eos.lr0)), float(np.exp(eos.lr1))
    # never below the triple point: there is no vapour-liquid equilibrium there,
    # and after the 2026-09-08 repair the surface is a constrained single phase
    T_t = TRIPLE_T[eos.species]
    T_lo = float(np.exp(eos.lt0)) if T_lo is None else T_lo
    T_lo = max(T_lo, T_t)

    def construct(T):
        def pz(rho):
            return eos.p(np.atleast_1d(rho), np.full(np.atleast_1d(rho).shape, T)) / (rho_c * R_s * T)

        def mu(rho):
            rho = np.atleast_1d(rho)
            t = np.full(rho.shape, T)
            return (eos.free_energy(rho, t) + eos.p(rho, t) / rho) / (R_s * T)
        return _maxwell_construction(pz, mu, rho_c, T, rho_lo=r_lo, rho_hi=min(3.5 * rho_c, r_hi))

    def has_loop(T):
        return np.isfinite(construct(T)['rho_sv'])

    # critical temperature of the fit
    a, b = 0.9 * it.T_red, 1.15 * it.T_red
    if not has_loop(a):
        T_c = a
    elif has_loop(b):
        T_c = b
    else:
        for _ in range(30):
            m = 0.5 * (a + b)
            if has_loop(m):
                a = m
            else:
                b = m
        T_c = a
    Tg = np.exp(np.linspace(np.log(T_lo), np.log(0.999 * T_c), n_T))
    P = np.full(n_T, np.nan); rl = np.full(n_T, np.nan); rv = np.full(n_T, np.nan)
    rl_star = np.full(n_T, np.nan)
    P_fit = np.full(n_T, np.nan); rl_fit = np.full(n_T, np.nan); rv_fit = np.full(n_T, np.nan)
    for i, T in enumerate(Tg):
        m = construct(float(T))
        # liquid bracket: above the fit's own liquid spinodal (the last density
        # at which dP/drho < 0), and above rho' (or, where the vapour branch is
        # below the box, the last crossing from the dense side of P = P(rho_lo, T));
        # on [rho_l*, rho_hi] the isotherm is then strictly increasing
        rr = np.exp(np.linspace(np.log(r_lo), np.log(min(3.5 * rho_c, r_hi)), 6000))
        pp = eos.p(rr, np.full(rr.shape, float(T)))
        desc = np.flatnonzero(np.diff(pp) < 0)
        rho_sl = rr[int(desc[-1]) + 1] if desc.size else r_lo
        # THE BRANCH BOUNDARY MUST COME FROM THE FIT'S OWN SURFACE, not from the
        # reference coexistence curve.  Taking P_sat from the reference was tried
        # (2026-09-08) and broke the rho -> P -> rho round trip by 0.89 in ln rho:
        # where the reference's P_sat lies above the peak of the fit's own loop,
        # the vapour branch has no root at that pressure at all.  Self-consistency
        # of the delivered tables outranks agreement with the reference dome, and
        # the reference dome is reported separately (P_fit, rho_l_fit, rho_v_fit
        # below, and stage S2 of the validation suite).
        P_fit[i] = m['P'] * rho_c * R_s * T if m['ok'] else np.nan
        rl_fit[i] = m['rho_l'] if m['ok'] else np.nan
        rv_fit[i] = m['rho_v'] if m['ok'] else np.nan
        if m['ok']:
            P[i], rl[i], rv[i] = m['P'] * rho_c * R_s * T, m['rho_l'], m['rho_v']
            rl_star[i] = max(m['rho_l'], rho_sl)
            rv[i] = min(rv[i], m['rho_sv'])
        else:
            below = np.flatnonzero(pp < pp[0])
            j = int(below[-1]) if below.size else 0
            rl_star[i] = max(rr[min(j + 1, rr.size - 1)], rho_sl)
        # never hand the solver a bracket end with P <= 0: walk up to the first
        # scanned density above rl_star where P > 0
        k = int(np.searchsorted(rr, rl_star[i]))
        while k < rr.size - 1 and pp[k] <= 0:
            k += 1
        rl_star[i] = max(rl_star[i], rr[k])
    if verbose:
        ok = np.isfinite(P)
        okf = np.isfinite(P_fit)
        print(f'  [{eos.species}] fit coexistence: T_c = {T_c:.2f} K (reference {it.T_red:.2f});'
              f' the fit resolves its own vapour root for T >= '
              f'{Tg[okf].min() if okf.any() else np.nan:.1f} K'
              + ('' if okf.any() else ' -- NO LOOP ANYWHERE: the surface is single phase'
                                      ' over the whole box, see the limitations'))
    eos.sat = dict(T=Tg, P=P, rho_l=rl, rho_v=rv, rho_l_star=rl_star, T_c=float(T_c),
                   T_t=float(T_t), P_fit=P_fit, rho_l_fit=rl_fit, rho_v_fit=rv_fit)
    return eos.sat


def reference_pseudodata(species, n_rho=64, n_T=22, rho_max=1.2, phase_cuts=True,
                         return_mask=False, gamma_max=8.0):
    """
    P, U, S from the reference EOS on a grid inside its validity domain, in OUR
    gauge.  Points with P <= 0 or (dP/drho)_T <= 0 are rejected.

    That filter alone is NOT enough for methane, whose reference range starts
    below T_c: it keeps the metastable branches inside the vapour-liquid dome
    (where SW91's Eq. 5.3 is a smooth continuation with no physical meaning;
    measured u/(R_s T) = -169, s/R_s = -148 at rho = 0.176 g/cm^3, 136 K) and
    it keeps states beyond the melting line (the rho = 0.527 isochore below
    163 K).  With `phase_cuts` (v3 default) both are removed using the SW91
    auxiliary equations.  The P <= 1000 MPa cap means the grid reaches only
    rho ~0.53 (CH4) / 0.69 (NH3) g/cm^3, not `rho_max`.
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
    if phase_cuts:
        # Cut only what the fit leaves unconstrained (`two_phase_region`): from
        # the vapour spinodal to the saturated liquid density, above T_t.  The
        # metastable supersaturated vapour between rho'' and rho_sv is KEPT:
        # it is locally stable, the reference equations are smooth and sane
        # there (measured Z 0.34-0.99, u 0.96-2.99 R_s T, (dP/drho)_T > 0), and
        # it is the only data in the cold dilute band that the fit would
        # otherwise have to invent.  Also cut the melting line (methane;
        # SW91 Eq. 3.7) and a narrow band around the critical point where
        # Gamma -> 0 and no floor can be satisfied.
        good &= ~two_phase_region(species, r, t, crit_band=0.0)
        good &= ~((np.abs(t / it.T_red - 1.0) < 0.03)
                  & (r > 0.4 * it.rho_red) & (r < 2.5 * it.rho_red))
        if species == 'methane':
            from eos import ch4_setzmann_eos as m
            good &= p <= m.melting_pressure(t)
        # the stiff cold liquid next to the saturation line: within ~0.2 in
        # ln rho of rho'(T) the pressure rises from P_sat ~ 1e-3 rho R_s T to
        # ~1 GPa, i.e. Gamma = K_T/P of 10-40, which a cubic B-spline with
        # knots 0.3 apart in ln rho cannot follow (the unconstrained fit
        # reproduces these points only with c_v wiggles of 40-70%, and the
        # constrained fit's misfit there propagated into the reference/DFT
        # seam).  These states (liquid below ~0.3 GPa) play no role in a
        # planetary interior; the fit is a stable interpolant there.
        if gamma_max is not None:
            good &= (r * dpdr / p) <= gamma_max
    if return_mask:
        return r, t, p, u, s, good
    return r[good], t[good], p[good], u[good], s[good]


def reference_valid(species, rho, T, phase_cuts=True, gamma_max=8.0):
    """
    Where the reference equation of state is used as data: the exact predicate
    `reference_pseudodata` applies, evaluated at arbitrary (rho, T).

    Exists so that anything else which draws or quotes the reference -- the
    figures above all -- applies the SAME cuts the fit was trained under.
    Figure 2 previously tested only T-range, P > 0 and P <= P_max, and so drew
    the reference across states the fit was never asked to reproduce: the whole
    CH4 rho = 0.6 g/cm^3 curve lies above SW91's own melting pressure (solid
    methane), and NH3 at 0.736 g/cm^3 below 325 K has Gamma_ref up to 24.5.
    Those stretches then read as fit-versus-reference disagreement when they
    are nothing of the kind.
    """
    it = IDEAL[species]
    lim = REF_LIMITS[species]
    r = np.atleast_1d(np.asarray(rho, dtype=float))
    t = np.atleast_1d(np.asarray(T, dtype=float))
    r, t = np.broadcast_arrays(r, t)
    with np.errstate(all='ignore'):
        _, Dar, _ = _reference_residual(species, r, t)
        p = r * it.R_s * t * (1.0 + Dar)
        h = 1e-5
        _, Dp, _ = _reference_residual(species, r * (1 + h), t)
        _, Dm, _ = _reference_residual(species, r * (1 - h), t)
        dpdr = ((1 + h) * (1.0 + Dp) - (1 - h) * (1.0 + Dm)) / (2 * h) * it.R_s * t
    good = (np.isfinite(p) & (p > 0) & (dpdr > 0)
            & (t >= lim['T_lo']) & (t <= lim['T_hi']) & (p <= lim['P_max']))
    if phase_cuts:
        good &= ~two_phase_region(species, r, t, crit_band=0.0)
        good &= ~((np.abs(t / it.T_red - 1.0) < 0.03)
                  & (r > 0.4 * it.rho_red) & (r < 2.5 * it.rho_red))
        if species == 'methane':
            from eos import ch4_setzmann_eos as m
            good &= p <= m.melting_pressure(t)
        if gamma_max is not None:
            good &= (r * dpdr / p) <= gamma_max
    return good


def reference_cv(species, rho, T, h=1e-4):
    """c_v of the reference EOS in our gauge, by central differences of u in T
    (the reference modules do not expose alpha^r_tautau with array broadcasting).
    u = u_ideal(ours) + R_s T tau alpha^r_tau is analytic, so the O(h^2) error
    at h = 1e-4 is ~1e-8 relative."""
    it = IDEAL[species]
    rho = np.atleast_1d(np.asarray(rho, float))
    T = np.atleast_1d(np.asarray(T, float))
    out = []
    for sgn in (1.0, -1.0):
        t = T * (1.0 + sgn * h)
        _, _, tau_ar_t = _reference_residual(species, rho, t)
        out.append(it.u(rho, t) + it.R_s * t * tau_ar_t)
    return (out[0] - out[1]) / (2.0 * h * T)


def reference_gamma(species, rho, T, h=1e-5):
    """Gamma = (dlnP/dlnrho)_T of the reference EOS, by central differences."""
    it = IDEAL[species]
    rho, T = np.atleast_1d(rho), np.atleast_1d(T)
    with np.errstate(all='ignore'):
        _, Dp, _ = _reference_residual(species, rho * (1 + h), T)
        _, Dm, _ = _reference_residual(species, rho * (1 - h), T)
        pp = (1 + h) * (1.0 + Dp)
        pm = (1 - h) * (1.0 + Dm)
        _, D0, _ = _reference_residual(species, rho, T)
        p0 = 1.0 + D0
    return (pp - pm) / (2 * h) / p0


def reference_second_virial(species, T):
    """B(T) of the reference EOS [cm^3/g], from the residual at delta -> 0."""
    it = IDEAL[species]
    T = np.atleast_1d(T).astype(float)
    rho0 = 1.0e-7
    _, Dar, _ = _reference_residual(species, np.full(T.size, rho0), T)
    return Dar / rho0


# ---------------------------------------------------------------------------
# DFT isochore fits: the cold intercept, the thermal-pressure slope, and the
# (quantum-corrected) energy at the 1000 K junction.
# ---------------------------------------------------------------------------
def dft_isochore_fits(species, T_lo=1000.0, T_hi=6000.0, quantum=True):
    """Per DFT isochore: rho, P_cold intercept, slope b = dP/dT, P(1000 K) on
    the line, u(1000 K) from a linear fit of u over [1000, 2000] K, and the
    point count.  Least squares of P = P_cold + b T over [T_lo, T_hi] is
    linear to a few percent on every isochore (Part II of the lab note)."""
    r, t, p, u = load_dft(species, quantum=quantum)
    out = []
    for r0 in np.unique(r):
        m = np.isclose(r, r0) & (t >= T_lo) & (t <= T_hi)
        m2 = np.isclose(r, r0) & (t <= 2000.0)
        if m.sum() < 3 or m2.sum() < 2:
            continue
        b, p_cold = np.polyfit(t[m], p[m], 1)
        cu, u0 = np.polyfit(t[m2], u[m2], 1)
        out.append((r0, p_cold, b, p_cold + b * 1000.0, u0 + cu * 1000.0, int(m.sum())))
    a = np.array(out, dtype=float)
    return dict(rho=a[:, 0], p_cold=a[:, 1], b=a[:, 2], p_1000=a[:, 3],
                u_1000=a[:, 4], n=a[:, 5].astype(int))


# ---------------------------------------------------------------------------
# The structural cold curve
# ---------------------------------------------------------------------------
class ColdCurve:
    """
    T-independent cold curve u_cold(rho) added to F, with P_cold = rho^2 u_cold'.

    Construction (all parameters fixed BEFORE the fit, so every fitted quantity
    stays affine in the coefficients):

      * a Vinet form P_V(rho; rho_0, K_0, K_0') fitted in ln space to the DFT
        isochore intercepts (Vinet et al. 1987), scaled by `scale` (0.97) so
        that no DFT point lies below it -- the isochores are slightly convex,
        so a raw intercept overshoots P(T) by up to 0.9%;
      * a C^inf onset S(rho) = [1 + tanh((ln rho - ln rho_c)/w)]/2 rising over
        [rho_a, rho_b], placed ABOVE the highest reference density, so that the
        reference liquid sees no cold pressure and P_cold >= 0 everywhere
        (the ideal gas + cold curve, theta = 0, must stay strictly feasible for
        every stability floor: a tension region would put Z(0) ~ -40 at 50 K).
        The Vinet rho_0 (0.33 CH4, 0.57 NH3) lies below the onset, so the
        product is taken only for rho >= rho_0 and is zero below;
      * above the DFT top density rho_top a continuation with
        Gamma_c(rho) = 5/3 + (Gamma(rho_top) - 5/3) sqrt(rho_top/rho), anchored to
        the Vinet's own value and slope at rho_top (C^1) and blended C^2 over
        [0.9, 1.1] rho_top.  5/3 is the free-electron limit; the v2
        `highdensity_pseudodata` used the same relaxation but, applied per
        isotherm, it made hot isotherms cross cold ones (F19).

    Representation: the exact integral u_cold(x) = int P_cold/rho dx (x = ln rho)
    is computed by cumulative Simpson on a dense grid and stored as a k = 5
    interpolating B-spline; P_cold = rho u', dP_cold/drho = u' + u'' are the
    spline's own derivatives, so P = rho^2 dF/drho, the Maxwell relation and the
    Legendre identity hold to machine precision and Gamma is C^2 everywhere (a
    differentiated cubic spline would leave kinks at every knot).
    """

    def __init__(self, species, rho0, K0, K0p, rho_a, rho_b, rho_top,
                 scale=0.97, gamma_inf=5.0 / 3.0, blend=(0.9, 1.1),
                 onset_edge=0.98, x_lo=np.log(1e-6), x_hi=np.log(300.0), n=24001):
        self.species = species
        self.rho0, self.K0, self.K0p = float(rho0), float(K0), float(K0p)
        self.rho_a, self.rho_b, self.rho_top = float(rho_a), float(rho_b), float(rho_top)
        self.scale, self.gamma_inf = float(scale), float(gamma_inf)
        self.blend = (float(blend[0]), float(blend[1]))
        self.onset_edge = float(onset_edge)
        self._x_lo, self._x_hi, self._n = float(x_lo), float(x_hi), int(n)
        self._build()

    # -- the analytic model pieces -----------------------------------------
    def _vinet(self, rho):
        rho = np.asarray(rho, dtype=float)
        x = (self.rho0 / np.maximum(rho, 1e-300)) ** (1.0 / 3.0)
        return (3.0 * self.K0 * (1.0 - x) / x ** 2
                * np.exp(1.5 * (self.K0p - 1.0) * (1.0 - x)))

    def _onset(self, rho):
        lc = 0.5 * (np.log(self.rho_a) + np.log(self.rho_b))
        w = 0.5 * np.log(self.rho_b / self.rho_a) / np.arctanh(self.onset_edge)
        return 0.5 * (1.0 + np.tanh((np.log(np.asarray(rho, float)) - lc) / w))

    def _low(self, rho):
        """scale * Vinet * onset for rho >= rho_0, zero below."""
        rho = np.asarray(rho, dtype=float)
        return np.where(rho > self.rho0, self.scale * self._vinet(rho) * self._onset(rho), 0.0)

    def _low_gamma(self, rho, h=1e-6):
        rho = np.asarray(rho, dtype=float)
        return (np.log(self._low(rho * (1 + h))) - np.log(self._low(rho * (1 - h)))) / (2 * h)

    def _cont(self, rho):
        """Gamma -> gamma_inf continuation above rho_top, C^1 at rho_top."""
        rho = np.asarray(rho, dtype=float)
        L = np.log(np.maximum(rho, self.rho_top) / self.rho_top)
        g_top = float(self._low_gamma(self.rho_top))
        integral = self.gamma_inf * L + 2.0 * (g_top - self.gamma_inf) * (1.0 - np.exp(-0.5 * L))
        return float(self._low(self.rho_top)) * np.exp(integral)

    def p_model(self, rho):
        """The analytic cold pressure the spline is built from [dyn/cm^2]."""
        rho = np.asarray(rho, dtype=float)
        lo, hi = self.blend[0] * self.rho_top, self.blend[1] * self.rho_top
        t = np.clip((np.log(rho) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0)
        w = t * t * t * (t * (6.0 * t - 15.0) + 10.0)          # C^2 smoothstep
        return (1.0 - w) * self._low(rho) + w * self._cont(rho)

    # -- the spline representation -------------------------------------------
    def _build(self):
        x = np.linspace(self._x_lo, self._x_hi, self._n)
        rho = np.exp(x)
        f = self.p_model(rho) / rho                            # du/dx = P/rho
        u = _cumsimp(f, x=x, initial=0.0)
        self._u_spl = _mkspl(x, u, k=5)
        self._du_spl = self._u_spl.derivative(1)
        self._d2u_spl = self._u_spl.derivative(2)

    def u(self, rho):
        """u_cold [erg/g]; zero for rho below the onset."""
        return self._u_spl(np.log(np.asarray(rho, dtype=float)))

    def p(self, rho):
        """P_cold = rho^2 du/drho = rho du/dx [dyn/cm^2]."""
        rho = np.asarray(rho, dtype=float)
        return rho * self._du_spl(np.log(rho))

    def dpdrho(self, rho):
        """dP_cold/drho = u'(x) + u''(x) [dyn cm/g]."""
        x = np.log(np.asarray(rho, dtype=float))
        return self._du_spl(x) + self._d2u_spl(x)

    def bulk_modulus(self, rho):
        """K_cold = rho dP_cold/drho [dyn/cm^2]."""
        rho = np.asarray(rho, dtype=float)
        return rho * self.dpdrho(rho)

    def gamma(self, rho):
        """Gamma_cold = dlnP_cold/dlnrho (nan where P_cold = 0)."""
        rho = np.asarray(rho, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return rho * self.dpdrho(rho) / self.p(rho)

    # -- persistence -------------------------------------------------------------
    @property
    def params(self):
        return dict(species=self.species, rho0=self.rho0, K0=self.K0, K0p=self.K0p,
                    rho_a=self.rho_a, rho_b=self.rho_b, rho_top=self.rho_top,
                    scale=self.scale, gamma_inf=self.gamma_inf, blend=list(self.blend),
                    onset_edge=self.onset_edge, x_lo=self._x_lo, x_hi=self._x_hi,
                    n=self._n)

    @classmethod
    def from_params(cls, p):
        p = dict(p)
        return cls(p.pop('species'), p.pop('rho0'), p.pop('K0'), p.pop('K0p'),
                   p.pop('rho_a'), p.pop('rho_b'), p.pop('rho_top'), **p)

    @classmethod
    def from_species(cls, species, scale=0.97, onset=None, quantum=True, **kw):
        """Fit the Vinet form to the DFT isochore intercepts and build.

        `onset` is (rho_a, rho_b).  The defaults switch the Vinet on across
        its own rho_0 (0.33 CH4, 0.57 NH3) so that it carries the full T -> 0
        pressure of every DFT isochore (within 5%; the first v3 onsets,
        [0.45, 1.30] and [0.60, 1.50], suppressed it at 0.6-0.8 g/cm^3 and
        left 2 GPa of cold pressure for the spline to supply as a steep
        low-temperature rise, which is what the Grueneisen ceiling then
        fought).  At the reference's top density the Vinet exceeds the cold
        liquid's pressure (1.0 vs 0.44 GPa at 0.527 g/cm^3, 178 K for CH4): the
        spline's thermal part is negative there, which is allowed.
        """
        fits = dft_isochore_fits(species, quantum=quantum)
        m = fits['p_cold'] > 0
        rr, pc = fits['rho'][m], fits['p_cold'][m]

        def resid(q):
            rho0, K0, K0p = q
            x = (rho0 / rr) ** (1.0 / 3.0)
            pv = 3.0 * K0 * (1.0 - x) / x ** 2 * np.exp(1.5 * (K0p - 1.0) * (1.0 - x))
            return np.log(np.maximum(pv, 1e-30)) - np.log(pc)

        q0 = np.array([0.6 * rr.min(), 1.0e10, 6.0])
        sol = _lsq(resid, q0, bounds=([0.05 * rr.min(), 1e7, 2.0],
                                      [0.99 * rr.min(), 1e13, 15.0]))
        rho0, K0, K0p = sol.x
        if onset is None:
            # A steep onset ([0.55, 0.90] for CH4) gave P_cold a local Gamma of 14
            # between 0.6 and 0.8 g/cm^3 against a physical 3.3, and the spline
            # rang trying to cancel it (+86% at 0.8 g/cm^3, 1000 K).  Over a
            # factor ~2.5-2.9 in density the onset's stiffness stays comparable to
            # the fluid's, and the cold pressure at the highest kept reference
            # density is still <10% of the reference pressure there.
            # The onset must sit ABOVE the densest kept reference point, or the
            # cold pressure exceeds the reference pressure there and the guard
            # P >= 0.5 P_cold fights the data.  Admitting the metastable
            # vapour band (2026-09-08) raised the NH3 ceiling to 0.797 g/cm^3,
            # so its onset moved from 0.55 to 0.85; CH4 reaches only 0.529.
            onset = {'methane': (0.42, 0.71), 'ammonia': (0.85, 1.30)}[species]
        cc = cls(species, rho0, K0, K0p, onset[0], onset[1], float(rr.max()),
                 scale=scale, **kw)
        cc.fit_rms_ln = float(np.sqrt(np.mean(sol.fun ** 2)))
        cc.fit_points = (rr, pc)
        return cc


# ---------------------------------------------------------------------------
# Debye function and the quasi-harmonic thermal anchor
# ---------------------------------------------------------------------------
_DEBYE_Y = np.exp(np.linspace(np.log(1e-4), np.log(2e3), 400))
_DEBYE_TAB = None


def debye_energy_function(y):
    """D_E(y) = (3/y^3) int_0^y t^3/(e^t - 1) dt, so that E = 3 N k T D_E(theta/T)
    for a Debye solid.  Tabulated once by quadrature and interpolated in
    log y; series limits outside the table."""
    global _DEBYE_TAB
    if _DEBYE_TAB is None:
        vals = np.empty_like(_DEBYE_Y)
        for i, yy in enumerate(_DEBYE_Y):
            vals[i] = 3.0 / yy ** 3 * _quad(lambda t: t ** 3 / np.expm1(t), 0.0, yy,
                                          limit=200)[0]
        _DEBYE_TAB = _mkspl(np.log(_DEBYE_Y), vals, k=3)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    out = np.empty_like(y)
    lo = y < _DEBYE_Y[0]
    hi = y > _DEBYE_Y[-1]
    mid = ~(lo | hi)
    out[lo] = 1.0 - 3.0 * y[lo] / 8.0 + y[lo] ** 2 / 20.0
    out[hi] = np.pi ** 4 / 5.0 / y[hi] ** 3
    out[mid] = _DEBYE_TAB(np.log(y[mid]))
    return out


# hbar/k_B [K s], Avogadro [1/mol]
_HBAR_OVER_KB = 7.638232e-12
_N_AVOGADRO = 6.02214076e23


def debye_temperature(cold, rho, M, poisson_factor=0.85):
    """theta_D(rho) from the cold-curve bulk modulus [K].

    theta_D = (hbar/k_B) (6 pi^2 n)^{1/3} v_m with n the MOLECULE number density
    and v_m ~ 0.85 sqrt(K/rho) a Debye mean sound speed (the factor covers the
    longitudinal/transverse average for a Poisson ratio near 0.3).  This ties
    the anchor's thermal model to the same cold curve that carries its
    pressure; it is a model, documented as such.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    n = rho * _N_AVOGADRO / M
    K = np.maximum(cold.bulk_modulus(rho), 0.0)
    v = poisson_factor * np.sqrt(K / rho)
    return _HBAR_OVER_KB * (6.0 * np.pi ** 2 * n) ** (1.0 / 3.0) * v


def anchor_pseudodata(species, cold, T_lo=50.0, T_junction=1000.0, T_hi=30000.0,
                      rho_hi=60.0, n_T_low=8, n_rho_ext=8, n_T_ext=10, quantum=True,
                      gamma_int=0.0):
    """
    Quasi-harmonic (Mie-Grueneisen / Debye) pseudo-data below the DFT
    temperature floor and above the DFT density ceiling.  Replaces both the
    v2 `lowT_pseudodata` (P linear in T down to 50 K, which violates the third
    law and gave Grueneisen parameters up to 182) and `highdensity_pseudodata`
    (per-isotherm continuation whose isotherms crossed).

    On every DFT isochore rho_j (and on log-spaced extended isochores up to
    rho_hi, using the continued cold curve):

        P(rho,T) = P_cold(rho) + rho gamma_latt(rho) 6 R_s T D_E(theta_D/T)
                                + rho gamma_int E_int(T)
        U(rho,T) = u_ref(rho) + [E_th(rho,T) - E_th(rho_j, T_junction)]

    with E_th = 6 R_s T D_E(theta_D/T) + E_int(T) (six external modes Debye,
    internal modes Planck-Einstein), gamma_latt = b/(6 rho R_s) from the DFT
    slope b(rho) (held at its rho_top value beyond the DFT range), gamma_int = 0
    so that the uncorrected DFT pressures stay consistent at the junction, and
    u_ref the quantum-corrected DFT energy at (rho_j, T_junction) -- i.e. the
    energy is RELATIVE to the DFT gauge and rides on the same Delta_u column,
    never an absolute claim (the cold curve carries no cohesive energy).  As
    T -> 0 both dP/dT and c_v vanish, so the entropy stays finite.

    Returned sigma_P is per row: max(0.10, 3 (1 - P_cold/P_1000)), so isochores
    whose pressure is more than 10% thermal at 1000 K make no absolute claim on
    the cold pressure (the intercept of a hot linear fit is not a measurement
    of it).

    Known limitation: theta_D reaches ~3000 K at rho_top, so the anchor's
    lattice modes are largely frozen at 1000 K where the classical DFT has them
    fully excited; the c_v mismatch at the junction is 30-50% for rho >= 3.
    """
    it = IDEAL[species]
    R_s, M = it.R_s, it.M
    fits = dft_isochore_fits(species, quantum=quantum)
    rho_j, b_j, p1000_j, u1000_j = fits['rho'], fits['b'], fits['p_1000'], fits['u_1000']
    rho_top = float(rho_j.max())
    # the DFT pressure AT the junction where a 1000 K point exists (the hot
    # linear fit misses it by a factor 2 on the lowest NH3 isochore, whose
    # isochore is strongly convex); the anchor is normalised to it
    r_d, t_d, p_d, _ = load_dft(species, quantum=quantum)
    pj_data = np.array([p_d[np.isclose(r_d, r0) & np.isclose(t_d, T_junction)][0]
                        if np.any(np.isclose(r_d, r0) & np.isclose(t_d, T_junction)) else p1
                        for r0, p1 in zip(rho_j, p1000_j)])

    def gamma_latt(rho):
        rho = np.atleast_1d(rho)
        g = np.interp(rho, rho_j, b_j / (6.0 * rho_j * R_s))
        return np.where(rho > rho_top, b_j[-1] / (6.0 * rho_top * R_s), g)

    def e_th(rho, T):
        th = debye_temperature(cold, rho, M)
        return 6.0 * R_s * T * debye_energy_function(th / T) + e_intra(species, T)

    def p_th(rho, T):
        th = debye_temperature(cold, rho, M)
        return rho * (gamma_latt(rho) * 6.0 * R_s * T * debye_energy_function(th / T)
                      + gamma_int * e_intra(species, T))

    def p_th_shape(rho, T):
        # T D_E(theta_D/T): the lattice thermal-pressure shape, -> T at high T
        th = debye_temperature(cold, rho, M)
        return T * debye_energy_function(th / T)

    rows = []
    # (a) below the DFT temperature floor on the DFT isochores.  The thermal
    # pressure is (P_DFT(1000 K) - P_cold) scaled by the Debye shape, so the
    # anchor reproduces the DFT at the junction BY CONSTRUCTION and only its
    # T -> 0 behaviour (dP/dT -> 0) is a model.
    T_low = np.exp(np.linspace(np.log(T_lo), np.log(T_junction), n_T_low, endpoint=False))
    for r0, p1000, pj, u1000 in zip(rho_j, p1000_j, pj_data, u1000_j):
        pc = float(cold.p(r0))
        # the model is trusted in proportion to how cold-dominated the isochore is:
        # sigma of the thermal-pressure residual (in units of the junction thermal
        # pressure) is 0.15 where P_cold carries the pressure and 0.65 where the
        # pressure is all thermal (rho = 0.6 g/cm^3 for CH4), where a linear
        # thermal pressure down to 50 K is a guess that must not fight the
        # reference liquid next to it
        sig_p = 0.15 + 0.5 * max(0.0, 1.0 - pc / pj)
        e_junc = float(e_th(r0, np.array([T_junction]))[0])
        shape_j = float(p_th_shape(r0, np.array([T_junction]))[0])
        for Tq in T_low:
            p_row = pc + (pj - pc) * float(p_th_shape(r0, np.array([Tq]))[0]) / shape_j
            rows.append((r0, Tq, p_row,
                         u1000 + float(e_th(r0, np.array([Tq]))[0]) - e_junc, sig_p, 0,
                         pj - pc, u1000))
    # (b) above the DFT density ceiling, all temperatures
    u_top = u1000_j[-1]
    e_top = float(e_th(rho_top, np.array([T_junction]))[0])
    c_top = u_top - float(cold.u(rho_top)) - e_top
    rho_ext = np.exp(np.linspace(np.log(rho_top * 1.15), np.log(rho_hi), n_rho_ext))
    T_ext = np.exp(np.linspace(np.log(T_lo), np.log(T_hi), n_T_ext))
    for r0 in rho_ext:
        pc = float(cold.p(r0))
        pth_j = float(p_th(r0, np.array([T_junction]))[0])
        for Tq in T_ext:
            # normalise by the anchor's own thermal pressure at this T once above
            # the junction (a fixed 1000 K normaliser made these model rows 20x
            # tighter than the DFT at 20000 K and pulled the hottest isotherm down)
            pth_here = max(pth_j, float(p_th(r0, np.array([Tq]))[0]))
            rows.append((r0, Tq, pc + float(p_th(r0, np.array([Tq]))[0]),
                         float(cold.u(r0)) + float(e_th(r0, np.array([Tq]))[0]) + c_top,
                         0.15, 1, pth_here,
                         float(cold.u(r0)) + float(e_th(r0, np.array([T_junction]))[0]) + c_top))
    a = np.array(rows, dtype=float)
    return dict(rho=a[:, 0], T=a[:, 1], p=a[:, 2], u=a[:, 3], sig_p=a[:, 4],
                ext=a[:, 5].astype(bool), p_th_junction=a[:, 6], u_junction=a[:, 7],
                T_junction=T_junction,
                gamma_latt=gamma_latt, e_th=e_th, p_th=p_th,
                p_th_shape=p_th_shape, p_junction=dict(zip(rho_j, pj_data)),
                theta_D=lambda rho: debye_temperature(cold, rho, M))


# ---------------------------------------------------------------------------
# Second-virial pseudo-data for the hot dilute wedge
# ---------------------------------------------------------------------------
_LJ_TSTAR = np.exp(np.linspace(np.log(0.3), np.log(400.0), 160))
_LJ_BSTAR = None


def lj_bstar(Tstar):
    """Reduced Lennard-Jones second virial coefficient B*(T*) = B/(2 pi N_A sigma^3/3),
    B* = 3 int_0^inf (1 - exp(-u*/T*)) x^2 dx with u* = 4 (x^-12 - x^-6)."""
    global _LJ_BSTAR
    if _LJ_BSTAR is None:
        vals = np.empty_like(_LJ_TSTAR)
        for i, ts in enumerate(_LJ_TSTAR):
            f = lambda x: (1.0 - np.exp(-4.0 * (x ** -12 - x ** -6) / ts)) * x ** 2
            vals[i] = 3.0 * (_quad(f, 1e-3, 1.0, limit=200)[0]
                             + _quad(f, 1.0, 60.0, limit=200)[0])
        _LJ_BSTAR = _mkspl(np.log(_LJ_TSTAR), vals, k=3)
    return _LJ_BSTAR(np.log(np.clip(np.asarray(Tstar, float), _LJ_TSTAR[0], _LJ_TSTAR[-1])))


def virial_pseudodata(species, T_max=1.0e4, n_T=16, rho_pts=(1e-3, 3e-3, 1e-2, 3e-2),
                      sig_pts=(0.005, 0.005, 0.01, 0.03), n_fit=40):
    """
    Z = 1 + B(T) rho at low density for T between the reference ceiling and
    `T_max`, with B(T) from a Lennard-Jones (epsilon, sigma) fitted to the
    reference EOS's own B(T) over its validity range.  The information enters
    the fit through the delta^1 block A(y), which is global in density, so it
    pins the hot dilute wedge that neither source covers (v2: Z = 0.75 at
    0.03 g/cm^3 and 10^4 K, Gamma_1 = 0.26 at 3000 K, 1.3 kbar).  The dilute
    limit stays molecular: at 1 bar and 10^4 K the real Z is ~5 from
    dissociation and ionisation, which is a limitation of the ideal term, not
    of this class.
    """
    it = IDEAL[species]
    lim = REF_LIMITS[species]
    M = it.M
    Tf = np.exp(np.linspace(np.log(lim['T_lo']), np.log(lim['T_hi']), n_fit))
    Bf = reference_second_virial(species, Tf)          # cm^3/g

    def model(q, T):
        eps, sig = q
        b0 = 2.0 * np.pi * _N_AVOGADRO * (sig * 1e-8) ** 3 / 3.0 / M   # cm^3/g
        return b0 * lj_bstar(T / eps)

    def resid(q):
        return model(q, Tf) - Bf

    guess = {'methane': (148.0, 3.73), 'ammonia': (350.0, 3.0)}[species]
    sol = _lsq(resid, np.array(guess), bounds=([20.0, 1.5], [3000.0, 6.0]))
    eps, sig = sol.x
    T = np.exp(np.linspace(np.log(lim['T_hi']), np.log(T_max), n_T))
    B = model(sol.x, T)
    R, TT = np.meshgrid(np.array(rho_pts, float), T)
    SG, _ = np.meshgrid(np.array(sig_pts, float), T)
    Z = 1.0 + np.interp(TT.ravel(), T, B) * R.ravel()
    return dict(rho=R.ravel(), T=TT.ravel(), Z=Z, sig=SG.ravel(), eps=float(eps), sigma=float(sig),
                B_fit_rms=float(np.sqrt(np.mean(sol.fun ** 2))), T_fit=Tf, B_ref=Bf,
                B=lambda t: model(sol.x, np.atleast_1d(t)))


# ---------------------------------------------------------------------------
# Floors and ceilings that depend on the state
# ---------------------------------------------------------------------------
def _gamma_floor(rho, T, T_c, lo=1e-3, hi=0.90, rho_ramp=0.6, width=None,
                 T_ramp=2.2, T_width=None, rho_c=None, hi_dilute=0.95):
    """
    Two-dimensional floor on Gamma = (dlnP/dlnrho)_T.

    Gamma genuinely approaches zero only near the critical point; the reference
    EOS's Gamma < 0.9 occupies rho in [0.01, 1.6] rho_c and T up to 1.57 T_c
    (CH4; similar for NH3), and Gamma_ref >= 0.98 above 2 T_c.  The v2 floor
    ramped off in density alone, so it also released the hot dilute wedge at
    1000-10000 K, where the fit then softened to Gamma_1 = 0.26 (F9).  Here
    the floor is released only where BOTH the density ramp (centre 0.6 g/cm^3,
    width ln 3, as in v2) AND a temperature ramp (centre 2.2 T_c, width ln 1.5)
    are off; it is checked against the reference Gamma on every kept point
    before fitting (margin 0.05).  `hi` may exceed 1 only where the cold curve
    makes theta = 0 feasible; the caller verifies feasibility.
    """
    if width is None:
        width = np.log(3.0)
    if T_width is None:
        T_width = np.log(1.5)
    if rho_c is not None:
        # v2's 0.6 g/cm^3 centre is 3.7 rho_c for CH4; keep that ratio per species
        # (0.86 g/cm^3 for NH3), otherwise the floor bites NH3's near-critical
        # reference points at 1.7 rho_c and 1.02 T_c, where Gamma_ref ~ 0.1.
        rho_ramp = 3.7 * rho_c
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    s_r = 0.5 * (1.0 + np.tanh(np.log(rho / rho_ramp) / width))
    s_t = 0.5 * (1.0 + np.tanh(np.log(T / (T_ramp * T_c)) / T_width))
    on = 1.0 - (1.0 - s_r) * (1.0 - s_t)
    floor = lo + (hi - lo) * on
    # In the hot dilute wedge (T well above the Boyle temperature, rho below the
    # density ramp) the virial series gives Gamma = 1 + B rho + ... >= 1, so the
    # floor rises to `hi_dilute`: a floor of 0.9 alone let Z fall to 0.75 over two
    # decades of density in the first v3 fit.  Ramps on above ~4 T_c, where the
    # reference Gamma exceeds 0.98, so no kept reference point is touched.
    s_t2 = 0.5 * (1.0 + np.tanh(np.log(T / (4.0 * T_c)) / np.log(1.3)))
    floor = floor + (hi_dilute - hi) * s_t2 * (1.0 - s_r)
    return floor


def _dpdrho_floor(rho, T, T_c, rho_c, lo=1e-4, hi=0.30):
    """
    Two-dimensional floor on (dP/drho)_T in units of R_s T.

    v2 (and the first v3 fits) used a uniform 0.3 R_s T.  Near the critical
    point the reference EOS itself has (dP/drho)_T = Gamma Z R_s T with
    Gamma ~ 0.1 and Z ~ 0.2 (Z_c = 0.29 CH4, 0.24 NH3), i.e. ~0.02 R_s T --
    fifteen times BELOW that floor -- so the fit could not reproduce the
    near-critical reference points (P residuals of 75-127% at rho ~ 0.2-0.4
    g/cm^3, T ~ 1.02 T_c) and the deformation propagated through the
    subcritical hole into the reference/DFT seam, where it collapsed c_v
    (0.3-0.5 R_s at 0.6-0.75 g/cm^3) and, via the Grueneisen ceiling, put
    +50-100% bumps on the 1000 K DFT isotherm.  Here the floor keeps its 0.3
    value only where the density AND temperature ramps of `_gamma_floor` are
    on, and relaxes to `lo` (a pure stability margin) in the near-critical
    and two-phase regions.  Checked against the reference (dP/drho)_T on every
    kept point before fitting.
    """
    # Temperature ramp only.  A density ramp (as in `_gamma_floor`) still put
    # 0.06 R_s T on NH3's near-critical reference points at 1.3-2 rho_c and
    # 1.02 T_c, three times their Gamma Z ~ 0.02; and in the dense region the
    # Gamma floor already implies (dP/drho)_T >= 0.9 P/rho >> 0.3 R_s T, so
    # nothing is lost by releasing this floor there.
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    s_t = 0.5 * (1.0 + np.tanh(np.log(T / (2.2 * T_c)) / np.log(1.5)))
    return lo + (hi - lo) * s_t + 0.0 * rho


def _z_floor(rho, T, T_c, lo=1e-5, hi=0.01):
    """Floor on Z = P/(rho R_s T): `hi` above ~2.2 T_c (the dilute-gas guard),
    relaxing to `lo` below it, where the saturated liquid of the two-phase
    construction has Z = P_sat/(rho' R_s T) ~ 1e-4 (NH3 at 200 K)."""
    T = np.asarray(T, dtype=float)
    s_t = 0.5 * (1.0 + np.tanh(np.log(T / (2.2 * T_c)) / np.log(1.5)))
    return lo + (hi - lo) * s_t + 0.0 * np.asarray(rho, dtype=float)


def _cv_ceiling(T, R_s, c_lo=8.0, c_hi=30.0, T0=300.0, T1=1500.0):
    """Ceiling on c_v [erg/(g K)]: c_lo R_s below T0 rising log-linearly to
    c_hi R_s above T1.  The reference liquids have c_v = 3.5-4.5 R_s below
    300 K and the quasi-harmonic anchor at most ~6.5 R_s; the DFT reaches
    16-26 R_s above 1000 K with dissociation.  Without it the spline put
    c_v = 40 R_s spikes into the T = 50 K edge, which cost 10 R_s of entropy.
    Affine in the coefficients (c_v is), and the ideal gas (3-4 R_s) is
    strictly inside."""
    T = np.asarray(T, dtype=float)
    f = np.clip(np.log(T / T0) / np.log(T1 / T0), 0.0, 1.0)
    return R_s * (c_lo + (c_hi - c_lo) * f)


def _gruneisen_ceiling(rho, rho_top, g_lo=2.0, g_hi=4.0, span=3.0):
    """Ceiling g(rho) on gamma_G = (dP/dT)_rho / (rho c_v): g_hi up to rho_top,
    falling log-linearly to g_lo at span*rho_top.  Reference gamma_G is at
    most 2.9 (in the solid region the melting cut removes; p99 = 2.1) and the
    DFT gives 0.2-0.9 at 1000 K, so this is a runaway stop, not a data
    constraint: v2 reached 182."""
    rho = np.asarray(rho, dtype=float)
    f = np.clip(np.log(rho / rho_top) / np.log(span), 0.0, 1.0)
    return g_hi + (g_lo - g_hi) * f



# ---------------------------------------------------------------------------
# Post-fit gauges
# ---------------------------------------------------------------------------
def _box_grid(eos, rho_lo=None, rho_hi=None, T_lo=None, T_hi=None, n=400):
    rho_lo = np.exp(eos.lr0) if rho_lo is None else rho_lo
    rho_hi = np.exp(eos.lr1) if rho_hi is None else rho_hi
    T_lo = np.exp(eos.lt0) if T_lo is None else T_lo
    T_hi = np.exp(eos.lt1) if T_hi is None else T_hi
    R, T = np.meshgrid(np.exp(np.linspace(np.log(rho_lo), np.log(rho_hi), n)),
                       np.exp(np.linspace(np.log(T_lo), np.log(T_hi), n)))
    return R.ravel(), T.ravel()


def solve_s_offset(eos, n=400, margin_Rs=0.5, exclude=None, **dom):
    """
    Choose the additive entropy gauge so s > 0 over the whole evaluation domain.

    Entropy is physical only up to an additive constant (a1 is pure gauge and cancels
    exactly out of U, P and c_v), so this changes no thermodynamics whatsoever -- it
    only moves the zero.  Doing it AFTER the fit means the requirement costs the fit
    nothing.  Records the third-law minimum (`eos.s_min_thirdlaw`, erg/(g K)) as
    well: in v3 it should be close to zero or positive; in v2 it sat at the
    -100 R_s fit bound.
    """
    r, t = _box_grid(eos, n=n, **dom)
    if exclude is not None:
        keep = ~exclude(r, t)
        r, t = r[keep], t[keep]
    eos.s_offset = 0.0
    s_min = float(eos.s(r, t).min())
    eos.s_min_thirdlaw = s_min
    eos.s_offset = max(0.0, margin_Rs * eos.R_s - s_min)
    return eos.s_offset


def solve_u_offset(eos, n=400, margin_frac=0.1, exclude=None, **dom):
    """
    Choose the additive energy gauge so u > 0 over the whole evaluation domain
    (the consumers store log10 u).  A constant in F shifts F and U only; S, P,
    c_v and every derivative are untouched.  Records `eos.u_min_gauge0`.
    """
    r, t = _box_grid(eos, n=n, **dom)
    if exclude is not None:
        keep = ~exclude(r, t)
        r, t = r[keep], t[keep]
    eos.u_offset = 0.0
    u_min = float(eos.u(r, t).min())
    eos.u_min_gauge0 = u_min
    margin = margin_frac * eos.R_s * np.exp(eos.lt0)
    eos.u_offset = max(0.0, margin - u_min)
    return eos.u_offset


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def fit(species, n_r=20, n_t=13, k=4, n_per_knot=4, n_rho_ref=64, gamma_max=8.0,
        rho_lo=1e-3, rho_hi=60.0, T_lo=50.0, T_hi=30000.0,
        # anc_pth: sigma of the anchor's THERMAL pressure relative to its value at
        # the 1000 K junction (so the rows force dP/dT -> 0 as T -> 0; normalised by
        # the total pressure they could not, measured: P_th(50 K)/P_th(1000 K) stayed
        # at 0.4-0.6).  anc_u: ABSOLUTE energy sigma in units of R_s * 1000 K; the
        # first v3 fit used 0.3 R_s T, which at 50 K is 15 R_s K and let 176 model
        # rows outweigh the DFT energies by 20x.
        sig=dict(ref_p=0.002, ref_u=0.02, ref_s=0.02, ref_cv=0.03, dome_p=None, dome_u=None,
            dome_s=None, dft_p=0.02, dft_u=0.3,
                 anc_pth=0.15, anc_u=0.1, vir_z=0.005),
        wt=dict(ref=1.0, dome=0.5, dft=1.0, anc_p=0.5, anc_u=1.0, vir=0.2),
        # floors are multiples of strictly-positive scales:
        # Z [-], dpdrho [R_s T], dpdT [R_s rho], cv [R_s], s [R_s]; gamma is the
        # Gamma floor's high value; pcold is the fraction f in P >= f P_cold;
        # grun is the Grueneisen ceiling below rho_top.
        floors=dict(Z=0.01, dpdrho=0.30, dpdT=0.001, cv=0.2, s=-5.0,
                    gamma=0.90, pcold=0.5, grun=4.0, cv_max=(8.0, 30.0), loop_p=3.0, loop_dpdrho=10.0),
        cold=True, cold_onset=None, quantum=True, phase_cuts=True, virial=True, warp=True,
        dome_dilate=(0.9, 1.05), prior_sat=None,
        lam=0.1, ridge=1e-5, audit_per_knot=24, max_cuts=20, verbose=True,
        return_prechecks=False, dry_run=False):
    """
    Fit alpha_exc for one species by convex QP (v3).

    Objective: sum over data classes of (w_k / N_k) * ||r_k / sigma_k||^2, with
    dimensionless residuals and each class divided by its own point count so a
    class cannot buy influence by being densely sampled.  Four classes:
    reference pseudo-data (P, U, S; phase cuts for CH4), DFT-MD (P as
    published, U quantum-corrected, sigma_U ramping with T), the quasi-harmonic
    anchor (P, U below 1000 K and above the DFT density ceiling), and the
    second-virial class (Z in the hot dilute wedge).

    Constraints, all affine in the coefficients: Z, (dP/drho)_T, (dP/dT)_rho,
    c_v and s above floors; Gamma above a two-dimensional floor; P >= f P_cold;
    and gamma_G below a ceiling.  There is NO floor on u (energy is gauge; see
    `solve_u_offset`).  theta = 0, the re-gauged ideal gas plus the cold curve,
    is strictly feasible for every row -- verified and reported per family
    before the barrier starts, since the barrier needs an interior point.
    """
    cc = ColdCurve.from_species(species, onset=cold_onset, quantum=quantum) if cold else None
    # knot stretching centred between the reference ceiling and the DFT floor
    warp_rho = None
    if warp:
        r_ref_max = float(reference_pseudodata(species, n_rho=n_rho_ref, gamma_max=gamma_max,
                                               phase_cuts=phase_cuts)[0].max())
        r_dft_min = float(load_dft(species)[0].min())
        warp_rho = float(np.sqrt(r_ref_max * r_dft_min))
    eos = HelmholtzEOS(species, rho_lo, rho_hi, T_lo, T_hi, n_r, n_t, k, cold=cc,
                       warp_rho=warp_rho)
    n_A, n_s = eos.n_A, eos.n_coef
    n_th = n_s + 1                      # + free Delta_u for the DFT energy zero
    Rs = eos.R_s
    T_c = eos.T_red
    rho_top = cc.rho_top if cc is not None else float(load_dft(species)[0].max())

    rows, rhs, tags = [], [], []

    def add(M, y, w, tag):
        """Append rows scaled by sqrt(w / N) so each class is count-normalised."""
        if len(y) == 0:
            return
        f = np.sqrt(w / len(y))
        rows.append(M * f)
        rhs.append(np.asarray(y, float) * f)
        tags.append((tag, len(y)))

    zero = lambda n: np.zeros((n, 1))

    # ---- class 1: reference pseudo-data (P, U, S) -------------------------
    r1, t1, p1, u1, s1 = reference_pseudodata(species, n_rho=n_rho_ref, gamma_max=gamma_max,
                                             phase_cuts=phase_cuts)
    i, M = eos.design(r1, t1, 'p')
    add(np.hstack([M / p1[:, None], zero(len(r1))]) / sig['ref_p'],
        (1.0 - i / p1) / sig['ref_p'], wt['ref'], 'ref_P')
    i, M = eos.design(r1, t1, 'u')
    sc = (Rs * t1)[:, None]
    add(np.hstack([M / sc, zero(len(r1))]) / sig['ref_u'],
        ((u1 - i) / (Rs * t1)) / sig['ref_u'], wt['ref'], 'ref_U')
    i, M = eos.design(r1, t1, 's')
    add(np.hstack([M / Rs, zero(len(r1))]) / sig['ref_s'],
        ((s1 - i) / Rs) / sig['ref_s'], wt['ref'], 'ref_S')
    # c_v rows (v3): the second T-derivative is otherwise pinned only through
    # the T-spacing of the U and S rows, which near the reference's top density
    # (one or two temperatures per isochore) left c_v free to collapse to the
    # 0.2 R_s floor; the Grueneisen ceiling then acted on that collapse.
    if sig.get('ref_cv'):
        cv1 = reference_cv(species, r1, t1)
        ok = np.isfinite(cv1) & (cv1 > 0)
        i, M = eos.design(r1[ok], t1[ok], 'cv')
        add(np.hstack([M / cv1[ok][:, None], zero(int(ok.sum()))]) / sig['ref_cv'],
            (1.0 - i / cv1[ok]) / sig['ref_cv'], wt['ref'], 'ref_CV')

    # ---- class 2: DFT-MD, scattered (no gridding, no differencing) --------
    r2, t2, p2, u2 = load_dft(species, quantum=quantum)
    i, M = eos.design(r2, t2, 'p')
    add(np.hstack([M / p2[:, None], zero(len(r2))]) / sig['dft_p'],
        (1.0 - i / p2) / sig['dft_p'], wt['dft'], 'dft_P')
    i, M = eos.design(r2, t2, 'u')
    sc = (Rs * t2)[:, None]
    su = (dft_sigma_u(t2, sig_lo=sig['dft_u']))[:, None]
    col = np.full((len(r2), 1), -1.0) / sc      # d r_u / d Delta_u
    add(np.hstack([M / sc, col]) / su,
        ((u2 - i) / (Rs * t2)) / su[:, 0], wt['dft'], 'dft_U')

    # ---- class 1b: two-phase (Maxwell) construction inside the dome -------
    # OFF by default since 2026-09-08.  These rows assert P = P_sat, i.e. they
    # ask the analytic surface to be its own convex hull; that is the job of
    # the Maxwell construction applied to the FINISHED surface (`tabulate`,
    # `march_pt`), not of F itself.  The reference equations carry a van der
    # Waals loop inside the dome and so should the fit.  Set sig['dome_p'] to
    # re-enable.
    dome = dome_pseudodata(species) if (phase_cuts and sig.get('dome_p')) else None
    if dome is not None:
        r5, t5, p5, u5, s5 = dome['rho'], dome['T'], dome['p'], dome['u'], dome['s']
        # P in ABSOLUTE units of rho R_s T (sigma_Z): the saturation pressure
        # is 1e-4 rho R_s T at the liquid edge and a relative sigma there would
        # demand that the spline cancel the ideal-gas pressure to 1e-6 -- rows
        # of enormous weight that it cannot satisfy and that then dominate
        # everything else.  What the dome needs is a SMALL, FLAT pressure.
        i, M = eos.design(r5, t5, 'Z')
        add(np.hstack([M, zero(len(r5))]) / sig['dome_p'],
            (p5 / (r5 * Rs * t5) - i) / sig['dome_p'], wt['dome'], 'dome_P')
        # lever-rule U and S rows are available but OFF by default: the phase
        # boundary is a kink in F that the spline cannot reproduce, and with
        # them the unconstrained fit's reference c_v residual (p95) rose from
        # 1.6% to 40%.  The dome only needs a small, flat pressure.
        if sig.get('dome_u'):
            i, M = eos.design(r5, t5, 'u')
            sc = (Rs * t5)[:, None]
            add(np.hstack([M / sc, zero(len(r5))]) / sig['dome_u'],
                ((u5 - i) / (Rs * t5)) / sig['dome_u'], wt['dome'], 'dome_U')
        if sig.get('dome_s'):
            i, M = eos.design(r5, t5, 's')
            add(np.hstack([M / Rs, zero(len(r5))]) / sig['dome_s'],
                ((s5 - i) / Rs) / sig['dome_s'], wt['dome'], 'dome_S')

    # ---- class 3: quasi-harmonic anchor (P and U) --------------------------
    anc = anchor_pseudodata(species, cc, T_lo=T_lo, T_hi=T_hi, rho_hi=rho_hi,
                            quantum=quantum) if cc is not None else None
    if anc is not None and phase_cuts:
        # no anchor rows inside the dome (NH3's 0.5 g/cm^3 isochore lies below
        # rho_0 of the cold curve and inside the two-phase region for T < T_c)
        keep = ~in_two_phase(species, anc['rho'], anc['T'])
        anc = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v)
               for k, v in anc.items()}
    if anc is not None:
        r3, t3, p3, u3 = anc['rho'], anc['T'], anc['p'], anc['u']
        # P rows: residual (P_fit - P_anchor) in units of the junction thermal pressure
        pthj = anc['p_th_junction']
        sp3 = anc['sig_p'] * (sig['anc_pth'] / 0.15)     # per-row sigma, scaled by the kw
        i, M = eos.design(r3, t3, 'p')
        add(np.hstack([M / pthj[:, None], zero(len(r3))]) / sp3[:, None],
            ((p3 - i) / pthj) / sp3, wt['anc_p'], 'anc_P')
        # U rows as ISOCHORIC DIFFERENCES relative to the junction temperature,
        #     u(rho, T) - u(rho, T_j) = E_th(rho, T) - E_th(rho, T_j),
        # with sigma = anc_u * |Delta E_th| + 20 R_s K.  These pin c_v(T) along
        # each isochore to the quasi-harmonic model (~10%) without any claim on
        # the absolute energy, so they need no Delta_u column and cannot drag
        # the DFT gauge.  The earlier absolute rows (sigma 500 R_s K, weight 0.3
        # over 176 rows) were too weak to matter: c_v in the solid corner then
        # collapsed to its floor and the Grueneisen ceiling acted on the
        # collapse rather than on physics.
        i, M = eos.design(r3, t3, 'u')
        i_j, M_j = eos.design(r3, np.full(len(r3), anc['T_junction']), 'u')
        du = u3 - anc['u_junction']
        sd = (sig['anc_u'] * np.abs(du) + 20.0 * Rs)[:, None]
        add(np.hstack([(M - M_j) / sd, zero(len(r3))]),
            ((du - (i - i_j)) / sd[:, 0]), wt['anc_u'], 'anc_dU')

    # ---- class 4: second virial in the hot dilute wedge --------------------
    vir = virial_pseudodata(species) if virial else None
    if vir is not None:
        r4, t4, z4, sg4 = vir['rho'], vir['T'], vir['Z'], vir['sig']
        i, M = eos.design(r4, t4, 'Z')
        add(np.hstack([M, zero(len(r4))]) / sg4[:, None],
            (z4 - i) / sg4, wt['vir'], 'vir_Z')

    A = np.vstack(rows)
    b = np.concatenate(rhs)

    # ---- column scaling ---------------------------------------------------
    # delta^2 spans ~14 decades over the box, so the raw design matrix has
    # cond ~ 5e16 and the optimal coefficients are O(1e6).  Solving in scaled
    # coordinates z = cs * theta makes the columns unit-norm, which is what lets
    # the roughness penalty and the barrier both act on an O(1) quantity.
    cs = np.linalg.norm(A, axis=0)
    cs[cs <= 0] = 1.0
    As = A / cs

    # ---- constraints ------------------------------------------------------
    # Log-UNIFORM collocation, several nodes per knot interval, tied to the
    # knots so refining the basis refines the collocation.
    n_cr = n_per_knot * (n_r - k) + 1
    n_ct = n_per_knot * (n_t - k) + 1
    rc = np.exp(np.linspace(np.log(rho_lo), np.log(rho_hi), n_cr))
    tc = np.exp(np.linspace(np.log(T_lo), np.log(T_hi), n_ct))
    RC, TC = np.meshgrid(rc, tc)
    rq, tq = RC.ravel(), TC.ravel()
    # no constraints inside the (dilated) reference dome and the critical band
    free = (two_phase_region(species, rq, tq, dilate_v=dome_dilate[0], dilate_l=dome_dilate[1],
                             prior_sat=prior_sat)
            if phase_cuts else np.zeros(rq.size, bool))
    rq_free, tq_free = rq[free], tq[free]
    rq, tq = rq[~free], tq[~free]

    # Each row is normalised by a STRICTLY POSITIVE natural scale -- never by
    # the ideal-gas value of the quantity itself (s_ideal contains -R_s ln delta
    # and turns negative at high density; dividing an inequality by a negative
    # number flips it).
    _QSCALE = {'Z': lambda r, t: np.ones_like(r),
               'dpdrho': lambda r, t: Rs * t,
               'dpdT': lambda r, t: Rs * r,
               'cv': lambda r, t: np.full_like(r, Rs),
               's': lambda r, t: np.full_like(r, Rs)}
    _FIT_CONSTRAINED = tuple(_QSCALE)

    def _floor_of(q, r, t):
        """Floor of quantity q (in its _QSCALE units) at (r, t): the 2-D
        (dP/drho)_T floor, or the scalar floors for the rest."""
        if q == 'dpdrho':
            return _dpdrho_floor(r, t, T_c, eos.rho_red, lo=floors.get('dpdrho_lo', 1e-4),
                                 hi=floors['dpdrho'])
        if q == 'Z':
            return _z_floor(r, t, T_c, lo=floors.get('Z_lo', 1e-5), hi=floors['Z'])
        return np.full(len(np.atleast_1d(r)), float(floors[q]))

    def gamma_rows(r, t):
        """Gamma >= g(rho,T)  <=>  rho (dP/drho) - g P >= 0, normalised by rho R_s T."""
        g = _gamma_floor(r, t, T_c, hi=floors['gamma'], rho_c=eos.rho_red)
        i_d, M_d = eos.design(r, t, 'dpdrho')
        i_p, M_p = eos.design(r, t, 'p')
        scale = (r * Rs * t)[:, None]
        G = np.hstack([(r[:, None] * M_d - g[:, None] * M_p) / scale, zero(len(r))])
        h = (g * i_p - r * i_d) / scale[:, 0]
        return G, h

    def gamma_violates(r, t):
        return (r * eos.dpdrho(r, t)
                - _gamma_floor(r, t, T_c, hi=floors['gamma'], rho_c=eos.rho_red) * eos.p(r, t)) < 0.0

    def pcold_rows(r, t):
        """P >= f P_cold  <=>  M_p theta >= f P_cold - i_p, normalised by rho R_s T.
        The level guard a Gamma floor cannot provide: v2 undershot its own
        continuation 20x at 60 g/cm^3 with Gamma pinned at the floor."""
        i_p, M_p = eos.design(r, t, 'p')
        pc = cc.p(r) if cc is not None else np.zeros_like(r)
        scale = (r * Rs * t)[:, None]
        G = np.hstack([M_p / scale, zero(len(r))])
        h = (floors['pcold'] * pc - i_p) / scale[:, 0]
        return G, h

    def pcold_violates(r, t):
        pc = cc.p(r) if cc is not None else np.zeros_like(r)
        return (eos.p(r, t) - floors['pcold'] * pc) < 0.0

    def grun_rows(r, t):
        """gamma_G <= g  <=>  g rho T c_v - T dP/dT >= 0, normalised by rho R_s T."""
        g = _gruneisen_ceiling(r, rho_top, g_hi=floors['grun'])
        i_c, M_c = eos.design(r, t, 'cv')
        i_t, M_t = eos.design(r, t, 'dpdT')
        scale = (r * Rs * t)[:, None]
        G = np.hstack([((g * r * t)[:, None] * M_c - t[:, None] * M_t) / scale, zero(len(r))])
        h = (t * i_t - g * r * t * i_c) / scale[:, 0]
        return G, h

    def grun_violates(r, t):
        g = _gruneisen_ceiling(r, rho_top, g_hi=floors['grun'])
        return (t * eos.dpdT(r, t) - g * r * t * eos.cv(r, t)) > 0.0

    def cvmax_rows(r, t):
        """c_v <= c_max(T)  <=>  c_max - i_cv - M_cv theta >= 0, normalised by R_s."""
        cmax = _cv_ceiling(t, Rs, *floors['cv_max'])
        i_c, M_c = eos.design(r, t, 'cv')
        G = np.hstack([-M_c / Rs, zero(len(r))])
        h = (i_c - cmax) / Rs
        return G, h

    def cvmax_violates(r, t):
        return eos.cv(r, t) > _cv_ceiling(t, Rs, *floors['cv_max'])

    def loop_rows(r, t):
        """Bounded van der Waals loop inside the constraint-free region.

        Stability is NOT imposed there (the surface must be free to carry the
        loop the reference equations carry), but the loop's depth is bounded so
        that a data-free region cannot host an arbitrarily large excursion:
            P            >= -a rho R_s T
            (dP/drho)_T  >= -b R_s T
        Both are affine in the coefficients and theta = 0 (ideal gas plus the
        cold curve, Z = 1) is strictly interior.  Without them the freed region
        reached Z = -30 at 0.1 g/cm^3 and 50 K."""
        a, b = floors['loop_p'], floors['loop_dpdrho']
        i_p, M_p = eos.design(r, t, 'p')
        i_d, M_d = eos.design(r, t, 'dpdrho')
        sp_ = (r * Rs * t)[:, None]
        sd_ = (Rs * t)[:, None]
        G = np.vstack([np.hstack([M_p / sp_, zero(len(r))]),
                       np.hstack([M_d / sd_, zero(len(r))])])
        h = np.concatenate([(-a * r * Rs * t - i_p) / sp_[:, 0],
                            (-b * Rs * t - i_d) / sd_[:, 0]])
        return G, h

    def loop_violates(r, t):
        return ((eos.p(r, t) < -floors['loop_p'] * r * Rs * t)
                | (eos.dpdrho(r, t) < -floors['loop_dpdrho'] * Rs * t))

    def constraint_rows(r, t, quantities=_FIT_CONSTRAINED):
        """Rows of G, h and family labels enforcing every floor at (r, t)."""
        Gr, Hr, fam = [], [], []
        for q in quantities:
            scale = _QSCALE[q](r, t)
            assert scale.min() > 0, f'constraint scale for {q} must be strictly positive'
            i, M = eos.design(r, t, q)
            Gr.append(np.hstack([M, zero(len(r))]) / scale[:, None])
            Hr.append(_floor_of(q, r, t) - i / scale)
            fam.append((q, len(r)))
        for name, fn in (('gamma', gamma_rows), ('pcold', pcold_rows), ('grun', grun_rows),
                         ('cvmax', cvmax_rows)):
            Gg, hg = fn(r, t)
            Gr.append(Gg)
            Hr.append(hg)
            fam.append((name, len(r)))
        return np.vstack(Gr), np.concatenate(Hr), fam

    Gc, hc, fam0 = constraint_rows(rq, tq)
    if rq_free.size:
        Gl, hl = loop_rows(rq_free, tq_free)
        Gc = np.vstack([Gc, Gl])
        hc = np.concatenate([hc, hl])
        fam0 = list(fam0) + [('loop', Gl.shape[0])]

    # ---- pre-fit feasibility report ----------------------------------------
    # slack at theta = 0 is -h per row; report the minimum per family.  Every
    # family must be strictly positive or the barrier cannot start.
    prechecks = {}
    pos = 0
    for name, n in fam0:
        prechecks[f'slack0_{name}'] = float((-hc[pos:pos + n]).min())
        pos += n
    if cc is not None:
        for r_, p_, tag in ((r1, p1, 'ref'), (r2, p2, 'dft')):
            prechecks[f'data_minus_cold_{tag}_min_Z'] = float(
                ((p_ - cc.p(r_)) / (r_ * Rs * t1 if tag == 'ref' else r_ * Rs * t2)).min())
        g_ref = reference_gamma(species, r1, t1)
        prechecks['gamma_floor_margin_ref'] = float(
            (g_ref - _gamma_floor(r1, t1, T_c, hi=floors['gamma'], rho_c=eos.rho_red)).min())
        # (dP/drho)_T of the reference against its 2-D floor, in units of R_s T
        h = 1e-5
        _, Dp, _ = _reference_residual(species, r1 * (1 + h), t1)
        _, Dm, _ = _reference_residual(species, r1 * (1 - h), t1)
        dpdr_ref = ((1 + h) * (1.0 + Dp) - (1 - h) * (1.0 + Dm)) / (2 * h) * Rs * t1
        prechecks['dpdrho_floor_margin_ref'] = float(
            (dpdr_ref / (Rs * t1) - _floor_of('dpdrho', r1, t1)).min())
        prechecks['z_floor_margin_ref'] = float(
            (p1 / (r1 * Rs * t1) - _floor_of('Z', r1, t1)).min())
        if dome is not None:
            prechecks['z_floor_margin_dome'] = float(
                (p5 / (r5 * Rs * t5) - _floor_of('Z', r5, t5)).min())
        if anc is not None:
            # anchor monotonicity: dP/dT > 0 along every extended isochore
            dp = []
            for r0 in np.unique(anc['rho']):
                m = np.isclose(anc['rho'], r0)
                o = np.argsort(anc['T'][m])
                dp.append(np.diff(anc['p'][m][o]).min())
            prechecks['anchor_min_dPdT_step'] = float(min(dp))
    if verbose:
        print(f'  [{species}] pre-fit: ' + ', '.join(f'{k}={v:.3g}' for k, v in prechecks.items()))
    bad0 = [k for k, v in prechecks.items() if k.startswith('slack0_') and v <= 0]
    assert not bad0, f'theta = 0 is not strictly feasible for {bad0}'
    bad1 = [k for k in ('gamma_floor_margin_ref', 'dpdrho_floor_margin_ref',
                        'z_floor_margin_ref', 'z_floor_margin_dome')
            if k in prechecks and prechecks[k] < 0]
    assert not bad1, f'a floor is violated by the reference data itself: {bad1}'

    # ---- regularisation ---------------------------------------------------
    # Roughness (lam): squared second differences between neighbouring
    # coefficients; its null space is the bilinear functions.  Tikhonov
    # (ridge): shrinks alpha_exc toward zero, i.e. the ideal gas plus the cold
    # curve.  Both calibrated against the data quadratic form at the
    # unconstrained solution.
    Dr, Dt = _diff2(n_r), _diff2(n_t)
    Rx = np.kron(Dr, np.eye(n_t))       # Psi is flattened row-major: idx = i*n_t + j
    Ry = np.kron(np.eye(n_r), Dt)
    Pen = np.zeros((n_th, n_th))
    Pen[:n_A, :n_A] = Dt.T @ Dt
    Pen[n_A:n_s, n_A:n_s] = Rx.T @ Rx + Ry.T @ Ry
    Tik = np.eye(n_th)
    Tik[-1, -1] = 0.0                   # Delta_u is physical; never shrink it

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
    # unconstrained solution diagnostics (cheap, informative)
    th_lsq = z0 / cs
    eos.lsq_theta = th_lsq
    eos.coef, eos.delta_u_dft = th_lsq[:n_s], th_lsq[n_s]
    prechecks['lsq_dft_p_median_pct'] = float(np.median(np.abs(eos.p(r2, t2) / p2 - 1)) * 100)
    prechecks['lsq_ref_p_median_pct'] = float(np.median(np.abs(eos.p(r1, t1) / p1 - 1)) * 100)
    eos.coef = np.zeros(n_s)
    if dry_run:
        eos.cold_curve = cc
        eos.anchor = anc
        eos.virial = vir
        eos.dome = dome
        eos.prechecks = prechecks
        eos.classes = dict(tags)
        # diagnostic handles: the assembled QP and the constraint generators, so
        # a caller can re-solve with families removed without re-running fit()
        eos.qp = dict(Hs=Hs, gs=gs, cs=cs, H0=H0, dat0=dat0, As=As, b=b, tags=list(tags),
                      Gc=Gc, hc=hc, fam0=fam0, n_s=n_s, n_th=n_th)
        eos.constraint_rows = constraint_rows
        eos.violates = dict(gamma=gamma_violates, pcold=pcold_violates,
                            grun=grun_violates, cvmax=cvmax_violates)
        return eos, prechecks

    # ---- solve, with a cutting-plane loop ---------------------------------
    # Solve, audit on a dense grid that resolves the knots, append only the
    # points that violate (dilated into their 3x3 neighbourhood), re-solve.
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
        assert (Gs @ z0 - hcur).min() > 0, 'theta = 0 should be strictly feasible'
        z, info = _solve_qp_interior(Hs, gs, Gs, hcur, z0)
        eos.coef, eos.delta_u_dft = (z / cs)[:n_s], (z / cs)[n_s]

        raq, taq = audit_grid(0.5 * (rounds % 2) + 0.17 * (rounds % 3))
        free_a = (two_phase_region(species, raq, taq, dilate_v=dome_dilate[0],
                                   dilate_l=dome_dilate[1], prior_sat=prior_sat)
                  if phase_cuts else np.zeros(raq.size, bool))
        raq_f, taq_f = raq[free_a], taq[free_a]
        raq, taq = raq[~free_a], taq[~free_a]
        bad = np.zeros(len(raq), bool)
        for q in _FIT_CONSTRAINED:
            bad |= getattr(eos, q)(raq, taq) / _QSCALE[q](raq, taq) < _floor_of(q, raq, taq)
        bad |= gamma_violates(raq, taq)
        bad |= pcold_violates(raq, taq)
        bad |= grun_violates(raq, taq)
        bad |= cvmax_violates(raq, taq)
        bad_f = loop_violates(raq_f, taq_f) if raq_f.size else np.zeros(0, bool)
        if not bad.any() and not bad_f.any():
            clean = True
            break
        # dilate each violating node into its neighbourhood (in ln rho, ln T)
        # by distance rather than by array index, since dome nodes were dropped
        lra, lta = np.log(raq), np.log(taq)
        dra = np.log(rho_hi / rho_lo) / (n_ar - 1)
        dta = np.log(T_hi / T_lo) / (n_at - 1)
        D = bad.copy()
        for i in np.flatnonzero(bad):
            D |= (np.abs(lra - lra[i]) <= 1.01 * dra) & (np.abs(lta - lta[i]) <= 1.01 * dta)
        bad = D
        if not bad.any() and bad_f.any():    # only loop bounds were violated
            Gvf, hvf = loop_rows(raq_f[bad_f], taq_f[bad_f])
            Gs = np.vstack([Gs, Gvf / cs])
            hcur = np.concatenate([hcur, hvf])
            n_added += int(bad_f.sum())
            continue
        Gv, hv, _ = constraint_rows(raq[bad], taq[bad])
        Gs = np.vstack([Gs, Gv / cs])
        hcur = np.concatenate([hcur, hv])
        n_added += int(bad.sum())
        if bad_f.any():                      # loop-bound violations in the freed region
            Gvf, hvf = loop_rows(raq_f[bad_f], taq_f[bad_f])
            Gs = np.vstack([Gs, Gvf / cs])
            hcur = np.concatenate([hcur, hvf])
            n_added += int(bad_f.sum())
    th = z / cs
    eos.audit_clean = clean

    eos.coef = th[:n_s]
    eos.delta_u_dft = th[n_s]
    eos.cond = float(np.linalg.cond(As))
    eos.fit_info = info
    eos.cut_rounds, eos.cut_points = rounds, n_added
    eos.classes = dict(tags)
    eos.prechecks = prechecks
    eos.virial_params = (dict(eps=vir['eps'], sigma=vir['sigma'], B_fit_rms=vir['B_fit_rms'])
                         if vir is not None else None)
    # the fit's own coexistence curve first; the gauges are then solved over
    # every state the tables can contain, i.e. everything outside the fit's
    # own dome (the dilated reference dome used for the constraints is wider)
    eos.dome_dilate = tuple(dome_dilate)
    eos.prior_sat = prior_sat
    eos_saturation(eos, verbose=verbose)
    excl = (lambda r, t: eos.in_two_phase(r, t)) if phase_cuts else None
    solve_s_offset(eos, exclude=excl)
    solve_u_offset(eos, exclude=excl)
    slack = Gs @ z - hcur
    eos.n_violations = int((slack < -1e-9).sum())
    # which families are active at the solution (min normalised slack)
    if verbose:
        print(f'  [{species}] classes: ' + ', '.join(f'{t}={n}' for t, n in tags))
        print(f'  [{species}] {n_th} params, {len(b)} data rows, {len(hc)} constraints'
              f' (+{n_added} cut in {rounds} rounds,'
              f' audit {"CLEAN" if clean else "NOT CLEAN"}) -> residual violations'
              f' {eos.n_violations}, Delta_u = {eos.delta_u_dft:+.4e} erg/g,'
              f' max|coef| = {np.abs(eos.coef).max():.3e}, min slack = {slack.min():.3e},'
              f' s_min(third law) = {eos.s_min_thirdlaw / Rs:+.3f} R_s,'
              f' u_offset = {eos.u_offset:.3e} erg/g')
    return (eos, prechecks) if return_prechecks else eos


def _solve_qp_interior(H, g, G, h, x0, mu0=1.0, mu_min=1e-12, tol=1e-10,
                       max_newton=200):
    """
    Minimise 0.5 x'Hx - g'x subject to Gx >= h, by a primal log-barrier method.

        phi_mu(x) = 0.5 x'Hx - g'x - mu * sum(log(Gx - h))
        grad      = Hx - g - mu G' s          s_i = 1/(G_i x - h_i)
        hess      = H + mu G' diag(s^2) G

    SLSQP is not usable at this size -- its dense LSQ subproblem gives up with
    "More than 3*n iterations" once the constraint count reaches ~2000 -- and
    trust-constr is ~60x slower for the same answer.  Relies on x0 being
    STRICTLY feasible.  The line search is a proper Armijo test (v2's was
    garbled into "any decrease").  Returns the final barrier gap m*mu, an
    upper bound on the suboptimality of the returned point.
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
            f0 = 0.5 * x @ H @ x - g @ x - mu * np.log(r).sum()
            slope = float(grad @ dx)
            step = 1.0
            for _ in range(60):
                xt = x + step * dx
                rt = G @ xt - h
                if rt.min() > 0:
                    ft = 0.5 * xt @ H @ xt - g @ xt - mu * np.log(rt).sum()
                    if ft <= f0 + 1e-4 * step * slope:
                        break
                step *= 0.5
            else:
                break
            x = x + step * dx
            n_steps += 1
            if np.abs(step * dx).max() < tol * max(1.0, np.abs(x).max()):
                break
        mu *= 0.1
    return x, dict(newton_steps=n_steps, min_slack=float((G @ x - h).min()),
                   barrier_gap=float(len(h) * mu * 10.0))
