"""Analytic ideal-gas EOS with H2 dissociation, for the low-density extension.

Roughly half of Brianna's (rho, T) rhomboid lies outside where CD/CMS are
defined: logrho reaches -15 (the raw H-He tables are clamped at -9 and the
saved rho-T table is +2 dex wrong by -15), logT reaches 1.7 (the raw tables
stop at 2.0 and their logT = 2.00 helium isotherm is broken), and logP reaches
-5.7 (the saved P-T table starts at 5).  Down there the gas is an ideal
mixture, so we can compute it exactly instead of extrapolating a table.

Model
-----
* H2: rigid rotor summed over J with ortho/para in equilibrium
  (nuclear-spin weights 1 even / 3 odd, then divided by 4 so nuclear spin is
  excluded from the entropy, the usual chemical convention), harmonic
  vibration on the fundamental, dissociation energy D0 measured from v=0,J=0.
* H: electronic g = 2.  He: monatomic.  H2O: vapor, classical asymmetric top
  with sigma = 2 and three harmonic modes.
* H2 <-> 2H in Saha equilibrium.  At fixed (rho, T) this is a quadratic with a
  closed-form root, so the whole model is analytic and vectorized -- no
  iteration anywhere.
* Neglected, and guarded against by the CLI: ionization (x_e ~ 3e-5 on
  Brianna's domain), water condensation, water dissociation above ~2500 K.

Against Brianna's SCvH tables at logrho < -8 this reproduces logP to about
1e-3 dex and log S to a few times 1e-4 dex, dissociation ridge included.

Entropy convention
------------------
``ideal_state`` returns the physically correct S.  ORCHARD's VAL entropy adds
an ideal mixing term that treats hydrogen as atomic (``_m_h_atomic = 1``) even
where it is molecular, which is +0.041 k_B/amu too high in molecular gas.  To
keep the blend seamless, ``smix_orchard_delta`` reproduces that convention so
the extension can carry it too (the user's choice: the delivered tables stay
consistent with ORCHARD's interior EOS).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
K_B = 1.380649e-16            # erg/K
H_PLANCK = 6.62607015e-27     # erg s
AMU = 1.66053906660e-24       # g
EV = 1.602176634e-12          # erg
M_E = 9.1093837015e-28        # g
C2 = 1.438776877              # cm K, hc/k

M_H = 1.00782503 * AMU
M_H2 = 2.01565006 * AMU
M_HE = 4.002602 * AMU
M_H2O = 18.010565 * AMU

# H2 internal structure
THETA_ROT_H2 = C2 * 59.322    # 85.35 K, from B0 = 59.322 cm^-1
THETA_VIB_H2 = C2 * 4161.17   # 5987 K, from the fundamental
D0_H2 = 4.47813 * EV
J_MAX = 60                    # rotational levels; convergence checked in tests

# H2O vapor
SIGMA_H2O = 2
THETA_ROT_H2O = (C2 * 27.8806, C2 * 14.5216, C2 * 9.2778)     # 40.11, 20.89, 13.35 K
THETA_VIB_H2O = (C2 * 3657.05, C2 * 1594.75, C2 * 3755.93)    # 5261.7, 2294.5, 5403.9 K

CHI_H = 13.598434 * EV        # hydrogen ionization potential

# ORCHARD's molecular weights for the ideal entropy of mixing
# (eos_class.py:1388-1394); used so the convention cancels exactly at the seam.
ORCHARD_M_H_ATOMIC = 1.0
ORCHARD_M_HE = 4.0026
ORCHARD_M_WATER = 18.015


# ------------------------------------------------------------ partition fns
def _lam3_inv(mass, t):
    """Lambda^-3 = (2 pi m k T / h^2)^{3/2}, in cm^-3."""
    return (2.0 * np.pi * mass * K_B * t / H_PLANCK ** 2) ** 1.5


def h2_rotation(t, theta_rot=THETA_ROT_H2, j_max=J_MAX):
    """Quantum rigid rotor, ortho/para in equilibrium, nuclear spin excluded.

    Returns ``(ln q_rot, <E_rot>/kT)``.  The nuclear-spin weights (1 for even
    J, 3 for odd J) set the equilibrium ortho/para ratio; dividing q by 4
    removes the spin degeneracy from the entropy without changing any
    population or the dissociation equilibrium (the same factor cancels
    against g_H = 2 rather than 4 in the Saha constant).
    """
    t = np.asarray(t, dtype=float)
    j = np.arange(j_max + 1)
    w = (2 * j + 1) * np.where(j % 2 == 0, 1.0, 3.0)
    e = theta_rot * j * (j + 1)                       # level energy / k, in K
    x = -e[:, None] / np.ravel(t)[None, :]
    terms = w[:, None] * np.exp(x)
    q = terms.sum(axis=0)
    e_mean = (terms * (e[:, None] / np.ravel(t)[None, :])).sum(axis=0) / q
    ln_q = np.log(q) - np.log(4.0)
    return ln_q.reshape(t.shape), e_mean.reshape(t.shape)


def _harmonic(t, theta):
    """Harmonic oscillator measured from v = 0: ``(ln q_vib, <E_vib>/kT)``."""
    x = theta / np.asarray(t, dtype=float)
    return -np.log1p(-np.exp(-x)), x / np.expm1(x)


def h2o_internal(t):
    """Water vapor internal entropy per molecule, ``s_int = ln q + <E>/kT``."""
    t = np.asarray(t, dtype=float)
    ta, tb, tc = THETA_ROT_H2O
    ln_q_rot = (np.log(np.sqrt(np.pi) / SIGMA_H2O)
                + 0.5 * np.log(t ** 3 / (ta * tb * tc)))
    s = ln_q_rot + 1.5
    for theta in THETA_VIB_H2O:
        ln_q, e = _harmonic(t, theta)
        s = s + ln_q + e
    return s


# ------------------------------------------------------------------- Saha
def dissociation_fraction(logrho, logt, x_h_mass, theta_vib=THETA_VIB_H2,
                          theta_rot=THETA_ROT_H2, d0=D0_H2):
    """Atomic fraction of hydrogen nuclei, ``x = n_H / n_H,total``.

    Solves ``n_H^2 = K n_H2`` with ``n_H + 2 n_H2 = n_Ht``.  Everything is done
    through ``logaddexp`` because ``exp(-D0/kT)`` underflows hard at 50 K
    (ln K ~ -1000) and ``8 n_Ht / K`` would overflow.
    """
    rho = 10.0 ** np.asarray(logrho, dtype=float)
    t = 10.0 ** np.asarray(logt, dtype=float)
    n_ht = x_h_mass * rho / M_H

    ln_q_rot, _ = h2_rotation(t, theta_rot=theta_rot)
    ln_q_vib, _ = _harmonic(t, theta_vib)
    # K = Lambda_H^-6 / Lambda_H2^-3 * g_H^2 / q_int(H2) * exp(-D0/kT)
    ln_k = (2.0 * np.log(_lam3_inv(M_H, t)) - np.log(_lam3_inv(M_H2, t))
            + 2.0 * np.log(2.0) - (ln_q_rot + ln_q_vib) - d0 / (K_B * t))

    with np.errstate(divide='ignore'):
        u = np.log(8.0 * n_ht) - ln_k                  # = ln(8 n_Ht / K)
    a = 0.5 * np.logaddexp(0.0, u)                     # = ln sqrt(1 + 8n/K)
    e = np.exp(-a)                                     # in (0, 1]
    return np.where(n_ht > 0, 2.0 * e / (1.0 + e), 0.0)


def ionization_fraction(logrho, logt, x_h_mass):
    """Upper bound on the electron fraction, assuming hydrogen is all atomic.

    Only used to warn when a user-requested domain leaves the regime where
    neglecting ionization is safe.
    """
    rho = 10.0 ** np.asarray(logrho, dtype=float)
    t = 10.0 ** np.asarray(logt, dtype=float)
    n_ht = np.maximum(x_h_mass * rho / M_H, 1e-300)
    k = _lam3_inv(M_E, t) * np.exp(-CHI_H / (K_B * t)) / n_ht
    # k underflows to 0 below ~1000 K, where the answer is simply x_e = 0.
    with np.errstate(divide='ignore', over='ignore'):
        return np.where(k > 0, 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / np.where(k > 0, k, 1.0))), 0.0)


# ------------------------------------------------------------- mixing terms
def _xlogx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def _smix(f_h, f_he, f_water, m_h):
    """ORCHARD's ideal entropy of mixing, k_B/amu (mirrors eos_class.py:1517)."""
    n_h = f_h / m_h
    n_he = f_he / ORCHARD_M_HE
    n_w = np.where(np.asarray(f_water) > 0, np.asarray(f_water) / ORCHARD_M_WATER, 0.0)
    ntot = n_h + n_he + n_w
    x_h, x_he, x_w = n_h / ntot, n_he / ntot, n_w / ntot
    q = m_h * x_h + ORCHARD_M_HE * x_he + ORCHARD_M_WATER * x_w
    return -(_xlogx(x_h) + _xlogx(x_he) + _xlogx(x_w)) / q


def smix_orchard_delta(f_h, f_he, f_water, x_diss):
    """ORCHARD's mixing convention minus the physical one, in erg/(g K).

    ORCHARD always uses atomic hydrogen (m_h = 1); the physical value uses
    ``m_h = 2/(1 + x)``, which reproduces the true particle count when a
    fraction ``x`` of the hydrogen nuclei are free atoms.  The difference is
    +0.041 k_B/amu in molecular gas at Y' = 0.25 and vanishes as H dissociates.
    """
    m_h_eff = 2.0 / (1.0 + np.asarray(x_diss, dtype=float))
    delta = (_smix(f_h, f_he, f_water, ORCHARD_M_H_ATOMIC)
             - _smix(f_h, f_he, f_water, m_h_eff))
    return delta * (K_B / AMU)


# ------------------------------------------------------------------- state
def ideal_state(logrho, logt, y_prime, z=0.0, smix='orchard',
                theta_vib=THETA_VIB_H2, theta_rot=THETA_ROT_H2, d0=D0_H2):
    """Ideal H2/H/He/H2O-vapor mixture at (rho, T).

    Parameters
    ----------
    logrho, logt : array_like
        log10 density (g/cm^3) and log10 temperature (K); broadcast together.
    y_prime : float or array_like
        Helium fraction of the H-He sub-mixture, Y' = Y/(1-Z).
    z : float or array_like
        Metal (water) mass fraction.
    smix : {'orchard', 'true'}
        Entropy-of-mixing convention.  'orchard' adds the atomic-hydrogen
        offset so the extension matches ORCHARD's VAL entropy at the blend
        seam; 'true' is the physical value (what SCvH has).

    Returns
    -------
    dict with ``logp`` (log10 dyn/cm^2), ``s`` (erg/(g K)), ``x_diss``,
    ``mu`` (mean molecular weight in amu) and ``x_e`` (ionization bound).
    """
    lr, lt, yp, zz = np.broadcast_arrays(
        np.asarray(logrho, dtype=float), np.asarray(logt, dtype=float),
        np.asarray(y_prime, dtype=float), np.asarray(z, dtype=float))
    rho = 10.0 ** lr
    t = 10.0 ** lt

    f_h = (1.0 - yp) * (1.0 - zz)
    f_he = yp * (1.0 - zz)
    f_w = zz

    x = dissociation_fraction(lr, lt, f_h, theta_vib=theta_vib,
                              theta_rot=theta_rot, d0=d0)
    n_ht = f_h * rho / M_H
    n_h = x * n_ht
    n_h2 = 0.5 * (1.0 - x) * n_ht
    n_he = f_he * rho / M_HE
    n_w = f_w * rho / M_H2O

    ln_q_rot, e_rot = h2_rotation(t, theta_rot=theta_rot)
    ln_q_vib, e_vib = _harmonic(t, theta_vib)
    s_int_h2 = ln_q_rot + e_rot + ln_q_vib + e_vib
    s_int_h2o = h2o_internal(t)

    n_tot = n_h + n_h2 + n_he + n_w
    p = n_tot * K_B * t

    # S/k = sum_i n_i [ ln(g_i Lambda_i^-3 / n_i) + 5/2 + s_int_i ]
    s_sum = np.zeros_like(rho)
    for n_i, mass, g_i, s_int in ((n_h, M_H, 2.0, 0.0),
                                  (n_h2, M_H2, 1.0, s_int_h2),
                                  (n_he, M_HE, 1.0, 0.0),
                                  (n_w, M_H2O, 1.0, s_int_h2o)):
        safe = n_i > 0
        n_safe = np.where(safe, n_i, 1.0)
        term = np.log(g_i * _lam3_inv(mass, t) / n_safe) + 2.5 + s_int
        s_sum = s_sum + np.where(safe, n_i * term, 0.0)
    s = K_B * s_sum / rho

    if smix == 'orchard':
        s = s + smix_orchard_delta(f_h, f_he, f_w, x)
    elif smix != 'true':
        raise ValueError(f"smix must be 'orchard' or 'true', got {smix!r}")

    return {
        'logp': np.log10(p),
        's': s,
        'x_diss': x,
        'mu': rho / (n_tot * AMU),
        'x_e': ionization_fraction(lr, lt, f_h),
    }
