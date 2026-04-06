import numpy as np
import os

"""
    Analytic Helmholtz free energy EOS for Ammonia (NH3).

    Based on Gao et al. (2023) — fundamental equation of state
    for ammonia valid from triple point (195.49 K) to 725 K.

    All outputs in CGS:
      - P in dyn/cm^2
      - u in erg/g
      - s in erg/g/K
      - f (free energy) in erg/g

    Authors: Roberto Tejada Arevalo
"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Unit conversions
_J2erg = 1.0e7          # 1 J = 1e7 erg

# Critical point
Tc = 405.56       # K
rhoc = 13.696     # mol/dm^3 (= mol/L)
M = 17.03052      # g/mol
R = 8.314462618   # J/(mol*K) — CODATA 2018

# Ideal gas coefficients
c0 = 4.0
a1 = -6.59406093943886
a2 = 5.60101151987913
v = np.array([2.224, 3.148, 0.9579])
u_k = np.array([1646.0, 3965.0, 7231.0])  # K

# Residual — 20 terms total
n = np.array([
    6.132232e-03, 1.7395866, -2.2261792, -0.30127553,
    0.08967023, -0.076387037, -0.84063963, -0.27026327,
    6.212578, -5.7844357, 2.4817542, -2.3739168,
    0.01493697, -3.7749264, 6.254348e-04, -1.7359e-05,
    -0.13462033, 0.07749072839, -1.6909858, 0.93739074])

t_exp = np.array([
    1.0, 0.382, 1.0, 1.0,
    0.677, 2.915, 3.51, 1.063,
    0.655, 1.3, 3.1, 1.4395,
    1.623, 0.643, 1.13, 4.5,
    1.0, 4.0, 4.3315, 4.015])

d_exp = np.array([
    4, 1, 1, 2,
    3, 3, 2, 3,
    1, 1, 1, 2,
    2, 1, 3, 3,
    1, 1, 1, 1], dtype=float)

l_exp = np.array([2, 2, 1], dtype=float)  # for terms i=6,7,8

eta = np.array([0.42776, 0.6424, 0.8175, 0.7995, 0.91,
                0.3574, 1.21, 4.14, 22.56, 22.68, 2.8452, 2.8342])

beta = np.array([1.708, 1.4865, 2.0915, 2.43, 0.488,
                 1.1, 0.85, 1.14, 945.64, 993.85, 0.3696, 0.2962])

gamma = np.array([1.036, 1.2777, 1.083, 1.2906, 0.928,
                   0.934, 0.919, 1.852, 1.05897, 1.05277, 1.108, 1.313])

epsilon = np.array([-0.0726, -0.1274, 0.7527, 0.57, 2.2,
                     -0.243, 2.96, 3.02, 0.9574, 0.9576, 0.4478, 0.44689])

b_assoc = np.array([1.244, 0.6826])  # for terms i=19,20 only


# =====================================================================
# Helpers
# =====================================================================

def _reduced(rho, T):
    """Convert rho [g/cm^3] and T [K] to reduced (delta, tau)."""
    rho_mol = rho * 1000.0 / M   # g/cm^3 -> mol/L
    delta = rho_mol / rhoc
    tau = Tc / T
    return delta, tau


# =====================================================================
# Ideal gas: alpha^o and its tau-derivative
# =====================================================================

def _alpha_ideal(delta, tau):
    """Ideal-gas dimensionless Helmholtz energy alpha^o(delta, tau)."""
    planck = np.sum(v * np.log(1.0 - np.exp(-u_k * tau / Tc)))
    return a1 + a2 * tau + np.log(delta) + (c0 - 1.0) * np.log(tau) + planck


def _alpha_ideal_tau(tau):
    """d(alpha^o)/d(tau) at constant delta."""
    # sum_k v_k * (u_k/Tc) / (exp(u_k*tau/Tc) - 1)
    x = u_k * tau / Tc
    planck_tau = np.sum(v * (u_k / Tc) / (np.exp(x) - 1.0))
    return a2 + (c0 - 1.0) / tau + planck_tau


# =====================================================================
# Residual: alpha^r and its partial derivatives
# =====================================================================

def _alpha_residual(delta, tau):
    """Residual dimensionless Helmholtz energy alpha^r(delta, tau)."""
    ar = 0.0

    # Block 1: i=1..5 (indices 0..4)
    for i in range(5):
        ar += n[i] * delta**d_exp[i] * tau**t_exp[i]

    # Block 2: i=6..8 (indices 5..7)
    for i in range(3):
        idx = i + 5
        ar += n[idx] * delta**d_exp[idx] * tau**t_exp[idx] * np.exp(-delta**l_exp[i])

    # Block 3: i=9..18 (indices 8..17), gaussian params j=0..9
    for i in range(10):
        idx = i + 8
        j = i
        ar += (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
               * np.exp(-eta[j] * (delta - epsilon[j])**2
                        - beta[j] * (tau - gamma[j])**2))

    # Block 4: i=19..20 (indices 18..19), gaussian params j=10..11, b_assoc j2=0..1
    for i in range(2):
        idx = i + 18
        j = i + 10
        j2 = i
        denom = beta[j] * (tau - gamma[j])**2 + b_assoc[j2]
        ar += (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
               * np.exp(-eta[j] * (delta - epsilon[j])**2
                        + 1.0 / denom))

    return ar


def _alpha_r_delta(delta, tau):
    """d(alpha^r)/d(delta) at constant tau."""
    ard = 0.0

    # Block 1: n_i * d_i * delta^(d_i-1) * tau^t_i
    for i in range(5):
        ard += n[i] * d_exp[i] * delta**(d_exp[i] - 1.0) * tau**t_exp[i]

    # Block 2: n_i * tau^t_i * exp(-delta^l_i) * (d_i*delta^(d_i-1) - l_i*delta^(d_i+l_i-1))
    for i in range(3):
        idx = i + 5
        exp_term = np.exp(-delta**l_exp[i])
        ard += (n[idx] * tau**t_exp[idx] * exp_term
                * (d_exp[idx] * delta**(d_exp[idx] - 1.0)
                   - l_exp[i] * delta**(d_exp[idx] + l_exp[i] - 1.0)))

    # Block 3: base_term * (d_i/delta - 2*eta_i*(delta-eps_i))
    for i in range(10):
        idx = i + 8
        j = i
        base = (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
                * np.exp(-eta[j] * (delta - epsilon[j])**2
                         - beta[j] * (tau - gamma[j])**2))
        ard += base * (d_exp[idx] / delta - 2.0 * eta[j] * (delta - epsilon[j]))

    # Block 4: base_term * (d_i/delta - 2*eta_i*(delta-eps_i))
    for i in range(2):
        idx = i + 18
        j = i + 10
        j2 = i
        denom = beta[j] * (tau - gamma[j])**2 + b_assoc[j2]
        base = (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
                * np.exp(-eta[j] * (delta - epsilon[j])**2
                         + 1.0 / denom))
        ard += base * (d_exp[idx] / delta - 2.0 * eta[j] * (delta - epsilon[j]))

    return ard


def _alpha_r_tau(delta, tau):
    """d(alpha^r)/d(tau) at constant delta."""
    art = 0.0

    # Block 1: n_i * t_i * delta^d_i * tau^(t_i-1)
    for i in range(5):
        art += n[i] * t_exp[i] * delta**d_exp[i] * tau**(t_exp[i] - 1.0)

    # Block 2: n_i * t_i * delta^d_i * tau^(t_i-1) * exp(-delta^l_i)
    for i in range(3):
        idx = i + 5
        art += (n[idx] * t_exp[idx] * delta**d_exp[idx] * tau**(t_exp[idx] - 1.0)
                * np.exp(-delta**l_exp[i]))

    # Block 3: base_term * (t_i/tau - 2*beta_i*(tau-gamma_i))
    for i in range(10):
        idx = i + 8
        j = i
        base = (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
                * np.exp(-eta[j] * (delta - epsilon[j])**2
                         - beta[j] * (tau - gamma[j])**2))
        art += base * (t_exp[idx] / tau - 2.0 * beta[j] * (tau - gamma[j]))

    # Block 4: base_term * (t_i/tau - 2*beta_i*(tau-gamma_i) / (beta_i*(tau-gamma_i)^2 + b_i)^2)
    for i in range(2):
        idx = i + 18
        j = i + 10
        j2 = i
        denom = beta[j] * (tau - gamma[j])**2 + b_assoc[j2]
        base = (n[idx] * delta**d_exp[idx] * tau**t_exp[idx]
                * np.exp(-eta[j] * (delta - epsilon[j])**2
                         + 1.0 / denom))
        # derivative of 1/denom wrt tau = -2*beta_j*(tau-gamma_j) / denom^2
        art += base * (t_exp[idx] / tau - 2.0 * beta[j] * (tau - gamma[j]) / denom**2)

    return art


# =====================================================================
# Public API
# =====================================================================

def free_energy(rho, T):
    """Specific Helmholtz free energy [erg/g]."""
    delta, tau = _reduced(rho, T)
    alpha = _alpha_ideal(delta, tau) + _alpha_residual(delta, tau)
    a_molar = alpha * R * T   # J/mol
    return a_molar * _J2erg / M  # erg/g


def get_p_rhot(rho, T):
    """Pressure [dyn/cm^2]."""
    delta, tau = _reduced(rho, T)
    rho_mol = rho * 1000.0 / M  # mol/L
    # rho_mol in mol/L; need mol/m^3 for Pa: multiply by 1000
    P_Pa = rho_mol * 1000.0 * R * T * (1.0 + delta * _alpha_r_delta(delta, tau))
    return P_Pa * 10.0  # Pa -> dyn/cm^2


def get_u_rhot(rho, T):
    """Internal energy [erg/g]."""
    delta, tau = _reduced(rho, T)
    u_molar = R * T * tau * (_alpha_ideal_tau(tau) + _alpha_r_tau(delta, tau))
    return u_molar * _J2erg / M  # erg/g


def get_s_rhot(rho, T):
    """Entropy [erg/(g*K)]."""
    delta, tau = _reduced(rho, T)
    ao = _alpha_ideal(delta, tau)
    ar = _alpha_residual(delta, tau)
    ao_tau = _alpha_ideal_tau(tau)
    ar_tau = _alpha_r_tau(delta, tau)
    s_molar = R * (tau * (ao_tau + ar_tau) - ao - ar)
    return s_molar * _J2erg / M  # erg/(g*K)
