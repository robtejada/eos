import numpy as np
import os

"""
    Analytic Helmholtz free energy EOS for Methane (CH4).

    Based on Setzmann & Wagner (1991) — 40-term fundamental equation
    valid from triple point (90.694 K) to 625 K, up to 1000 MPa.

    All outputs in CGS:
      - P in dyn/cm^2
      - u in erg/g
      - s in erg/(g*K)
      - f (free energy) in erg/g

    Authors: Roberto Tejada Arevalo
"""

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Critical constants
Tc = 190.564          # K
rhoc = 162.66         # kg/m^3 (0.16266 g/cm^3)
R_specific = 518.2705 # J/(kg*K)  = R_universal / M_CH4
_R_cgs = R_specific * 1.0e4  # erg/(g*K):  J/(kg*K) * 1e7 erg/J / 1e3 g/kg

# =====================================================================
# Ideal-gas Helmholtz coefficients (Setzmann & Wagner 1991, Eq. 5.2)
# =====================================================================
_a = np.array([
    9.91243972,    # a1
    -6.33270087,   # a2
    3.0016,        # a3
    0.008449,      # a4
    4.6942,        # a5
    3.4865,        # a6
    1.6572,        # a7
    1.4115         # a8
])

_theta = np.array([
    3.40043240,    # theta_4
    10.26951575,   # theta_5
    20.43932747,   # theta_6
    29.93744884,   # theta_7
    79.13351945    # theta_8
])

# =====================================================================
# Residual coefficients (Setzmann & Wagner 1991, Table 35)
# =====================================================================

# Block A: i = 1..13
_n_A = np.array([
    0.4367901028e-1,  0.6709236199,  -0.1765577859e1,  0.8582330241,
    -0.1206513052e1,  0.5120467220,  -0.4000010791e-3,
    -0.1247842423e-1, 0.3100269701e-1, 0.1754748522e-2,
    -0.3171921605e-5, -0.2240346840e-5, 0.2947056156e-6
])
_d_A = np.array([1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 8, 9, 10], dtype=float)
_t_A = np.array([-0.5, 0.5, 1.0, 0.5, 1.0, 1.5, 4.5,
                 0.0, 1.0, 3.0, 1.0, 3.0, 3.0])

# Block B: i = 14..36
_n_B = np.array([
    0.1830487909,    0.1511883679,   -0.4289363877,    0.6894002446e-1,
    -0.1408313996e-1, -0.3063054830e-1, -0.2969906708e-1, -0.1932040831e-1,
    -0.1105739959,    0.9952548995e-1,  0.8548437825e-2, -0.6150555662e-1,
    -0.4291792423e-1, -0.1813207290e-1,  0.3445904760e-1, -0.2385919450e-2,
    -0.1159094939e-1,  0.6641693602e-1, -0.2371549590e-1, -0.3961624905e-1,
    -0.1387292044e-1,  0.3389489599e-1, -0.2927378753e-2
])
_d_B = np.array([1, 1, 1, 2, 4, 5, 6, 1, 2, 3, 4, 4,
                 3, 5, 5, 8, 2, 3, 4, 4, 4, 5, 6], dtype=float)
_t_B = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 5.0,
                 5.0, 5.0, 2.0, 4.0, 12.0, 8.0, 10.0, 10.0,
                 10.0, 14.0, 12.0, 18.0, 22.0, 18.0, 14.0])
_c_B = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4], dtype=float)

# Block C: i = 37..40 (Gaussian terms)
_n_C = np.array([0.9324799946e-4, -0.6287171518e1,
                 0.1271069467e2,  -0.6423953466e1])
_d_C = np.array([2, 0, 0, 0], dtype=float)
_t_C = np.array([2.0, 0.0, 1.0, 2.0])
_alpha_C = np.array([20., 40., 40., 40.])
_beta_C = np.array([200., 250., 250., 250.])
_gamma_C = np.array([1.07, 1.11, 1.11, 1.11])
_Delta_C = np.array([1., 1., 1., 1.])


# =====================================================================
# Helpers
# =====================================================================

def _reduced(rho, T):
    """Convert rho [g/cm^3] and T [K] to reduced variables (delta, tau)."""
    rho_kgm3 = rho * 1000.0
    delta = rho_kgm3 / rhoc
    tau = Tc / T
    return delta, tau


# =====================================================================
# Ideal gas: Phi^o and d(Phi^o)/d(tau)
# =====================================================================

def _phi_ideal(delta, tau):
    """Ideal-gas dimensionless Helmholtz energy Phi^o(delta, tau)."""
    planck = 0.0
    for i in range(5):
        planck += _a[i + 3] * np.log(1.0 - np.exp(-_theta[i] * tau))
    return np.log(delta) + _a[0] + _a[1] * tau + _a[2] * np.log(tau) + planck


def _phi_ideal_tau(tau):
    """d(Phi^o)/d(tau) at constant delta."""
    planck_tau = 0.0
    for i in range(5):
        planck_tau += _a[i + 3] * _theta[i] / (np.exp(_theta[i] * tau) - 1.0)
    return _a[1] + _a[2] / tau + planck_tau


# =====================================================================
# Residual: Phi^r and its partial derivatives
# =====================================================================

def _phi_residual(delta, tau):
    """Residual dimensionless Helmholtz energy Phi^r(delta, tau)."""
    ar = 0.0

    # Block A: i=1..13
    for i in range(13):
        ar += _n_A[i] * delta**_d_A[i] * tau**_t_A[i]

    # Block B: i=14..36
    for i in range(23):
        ar += _n_B[i] * delta**_d_B[i] * tau**_t_B[i] * np.exp(-delta**_c_B[i])

    # Block C: i=37..40
    for i in range(4):
        ar += (_n_C[i] * delta**_d_C[i] * tau**_t_C[i]
               * np.exp(-_alpha_C[i] * (delta - _Delta_C[i])**2
                        - _beta_C[i] * (tau - _gamma_C[i])**2))

    return ar


def _phi_r_delta(delta, tau):
    """d(Phi^r)/d(delta) at constant tau."""
    ard = 0.0

    # Block A: n_i * d_i * delta^(d_i-1) * tau^t_i
    for i in range(13):
        ard += _n_A[i] * _d_A[i] * delta**(_d_A[i] - 1.0) * tau**_t_A[i]

    # Block B: n_i * tau^t_i * exp(-delta^c_i) * delta^(d_i-1) * (d_i - c_i * delta^c_i)
    for i in range(23):
        exp_term = np.exp(-delta**_c_B[i])
        ard += (_n_B[i] * tau**_t_B[i] * exp_term
                * delta**(_d_B[i] - 1.0)
                * (_d_B[i] - _c_B[i] * delta**_c_B[i]))

    # Block C: base * (d_i/delta - 2*alpha_i*(delta - Delta_i))
    # Note: d_C[1..3] = 0, so delta^0 = 1 and d_i/delta = 0
    for i in range(4):
        base = (_n_C[i] * delta**_d_C[i] * tau**_t_C[i]
                * np.exp(-_alpha_C[i] * (delta - _Delta_C[i])**2
                         - _beta_C[i] * (tau - _gamma_C[i])**2))
        if _d_C[i] == 0.0:
            # d(delta^0)/d(delta) = 0, so only the Gaussian derivative survives
            ard += base * (-2.0 * _alpha_C[i] * (delta - _Delta_C[i]))
        else:
            ard += base * (_d_C[i] / delta - 2.0 * _alpha_C[i] * (delta - _Delta_C[i]))

    return ard


def _phi_r_tau(delta, tau):
    """d(Phi^r)/d(tau) at constant delta."""
    art = 0.0

    # Block A: n_i * t_i * delta^d_i * tau^(t_i-1)
    for i in range(13):
        if _t_A[i] == 0.0:
            # d(tau^0)/d(tau) = 0
            continue
        art += _n_A[i] * _t_A[i] * delta**_d_A[i] * tau**(_t_A[i] - 1.0)

    # Block B: n_i * t_i * delta^d_i * tau^(t_i-1) * exp(-delta^c_i)
    for i in range(23):
        if _t_B[i] == 0.0:
            continue
        art += (_n_B[i] * _t_B[i] * delta**_d_B[i] * tau**(_t_B[i] - 1.0)
                * np.exp(-delta**_c_B[i]))

    # Block C: base * (t_i/tau - 2*beta_i*(tau - gamma_i))
    # Note: t_C[1] = 0, so t_i/tau = 0 for that term; tau^0 = 1
    for i in range(4):
        base = (_n_C[i] * delta**_d_C[i] * tau**_t_C[i]
                * np.exp(-_alpha_C[i] * (delta - _Delta_C[i])**2
                         - _beta_C[i] * (tau - _gamma_C[i])**2))
        if _t_C[i] == 0.0:
            art += base * (-2.0 * _beta_C[i] * (tau - _gamma_C[i]))
        else:
            art += base * (_t_C[i] / tau - 2.0 * _beta_C[i] * (tau - _gamma_C[i]))

    return art


# =====================================================================
# Public API (all CGS)
# =====================================================================

def free_energy(rho, T):
    """Specific Helmholtz free energy [erg/g]."""
    delta, tau = _reduced(rho, T)
    phi = _phi_ideal(delta, tau) + _phi_residual(delta, tau)
    return phi * _R_cgs * T  # erg/g


def get_p_rhot(rho, T):
    """Pressure [dyn/cm^2]."""
    delta, tau = _reduced(rho, T)
    return rho * _R_cgs * T * (1.0 + delta * _phi_r_delta(delta, tau))


def get_u_rhot(rho, T):
    """Internal energy [erg/g]."""
    delta, tau = _reduced(rho, T)
    return _R_cgs * T * tau * (_phi_ideal_tau(tau) + _phi_r_tau(delta, tau))


def get_s_rhot(rho, T):
    """Entropy [erg/(g*K)]."""
    delta, tau = _reduced(rho, T)
    phi_o = _phi_ideal(delta, tau)
    phi_r = _phi_residual(delta, tau)
    phi_o_tau = _phi_ideal_tau(tau)
    phi_r_tau = _phi_r_tau(delta, tau)
    return _R_cgs * (tau * (phi_o_tau + phi_r_tau) - phi_o - phi_r)
