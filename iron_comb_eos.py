"""
iron_eos_comb.py

Phase-mixed iron EOS.
Selects / blends solid and liquid iron EOS using a smooth transition
around the melting curve T_melt(P).

Liquid: ichikawa_iron_eos.Fe_EOS
Solid : dorogo_iron_eos.Fe_EOS(phase="hcp")

Public API (all return cgs / same units as the underlying classes):
    get_rho_pt(P, T)
    get_s_pt(P, T)
    get_u_pt(P, T)
    get_alpha_pt(P, T)
    get_cp_pt(P, T)
    get_cv_pt(P, T)

    get_t_sp(S, P)
    get_rho_sp(S, P)
    get_u_sp(S, P)

Assumptions:
- Underlying classes expose:
    get_rho_pt(P, T)
    get_s_pt(P, T)
    get_u_pt(P, T)
    get_alpha_pt(P, T)
    get_CP_pt(P, T)
    get_CV_pt(P, T)
    get_T_sp(S, P)
    get_rho_sp(S, P)
    get_u_sp(S, P)
- Units match between solid and liquid implementations:
    P in Pa
    T in K
    rho in kg/m^3
    S in erg/g/K
    U in erg/g
    Cp, Cv in erg/g/K
    alpha in 1/K
"""

from __future__ import annotations

import numpy as np

from eos import ichikawa_iron_eos, dorogo_iron_eos


class IRON_COMBINED_EOS:
    """
    Wrapper that smoothly blends solid and liquid EOS at T_melt(P).

    Define weight w in [0,1] where w=0 is solid and w=1 is liquid.
    We use a tanh switch with width dT = frac_width * T_melt.

    For PT queries:
        X = (1-w)*X_solid + w*X_liquid

    For SP queries:
        1. Compute T_solid = solid.get_T_sp(S,P)
           and T_liquid = liquid.get_T_sp(S,P).
        2. Compare both to T_melt(P). Use same smooth switch logic,
           evaluated using T_avg = 0.5*(T_solid+T_liquid).
        3. Blend all SP quantities with that weight.
    """

    def __init__(self, frac_width: float = 0.05, dT: float = 500.0):
        """
        frac_width sets the smoothing half-width as a fraction of T_melt.
        dT = frac_width * T_melt.
        """
        self.solid = dorogo_iron_eos.Fe_EOS(phase="hcp")
        # self.liquid = ichikawa_iron_eos.Fe_EOS()
        self.liquid = dorogo_iron_eos.Fe_EOS(phase="liquid")
        self.frac_width = float(frac_width)
        self.dT = float(dT)
    # ---------- helpers ----------

    @staticmethod
    def _as_arrays(a, b):
        A = np.array(a, ndmin=1, dtype=float)
        B = np.array(b, ndmin=1, dtype=float)
        A, B = np.broadcast_arrays(A, B)
        return A, B

    @staticmethod
    def _as_array_single(x):
        return np.array(x, ndmin=1, dtype=float)

    def get_T_melt(self, P):
        """
        Zhang et al. (2015) melt curve used in ichikawa_iron_eos.
        P in Pa. Return T_melt(P) in K.
        """
        P_arr = self._as_array_single(P)
        P_GPa = P_arr * 1e-9
        Tm = self.liquid.get_T_melt(P_GPa)
        return Tm.reshape(P_arr.shape)

    def get_S_liq_at_melt(self, P):
        """
        Return S_liq(P, T_melt) in erg/g/K.
        """
        P_arr = self._as_array_single(P)
        Tm = self.get_T_melt(P_arr)
        S_liq = self.liquid.get_s_pt(P_arr, Tm)
        return S_liq.reshape(P_arr.shape)

    def get_S_sol_at_melt(self, P):
        """
        Return S_sol(P, T_melt) in erg/g/K.
        """
        P_arr = self._as_array_single(P)
        Tm = self.get_T_melt(P_arr)
        S_sol = self.solid.get_s_pt(P_arr, Tm)
        return S_sol.reshape(P_arr.shape)

    def _blend_weight_PT(self, P_arr, T_arr):
        """
        Compute w_liq(P,T) using tanh around T_melt(P).

        w_liq = 0.5 * [1 + tanh((T - Tm)/dT)]
        with dT = frac_width * Tm.
        Clamp dT to at least 1 K to avoid zero-width at low P.
        """
        Tm = self.get_T_melt(P_arr)
        dT = np.maximum(self.frac_width * Tm, 1.0)
        arg = (T_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)

    def _blend_weight_given_T(self, P_arr, T_trial_arr):
        """
        Same as _blend_weight_PT but caller supplies T_trial_arr directly.
        Used for SP-based quantities where we estimate T first.
        """
        Tm = self.get_T_melt(P_arr)
        dT = self.dT
        arg = (T_trial_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)
    
    def _blend_weight_given_rhoT(self, P_arr, rho_trial_arr, T_trial_arr):
        """
        Same as _blend_weight_PT but caller supplies rho_trial_arr and T_trial_arr directly.
        Used for SP-based quantities where we estimate T first.
        """
        Tm = self.get_T_melt(P_arr)
        dT = self.dT
        arg = (T_trial_arr - Tm) / dT
        w = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(w, 0.0, 1.0)

    # ---------- PT interface ----------

    def get_rho_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        rho_s = self.solid.get_rho_pt(P_arr, T_arr, tab=solid_tab)
        rho_l = self.liquid.get_rho_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        rho_mix = (1.0 - w) * rho_s + w * rho_l
        return rho_mix.reshape(P_arr.shape)

    def get_s_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        s_s = self.solid.get_s_pt(P_arr, T_arr, tab=solid_tab)
        s_l = self.liquid.get_s_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        s_mix = (1.0 - w) * s_s + w * s_l
        return s_mix.reshape(P_arr.shape)

    def get_u_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        u_s = self.solid.get_u_pt(P_arr, T_arr, tab=solid_tab)
        u_l = self.liquid.get_u_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        u_mix = (1.0 - w) * u_s + w * u_l
        return u_mix.reshape(P_arr.shape)

    def get_alpha_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        a_s = self.solid.get_alpha_pt(P_arr, T_arr, tab=solid_tab)
        a_l = self.liquid.get_alpha_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        a_mix = (1.0 - w) * a_s + w * a_l
        return a_mix.reshape(P_arr.shape)

    def get_cp_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        cp_s = self.solid.get_CP_pt(P_arr, T_arr, tab=solid_tab)
        cp_l = self.liquid.get_CP_pt(P_arr, T_arr, tab=liquid_tab)
        w = self._blend_weight_PT(P_arr, T_arr)
        cp_mix = (1.0 - w) * cp_s + w * cp_l
        return cp_mix.reshape(P_arr.shape)

    def get_CP_pt(self, P, T, solid_tab=False, liquid_tab=True):
        return self.get_cp_pt(P, T, solid_tab=solid_tab, liquid_tab=liquid_tab)

    def get_cv_pt(self, P, T, solid_tab=False, liquid_tab=True):
        P_arr, T_arr = self._as_arrays(P, T)

        cv_s = self.solid.get_CV_pt(P_arr, T_arr, tab=solid_tab)
        cv_l = self.liquid.get_CV_pt(P_arr, T_arr, tab=liquid_tab)

        w = self._blend_weight_PT(P_arr, T_arr)
        cv_mix = (1.0 - w) * cv_s + w * cv_l
        return cv_mix.reshape(P_arr.shape)

    def get_CV_pt(self, P, T, solid_tab=False, liquid_tab=True):
        return self.get_cv_pt(P, T, solid_tab=solid_tab, liquid_tab=liquid_tab)

    # ---------- SP interface ----------

    def _get_phase_weights_SP(self, S, P, solid_tab=False, liquid_tab=True):
        """
        Internal step for SP queries.

        1. Compute T_s(P,S) and T_l(P,S).
        2. Take T_avg = 0.5*(T_s + T_l).
        3. Compute w_liq from P and T_avg.
        """
        S_arr, P_arr = self._as_arrays(S, P)

        rho_s, T_s = self.solid.get_rhot_sp_2d_inv(S_arr, P_arr) # solid direct inversion
        rho_l, T_l = self.liquid.get_rho_sp(S_arr, P_arr, tab=liquid_tab), self.liquid.get_T_sp(S_arr, P_arr, tab=liquid_tab)

        T_avg = 0.5 * (T_s + T_l)
        w = self._blend_weight_given_T(P_arr, T_avg)


        return w.reshape(P_arr.shape), rho_s.reshape(P_arr.shape), rho_l.reshape(P_arr.shape), T_s.reshape(P_arr.shape), T_l.reshape(P_arr.shape)

    def get_t_sp(self, S, P, solid_tab=False, liquid_tab=True):
        S_arr, P_arr = self._as_arrays(S, P)

        w, _, _, T_s, T_l = self._get_phase_weights_SP(S_arr, P_arr, 
                                                 solid_tab=solid_tab, liquid_tab=liquid_tab)
        T_mix = (1.0 - w) * T_s + w * T_l
        return T_mix.reshape(P_arr.shape)

    def get_T_sp(self, S, P, solid_tab=False, liquid_tab=True):
        return self.get_t_sp(S, P, solid_tab=solid_tab, liquid_tab=liquid_tab)

    def get_rho_sp(self, S, P, solid_tab=False, liquid_tab=True):
        S_arr, P_arr = self._as_arrays(S, P)

        w, rho_s, rho_l, _, _ = self._get_phase_weights_SP(S_arr, P_arr, 
                                             solid_tab=solid_tab, liquid_tab=liquid_tab)
        
        rho_mix = 1/( (1.0 - w)/rho_s + w/rho_l )
        return rho_mix.reshape(P_arr.shape)

    def get_u_sp(self, S, P, solid_tab=False, liquid_tab=True):
        S_arr, P_arr = self._as_arrays(S, P)

        w, rho_s, rho_l, T_s, T_l = self._get_phase_weights_SP(S_arr, P_arr, 
                                             solid_tab=solid_tab, liquid_tab=liquid_tab)

        u_s = self.solid.get_u_rhot(rho_s, T_s)
        u_l = self.liquid.get_u_rhot(rho_l, T_l)

        u_mix = (1.0 - w) * u_s + w * u_l
        return u_mix.reshape(P_arr.shape)
