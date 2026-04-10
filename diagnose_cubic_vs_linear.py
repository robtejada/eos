#!/usr/bin/env python
"""Diagnostic: compare linear vs cubic RGI interpolation for FD derivatives.

Tests whether cubic interpolation on inversion tables (S-P, rho-P)
produces smoother finite-difference derivatives than linear.

Usage:
    cd /Users/arevalo/orchard && python eos/diagnose_cubic_vs_linear.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---- Configuration ----
YP = 0.245
Z  = 0.02
LOGP = np.linspace(6.5, 13.5, 100)  # stay 0.5 inside table edges
LOGT_VALUES = np.arange(2.0, 4.3, 0.1)   # isotherms, away from table edges

DERIVATIVES = [
    ('nabla_ad', 'get_nabla_ad', r'$\nabla_{\rm ad}$'),
    ('gamma1',   'get_gamma1',   r'$\Gamma_1$'),
    ('dsdy_rhop','get_dsdy_rhop',r'$\partial S/\partial Y|_{\rho,P}$'),
]


def eval_deriv_isotherms(eos, method_name, logp, logt_vals, yp, z,
                         method='finite_difference', **kw):
    """Evaluate derivative along isotherms. Returns (n_T, n_P) array."""
    func = getattr(eos, method_name)
    out = np.full((len(logt_vals), len(logp)), np.nan)
    for i, lgt in enumerate(logt_vals):
        lgt_arr = np.full_like(logp, lgt)
        try:
            vals = func(logp, lgt_arr, yp, z, method=method, **kw)
            out[i] = np.atleast_1d(vals)
        except Exception:
            pass
    return out


def time_derivatives(eos, logp, logt_vals, yp, z, method='finite_difference'):
    """Time all derivatives and return dict of wall-clock times."""
    times = {}
    for name, func_name, _ in DERIVATIVES:
        func = getattr(eos, func_name)
        t0 = time.perf_counter()
        for lgt in logt_vals:
            lgt_arr = np.full_like(logp, lgt)
            try:
                func(logp, lgt_arr, yp, z, method=method)
            except Exception:
                pass
        times[name] = time.perf_counter() - t0
    return times


# ---- Load EOS with both methods ----
from eos.eos_class import hhe_z_mixtures, erg_to_kbbar

print("Loading EOS with interp_method='linear' ...")
eos_lin = hhe_z_mixtures(pt_tab=True, inv_tab=True, srho_tab=True,
                         xi_transform=False, interp_method='linear')
print("Loading EOS with interp_method='cubic' ...")
eos_cub = hhe_z_mixtures(pt_tab=True, inv_tab=True, srho_tab=True,
                         xi_transform=False, interp_method='cubic')
print("Done.\n")

# ---- Timing ----
print("Timing (FD method, all isotherms) ...")
t_lin = time_derivatives(eos_lin, LOGP, LOGT_VALUES, YP, Z)
t_cub = time_derivatives(eos_cub, LOGP, LOGT_VALUES, YP, Z)
print(f"{'Derivative':<15} {'Linear (s)':>12} {'Cubic (s)':>12} {'Ratio':>8}")
print("-" * 50)
for name, _, _ in DERIVATIVES:
    ratio = t_cub[name] / t_lin[name] if t_lin[name] > 0 else float('inf')
    print(f"{name:<15} {t_lin[name]:>12.3f} {t_cub[name]:>12.3f} {ratio:>8.1f}x")
print()

# ---- Compute derivatives ----
print("Computing derivatives ...")
cmap = plt.cm.viridis
norm = Normalize(vmin=LOGT_VALUES.min(), vmax=LOGT_VALUES.max())

for deriv_name, func_name, label in DERIVATIVES:
    print(f"  {deriv_name} ...")
    kw = dict(h=0.2) if deriv_name in ('nabla_ad', 'gamma1') else {}

    fd_lin = eval_deriv_isotherms(eos_lin, func_name, LOGP, LOGT_VALUES,
                                  YP, Z, method='finite_difference', **kw)
    fd_cub = eval_deriv_isotherms(eos_cub, func_name, LOGP, LOGT_VALUES,
                                  YP, Z, method='finite_difference', **kw)
    id_lin = eval_deriv_isotherms(eos_lin, func_name, LOGP, LOGT_VALUES,
                                  YP, Z, method='identity')

    # Scale dsdy_rhop to kb/baryon
    if 'dsdy' in deriv_name or 'dsdz' in deriv_name:
        fd_lin *= erg_to_kbbar
        fd_cub *= erg_to_kbbar
        id_lin *= erg_to_kbbar

    # Use identity result to set a reasonable y-range
    all_id = id_lin[np.isfinite(id_lin)]
    if len(all_id) > 10:
        ylo = np.percentile(all_id, 1)
        yhi = np.percentile(all_id, 99)
        margin = (yhi - ylo) * 0.2
        ylims = (ylo - margin, yhi + margin)
    else:
        ylims = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, lgt in enumerate(LOGT_VALUES):
        c = cmap(norm(lgt))
        axes[0].plot(LOGP, fd_lin[i], '-', color=c, lw=0.8, alpha=0.8)
        axes[1].plot(LOGP, fd_cub[i], '-', color=c, lw=0.8, alpha=0.8)
        axes[2].plot(LOGP, id_lin[i], '-', color=c, lw=0.8, alpha=0.8)

    axes[0].set_title(f'FD (linear RGI)')
    axes[1].set_title(f'FD (cubic RGI)')
    axes[2].set_title(f'Identity (VAL)')

    if ylims is not None:
        for ax in axes:
            ax.set_ylim(ylims)

    for ax in axes:
        ax.set_xlabel(r'$\log_{10}\,P$ [dyn/cm$^2$]')
    axes[0].set_ylabel(label)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label=r'$\log_{10}\,T$ [K]',
                 shrink=0.8)

    fig.suptitle(f"Y' = {YP}, Z = {Z}", fontsize=12)
    plt.tight_layout()
    fname = f'diagnose_{deriv_name}_cubic_vs_linear.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"    Saved {fname}")
    plt.close(fig)

print("\nDone.")
