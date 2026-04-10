#!/usr/bin/env python
"""Verification script for hhe_z_mixtures derivative restructuring.

Compares finite-difference and identity methods for all composition
derivatives on a logP x logT grid at fixed Y' and Z.

Usage:
    python verify_derivatives.py
"""

import sys
import os
# Ensure parent directory is on path so 'eos' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from eos.eos_class import hhe_z_mixtures

# ---- Configuration ----
YP = 0.27        # Y' = Y/(1-Z)
Z  = 0.02        # total metal fraction
LOGP_GRID = np.linspace(8.0, 13.5, 30)
LOGT_GRID = np.linspace(2.5, 4.3, 30)

# ---- Initialize EOS with all tables (square) ----
print("Loading EOS with pt_tab=True, inv_tab=True, srho_tab=True, "
      "xi_transform=False ...")
eos = hhe_z_mixtures(pt_tab=True, inv_tab=True, srho_tab=True,
                     xi_transform=False)
print("Done.\n")


def compute_derivative_grid(get_func, logp_arr, logt_arr, yp, z,
                            method='finite_difference'):
    """Compute a derivative on a 2D grid of (logP, logT)."""
    nP, nT = len(logp_arr), len(logt_arr)
    out = np.full((nP, nT), np.nan)
    for i, lgp in enumerate(logp_arr):
        for j, lgt in enumerate(logt_arr):
            try:
                val = get_func(lgp, lgt, yp, z, method=method)
                out[i, j] = float(np.atleast_1d(val)[0])
            except Exception:
                pass
    return out


def plot_comparison(name, fd_grid, id_grid, logp_arr, logt_arr, save_prefix):
    """Plot FD vs identity and fractional difference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    P, T = np.meshgrid(logp_arr, logt_arr, indexing='ij')

    # FD result
    ax = axes[0]
    im = ax.pcolormesh(T, P, fd_grid, shading='auto')
    ax.set_xlabel('log T [K]')
    ax.set_ylabel('log P [dyn/cm²]')
    ax.set_title(f'{name}\n(finite_difference)')
    plt.colorbar(im, ax=ax)

    # Identity result
    ax = axes[1]
    im = ax.pcolormesh(T, P, id_grid, shading='auto')
    ax.set_xlabel('log T [K]')
    ax.set_ylabel('log P [dyn/cm²]')
    ax.set_title(f'{name}\n(identity)')
    plt.colorbar(im, ax=ax)

    # Fractional difference
    ax = axes[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(np.abs(fd_grid) > 1e-30, fd_grid, np.nan)
        frac_diff = (id_grid - fd_grid) / denom
    vmax = np.nanpercentile(np.abs(frac_diff), 95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.pcolormesh(T, P, frac_diff, shading='auto',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('log T [K]')
    ax.set_ylabel('log P [dyn/cm²]')
    ax.set_title(f'Fractional diff\n(identity - FD) / FD')
    plt.colorbar(im, ax=ax)

    fig.suptitle(f"Y' = {YP}, Z = {Z}", fontsize=12)
    plt.tight_layout()
    fname = f'{save_prefix}_{name.replace("/", "_").replace("|", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ---- Derivatives to verify ----
derivatives = [
    ('dsdy_rhop', eos.get_dsdy_rhop),
    ('dsdz_rhop', eos.get_dsdz_rhop),
    ('dtdy_srho', eos.get_dtdy_srho),
    ('dtdz_srho', eos.get_dtdz_srho),
    ('dudy_srho', eos.get_dudy_srho),
    ('dudz_srho', eos.get_dudz_srho),
]

print("Computing derivatives on logP x logT grid ...")
for name, func in derivatives:
    print(f"  {name} ...")
    fd_grid = compute_derivative_grid(func, LOGP_GRID, LOGT_GRID, YP, Z,
                                      method='finite_difference')
    id_grid = compute_derivative_grid(func, LOGP_GRID, LOGT_GRID, YP, Z,
                                      method='identity')
    plot_comparison(name, fd_grid, id_grid, LOGP_GRID, LOGT_GRID,
                    'verify_deriv')

print("\nDone. Plots saved as verify_deriv_*.png")
