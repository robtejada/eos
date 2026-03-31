#!/usr/bin/env python
"""
EOS Table Inversion Script
==========================

Inverts the P-T basis EOS tables to any target basis:
    sp   : T(S, P, Y', Z)          — entropy-pressure
    rhot : P(ρ, T, Y', Z)          — density-temperature
    rhop : T(ρ, P, Y', Z)          — density-pressure  (ξ-mapped on ρ)
    srho : P,T(S, ρ, Y', Z)        — entropy-density   (ξ-mapped on S, 2-D)

Usage
-----
    python eos_inversions.py --basis sp --hhe_eos cd --z_eos water
    python eos_inversions.py --basis rhot --hhe_eos cms
    python eos_inversions.py --basis srho --hhe_eos cd --z_eos water --logp_lo 6 --logp_hi 14

All parameters have sensible defaults. Tables are saved to the auto-load
paths used by ``hhe_z_mixtures(tab=True)``.
"""

import argparse
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eos.eos_class import hhe_z_mixtures


# =====================================================================
# Defaults
# =====================================================================

DEFAULTS = {
    'hhe_eos':    'cd',
    'z_eos':      'aqua',
    'hg':         True,
    'smooth_hhe': True,
    'smooth_z':   True,
    'mu_h_vary':  True,

    # P-T grid
    'logp_lo':    5.0, # 0.1 bar
    'logp_hi':    15.0, # 1000 Mbar
    'logp_step':  0.05,
    'logt_lo':    2.0,
    'logt_hi':    6.0,

    # ρ grid (for rhot, rhop, srho)
    'logrho_lo':  -8.0,
    'logrho_hi':  2.0,
    'logrho_step': 0.05,

    # ξ resolution (for sp, rhop, srho)
    'n_xi':       100,

    # Y' and Z grids
    'y_lo':       0.05,
    'y_hi':       1.0,
    'y_step':     0.05,
    'z_lo':       0.00,
    'z_hi':       1.0,
    'z_step':     0.02,
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--basis', required=True,
                   choices=['sp', 'rhot', 'rhop', 'srho'],
                   help='Target basis for inversion')

    # EOS selection
    p.add_argument('--hhe_eos', default=DEFAULTS['hhe_eos'],
                   choices=['cms', 'cd'],
                   help=f"H-He EOS (default: {DEFAULTS['hhe_eos']})")
    p.add_argument('--z_eos', default=DEFAULTS['z_eos'],
                   help=f"Z EOS label for filename (default: {DEFAULTS['z_eos']})")
    p.add_argument('--no_hg', action='store_true',
                   help='Disable HG23 non-ideal mixing')
    p.add_argument('--no_smooth', action='store_true',
                   help='Disable H-He and Z smoothing')
    p.add_argument('--no_mu_h_vary', action='store_true',
                   help='Disable P-T dependent mu_H')
    p.add_argument('--species', nargs='+',
                   default=['water'],
                   help="Z species for val_mixtures (default: water)")

    # P grid
    p.add_argument('--logp_lo', type=float, default=DEFAULTS['logp_lo'],
                   help=f"log10 P_min [dyn/cm²] (default: {DEFAULTS['logp_lo']})")
    p.add_argument('--logp_hi', type=float, default=DEFAULTS['logp_hi'],
                   help=f"log10 P_max [dyn/cm²] (default: {DEFAULTS['logp_hi']})")
    p.add_argument('--logp_step', type=float, default=DEFAULTS['logp_step'],
                   help=f"logP step (default: {DEFAULTS['logp_step']})")

    # T range
    p.add_argument('--logt_lo', type=float, default=DEFAULTS['logt_lo'],
                   help=f"log10 T_min [K] (default: {DEFAULTS['logt_lo']})")
    p.add_argument('--logt_hi', type=float, default=DEFAULTS['logt_hi'],
                   help=f"log10 T_max [K] (default: {DEFAULTS['logt_hi']})")

    # ρ grid
    p.add_argument('--logrho_lo', type=float, default=DEFAULTS['logrho_lo'],
                   help=f"log10 rho_min [g/cm³] (default: {DEFAULTS['logrho_lo']})")
    p.add_argument('--logrho_hi', type=float, default=DEFAULTS['logrho_hi'],
                   help=f"log10 rho_max [g/cm³] (default: {DEFAULTS['logrho_hi']})")
    p.add_argument('--logrho_step', type=float, default=DEFAULTS['logrho_step'],
                   help=f"logrho step (default: {DEFAULTS['logrho_step']})")

    # ξ
    p.add_argument('--n_xi', type=int, default=DEFAULTS['n_xi'],
                   help=f"Number of xi grid points (default: {DEFAULTS['n_xi']})")

    # Y' grid
    p.add_argument('--y_lo', type=float, default=DEFAULTS['y_lo'],
                   help=f"Y' min (default: {DEFAULTS['y_lo']})")
    p.add_argument('--y_hi', type=float, default=DEFAULTS['y_hi'],
                   help=f"Y' max (default: {DEFAULTS['y_hi']})")
    p.add_argument('--y_step', type=float, default=DEFAULTS['y_step'],
                   help=f"Y' step (default: {DEFAULTS['y_step']})")

    # Z grid
    p.add_argument('--z_lo', type=float, default=DEFAULTS['z_lo'],
                   help=f"Z min (default: {DEFAULTS['z_lo']})")
    p.add_argument('--z_hi', type=float, default=DEFAULTS['z_hi'],
                   help=f"Z max (default: {DEFAULTS['z_hi']})")
    p.add_argument('--z_step', type=float, default=DEFAULTS['z_step'],
                   help=f"Z step (default: {DEFAULTS['z_step']})")

    # Output
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: auto from hhe_eos/z_eos)')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Build grids ---
    yvals = np.arange(args.y_lo, args.y_hi + args.y_step * 0.1, args.y_step)
    zvals = np.arange(args.z_lo, args.z_hi + args.z_step * 0.1, args.z_step)

    hg = not args.no_hg
    smooth = not args.no_smooth
    mu_h_vary = not args.no_mu_h_vary

    # --- Print summary ---
    print("=" * 65)
    print(f"EOS Table Inversion: P,T → {args.basis.upper()}")
    print("=" * 65)
    print(f"  H-He EOS:    {args.hhe_eos}")
    print(f"  Z EOS:       {args.z_eos}")
    print(f"  Species:     {args.species}")
    print(f"  HG23:        {hg}")
    print(f"  Smoothing:   {smooth}")
    print(f"  mu_H(P,T):   {mu_h_vary}")
    print(f"  logP range:  [{args.logp_lo}, {args.logp_hi}] step {args.logp_step}")
    print(f"  logT range:  [{args.logt_lo}, {args.logt_hi}]")
    if args.basis in ('rhot', 'rhop', 'srho'):
        print(f"  logrho range:[{args.logrho_lo}, {args.logrho_hi}] step {args.logrho_step}")
    if args.basis in ('sp', 'rhop', 'srho'):
        print(f"  n_xi:        {args.n_xi}")
    nY, nZ = len(yvals), len(zvals)
    print(f"  Y' grid:     [{yvals[0]:.3f}, {yvals[-1]:.3f}] step {args.y_step} ({nY} pts)")
    print(f"  Z grid:      [{zvals[0]:.3f}, {zvals[-1]:.3f}] step {args.z_step} ({nZ} pts)")

    # Estimate file size
    nP = len(np.arange(args.logp_lo, args.logp_hi + args.logp_step * 0.1, args.logp_step))
    nR = len(np.arange(args.logrho_lo, args.logrho_hi + args.logrho_step * 0.1, args.logrho_step))
    nxi = args.n_xi
    bytes_per_f32 = 4

    if args.basis == 'sp':
        n_cells = nxi * nP * nY * nZ
        n_arrays = 1   # logt_sp only
        bounds_cells = nP * nY * nZ * 2   # s_lo + s_hi
    elif args.basis == 'rhot':
        nT = len(np.arange(args.logt_lo, args.logt_hi + 0.01, args.logp_step))
        n_cells = nR * nT * nY * nZ
        n_arrays = 1   # logp_rhot only
        bounds_cells = 0
    elif args.basis == 'rhop':
        n_cells = nxi * nP * nY * nZ
        n_arrays = 1   # logt_rhop only
        bounds_cells = nP * nY * nZ * 2   # rho_lo + rho_hi
    elif args.basis == 'srho':
        n_cells = nxi * nR * nY * nZ
        n_arrays = 2   # logp_srho + logt_srho
        bounds_cells = nR * nY * nZ * 2   # s_lo + s_hi

    est_bytes = (n_cells * n_arrays + bounds_cells) * bytes_per_f32
    est_mb = est_bytes / 1e6
    print(f"  Est. size:   {n_cells * n_arrays:,} cells → "
          f"~{est_mb:.0f} MB (float32, uncompressed)")
    print("-" * 65)

    # --- Create EOS ---
    eos = hhe_z_mixtures(
        hhe_eos_name=args.hhe_eos,
        hg=hg,
        smooth_hhe=smooth,
        smooth_z=smooth,
        mu_h_vary=mu_h_vary,
        species_list=args.species,
        z_eos=args.z_eos,
        tab=False,
        logp_range=(args.logp_lo, args.logp_hi),
        logp_step=args.logp_step,
        logt_range=(args.logt_lo, args.logt_hi),
        logrho_range=(args.logrho_lo, args.logrho_hi),
        logrho_step=args.logrho_step,
        n_xi=args.n_xi,
    )

    # --- Build table ---
    t0 = time.time()

    if args.basis == 'sp':
        result = eos.build_sp_table(yvals, zvals, n_xi=args.n_xi)
        eos.save_sp_table(result, path=args.output)

    elif args.basis == 'rhot':
        result = eos.build_rhot_table(yvals, zvals)
        eos.save_rhot_table(result, path=args.output)

    elif args.basis == 'rhop':
        result = eos.build_rhop_table(yvals, zvals, n_xi=args.n_xi)
        eos.save_rhop_table(result, path=args.output)

    elif args.basis == 'srho':
        result = eos.build_srho_table(yvals, zvals, n_xi=args.n_xi)
        eos.save_srho_table(result, path=args.output)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print(f"\nTotal time: {hours}h {mins}m {secs:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
