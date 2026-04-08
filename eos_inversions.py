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
    'z_eos':      'aqua_revised',
    'hg':         True,
    'smooth_hhe': True,
    'smooth_z':   True,
    'mu_h_vary':  False,

    # P-T grid
    'logp_lo':    3.0, # 1 bar
    'logp_hi':    16.0, # 1000 Mbar
    'logp_step':  0.05,
    'logt_lo':    1.5,
    'logt_hi':    6.0,

    # ρ grid (for rhot, rhop, srho)
    'logrho_lo':  -6.0,
    'logrho_hi':  1.6,
    'logrho_step': 0.1,

    # S grid for square tables (SP, S-rho) — ADJUST THESE VALUES
    's_lo':       2.0,    # kb/baryon — lower entropy bound
    's_hi':       12.0,   # kb/baryon — upper entropy bound
    's_step':     0.1,    # kb/baryon — entropy step

    # ξ resolution (for xi-mapped tables only, used with --xi_transform)
    'n_xi':       150,

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
                   choices=['pt', 'sp', 'rhot', 'rhop', 'srho'],
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
                   default=['water_revised'],
                   help="Z species for val_mixtures (default: water_revised)")

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

    # ξ and table mode
    p.add_argument('--xi_transform', action='store_true',
                   help='Use xi-mapped adaptive tables instead of '
                        'square (default: square)')
    p.add_argument('--n_xi', type=int, default=DEFAULTS['n_xi'],
                   help=f"Number of xi grid points for xi-mapped "
                        f"tables (default: {DEFAULTS['n_xi']})")

    # S grid (for square SP and S-rho tables)
    p.add_argument('--s_lo', type=float, default=DEFAULTS['s_lo'],
                   help=f"S min [kb/baryon] for square SP/S-rho "
                        f"tables (default: {DEFAULTS['s_lo']})")
    p.add_argument('--s_hi', type=float, default=DEFAULTS['s_hi'],
                   help=f"S max [kb/baryon] for square SP/S-rho "
                        f"tables (default: {DEFAULTS['s_hi']})")
    p.add_argument('--s_step', type=float, default=DEFAULTS['s_step'],
                   help=f"S step [kb/baryon] for square SP/S-rho "
                        f"tables (default: {DEFAULTS['s_step']})")

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

    # S-ρ specific
    p.add_argument('--srho_basis', default='rhot',
                   choices=['rhot', 'sp'],
                   help="1-D decomposition for srho inversion "
                        "(default: rhot)")
    p.add_argument('--srho_use_tab', default=True,
                   type=lambda x: x.lower() not in ('false', '0', 'no'),
                   help="Use pre-computed rhot/sp table for srho "
                        "inversion (default: True). Set to False "
                        "to use per-point Newton-Raphson.")

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
    print(f"  Table mode:  {'xi-mapped' if args.xi_transform else 'square'}")
    print(f"  logP range:  [{args.logp_lo}, {args.logp_hi}] step {args.logp_step}")
    print(f"  logT range:  [{args.logt_lo}, {args.logt_hi}]")
    if args.basis in ('rhot', 'rhop', 'srho'):
        print(f"  logrho range:[{args.logrho_lo}, {args.logrho_hi}] step {args.logrho_step}")
    if args.xi_transform and args.basis in ('sp', 'rhop', 'srho'):
        print(f"  n_xi:        {args.n_xi}")
    if not args.xi_transform and args.basis in ('sp', 'srho'):
        nS = len(np.arange(args.s_lo, args.s_hi + args.s_step * 0.1, args.s_step))
        print(f"  S range:     [{args.s_lo}, {args.s_hi}] step {args.s_step} ({nS} pts)")
    if args.basis == 'srho':
        print(f"  srho basis:  {args.srho_basis}")
        print(f"  srho use_tab:{args.srho_use_tab}")
    nY, nZ = len(yvals), len(zvals)
    print(f"  Y' grid:     [{yvals[0]:.3f}, {yvals[-1]:.3f}] step {args.y_step} ({nY} pts)")
    print(f"  Z grid:      [{zvals[0]:.3f}, {zvals[-1]:.3f}] step {args.z_step} ({nZ} pts)")

    # Estimate file size
    nP = len(np.arange(args.logp_lo, args.logp_hi + args.logp_step * 0.1, args.logp_step))
    nR = len(np.arange(args.logrho_lo, args.logrho_hi + args.logrho_step * 0.1, args.logrho_step))
    nxi = args.n_xi
    nS = len(np.arange(args.s_lo, args.s_hi + args.s_step * 0.1, args.s_step))
    bytes_per_f32 = 4

    nT = len(np.arange(args.logt_lo, args.logt_hi + 0.01, args.logp_step))

    if args.basis == 'pt':
        n_cells = nP * nT * nY * nZ
        n_arrays = 3   # s_pt, logrho_pt, logu_pt
        bounds_cells = 0
    elif args.basis == 'sp':
        if args.xi_transform:
            n_cells = nxi * nP * nY * nZ
            bounds_cells = nP * nY * nZ * 2   # s_lo + s_hi
        else:
            n_cells = nS * nP * nY * nZ
            bounds_cells = 0
        n_arrays = 1   # logt_sp only
    elif args.basis == 'rhot':
        if args.xi_transform:
            n_cells = nxi * nT * nY * nZ
            bounds_cells = nT * nY * nZ * 2  # rho_lo + rho_hi
        else:
            n_cells = nR * nT * nY * nZ
            bounds_cells = 0
        n_arrays = 1   # logp_rhot only
    elif args.basis == 'rhop':
        if args.xi_transform:
            n_cells = nxi * nP * nY * nZ
            bounds_cells = nP * nY * nZ * 2   # rho_lo + rho_hi
        else:
            n_cells = nR * nP * nY * nZ
            bounds_cells = 0
        n_arrays = 1   # logt_rhop only
    elif args.basis == 'srho':
        if args.xi_transform:
            n_cells = nxi * nR * nY * nZ
            bounds_cells = nR * nY * nZ * 2   # s_lo + s_hi
        else:
            n_cells = nS * nR * nY * nZ
            bounds_cells = 0
        n_arrays = 2   # logp_srho + logt_srho

    est_bytes = (n_cells * n_arrays + bounds_cells) * bytes_per_f32
    est_mb = est_bytes / 1e6
    print(f"  Est. size:   {n_cells * n_arrays:,} cells → "
          f"~{est_mb:.0f} MB (float32, uncompressed)")
    print("-" * 65)

    # --- Create EOS ---
    # pt_tab=True uses the smooth P-T table for forward-model evaluation
    # during inversions. inv_tab=False avoids loading inverted tables
    # (which are what we're building). For --basis pt, both are off
    # since we build from raw VAL.
    #
    # For --basis srho, we set inv_tab=True so the required dependency
    # table (ρ-T or S-P) is auto-loaded, and srho_tab=False so the
    # old S-ρ table is NOT loaded (we're building a fresh one).
    _pt_tab  = (args.basis != 'pt')  # PT basis builds from VAL
    _inv_tab = (args.basis == 'srho' and args.srho_use_tab)
    eos = hhe_z_mixtures(
        hhe_eos_name=args.hhe_eos,
        hg=hg,
        smooth_hhe=smooth,
        smooth_z=smooth,
        mu_h_vary=mu_h_vary,
        species_list=args.species,
        z_eos=args.z_eos,
        pt_tab=_pt_tab,
        inv_tab=_inv_tab,
        srho_tab=False,
        xi_transform=args.xi_transform,
        logp_range=(args.logp_lo, args.logp_hi),
        logp_step=args.logp_step,
        logt_range=(args.logt_lo, args.logt_hi),
        logrho_range=(args.logrho_lo, args.logrho_hi),
        logrho_step=args.logrho_step,
        n_xi=args.n_xi,
    )

    # --- Build table ---
    t0 = time.time()

    if args.basis == 'pt':
        result = eos.build_pt_table(yvals, zvals)
        eos.save_pt_table(result, path=args.output)

    elif args.basis == 'sp':
        result = eos.build_sp_table(yvals, zvals, n_xi=args.n_xi,
                                    xi_transform=args.xi_transform,
                                    s_lo=args.s_lo, s_hi=args.s_hi,
                                    s_step=args.s_step)
        eos.save_sp_table(result, path=args.output)

    elif args.basis == 'rhot':
        result = eos.build_rhot_table(yvals, zvals, n_xi=args.n_xi,
                                      xi_transform=args.xi_transform)
        eos.save_rhot_table(result, path=args.output)

    elif args.basis == 'rhop':
        result = eos.build_rhop_table(yvals, zvals, n_xi=args.n_xi,
                                      xi_transform=args.xi_transform)
        eos.save_rhop_table(result, path=args.output)

    elif args.basis == 'srho':
        # The S-ρ inversion uses a 1-D decomposition.  When
        # use_tab=True, the required dependency table (ρ-T or S-P)
        # was auto-loaded via inv_tab=True above.  Verify it's there.
        if args.srho_use_tab:
            if args.srho_basis == 'rhot' and eos._logp_rhot_rgi is None:
                if args.xi_transform:
                    rhot_path = eos._table_path('rhot')
                else:
                    rhot_path = eos._table_path_square('rhot')
                print(f"ERROR: rho-T table not found at {rhot_path}")
                print("Build it first:  python eos_inversions.py "
                      "--basis rhot ...")
                print("Or set --srho_use_tab False to use "
                      "per-point Newton-Raphson.")
                sys.exit(1)
            if args.srho_basis == 'sp' and eos._logt_sp_rgi is None:
                if args.xi_transform:
                    sp_path = eos._table_path('sp')
                else:
                    sp_path = eos._table_path_square('sp')
                print(f"ERROR: S-P table not found at {sp_path}")
                print("Build it first:  python eos_inversions.py "
                      "--basis sp ...")
                print("Or set --srho_use_tab False to use "
                      "per-point Newton-Raphson.")
                sys.exit(1)

        result = eos.build_srho_table(yvals, zvals, n_xi=args.n_xi,
                                      xi_transform=args.xi_transform,
                                      s_lo=args.s_lo, s_hi=args.s_hi,
                                      s_step=args.s_step,
                                      basis=args.srho_basis,
                                      use_tab=args.srho_use_tab)
        eos.save_srho_table(result, path=args.output)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print(f"\nTotal time: {hours}h {mins}m {secs:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
