#!/usr/bin/env python
"""
EOS Table Builder / Inversion Script
====================================

Builds the P-T basis EOS table from the volume-addition-law forward
model, or inverts an existing P-T basis table to any of four target
bases:

    pt   : S, log10 ρ, log10 U on (logP, logT, Y', Z)   — forward
    sp   : T(S, P, Y', Z)                                — entropy-pressure
    rhot : P(ρ, T, Y', Z)                                — density-temperature
    rhop : T(ρ, P, Y', Z)                                — density-pressure
    srho : P, T(S, ρ, Y', Z)                             — entropy-density (2 outputs)

All inversion bases are stored on rectangular grids (the legacy
ξ-mapped option has been retired).

Build order
-----------
The P-T table must exist before any of the inversions can run, since
the inversions all use ``pt_tab=True`` to read the smooth P-T forward
model.  The S-ρ inversion additionally needs either the ρ-T or the
S-P table (see ``--srho_basis``).

Usage
-----
    # Build P-T forward table first (no per-component smoothing):
    python eos/eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised
    python eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised

    # Then build the inversions (any order, except srho is last):
    python eos_inversions.py --basis sp   --hhe_eos cd --z_eos aqua_revised
    python eos_inversions.py --basis rhot --hhe_eos cms
    python eos_inversions.py --basis rhop --hhe_eos cd --z_eos aqua_revised
    python eos_inversions.py --basis srho --hhe_eos cd --z_eos aqua_revised \\
                             --srho_basis rhot

    # Customise the P/T grid for any basis:
    python eos_inversions.py --basis srho --hhe_eos cd --z_eos aqua_revised \\
                             --logp_lo 6 --logp_hi 14

    # Turn on per-component smoothing of the H-He and/or Z tables
    # before VAL mixing (off by default; produces smoother PT
    # surfaces but blurs real features like H2 dissociation steps):
    python eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised \\
                             --smooth_hhe --smooth_z

All parameters have sensible defaults.  Tables are saved to the
auto-load paths used by ``hhe_z_mixtures(pt_tab=True, inv_tab=True)``.

Default toggles
---------------
    HG23 non-ideal mixing  : ON   (disable with --no_hg)
    P-T-dependent mu_H     : OFF  (enable  with --mu_h_vary)
    H-He smoothing         : OFF  (enable  with --smooth_hhe)
    Z smoothing            : OFF  (enable  with --smooth_z)
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
    'hg':         True,        # opt-out via --no_hg
    'smooth_hhe': False,       # opt-in via --smooth_hhe
    'smooth_z':   False,       # opt-in via --smooth_z
    'mu_h_vary':  False,       # opt-in via --mu_h_vary

    # P-T grid
    'logp_lo':    5.0, # 1 mbar
    'logp_hi':    15.1, # 1000 Mbar
    'logp_step':  0.05,
    'logt_lo':    1.3,
    'logt_hi':    6.0,

    # ρ grid (for rhot, rhop, srho)
    'logrho_lo':  -6.0,
    'logrho_hi':  1.76,
    'logrho_step': 0.1,

    # S grid for SP and S-ρ tables
    's_lo':       2.0,    # kb/baryon — lower entropy bound
    's_hi':       12.1,   # kb/baryon — upper entropy bound
    's_step':     0.1,    # kb/baryon — entropy step

    # Y' and Z grids
    'y_lo':       0.05,
    'y_hi':       1.05,
    'y_step':     0.05,
    'z_lo':       0.00,
    'z_hi':       1.02,
    'z_step':     0.02,
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--basis', required=True,
                   choices=['pt', 'sp', 'rhot', 'rhop', 'srho'],
                   help='Target basis (pt = forward; others = inversions)')

    # EOS selection
    p.add_argument('--hhe_eos', default=DEFAULTS['hhe_eos'],
                   choices=['cms', 'cd'],
                   help=f"H-He EOS (default: {DEFAULTS['hhe_eos']})")
    p.add_argument('--z_eos', default=DEFAULTS['z_eos'],
                   help=f"Z EOS label for filename (default: {DEFAULTS['z_eos']})")
    p.add_argument('--no_hg', action='store_true',
                   help='Disable HG23 non-ideal mixing (default: enabled)')
    p.add_argument('--smooth_hhe', action='store_true',
                   help='Enable Gaussian smoothing of the H-He '
                        'component table before VAL mixing '
                        '(default: disabled)')
    p.add_argument('--smooth_z', action='store_true',
                   help='Enable Gaussian smoothing of the Z '
                        'component tables before VAL mixing '
                        '(default: disabled)')
    p.add_argument('--mu_h_vary', action='store_true',
                   help='Enable P-T dependent hydrogen molecular '
                        'weight mu_H (default: disabled, i.e. '
                        'constant atomic mu_H)')
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

    # S grid (for SP and S-rho tables)
    p.add_argument('--s_lo', type=float, default=DEFAULTS['s_lo'],
                   help=f"S min [kb/baryon] for SP/S-rho "
                        f"tables (default: {DEFAULTS['s_lo']})")
    p.add_argument('--s_hi', type=float, default=DEFAULTS['s_hi'],
                   help=f"S max [kb/baryon] for SP/S-rho "
                        f"tables (default: {DEFAULTS['s_hi']})")
    p.add_argument('--s_step', type=float, default=DEFAULTS['s_step'],
                   help=f"S step [kb/baryon] for SP/S-rho "
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

    # Post-inversion smoothing
    p.add_argument('--smooth_sigma', type=float, default=0.0,
                   help="Gaussian smoothing sigma (grid cells) "
                        "applied to inversion tables after Hampel "
                        "filtering along the two physical axes. "
                        "Recommended: 0.5. (default: 0.0 = off)")

    # Output
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: auto from hhe_eos/z_eos)')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Build grids ---
    yvals = np.arange(args.y_lo, args.y_hi, args.y_step)
    zvals = np.arange(args.z_lo, args.z_hi, args.z_step)

    hg = not args.no_hg
    smooth_hhe = args.smooth_hhe
    smooth_z = args.smooth_z
    mu_h_vary = args.mu_h_vary

    # --- Print summary ---
    print("=" * 65)
    if args.basis == 'pt':
        print(f"EOS Table Build: P-T forward (S, ρ, U)")
    else:
        print(f"EOS Table Inversion: P,T → {args.basis.upper()}")
    print("=" * 65)
    print(f"  H-He EOS:    {args.hhe_eos}")
    print(f"  Z EOS:       {args.z_eos}")
    print(f"  Species:     {args.species}")
    print(f"  HG23:        {hg}")
    print(f"  Smooth H-He: {smooth_hhe}")
    print(f"  Smooth Z:    {smooth_z}")
    print(f"  mu_H(P,T):   {mu_h_vary}")
    print(f"  logP range:  [{args.logp_lo}, {args.logp_hi}] step {args.logp_step}")
    print(f"  logT range:  [{args.logt_lo}, {args.logt_hi}]")
    if args.basis in ('rhot', 'rhop', 'srho'):
        print(f"  logrho range:[{args.logrho_lo}, {args.logrho_hi}] step {args.logrho_step}")
    if args.basis in ('sp', 'srho'):
        nS = len(np.arange(args.s_lo, args.s_hi + args.s_step * 0.1, args.s_step))
        print(f"  S range:     [{args.s_lo}, {args.s_hi}] step {args.s_step} ({nS} pts)")
    if args.basis == 'srho':
        print(f"  srho basis:  {args.srho_basis}")
        print(f"  srho use_tab:{args.srho_use_tab}")
    if args.smooth_sigma > 0:
        print(f"  smooth_sigma:{args.smooth_sigma}")
    nY, nZ = len(yvals), len(zvals)
    print(f"  Y' grid:     [{yvals[0]:.3f}, {yvals[-1]:.3f}] step {args.y_step} ({nY} pts)")
    print(f"  Z grid:      [{zvals[0]:.3f}, {zvals[-1]:.3f}] step {args.z_step} ({nZ} pts)")

    # Estimate file size
    nP = len(np.arange(args.logp_lo, args.logp_hi, args.logp_step))
    nR = len(np.arange(args.logrho_lo, args.logrho_hi, args.logrho_step))
    nS = len(np.arange(args.s_lo, args.s_hi, args.s_step))
    nT = len(np.arange(args.logt_lo, args.logt_hi, args.logp_step))
    bytes_per_f32 = 4

    if args.basis == 'pt':
        n_cells = nP * nT * nY * nZ
        n_arrays = 3   # s_pt, logrho_pt, logu_pt
    elif args.basis == 'sp':
        n_cells = nS * nP * nY * nZ
        n_arrays = 1   # logt_sp only
    elif args.basis == 'rhot':
        n_cells = nR * nT * nY * nZ
        n_arrays = 1   # logp_rhot only
    elif args.basis == 'rhop':
        n_cells = nR * nP * nY * nZ
        n_arrays = 1   # logt_rhop only
    elif args.basis == 'srho':
        n_cells = nS * nR * nY * nZ
        n_arrays = 2   # logp_srho + logt_srho

    est_bytes = n_cells * n_arrays * bytes_per_f32
    est_mb = est_bytes / 1e6
    print(f"  Est. size:   {n_cells * n_arrays:,} cells → "
          f"~{est_mb:.0f} MB (float32, uncompressed)")
    print("-" * 65)

    # --- Create EOS ---
    # pt_tab=True uses the smooth P-T table for forward-model evaluation
    # during inversions.  inv_tab=False avoids loading inverted tables
    # (which are what we're building).  For --basis pt, both are off
    # since we build from raw VAL.
    #
    # For --basis srho with srho_use_tab=True, we set inv_tab=True so
    # the required dependency table (ρ-T or S-P) is auto-loaded.
    # srho_tab=False because we're building a fresh S-ρ table.
    _pt_tab  = (args.basis != 'pt')  # PT basis builds from VAL
    _inv_tab = (args.basis == 'srho' and args.srho_use_tab)
    eos = hhe_z_mixtures(
        hhe_eos_name=args.hhe_eos,
        hg=hg,
        smooth_hhe=smooth_hhe,
        smooth_z=smooth_z,
        mu_h_vary=mu_h_vary,
        species_list=args.species,
        z_eos=args.z_eos,
        pt_tab=_pt_tab,
        inv_tab=_inv_tab,
        srho_tab=False,
        logp_range=(args.logp_lo, args.logp_hi),
        logp_step=args.logp_step,
        logt_range=(args.logt_lo, args.logt_hi),
        logrho_range=(args.logrho_lo, args.logrho_hi),
        logrho_step=args.logrho_step,
    )

    # --- Build table ---
    t0 = time.time()

    if args.basis == 'pt':
        result = eos.build_pt_table(yvals, zvals)
        eos.save_pt_table(result, path=args.output)

    elif args.basis == 'sp':
        result = eos.build_sp_table(yvals, zvals,
                                    s_lo=args.s_lo, s_hi=args.s_hi,
                                    s_step=args.s_step,
                                    smooth_sigma=args.smooth_sigma)
        eos.save_sp_table(result, path=args.output)

    elif args.basis == 'rhot':
        result = eos.build_rhot_table(yvals, zvals,
                                      smooth_sigma=args.smooth_sigma)
        eos.save_rhot_table(result, path=args.output)

    elif args.basis == 'rhop':
        result = eos.build_rhop_table(yvals, zvals,
                                      smooth_sigma=args.smooth_sigma)
        eos.save_rhop_table(result, path=args.output)

    elif args.basis == 'srho':
        # The S-ρ inversion uses a 1-D decomposition.  When
        # use_tab=True, the required dependency table (ρ-T or S-P)
        # was auto-loaded via inv_tab=True above.  Verify it's there.
        if args.srho_use_tab:
            if args.srho_basis == 'rhot' and eos._logp_rhot_rgi is None:
                rhot_path = eos._table_path('rhot')
                print(f"ERROR: rho-T table not found at {rhot_path}")
                print("Build it first:  python eos_inversions.py "
                      "--basis rhot ...")
                print("Or set --srho_use_tab False to use "
                      "per-point Newton-Raphson.")
                sys.exit(1)
            if args.srho_basis == 'sp' and eos._logt_sp_rgi is None:
                sp_path = eos._table_path('sp')
                print(f"ERROR: S-P table not found at {sp_path}")
                print("Build it first:  python eos_inversions.py "
                      "--basis sp ...")
                print("Or set --srho_use_tab False to use "
                      "per-point Newton-Raphson.")
                sys.exit(1)

        result = eos.build_srho_table(yvals, zvals,
                                      s_lo=args.s_lo, s_hi=args.s_hi,
                                      s_step=args.s_step,
                                      basis=args.srho_basis,
                                      use_tab=args.srho_use_tab,
                                      smooth_sigma=args.smooth_sigma)
        eos.save_srho_table(result, path=args.output)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print(f"\nTotal time: {hours}h {mins}m {secs:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
