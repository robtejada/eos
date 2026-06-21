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
model.  The S-ρ inversion additionally needs the ρ-T inversion
table — it does a 1-D outer Newton in T with the inner P(ρ, T) step
served by the ρ-T table — so build ρ-T before S-ρ.

Usage
-----
    # Build P-T forward table first (no per-component smoothing):
    python eos/eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised

    # Then build the inversions (any order, except srho is last):
    python eos_inversions.py --basis sp   --hhe_eos cd --z_eos aqua_revised
    python eos_inversions.py --basis rhot --hhe_eos cms
    python eos_inversions.py --basis rhop --hhe_eos cd --z_eos aqua_revised
    python eos_inversions.py --basis srho --hhe_eos cd --z_eos aqua_revised

    # Customise the P/T grid for any basis:
    python eos_inversions.py --basis srho --hhe_eos cd --z_eos aqua_revised \\
                             --logp_lo 6 --logp_hi 14

    # Turn on per-component smoothing of the H-He and/or Z tables
    # before VAL mixing (off by default; produces smoother PT
    # surfaces but blurs real features like H2 dissociation steps):
    python eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised \\
                             --smooth_hhe --smooth_z

    # Turn on post-inversion smoothing (Hampel + Gaussian sigma=1) on
    # the inverted table itself:
    python eos_inversions.py --basis sp --hhe_eos cd --z_eos aqua_revised \\
                             --smooth_inverted

All parameters have sensible defaults.  Tables are always written to
(and auto-loaded from) the canonical ``*_<basis>_square.npz`` paths
used by ``hhe_z_mixtures(pt_tab=True, inv_tab=True)``.

Reference recipe (full smoothed pipeline used to generate the committed tables)
-------------------------------------------------------------------------------
The five commands below reproduce the production smooth tables for the
``cd + aqua_revised`` configuration.  Run in order — PT first (so its
output is on disk for the inversions to read), then ρ-T (so srho can
read it), then any of {sp, rhop, srho}.

    # 1) PT forward, with per-component smoothing of H-He and Z
    #    component tables before VAL mixing:
    python eos/eos_inversions.py --basis pt --hhe_eos cd --z_eos aqua_revised \\
                                 --smooth_hhe --smooth_z

    # 2) rho-T inversion on an extended (logP, logrho) grid, with
    #    post-inversion Hampel + Gaussian smoothing:
    python eos/eos_inversions.py --basis rhot --hhe_eos cd --z_eos aqua_revised \\
                                 --logp_lo 0.0 --logp_hi 16 --logrho_lo -8.0 \\
                                 --n_workers 10 --smooth_inverted

    # 3) S-P inversion:
    python eos/eos_inversions.py --basis sp --hhe_eos cd --z_eos aqua_revised \\
                                 --n_workers 10 --smooth_inverted

    # 4) rho-P inversion:
    python eos/eos_inversions.py --basis rhop --hhe_eos cd --z_eos aqua_revised \\
                                 --n_workers 10 --smooth_inverted

    # 5) S-rho inversion (extended S range to S=40 to cover low-P /
    #    high-T queries in the Sackur-Tetrode regime; finer S step,
    #    coarser Z step to keep the file size manageable):
    python eos/eos_inversions.py --basis srho --hhe_eos cd --z_eos aqua_revised \\
                                 --logrho_lo -8.0 --logp_lo 1.0 --logp_hi 16.0 \\
                                 --s_lo 2.0 --s_hi 40.0 --s_step 0.05 \\
                                 --z_step 0.05 \\
                                 --n_workers 10 --smooth_inverted

Rock mass fraction
------------------
By default the metal (Z) component is pure water (``--species
water_revised``).  To build a table set whose metal is a fixed
water/rock mixture, pass ``--f_rock`` — the rock (mg2sio4) mass fraction
*within* the metal budget Z (the nested sub-fraction ``_zr``, with
``_zm = _za = 0``).  For example a 50/50 water/rock set:

    for b in pt rhot sp rhop srho; do
        python eos/eos_inversions.py --basis $b --hhe_eos cd \\
            --z_eos aqua_revised --f_rock 0.5 [--other-grid-flags ...]
    done

When ``--f_rock > 0`` the script auto-adds 'mg2sio4' to the species list
and tags every output file with ``frock{f_rock:.2f}`` (e.g.
``cd_aqua_revised_pt_square_frock0.50.npz``) so the pure-water
``_square.npz`` tables are never overwritten.  Load such a set with
``hhe_z_mixtures(..., table_suffix='frock0.50')``.  ``--f_rock 0``
(default) reproduces the canonical pure-water tables and naming.

Default toggles
---------------
    HG23 non-ideal mixing      : ON   (disable with --no_hg)
    P-T-dependent mu_H         : OFF  (enable  with --mu_h_vary)
    H-He smoothing             : OFF  (enable  with --smooth_hhe)
    Z smoothing                : OFF  (enable  with --smooth_z)
    Post-inversion smoothing   : OFF  (enable  with --smooth_inverted)
    Rock fraction _zr          : 0.0  (set     with --f_rock)
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
    'logp_lo':    5.0, # 0.1 bar
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
    p.add_argument('--f_rock', type=float, default=0.0,
                   help="Rock (mg2sio4) mass fraction WITHIN the metal "
                        "budget Z, i.e. the nested sub-fraction _zr "
                        "(_zm=_za=0).  f_rock=0 -> pure water (default); "
                        "f_rock=0.5 -> 50/50 water/rock; f_rock=1 -> pure "
                        "rock.  When >0, 'mg2sio4' is auto-added to the "
                        "species list and the output filename is tagged "
                        "with 'frock{f_rock:.2f}' so it does not overwrite "
                        "the pure-water _square.npz tables.")

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

    # Post-inversion smoothing
    p.add_argument('--smooth_inverted', action='store_true',
                   help="Apply Hampel + light Gaussian (sigma=1 grid "
                        "cell) smoothing along the two physical axes "
                        "of the inverted table after NaN-fill.  "
                        "Composition axes (Y' and Z) are deliberately "
                        "not smoothed.  Default off -- builds a raw "
                        "inversion with no post-processing.  Disjoint "
                        "from --smooth_hhe / --smooth_z (those smooth "
                        "the underlying H-He / Z component tables).  "
                        "Has no effect for --basis pt.")

    # Parallelism
    p.add_argument('--n_workers', type=int, default=1,
                   help="Number of worker processes for the inversion "
                        "build (default: 1 = serial).  Y' rows are "
                        "split across workers.  Each worker constructs "
                        "its own EOS instance once at startup.  "
                        "Capped at the number of Y' rows.  Has no "
                        "effect for --basis pt (PT build is already "
                        "fully vectorised).")

    # Output
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: auto from hhe_eos/z_eos)')
    p.add_argument('--suffix', type=str, default='',
                   help="Optional tag appended to the auto-output "
                        "filename, e.g. --suffix highz produces "
                        "{hhe}_{z}_<basis>_square_highz.npz instead "
                        "of overwriting the canonical _square.npz "
                        "table. Pair with table_suffix='highz' when "
                        "loading from hhe_z_mixtures(...).")

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

    # --- Rock mass fraction (nested sub-fraction _zr; _zm = _za = 0) ---
    f_rock = float(args.f_rock)
    if not (0.0 <= f_rock <= 1.0):
        parser.error(f"--f_rock must be in [0, 1] (got {f_rock})")

    # The metal mixture needs the mg2sio4 EOS loaded whenever any rock is
    # present.  Auto-add it (canonical role key 'mg2sio4') if the user
    # did not include a rock species, so the rock fraction is not silently
    # dropped by val_mixtures' 'mg2sio4 in self.z' guard.
    species_list = list(args.species)
    _rock_names = {'mg2sio4', 'rock', 'forsterite'}
    if f_rock > 0.0 and not any(s.lower() in _rock_names for s in species_list):
        species_list.append('mg2sio4')
        print(f"Note: f_rock={f_rock:.3f} > 0 -> auto-added 'mg2sio4' "
              f"to species list: {species_list}")

    # Tag the output filename with the rock fraction so a 50/50 water/rock
    # table does not overwrite the pure-water _square.npz.  Combine with
    # any user --suffix (frock tag first).  f_rock=0 keeps the canonical
    # naming for backward compatibility.
    if f_rock > 0.0:
        rock_tag = f"frock{f_rock:.2f}"
        effective_suffix = (f"{rock_tag}_{args.suffix}"
                            if args.suffix else rock_tag)
    else:
        effective_suffix = args.suffix

    # --- Print summary ---
    print("=" * 65)
    if args.basis == 'pt':
        print(f"EOS Table Build: P-T forward (S, ρ, U)")
    else:
        print(f"EOS Table Inversion: P,T → {args.basis.upper()}")
    print("=" * 65)
    print(f"  H-He EOS:    {args.hhe_eos}")
    print(f"  Z EOS:       {args.z_eos}")
    print(f"  f_rock (_zr):{f_rock:.3f}")
    if effective_suffix:
        print(f"  Suffix:      {effective_suffix}  "
              f"(-> ..._square_{effective_suffix}.npz)")
    print(f"  Species:     {species_list}")
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
        print(f"  srho basis:  rho-T (1-D outer Newton in T)")
    if args.smooth_inverted and args.basis != 'pt':
        print(f"  smooth_inverted: True (Hampel + Gaussian sigma=1)")
    if args.n_workers > 1 and args.basis != 'pt':
        print(f"  n_workers:   {args.n_workers}")
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
    # pt_tab=True uses the P-T table for forward-model evaluation
    # during inversions.  inv_tab=False avoids loading inverted tables
    # (which are what we're building).  For --basis pt, both are off
    # since we build from raw VAL.
    #
    # For --basis srho we set inv_tab=True so the rho-T dependency
    # table auto-loads (build_srho_table does a 1-D outer Newton in
    # T using rho-T table lookups for the inner P(rho,T) step).
    # srho_tab=False because we're building a fresh S-ρ table.
    _pt_tab  = (args.basis != 'pt')  # PT basis builds from VAL
    _inv_tab = (args.basis == 'srho')
    eos = hhe_z_mixtures(
        hhe_eos_name=args.hhe_eos,
        hg=hg,
        smooth_hhe=smooth_hhe,
        smooth_z=smooth_z,
        mu_h_vary=mu_h_vary,
        species_list=species_list,
        z_eos=args.z_eos,
        pt_tab=_pt_tab,
        inv_tab=_inv_tab,
        srho_tab=False,
        logp_range=(args.logp_lo, args.logp_hi),
        logp_step=args.logp_step,
        logt_range=(args.logt_lo, args.logt_hi),
        logrho_range=(args.logrho_lo, args.logrho_hi),
        logrho_step=args.logrho_step,
        table_suffix=effective_suffix,
    )

    # --- Build table ---
    t0 = time.time()

    if args.basis == 'pt':
        result = eos.build_pt_table(yvals, zvals, _zr=f_rock)
        eos.save_pt_table(result, path=args.output)

    elif args.basis == 'sp':
        result = eos.build_sp_table(yvals, zvals, _zr=f_rock,
                                    s_lo=args.s_lo, s_hi=args.s_hi,
                                    s_step=args.s_step,
                                    smooth_inverted=args.smooth_inverted,
                                    n_workers=args.n_workers)
        eos.save_sp_table(result, path=args.output)

    elif args.basis == 'rhot':
        result = eos.build_rhot_table(yvals, zvals, _zr=f_rock,
                                      smooth_inverted=args.smooth_inverted,
                                      n_workers=args.n_workers)
        eos.save_rhot_table(result, path=args.output)

    elif args.basis == 'rhop':
        result = eos.build_rhop_table(yvals, zvals, _zr=f_rock,
                                      smooth_inverted=args.smooth_inverted,
                                      n_workers=args.n_workers)
        eos.save_rhop_table(result, path=args.output)

    elif args.basis == 'srho':
        # 1-D outer Newton in T using the pre-computed rho-T table.
        # build_srho_table itself raises a clear RuntimeError if the
        # rho-T table is missing, but verify here for a friendlier CLI
        # message.
        if eos._logp_rhot_rgi is None:
            rhot_path = eos._table_path('rhot')
            print(f"ERROR: rho-T table not found at {rhot_path}")
            print("Build it first:  python eos_inversions.py "
                  "--basis rhot ...")
            sys.exit(1)

        result = eos.build_srho_table(yvals, zvals, _zr=f_rock,
                                      s_lo=args.s_lo, s_hi=args.s_hi,
                                      s_step=args.s_step,
                                      smooth_inverted=args.smooth_inverted,
                                      n_workers=args.n_workers)
        eos.save_srho_table(result, path=args.output)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print(f"\nTotal time: {hours}h {mins}m {secs:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
