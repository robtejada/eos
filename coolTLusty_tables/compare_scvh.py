#!/usr/bin/env python
"""Compare the new CoolTLusty tables against Brianna's SCvH95 reference.

    python eos/coolTLusty_tables/compare_scvh.py [--dir DIR] [--yprime 0.25]

Writes ``compare_scvh_stats.txt``, ``compare_scvh_maps.png`` and
``compare_scvh_isochores.png``.

Both tables are read on the same nodes, so the comparison is exact, with no
interpolation.  Brianna's fill cells are masked: 322 cells with P = 0, 38 in
the last row with P = 6.0, and 304 with S = -1, 360 in total once the overlaps
are accounted for (18 cells carry P = 0 next to a valid S, so the mask is the
union, not a pairing).

Two differences are expected and are not errors:

* **Entropy, +0.001 to +0.003 dex.**  ORCHARD's VAL entropy adds an ideal
  mixing term computed with atomic hydrogen (``_m_h_atomic = 1``) even where
  hydrogen is molecular, which is about +0.041 k_B/amu.  The user chose to keep
  that convention so these tables stay consistent with ORCHARD's interior
  models, and the ideal-gas extension carries the same convention so there is
  no step at the blend seam.
* **Dense, cold gas (logrho > -2).**  SCvH, CD and CMS genuinely disagree there
  by up to a few tenths of a dex, and SCvH itself carries fill values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ctl_format as F                   # noqa: E402
import ctl_ideal as I                    # noqa: E402
import make_cooltlusty_tables as M       # noqa: E402

REF_DIR = HERE / 'brianna_originals'


def load_reference():
    hp, P = F.read_table(REF_DIR / 'ptab.dat')
    hs, S = F.read_table(REF_DIR / 'stab.dat')
    fill = (P == 0.0) | (P == 6.0) | (S == -1.0)
    return hp, P, S, fill


def regions(logrho_2d, logt_2d, x_diss, z_positive):
    w1 = M._band_weight(logrho_2d, M.BLEND_RHO)
    w2 = M._band_weight(logrho_2d, M.BLEND_RHO_Z) if z_positive else w1
    ideal = (w1 == 0) & (w2 == 0)
    blend = ((w1 > 0) & (w1 < 1)) | ((w2 > 0) & (w2 < 1))
    source = (w1 == 1) & (w2 == 1)
    dense = logrho_2d > -2.0
    return {
        'ideal model, molecular': ideal & (x_diss < 0.01),
        'ideal model, dissociating': ideal & (x_diss >= 0.01),
        'blend band': blend,
        'EOS source, logrho <= -2': source & ~dense,
        'EOS source, logrho > -2': source & dense,
        'ALL (masked)': np.ones(logrho_2d.shape, dtype=bool),
    }


def stats_block(name, dP, dS, sel, out):
    n = int(sel.sum())
    if n == 0:
        return
    def row(d):
        a = np.abs(d[sel])
        return (f'{np.median(d[sel]):+.5f} {np.median(a):.5f} '
                f'{np.percentile(a, 95):.5f} {a.max():.5f}')
    out.append(f'  {name:28s} n={n:6d}   dlogP {row(dP)}   dlogS {row(dS)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE))
    ap.add_argument('--yprime', type=float, default=0.25)
    args = ap.parse_args(argv)
    out_dir = Path(args.dir).resolve()

    href, PB, SB, fill = load_reference()
    lr1, lt = F.rhomboid_grid(href)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    st = I.ideal_state(LR, lt, args.yprime, 0.0, smix='true')
    x_diss = st['x_diss']

    lines = [
        'Comparison against Brianna\'s SCvH95 tables (identical nodes)',
        f'  grid: {href}',
        f'  masked fill cells: {int(fill.sum())} of {fill.size}',
        '',
        'Columns: median(signed)  median|.|  p95|.|  max|.|',
        '',
    ]

    pure = {}
    for eos_name in ('cd', 'cms'):
        ppath = out_dir / f'ptab_{eos_name}_Y{args.yprime:g}_Z0solar.dat'
        spath = out_dir / f'stab_{eos_name}_Y{args.yprime:g}_Z0solar.dat'
        if not ppath.exists():
            continue
        h, P = F.read_table(ppath)
        _, S = F.read_table(spath)
        if not h.matches(href, tol=1e-9):
            lines.append(f'{eos_name}: WARNING header differs from the reference')
        dP, dS = P - PB, S - SB
        pure[eos_name] = (P, S, dP, dS)
        lines.append(f'{eos_name} (pure H-He, Y\' = {args.yprime:g}) vs SCvH:')
        for name, sel in regions(LR, lt, x_diss, False).items():
            stats_block(name, dP, dS, sel & ~fill, lines)
        lines.append('')

    if 'cd' in pure and 'cms' in pure:
        d = np.abs(pure['cd'][0] - pure['cms'][0])
        ds = np.abs(pure['cd'][1] - pure['cms'][1])
        lines.append('cd vs cms (pure H-He): '
                     f'|dlogP| med {np.median(d):.5f} max {d.max():.5f}; '
                     f'|dlogS| med {np.median(ds):.5f} max {ds.max():.5f}')
        lines.append('')

    # metallicity series: the trend with Z, measured against our own Z = 0 table
    lines.append('Metallicity series (cms), relative to Z = 0:')
    for f_sun in (1.0, 3.16, 5.0, 10.0):
        ppath = out_dir / f'ptab_cms_Y{args.yprime:g}_Z{f_sun:g}solar.dat'
        spath = out_dir / f'stab_cms_Y{args.yprime:g}_Z{f_sun:g}solar.dat'
        if not ppath.exists() or 'cms' not in pure:
            continue
        _, P = F.read_table(ppath)
        _, S = F.read_table(spath)
        dp = P - pure['cms'][0]
        ds = S - pure['cms'][1]
        z = M.zsolar_to_z(f_sun)
        lines.append(f'  {f_sun:>5g}x solar (Z = {z:.6f}): dlogP median {np.median(dp):+.5f}, '
                     f'dlogS median {np.median(ds):+.5f}  '
                     f'(ideal-gas expectation for logP: '
                     f'{np.log10(_mu_ratio(args.yprime, 0.0) / _mu_ratio(args.yprime, z)):+.5f})')
    lines.append('')

    text = '\n'.join(lines)
    (out_dir / 'compare_scvh_stats.txt').write_text(text + '\n')
    print(text)

    _plot_maps(out_dir, LR, lt, pure, fill, x_diss)
    _plot_isochores(out_dir, href, lr1, lt, PB, SB, pure, args.yprime)
    print(f'wrote {out_dir / "compare_scvh_stats.txt"}, '
          f'{out_dir / "compare_scvh_maps.png"}, {out_dir / "compare_scvh_isochores.png"}')
    return 0


def _mu_ratio(y_prime, z):
    """Mean molecular weight of the molecular ideal mixture, for the Z trend."""
    f_h, f_he, f_w = (1 - y_prime) * (1 - z), y_prime * (1 - z), z
    return 1.0 / (f_h / 2.01565 + f_he / 4.002602 + f_w / 18.010565)


def _plot_maps(out_dir, LR, lt, pure, fill, x_diss):
    if not pure:
        return
    fig, axes = plt.subplots(2, len(pure), figsize=(7.0 * len(pure), 9.0),
                             squeeze=False, constrained_layout=True)
    for col, (eos_name, (P, S, dP, dS)) in enumerate(sorted(pure.items())):
        for row, (d, label, lim) in enumerate(((dP, r'$\Delta\log P$', 0.02),
                                               (dS, r'$\Delta\log S$', 0.005))):
            ax = axes[row][col]
            masked = np.ma.array(d, mask=fill)
            pc = ax.pcolormesh(LR, lt, masked, cmap='RdBu_r', vmin=-lim, vmax=lim,
                               shading='nearest')
            fig.colorbar(pc, ax=ax, label=f'{label} (ours - SCvH)')
            for edge in M.BLEND_RHO:
                ax.axvline(edge, color='k', lw=0.8, ls='--')
            ax.axhline(M.LOGT_ANCHOR, color='green', lw=0.8, ls=':')
            ax.contour(LR, lt, x_diss, levels=[0.5], colors='magenta', linewidths=1.0)
            ax.set_xlabel(r'$\log_{10}\rho\ \mathrm{[g\,cm^{-3}]}$')
            ax.set_ylabel(r'$\log_{10}T\ \mathrm{[K]}$')
            ax.set_title(f'{eos_name}: {label}  (dashed: blend band, '
                         f'dotted: T anchor, magenta: 50% H$_2$ dissociation)',
                         fontsize=9)
    fig.savefig(out_dir / 'compare_scvh_maps.png', dpi=130)
    plt.close(fig)


def _plot_isochores(out_dir, href, lr1, lt, PB, SB, pure, y_prime):
    rows = [-12.0, -8.0, -5.0, -3.0, -1.0]
    idx = [int(np.argmin(np.abs(lr1 - r))) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(idx)))
    for ax, which, ylabel in ((axes[0], 0, r'$\log_{10}P\ \mathrm{[dyn\,cm^{-2}]}$'),
                              (axes[1], 1, r'$\log_{10}S\ \mathrm{[k_B/m_H]}$')):
        ref = PB if which == 0 else SB
        for c, i in zip(colors, idx):
            ax.plot(lt[i], ref[i], color=c, lw=2.5, alpha=0.35,
                    label=f'SCvH, $\\log\\rho$={lr1[i]:.1f}')
            for eos_name, ls in (('cd', '-'), ('cms', '--')):
                if eos_name in pure:
                    ax.plot(lt[i], pure[eos_name][which][i], color=c, lw=1.1, ls=ls)
        ax.set_xlabel(r'$\log_{10}T\ \mathrm{[K]}$')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_title('Isochores: SCvH (thick) vs cd (solid) and cms (dashed)', fontsize=10)
    axes[1].set_title(f"Pure H-He, Y' = {y_prime:g}", fontsize=10)
    fig.savefig(out_dir / 'compare_scvh_isochores.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
