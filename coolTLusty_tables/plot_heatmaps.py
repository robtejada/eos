#!/usr/bin/env python
"""Side-by-side heatmaps of the SCvH reference and the new cd tables.

    python eos/coolTLusty_tables/plot_heatmaps.py [--dir DIR] [--eos cd]

Writes one figure per metallicity:

    heatmap_scvh_vs_cd_Z0solar.png      pure H-He, the like-for-like comparison
    heatmap_scvh_vs_cd_Z1solar.png      ... and the metal-enriched mixtures,
    heatmap_scvh_vs_cd_Z3.16solar.png       where the difference panel carries
    heatmap_scvh_vs_cd_Z5solar.png          both the change of EOS and the
    heatmap_scvh_vs_cd_Z10solar.png         change of composition
    heatmap_cd_metallicity_series.png   all five side by side

Each figure is two rows (log P, log S) by three columns (SCvH, ours, the
difference).  Within a row the two value panels share one color scale, so the
panels can be compared directly; the difference panel uses a diverging scale
centred on zero.  SCvH's fill cells are masked in grey everywhere, including in
the difference, so they cannot pull the scales around.
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
import make_cooltlusty_tables as M       # noqa: E402

REF_DIR = HERE / 'brianna_originals'

# Sequential ramp for magnitude (perceptually uniform semantic heat, with a
# scale legend on every panel); diverging warm/cool with a neutral midpoint for
# the signed differences.
CMAP_VALUE = 'magma'
CMAP_DIFF = 'RdBu_r'
GREY = '#b0b0b0'                          # masked (SCvH fill) cells

QUANTITIES = (
    ('log P', r'$\log_{10}P\ \mathrm{[dyn\,cm^{-2}]}$'),
    ('log S', r'$\log_{10}S\ \mathrm{[k_B/m_H]}$'),
)


def load_reference():
    hp, P = F.read_table(REF_DIR / 'ptab.dat')
    _, S = F.read_table(REF_DIR / 'stab.dat')
    fill = (P == 0.0) | (P == 6.0) | (S == -1.0)
    return hp, P, S, fill


def _panel(ax, LR, lt, data, mask, cmap, vmin, vmax, title, cbar_label, fig):
    cmap_obj = matplotlib.colormaps[cmap].with_extremes(bad=GREY)
    pc = ax.pcolormesh(LR, lt, np.ma.array(data, mask=mask), cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, shading='nearest', rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r'$\log_{10}\rho\ \mathrm{[g\,cm^{-3}]}$', fontsize=8)
    ax.set_ylabel(r'$\log_{10}T\ \mathrm{[K]}$', fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('0.7')
    return pc


def figure_one_metallicity(out_dir, eos_name, f_sun, href, PB, SB, fill, y_prime):
    ppath = out_dir / f'ptab_{eos_name}_Y{y_prime:g}_Z{f_sun:g}solar.dat'
    spath = out_dir / f'stab_{eos_name}_Y{y_prime:g}_Z{f_sun:g}solar.dat'
    if not ppath.exists():
        return None
    h, P = F.read_table(ppath)
    _, S = F.read_table(spath)
    lr1, lt = F.rhomboid_grid(href)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    z = M.zsolar_to_z(f_sun)

    # The dense end (logrho > -2) is where SCvH, CD and CMS all disagree by
    # tenths of a dex.  Left in, it sets the scale and washes out everything an
    # atmosphere actually samples, so the difference scale is set from the
    # thinner gas and the dense corner is allowed to saturate.
    atmospheric = (LR <= -2.0) & ~fill

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), constrained_layout=True)
    for row, ((name, unit), ref, ours) in enumerate(
            zip(QUANTITIES, (PB, SB), (P, S))):
        both = np.ma.array(np.stack([ref, ours]), mask=np.stack([fill, fill]))
        vmin, vmax = (float(np.percentile(both.compressed(), 0.5)),
                      float(np.percentile(both.compressed(), 99.5)))
        diff = ours - ref
        lim = max(float(np.percentile(np.abs(diff[atmospheric]), 99)), 1e-5)

        _panel(axes[row][0], LR, lt, ref, fill, CMAP_VALUE, vmin, vmax,
               f'SCvH95 reference — {name}', unit, fig)
        _panel(axes[row][1], LR, lt, ours, fill, CMAP_VALUE, vmin, vmax,
               f"{eos_name} — {name}   (Y' = {y_prime:g}, "
               + ('pure H-He)' if f_sun == 0 else f'Z = {f_sun:g}× solar)'), unit, fig)
        ax = axes[row][2]
        _panel(ax, LR, lt, diff, fill, CMAP_DIFF, -lim, lim,
               f'{eos_name} − SCvH   (scale = 99th pct at '
               + r'$\log\rho\leq-2$' + f', {lim:.4f};\n'
               + 'right of the dashed line every EOS disagrees, so it saturates)',
               rf'$\Delta$ {name}', fig)
        ax.axvline(-2.0, color='0.35', lw=0.9, ls='--')

    subtitle = ('pure H-He on both sides: this is the EOS change alone'
                if f_sun == 0 else
                f'SCvH is pure H-He, so the difference carries both the EOS change '
                f'and the metals (Z = {z:.6f}, pure water)')
    fig.suptitle(f'CoolTLusty EOS tables: SCvH95 vs ORCHARD/{eos_name}  —  {subtitle}\n'
                 f'grey = SCvH fill cells (masked)',
                 fontsize=11)
    path = out_dir / f'heatmap_scvh_vs_{eos_name}_Z{f_sun:g}solar.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_metallicity_series(out_dir, eos_name, f_suns, href, y_prime):
    """All metallicities on one shared scale, plus what each adds relative to Z = 0."""
    lr1, lt = F.rhomboid_grid(href)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    tables = {}
    for f_sun in f_suns:
        ppath = out_dir / f'ptab_{eos_name}_Y{y_prime:g}_Z{f_sun:g}solar.dat'
        spath = out_dir / f'stab_{eos_name}_Y{y_prime:g}_Z{f_sun:g}solar.dat'
        if ppath.exists():
            tables[f_sun] = (F.read_table(ppath)[1], F.read_table(spath)[1])
    if not tables:
        return None
    keys = sorted(tables)
    base = tables[keys[0]]
    no_mask = np.zeros(lt.shape, dtype=bool)

    fig, axes = plt.subplots(2, len(keys), figsize=(4.1 * len(keys), 8.2),
                             constrained_layout=True, squeeze=False)
    atmospheric = LR <= -2.0
    for row, (name, unit) in enumerate(QUANTITIES):
        diffs = [tables[k][row] - base[row] for k in keys[1:]]
        # One scale across the row so the metallicities can be compared, set
        # from the thinner gas so the dense edge doesn't flatten them all.
        lim = (max(float(np.percentile(np.abs(d[atmospheric]), 99)) for d in diffs)
               if diffs else 1e-5)
        vals = tables[keys[0]][row]
        _panel(axes[row][0], LR, lt, vals, no_mask, CMAP_VALUE,
               float(vals.min()), float(vals.max()),
               f'{eos_name}, pure H-He — {name}', unit, fig)
        for col, k in enumerate(keys[1:], start=1):
            _panel(axes[row][col], LR, lt, tables[k][row] - base[row], no_mask,
                   CMAP_DIFF, -lim, lim,
                   f'Z = {k:g}× solar minus pure H-He',
                   rf'$\Delta$ {name}', fig)
    fig.suptitle(f"ORCHARD/{eos_name} metallicity series (Y' = {y_prime:g}, pure water metals): "
                 f'the pure H-He table, then what each metallicity adds',
                 fontsize=11)
    path = out_dir / f'heatmap_{eos_name}_metallicity_series.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE))
    ap.add_argument('--eos', default='cd', choices=('cd', 'cms'))
    ap.add_argument('--yprime', type=float, default=0.25)
    ap.add_argument('--zsolar', nargs='+', type=float,
                    default=list(M.DEFAULT_ZSOLAR))
    args = ap.parse_args(argv)
    out_dir = Path(args.dir).resolve()

    href, PB, SB, fill = load_reference()
    written = []
    for f_sun in args.zsolar:
        path = figure_one_metallicity(out_dir, args.eos, f_sun, href, PB, SB,
                                      fill, args.yprime)
        if path:
            written.append(path)
    path = figure_metallicity_series(out_dir, args.eos, args.zsolar, href, args.yprime)
    if path:
        written.append(path)
    for p in written:
        print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
