#!/usr/bin/env python
"""Build CoolTLusty ``ptab``/``stab`` EOS tables from the ORCHARD EOS.

Usage (from anywhere; the repo root is added to sys.path automatically)
----------------------------------------------------------------------
    # the ten deliverable pairs: {cd, cms} x {0, 1, 3.16, 5, 10} x solar, Y' = 0.25
    python eos/coolTLusty_tables/make_cooltlusty_tables.py

    # one table on Brianna's default domain
    python eos/coolTLusty_tables/make_cooltlusty_tables.py --eos cms --yprime 0.25 --zsolar 3.16

    # a custom domain: logrho in [-10, 0.5], T from 100-9000 K at rho_min
    # to 200-20000 K at rho_max, on 250 density rows
    python eos/coolTLusty_tables/make_cooltlusty_tables.py --eos cd --zsolar 1 \
        --logrho -10 0.5 --T-at-rhomin 100 9000 --T-at-rhomax 200 20000 --nrho 250

Output is ``ptab_{eos}_Y{Y'}_Z{f}solar.dat`` (log10 P in cgs) and
``stab_…`` (log10 S), plus a ``.meta.json`` sidecar recording exactly how each
pair was built.

How a table is built
--------------------
Inside the domain of the H-He EOS, values come from the ORCHARD **VAL forward
model** -- ``get_logrho_pt_val`` inverted for P at each (rho, T) with
``_newton_1d_vec``, and ``get_s_pt_val`` for the entropy.  We use the forward
model rather than the pre-saved ``*_pt_square.npz`` tables because those carry
an interpolation artifact at logT 2.70-2.92 that inflates grad_ad by ~40%
(500-830 K, right where CoolTLusty computes atmospheric adiabats).

Outside it -- logrho below about -6, and logT below 2.25 where the raw helium
isotherms break down -- values come from the analytic ideal + Saha model in
``ctl_ideal``.  The two are joined by a C1 smoothstep in **logrho only**: a
weight that varied with P or T would leak ``R dw/dlnT`` into c_v and corrupt
grad_ad on the dissociation ridge, whereas with w = w(logrho) the blended
chi_T and c_v are exact convex combinations of the two models.

Below the temperature anchor the non-ideal excess is carried down with a
first-order Taylor expansion at fixed rho that satisfies the Maxwell relation
exactly: S_ex is frozen and P_ex(T) = P_ex(T_a) - (T-T_a) rho (dS_ex/dlnrho),
since (dP/dT)_rho = -rho (dS/dlnrho)_T.

Metals are pure water (``aqua_revised``); "N x solar" means Z/X scaling,
Z = f a/(1+f a) with a = Z_sun/(1-Z_sun) and Z_sun = 0.017 (Chen+ 2023), which
is the inverse of ORCHARD's ``utils.common.z_to_metallicity_factor``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # /Users/arevalo/orchard
sys.path.insert(0, str(ROOT))

import ctl_format as F       # noqa: E402
import ctl_ideal as I        # noqa: E402

Z_SUN = 0.017                # Z_SOLAR_CHEN23 (utils/const.py:182)

# Brianna's SCvH rhomboid.  The temperature bounds are stored as the log10
# values her header carries (50, 3000, 100 and 9000 K, but written to the
# precision of her file) so the default build reproduces her header exactly.
DEFAULT_DOMAIN = dict(
    logrho=(-15.0, 0.0),
    logt_at_rhomin=(1.69897, 3.4771212547197),
    logt_at_rhomax=(2.0, 3.9542425094393),
    nrho=300,
)
DEFAULT_EOS = ('cd', 'cms')
DEFAULT_ZSOLAR = (0.0, 1.0, 3.16, 5.0, 10.0)
DEFAULT_YPRIME = 0.25

BLEND_RHO = (-6.0, -5.0)     # ideal below, EOS above
BLEND_RHO_Z = (-7.0, -2.0)   # wider band for the water residual
LOGT_ANCHOR = 2.25           # below this the raw He isotherms are unusable
DLOGRHO_FD = 0.02            # central difference for dS_ex/dlnrho

EOS_KW = {'cd': dict(hhe_eos_name='cd', hg=False),
          'cms': dict(hhe_eos_name='cms', hg=True)}


# ------------------------------------------------------------------ helpers
def zsolar_to_z(f, z_sun=Z_SUN):
    """Z at f times solar, scaling Z/X (the inverse of z_to_metallicity_factor)."""
    a = z_sun / (1.0 - z_sun)
    return f * a / (1.0 + f * a)


def smoothstep(x):
    """C1 smoothstep on [0, 1]."""
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _band_weight(logrho, band):
    lo, hi = band
    return smoothstep((np.asarray(logrho, dtype=float) - lo) / (hi - lo))


@contextlib.contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_eos(name):
    """Construct ``hhe_z_mixtures`` for 'cd' or 'cms' with pure-water metals.

    The import must run from the repo root: ``eos/ppv_eos.py:43`` loads a data
    file through a relative path.  Everything after the import is cwd-agnostic.
    """
    with _chdir(ROOT):
        from eos.eos_class import hhe_z_mixtures
    return hhe_z_mixtures(species_list=['water_revised'], z_eos='aqua_revised',
                          y_prime=True, **EOS_KW[name])


# ------------------------------------------------------- the two EOS sources
ROOT_GUARD_DEV = 0.05        # dex; the saved rho-T table is good to ~0.04 here
ROOT_GUARD_STEP = 0.01       # dex, probe offset for the local validity test


def source_state(eos, logrho, logt, y_prime, z, stats=None):
    """ORCHARD VAL forward model at (rho, T).  Flat arrays in and out.

    Returns ``(logp, s, ok)``: log10 P (cgs), S in erg/(g K), and a mask that
    is False where the Newton inversion failed or the result is not finite.

    The VAL forward model has isolated dropouts at high density where
    ``logrho(logP)`` is discontinuous -- at logrho = -0.45, logT = 2.73, for
    instance, it returns -3.51 at logP = 11.10 between neighbours at -0.47 and
    -0.36.  Newton happily converges onto those spurious roots (the residual
    really is zero there), which puts a ~0.2 dex spike into the table.  So each
    root is checked for local sanity -- logrho must increase through it and
    stay near the target on both sides -- and against the smooth saved rho-T
    table.  Cells that fail fall back to the saved-table value, which is
    accurate to about 0.04 dex in this region, and are counted.
    """
    lr = np.ascontiguousarray(np.ravel(logrho), dtype=float)
    lt = np.ascontiguousarray(np.ravel(logt), dtype=float)
    seed = np.asarray(eos.get_logp_rhot(lr, lt, y_prime, z), dtype=float).ravel()

    def logrho_val(lp):
        return np.asarray(eos.val.get_logrho_pt_val(lp, lt, y_prime, z),
                          dtype=float).ravel()

    with np.errstate(all='ignore'):
        lp, conv = eos._newton_1d_vec(lambda lp: logrho_val(lp) - lr, seed,
                                      -2.0, 20.0, tol=1e-9, h=1e-4)
        d = ROOT_GUARD_STEP
        below, above = logrho_val(lp - d), logrho_val(lp + d)
        sane = ((above > below)
                & (np.abs(above - lr) < 0.1) & (np.abs(below - lr) < 0.1)
                & (np.abs(lp - seed) <= ROOT_GUARD_DEV)
                & np.isfinite(lp))
        lp = np.where(sane, lp, seed)
        s = np.asarray(eos.val.get_s_pt_val(lp, lt, y_prime, z), dtype=float).ravel()
    if stats is not None:
        stats['n_root_repaired'] = stats.get('n_root_repaired', 0) + int(np.sum(~sane))
    ok = np.asarray(conv, dtype=bool) & np.isfinite(lp) & np.isfinite(s) & (s > 0)
    return lp, s, ok


def _residual_at(eos, lr, lt, y_prime, z, smix, stats=None):
    """(P_excess [linear cgs], S_excess [erg/g/K], ok) of VAL over the ideal model."""
    lp, s, ok = source_state(eos, lr, lt, y_prime, z, stats=stats)
    idl = I.ideal_state(lr, lt, y_prime, z, smix=smix)
    p_ex = 10.0 ** lp - 10.0 ** idl['logp']
    s_ex = s - idl['s']
    return p_ex, s_ex, ok


def residuals(eos, logrho, logt, y_prime, z, smix='orchard',
              logt_anchor=LOGT_ANCHOR, need=None, stats=None):
    """Non-ideal excess of the EOS over the ideal model, extended below T_anchor.

    Returns ``(p_excess, s_excess, ok)`` as linear excesses (cgs) on the flat
    input arrays.  Above ``logt_anchor`` these are evaluated directly.  Below
    it, where the raw helium isotherms break down, the excess is carried down
    at fixed rho by the Maxwell-exact first-order expansion described in the
    module docstring.
    """
    lr = np.ravel(np.asarray(logrho, dtype=float))
    lt = np.ravel(np.asarray(logt, dtype=float))
    if need is None:
        need = np.ones(lr.shape, dtype=bool)

    p_ex = np.zeros(lr.shape)
    s_ex = np.zeros(lr.shape)
    ok = np.ones(lr.shape, dtype=bool)

    hi = need & (lt >= logt_anchor)
    if hi.any():
        p_ex[hi], s_ex[hi], ok[hi] = _residual_at(eos, lr[hi], lt[hi], y_prime, z,
                                                  smix, stats)

    lo = need & ~hi
    if lo.any():
        lr_lo = lr[lo]
        t_a = 10.0 ** logt_anchor
        anchor = np.full(lr_lo.shape, logt_anchor)
        p_a, s_a, ok_a = _residual_at(eos, lr_lo, anchor, y_prime, z, smix, stats)
        # dS_ex/dlnrho at the anchor, by central differences in logrho
        _, s_p, ok_p = _residual_at(eos, lr_lo + DLOGRHO_FD, anchor, y_prime, z, smix, stats)
        _, s_m, ok_m = _residual_at(eos, lr_lo - DLOGRHO_FD, anchor, y_prime, z, smix, stats)
        ds_dlnrho = (s_p - s_m) / (2.0 * DLOGRHO_FD * np.log(10.0))
        rho = 10.0 ** lr_lo
        t = 10.0 ** lt[lo]
        # (dP/dT)_rho = -rho (dS/dlnrho)_T  =>  Maxwell-exact to first order
        p_ex[lo] = p_a - (t - t_a) * rho * ds_dlnrho
        s_ex[lo] = s_a
        ok[lo] = ok_a & ok_p & ok_m
    return p_ex, s_ex, ok


# ------------------------------------------------------------------- build
def build_pair(eos, header, y_prime, z, blend_rho=BLEND_RHO,
               blend_rho_z=BLEND_RHO_Z, logt_anchor=LOGT_ANCHOR,
               s_unit=F.S_UNIT_MH, smix='orchard'):
    """Return ``(ptab, stab, info)`` on the header's rhomboid grid."""
    logrho_1d, logt_2d = F.rhomboid_grid(header)
    shape = logt_2d.shape
    lr = np.broadcast_to(logrho_1d[:, None], shape).ravel()
    lt = logt_2d.ravel()

    idl = I.ideal_state(lr, lt, y_prime, z, smix=smix)
    p_id = 10.0 ** idl['logp']

    w1 = _band_weight(lr, blend_rho)
    w2 = _band_weight(lr, blend_rho_z) if z > 0 else np.zeros_like(w1)

    p_ex = np.zeros_like(lr)
    s_ex = np.zeros_like(lr)
    ok = np.ones(lr.shape, dtype=bool)

    stats = {}
    need0 = (w1 > 0) | (w2 > 0)
    if need0.any():
        p0, s0, ok0 = residuals(eos, lr, lt, y_prime, 0.0, smix=smix,
                                logt_anchor=logt_anchor, need=need0, stats=stats)
        p_ex += w1 * p0
        s_ex += w1 * s0
        ok &= ok0 | ~need0
    if z > 0 and need0.any():
        pz, sz, okz = residuals(eos, lr, lt, y_prime, z, smix=smix,
                                logt_anchor=logt_anchor, need=need0, stats=stats)
        p_ex += w2 * (pz - p0)
        s_ex += w2 * (sz - s0)
        ok &= okz | ~need0

    p = p_id + p_ex
    s = idl['s'] + s_ex

    # Floors: the dense-cold corner (logrho > -2, logT < 2.3) has no trustworthy
    # source in SCvH, CD or CMS, and the Taylor term can in principle overshoot.
    p_floor = 1e-3 * p_id
    s_floor = 1e-3 * I.K_B / I.AMU
    n_pfloor = int(np.sum(p < p_floor))
    n_sfloor = int(np.sum(s < s_floor))
    p = np.maximum(p, p_floor)
    s = np.maximum(s, s_floor)

    ptab = np.log10(p).reshape(shape)
    stab = np.log10(s / s_unit).reshape(shape)

    info = {
        'n_cells': int(lr.size),
        'n_ideal_only': int(np.sum((w1 == 0) & (w2 == 0))),
        'n_blend': int(np.sum(((w1 > 0) & (w1 < 1)) | ((w2 > 0) & (w2 < 1)))),
        'n_source_only': int(np.sum((w1 == 1) & ((z == 0) | (w2 == 1)))),
        'n_source_failed': int(np.sum(~ok)),
        'n_pressure_floored': n_pfloor,
        'n_entropy_floored': n_sfloor,
        'n_below_anchor': int(np.sum((lt < logt_anchor) & need0)),
        'x_e_max': float(np.max(idl['x_e'])),
        'x_diss_max': float(np.max(idl['x_diss'])),
        'n_root_repaired': int(stats.get('n_root_repaired', 0)),
        'ptab_range': [float(ptab.min()), float(ptab.max())],
        'stab_range': [float(stab.min()), float(stab.max())],
    }
    return ptab, stab, info


# ------------------------------------------------------------------- output
def table_paths(out_dir, eos_name, y_prime, z_solar):
    stem = f'{eos_name}_Y{y_prime:g}_Z{z_solar:g}solar'
    return Path(out_dir) / f'ptab_{stem}.dat', Path(out_dir) / f'stab_{stem}.dat'


def make_header(y_prime, z, n_rho, logrho, logt_at_rhomin, logt_at_rhomax):
    """Header with the absolute helium mass fraction Y = Y'(1-Z).

    Y is rounded to 8 significant digits: sc_eos.f reads it but never uses it,
    and a full float repr just makes the line noisy.
    """
    y_abs = float(f'{y_prime * (1.0 - z):.8g}')
    return F.Header(y_abs, n_rho, logrho[0], logrho[1],
                    logt_at_rhomin[0], logt_at_rhomin[1],
                    logt_at_rhomax[0], logt_at_rhomax[1]).validate()


def _git_sha(path):
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain'],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f'{sha}{"-dirty" if dirty else ""}' if sha else None
    except Exception:
        return None


def _sha256(path):
    hsh = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            hsh.update(chunk)
    return hsh.hexdigest()


def write_pair(out_dir, eos_name, y_prime, z_solar, header, ptab, stab, info,
               s_unit_name, extra=None, force=True):
    ppath, spath = table_paths(out_dir, eos_name, y_prime, z_solar)
    for path in (ppath, spath):
        if path.exists() and not force:
            raise FileExistsError(f'{path} exists; pass --force to overwrite')
    F.write_table(ppath, header, ptab)
    F.write_table(spath, header, stab)
    meta = {
        'ptab': ppath.name, 'stab': spath.name,
        'eos': eos_name, 'hg': EOS_KW[eos_name]['hg'],
        'y_prime': y_prime, 'y_absolute': header.y,
        'z_solar': z_solar, 'z': zsolar_to_z(z_solar), 'z_sun': Z_SUN,
        'z_eos': 'aqua_revised (pure water)',
        'header': header.line.rstrip('\n'),
        'header_tokens': list(header.tokens),
        's_unit': s_unit_name,
        'source': 'VAL forward model (get_logrho_pt_val inverted, get_s_pt_val)',
        'mixing_entropy': "ORCHARD convention (_m_h_atomic = 1)",
        'blend_rho': list(BLEND_RHO), 'blend_rho_z': list(BLEND_RHO_Z),
        'logt_anchor': LOGT_ANCHOR,
        'effective_domain': {k: list(np.asarray(v, dtype=float))
                             for k, v in F.effective_domain(header).items()},
        'orchard_git': _git_sha(ROOT), 'eos_git': _git_sha(ROOT / 'eos'),
        'numpy': np.__version__, 'python': sys.version.split()[0],
        'sha256': {ppath.name: _sha256(ppath), spath.name: _sha256(spath)},
    }
    meta.update(info)
    if extra:
        meta.update(extra)
    (Path(out_dir) / f'{ppath.stem.replace("ptab_", "")}.meta.json').write_text(
        json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return ppath, spath


# ---------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        prog='make_cooltlusty_tables.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__.split('How a table is built')[0]))
    p.add_argument('--eos', nargs='+', choices=sorted(EOS_KW), default=list(DEFAULT_EOS),
                   help='H-He EOS: cd (Chabrier & Debras 2021) or cms (CMS19 + HG23)')
    p.add_argument('--yprime', type=float, default=DEFAULT_YPRIME,
                   help="helium fraction of the H-He sub-mixture, Y' = Y/(1-Z)")
    p.add_argument('--zsolar', nargs='+', type=float, default=list(DEFAULT_ZSOLAR),
                   help='metallicity in multiples of solar (Z_sun = 0.017); 0 = pure H-He')
    p.add_argument('--logrho', nargs=2, type=float, default=DEFAULT_DOMAIN['logrho'],
                   metavar=('LO', 'HI'), help='log10 density bounds (g/cm^3)')
    p.add_argument('--T-at-rhomin', nargs=2, type=float, dest='t_at_rhomin',
                   metavar=('LOW', 'HIGH'),
                   help='temperature range (K) at the low-density edge '
                        '(default 50 3000, as in Brianna\'s tables)')
    p.add_argument('--T-at-rhomax', nargs=2, type=float, dest='t_at_rhomax',
                   metavar=('LOW', 'HIGH'),
                   help='temperature range (K) at the high-density edge '
                        '(default 100 9000)')
    p.add_argument('--nrho', type=int, default=DEFAULT_DOMAIN['nrho'],
                   help=f'density rows (<= {F.MAX_ROWS}); T columns are fixed at {F.N_T}')
    p.add_argument('--s-unit', choices=sorted(F.S_UNITS), default='mH',
                   help="entropy unit: 'mH' matches Brianna's SCvH tables (default), "
                        "'amu' is the literal RCON convention (0.0034 dex lower)")
    p.add_argument('--out-dir', default=str(HERE))
    p.add_argument('--force', action='store_true', default=True,
                   help='overwrite existing tables (default)')
    p.add_argument('--no-force', dest='force', action='store_false')
    return p


def resolve_domain(args):
    """Header temperature bounds as log10 K, from --T-at-* or the defaults."""
    lo = (tuple(np.log10(args.t_at_rhomin)) if args.t_at_rhomin
          else DEFAULT_DOMAIN['logt_at_rhomin'])
    hi = (tuple(np.log10(args.t_at_rhomax)) if args.t_at_rhomax
          else DEFAULT_DOMAIN['logt_at_rhomax'])
    return lo, hi


def validate(args):
    if not 4 <= args.nrho <= F.MAX_ROWS:
        raise SystemExit(f'--nrho must be in [4, {F.MAX_ROWS}] (SL(330,100) in sc_eos.f)')
    if not args.logrho[0] < args.logrho[1]:
        raise SystemExit('--logrho needs LO < HI')
    for name, pair in (('--T-at-rhomin', args.t_at_rhomin), ('--T-at-rhomax', args.t_at_rhomax)):
        if pair is not None and not 0 < pair[0] < pair[1]:
            raise SystemExit(f'{name} needs 0 < LOW < HIGH')
    if not 0.0 <= args.yprime <= 1.0:
        raise SystemExit("--yprime must be in [0, 1]")
    if any(f < 0 for f in args.zsolar):
        raise SystemExit('--zsolar must be >= 0')


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    s_unit = F.S_UNITS[args.s_unit]

    logt_lo, logt_hi = resolve_domain(args)
    pairs = [(e, f) for e in args.eos for f in args.zsolar]
    print(f'{len(pairs)} table pair(s) -> {out_dir}')
    for eos_name in args.eos:
        t0 = time.time()
        eos = load_eos(eos_name)
        print(f"\n=== {eos_name} (hg={EOS_KW[eos_name]['hg']}) loaded in {time.time()-t0:.1f}s ===")
        for f in args.zsolar:
            z = zsolar_to_z(f)
            header = make_header(args.yprime, z, args.nrho, args.logrho,
                                 logt_lo, logt_hi)
            t0 = time.time()
            ptab, stab, info = build_pair(eos, header, args.yprime, z, s_unit=s_unit)
            ppath, spath = write_pair(out_dir, eos_name, args.yprime, f, header,
                                      ptab, stab, info, args.s_unit, force=args.force)
            print(f'  {f:>5g}x solar  Z={z:.6f}  Y={header.y:.6f}  ({time.time()-t0:.1f}s)'
                  f'  ptab {info["ptab_range"][0]:8.3f}..{info["ptab_range"][1]:7.3f}'
                  f'  stab {info["stab_range"][0]:7.3f}..{info["stab_range"][1]:6.3f}')
            print(f'         cells: {info["n_ideal_only"]} ideal, {info["n_blend"]} blended, '
                  f'{info["n_source_only"]} EOS; {info["n_below_anchor"]} below the T anchor; '
                  f'floors P/S {info["n_pressure_floored"]}/{info["n_entropy_floored"]}')
            if info['n_root_repaired']:
                print(f'         note: {info["n_root_repaired"]} VAL root(s) fell back to '
                      f'the saved rho-T table (isolated forward-model dropouts)')
            if info['n_source_failed']:
                print(f'         WARNING: {info["n_source_failed"]} cell(s) where the EOS '
                      f'source failed and the ideal model was used instead')
            if info['x_e_max'] > 1e-3:
                print(f'         WARNING: peak ionization fraction {info["x_e_max"]:.2e} > 1e-3; '
                      f'the ideal extension neglects ionization')
            if f > 0 and 10 ** np.max(F.rhomboid_grid(header)[1]) > 2500:
                print('         NOTE: water dissociation above ~2500 K is neglected in the '
                      'ideal extension')
            print(f'         wrote {ppath.name}, {spath.name}')

    dom = F.effective_domain(header)
    print('\nEffective LOOK domain (narrower than the header, sc_eos.f:200-203):')
    print(f'  logrho {dom["logrho"][0]:.4f} .. {dom["logrho"][1]:.4f}')
    print(f'  logT   {dom["logT_at_logrho_lo"][0]:.4f} .. {dom["logT_at_logrho_lo"][1]:.4f} '
          f'at the low-density edge')
    print(f'  logT   {dom["logT_at_logrho_hi"][0]:.4f} .. {dom["logT_at_logrho_hi"][1]:.4f} '
          f'at the high-density edge')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
