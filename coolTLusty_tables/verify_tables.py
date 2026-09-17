#!/usr/bin/env python
"""Quality gates for the CoolTLusty tables.  Exits non-zero if any gate fails.

    python eos/coolTLusty_tables/verify_tables.py [--dir DIR] [--quick]

Gates, in order of severity:

S0  CoolTLusty would misread the file
    - byte layout, F8.5 field widths, finite values, round trip
    - ptab and stab headers numerically identical (``SETTABL`` reads them into
      the same COMMON block, so the ptab header silently wins)
    - **the real Fortran reader**: ``SETTABL``/``LOOK`` are extracted verbatim
      from ``eos/scvh/eos/sc_eos.f``, compiled with gfortran, pointed at our
      files, and checked against the Python port at tens of thousands of
      points.  Run first on Brianna's own tables, which must pass.

S1  silently biased everywhere
    - grid convention and entropy unit, pinned against Brianna's SCvH tables
      through the analytic ideal model, with negative controls that must fail
      (linspace grid, per-amu entropy, wrong rotational constant)
    - Z(f) mapping, header Y = Y'(1-Z), cd/cms agreement in the extension

S2  local artifacts that would break CoolTLusty's derivatives
    - Maxwell relation, grad_ad range, blend-band smoothness, monotonicity

Only logrho <= -2 is gated.  Denser than that, no EOS -- SCvH, CD or CMS -- is
trustworthy: measured through LOOK the same way, Brianna's own table has a
Maxwell p95 of 0.47 and grad_ad running from -3.0 to +1.9 there, and it carries
fill values.  Those cells are counted and reported beside the SCvH benchmark.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ctl_format as F              # noqa: E402
import ctl_ideal as I               # noqa: E402
import make_cooltlusty_tables as M  # noqa: E402

SC_EOS_F = ROOT / 'eos' / 'scvh' / 'eos' / 'sc_eos.f'
REF_DIR = HERE / 'brianna_originals'

_results = []


def check(name, passed, detail=''):
    _results.append((name, bool(passed)))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    return passed


def info(name, detail):
    print(f'  [info] {name}  -- {detail}')


# ----------------------------------------------------------------- S0 format
def check_format(path):
    raw = Path(path).read_bytes()
    text = raw.decode('ascii')
    header, arr = F.read_table(path)
    lines = text.split('\n')
    ok = True
    ok &= check(f'{path.name}: line count = 1 + 10N',
                len(lines) - 1 == 1 + 10 * header.n_rho,
                f'{len(lines) - 1} lines, N={header.n_rho}')
    ok &= check(f'{path.name}: body lines are 80 chars, file ends in newline',
                text.endswith('\n') and all(len(ln) == 80 for ln in lines[1:-1]))
    ok &= check(f'{path.name}: ASCII, no CR or tab', b'\r' not in raw and b'\t' not in raw)
    ok &= check(f'{path.name}: values finite and inside F8.5',
                np.isfinite(arr).all() and arr.min() >= F.F8_MIN and arr.max() <= F.F8_MAX,
                f'range {arr.min():.5f} .. {arr.max():.5f}')
    # every field must re-format to exactly the bytes on disk
    rebuilt = F.format_table(header, arr)
    ok &= check(f'{path.name}: byte round trip through the writer',
                rebuilt == text)
    return ok, header, arr


# ------------------------------------------------------ S0 the Fortran reader
def build_fortran_harness(workdir, sc_eos=None):
    """Extract SETTABL/LOOK/SUBSOLD verbatim from sc_eos.f and compile them."""
    gfortran = shutil.which('gfortran') or '/opt/homebrew/bin/gfortran'
    if not Path(gfortran).exists():
        return None
    sc_eos = Path(sc_eos) if sc_eos else SC_EOS_F
    if not sc_eos.exists():
        return None
    src = sc_eos.read_text().splitlines(keepends=True)

    def grab(name):
        out, on = [], False
        for ln in src:
            if not on and re.search(rf'^\s*SUBROUTINE\s+{name}\b', ln, re.I):
                on = True
            if on:
                if 'PRINT *, DELTA' in ln:        # debug print inside LOOK
                    continue
                out.append(ln)
                if re.match(r'^\s{6,}END\s*$', ln.rstrip('\n'), re.I):
                    break
        if not out:
            raise RuntimeError(f'could not extract {name} from {sc_eos}')
        return ''.join(out)

    (workdir / 'lib.f').write_text(grab('SETTABL') + '\nC\n' + grab('LOOK')
                                   + '\nC\n' + grab('SUBSOLD'))
    (workdir / 'drv.f').write_text(
        '      PROGRAM DRV\n'
        '      IMPLICIT DOUBLE PRECISION (A-H,O-Z)\n'
        '      COMMON/FIRST/ VEOS\n'
        "      OPEN(10,FILE='probes.txt',STATUS='OLD')\n"
        "      OPEN(11,FILE='out.txt',STATUS='UNKNOWN')\n"
        '      CALL SETTABL\n'
        ' 10   READ(10,*,END=99) RL, TL\n'
        '      R = 10.D0**RL\n'
        '      T = 10.D0**TL\n'
        '      FP = -1.D99\n'
        '      FS = -1.D99\n'
        '      CALL LOOK(R,T,FP,FS)\n'
        "      WRITE(11,'(2ES27.17)') FP, FS\n"
        '      GO TO 10\n'
        ' 99   CONTINUE\n'
        '      CLOSE(11)\n'
        '      END\n')
    out = subprocess.run([gfortran, '-std=legacy', '-O2', '-o', 'drv', 'drv.f', 'lib.f'],
                         cwd=workdir, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'gfortran failed:\n{out.stderr[-2000:]}')
    return workdir / 'drv'


def fortran_roundtrip(label, ptab_path, stab_path, workdir, drv, n_random=20000, seed=11):
    """Compare the compiled LOOK against the Python port on the written files."""
    for target, src in (('ptabnew.dat', ptab_path), ('stabnew.dat', stab_path)):
        link = workdir / target
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(src).resolve())
    hp, P = F.read_table(ptab_path)
    hs, S = F.read_table(stab_path)
    rng = np.random.default_rng(seed)
    delta = rng.uniform(1.0, hp.n_rho - 1.0, n_random)
    lr = hp.r1 + delta * (hp.r2 - hp.r1) / hp.n_rho
    f = (lr - hp.r1) / (hp.r2 - hp.r1)
    alpha = hp.t1 + f * (hp.t12 - hp.t1)
    beta = (hp.t2 - hp.t1) + ((hp.t22 - hp.t12) - (hp.t2 - hp.t1)) * f
    lt = alpha + rng.uniform(0.02, 0.985, n_random) * beta
    lr_n, lt_n = F.rhomboid_grid(hp)
    ii, jj = np.meshgrid(np.arange(1, hp.n_rho - 1), np.arange(2, 99), indexing='ij')
    lr = np.concatenate([lr, np.broadcast_to(lr_n[:, None], lt_n.shape)[ii, jj].ravel()])
    lt = np.concatenate([lt, lt_n[ii, jj].ravel()])
    np.savetxt(workdir / 'probes.txt', np.column_stack([lr, lt]), fmt='%.17g')
    run = subprocess.run([str(drv)], cwd=workdir, capture_output=True, text=True)
    if run.returncode != 0:
        return check(f'{label}: compiled LOOK runs', False, run.stderr[-500:])
    fort = np.loadtxt(workdir / 'out.txt')
    off = fort[:, 0] == -1e99
    with np.errstate(invalid='ignore', divide='ignore'):
        lfp, lfs = np.log10(fort[:, 0]), np.log10(fort[:, 1])
    py_p, py_s = F.look(hp, P, lr, lt), F.look(hs, S, lr, lt)
    same_domain = np.array_equal(off, np.isnan(py_p))
    m = ~off
    dp = np.max(np.abs(lfp[m] - py_p[m])) if m.any() else 0.0
    ds = np.max(np.abs(lfs[m] - py_s[m])) if m.any() else 0.0
    ok = check(f'{label}: compiled Fortran LOOK == Python port',
               same_domain and dp < 1e-10 and ds < 1e-10,
               f'{m.sum()} probes, max |dlogP| {dp:.1e}, |dlogS| {ds:.1e}, '
               f'off-table sets {"match" if same_domain else "DIFFER"}')
    # interior nodes must come back as the stored values
    node = F.look(hp, P, np.broadcast_to(lr_n[:, None], lt_n.shape)[ii, jj], lt_n[ii, jj])
    ok &= check(f'{label}: LOOK reproduces stored values at interior nodes',
                np.nanmax(np.abs(node - P[ii, jj])) < 1e-10,
                f'max |diff| {np.nanmax(np.abs(node - P[ii, jj])):.1e}')
    return ok


# --------------------------------------------- S1 grid + units, vs Brianna
def check_grid_and_units():
    """Pin the grid convention and entropy unit against Brianna's SCvH tables.

    The analytic ideal model is the probe: in molecular gas it *is* the physics
    both tables describe, so any mismatch in node placement or entropy
    normalisation shows up immediately.
    """
    hb, PB = F.read_table(REF_DIR / 'ptab.dat')
    _, SB = F.read_table(REF_DIR / 'stab.dat')
    lr1, lt = F.rhomboid_grid(hb)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    fill = (PB == 0.0) | (PB == 6.0) | (SB == -1.0)
    st = I.ideal_state(LR, lt, 0.25, 0.0, smix='true')     # SCvH has the physical mixing
    mol = (st['x_diss'] < 0.01) & (LR < -9) & ~fill
    dP = np.abs(st['logp'] - PB)[mol]
    dS = np.abs(np.log10(st['s'] / F.S_UNIT_MH) - SB)[mol]
    ok = check('grid + units: ideal model reproduces SCvH in molecular gas',
               np.median(dP) < 5e-5 and np.percentile(dS, 95) < 1.5e-4,
               f'n={mol.sum()}, |dlogP| med {np.median(dP):.1e}, '
               f'|dlogS| med {np.median(dS):.1e} p95 {np.percentile(dS, 95):.1e}')

    # negative controls: each must be clearly worse, which proves the gate bites
    base = np.median(dS)
    ctl = {}
    ctl['per-amu entropy unit'] = np.median(
        np.abs(np.log10(st['s'] / F.S_UNIT_AMU) - SB)[mol])
    st2 = I.ideal_state(LR, lt, 0.25, 0.0, smix='true', theta_rot=87.6)
    ctl['theta_rot = 87.6 K (B_e)'] = np.median(
        np.abs(np.log10(st2['s'] / F.S_UNIT_MH) - SB)[mol])
    st3 = I.ideal_state(LR, lt, 0.25, 0.0, smix='orchard')
    ctl['ORCHARD mixing convention'] = np.median(
        np.abs(np.log10(st3['s'] / F.S_UNIT_MH) - SB)[mol])
    lr_ls = np.linspace(hb.r1, hb.r2, hb.n_rho)
    f_ls = (lr_ls - hb.r1) / (hb.r2 - hb.r1)
    al = hb.t1 + f_ls * (hb.t12 - hb.t1)
    be = (hb.t2 - hb.t1) + ((hb.t22 - hb.t12) - (hb.t2 - hb.t1)) * f_ls
    lt_ls = al[:, None] + np.linspace(0, 1, F.N_T)[None, :] * be[:, None]
    st4 = I.ideal_state(np.broadcast_to(lr_ls[:, None], lt_ls.shape), lt_ls,
                        0.25, 0.0, smix='true')
    ls_err = np.median(np.abs(st4['logp'] - PB)[mol])
    for label, val in ctl.items():
        ok &= check(f'negative control fails as expected: {label}', val > 5 * base,
                    f'{val:.1e} vs baseline {base:.1e} ({val / base:.0f}x)')
    ok &= check('negative control fails as expected: linspace grid',
                ls_err > 5 * np.median(dP), f'{ls_err:.1e} dex in logP')
    return ok


# ------------------------------------------------------- S2 thermodynamics
def derivatives(hp, P, hs, S, lr, lt, hr=0.05, ht=0.02):
    """Finite-difference log-derivatives through LOOK, as CoolTLusty would see them."""
    def d(tab, hh):
        return ((F.look(hh, tab, lr + hr, lt) - F.look(hh, tab, lr - hr, lt)) / (2 * hr),
                (F.look(hh, tab, lr, lt + ht) - F.look(hh, tab, lr, lt - ht)) / (2 * ht))
    chi_rho, chi_t = d(P, hp)
    s_rho, s_t = d(S, hs)
    p = 10.0 ** F.look(hp, P, lr, lt)
    s = 10.0 ** F.look(hs, S, lr, lt) * F.S_UNIT_MH
    rho, t = 10.0 ** lr, 10.0 ** lt
    # Maxwell: (dS/dlnrho)_T = -(P/rho T) (dlnP/dlnT)_rho  ->  ratio should be 1
    with np.errstate(invalid='ignore', divide='ignore'):
        maxwell = -(s * s_rho) * rho * t / (p * chi_t)
        grad_ad = 1.0 / (chi_t - chi_rho * s_t / s_rho)
    return dict(chi_rho=chi_rho, chi_t=chi_t, s_rho=s_rho, s_t=s_t,
                maxwell=maxwell, grad_ad=grad_ad)


def sample_points(hp, n=40000, seed=5):
    """Random interior (logrho, logT) points on a table's rhomboid."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(1.5, hp.n_rho - 1.5, n)
    lr = hp.r1 + delta * (hp.r2 - hp.r1) / hp.n_rho
    f = (lr - hp.r1) / (hp.r2 - hp.r1)
    alpha = hp.t1 + f * (hp.t12 - hp.t1)
    beta = (hp.t2 - hp.t1) + ((hp.t22 - hp.t12) - (hp.t2 - hp.t1)) * f
    return lr, alpha + rng.uniform(0.06, 0.94, n) * beta


def blend_regions(logrho, z_positive):
    """Split points by which model produced them, using the actual weights."""
    w1 = M._band_weight(logrho, M.BLEND_RHO)
    w2 = M._band_weight(logrho, M.BLEND_RHO_Z) if z_positive else w1
    return {'ideal model': (w1 == 0) & (w2 == 0),
            'blend band': ((w1 > 0) & (w1 < 1)) | ((w2 > 0) & (w2 < 1)),
            'EOS source': (w1 == 1) & (w2 == 1)}


# Limits calibrated against Brianna's SCvH table measured the same way (through
# LOOK, at logrho <= -2): SCvH gives p95 0.021 / 0.008 / 0.009 in these regions.
MAXWELL_LIMIT = {'ideal model': 0.025, 'blend band': 0.10, 'EOS source': 0.03}
GATED_LOGRHO = -2.0   # above this density no EOS here is trustworthy; see below


def check_thermo(label, ptab_path, stab_path, z_positive, n=40000, seed=5):
    """Maxwell relation and grad_ad, as CoolTLusty's own finite differences see them.

    Only logrho <= -2 is gated.  Denser than that every source -- SCvH, CD and
    CMS -- is rough: measured the same way, Brianna's own table has a Maxwell
    p95 of 0.47 and grad_ad running from -3.0 to +1.9 there.  Atmospheres never
    reach those densities, so that region is reported next to the SCvH
    benchmark rather than failed.
    """
    hp, P = F.read_table(ptab_path)
    hs, S = F.read_table(stab_path)
    lr, lt = sample_points(hp, n=n, seed=seed)
    d = derivatives(hp, P, hs, S, lr, lt)
    good = np.isfinite(d['maxwell']) & np.isfinite(d['grad_ad'])
    gated = good & (lr <= GATED_LOGRHO)
    ok = True
    for name, sel in blend_regions(lr, z_positive).items():
        m = gated & sel
        if not m.any():
            continue
        err = np.abs(d['maxwell'][m] - 1.0)
        lim = MAXWELL_LIMIT[name]
        p95 = np.percentile(err, 95)
        ok &= check(f'{label}: Maxwell relation, {name}', p95 < lim,
                    f'n={m.sum()}, |ratio-1| med {np.median(err):.4f} '
                    f'p95 {p95:.4f} (limit {lim})')
        g = d['grad_ad'][m]
        ok &= check(f'{label}: 0 < grad_ad < 0.45, {name}',
                    np.all((g > 0) & (g < 0.45)),
                    f'range {g.min():.3f} .. {g.max():.3f}, median {np.median(g):.3f}')
    m = good & (lr > GATED_LOGRHO)
    if m.any():
        info(f'{label}: logrho > {GATED_LOGRHO:g} (not gated; SCvH p95 there is 0.47)',
             f'n={m.sum()}, Maxwell |ratio-1| p95 '
             f'{np.percentile(np.abs(d["maxwell"][m] - 1), 95):.3f}, grad_ad median '
             f'{np.nanmedian(d["grad_ad"][m]):.3f}')
    return ok


def _mono_counts(arr, lr_2d, limit):
    """Violations of 'increases with T along a row', split at ``limit`` in logrho.

    Only strictly decreasing steps count.  Neighbouring nodes that come out
    equal are an artifact of rounding to F8.5, not of the physics, and are
    reported separately.
    """
    step = np.diff(arr, axis=1)
    bad = step < 0
    flat = step == 0
    gated = bad & (lr_2d[:, 1:] <= limit)
    dense = bad & (lr_2d[:, 1:] > limit)
    worst = step[bad].min() if bad.any() else 0.0
    return int(gated.sum()), int(dense.sum()), float(worst), int(flat.sum())


def check_monotonicity(label, ptab_path, stab_path, reference=None):
    """P and S must rise with T along each row, for logrho <= GATED_LOGRHO.

    Denser than that, CD/CMS carry real non-monotonic structure around H2
    dissociation and pressure ionisation.  We report those counts beside the
    SCvH reference, which has far more of them (156 in P and 131 in S, the
    worst being a 2.0 dex drop in S).
    """
    hp, P = F.read_table(ptab_path)
    hs, S = F.read_table(stab_path)
    lr1, lt = F.rhomboid_grid(hp)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    ok = True
    for name, arr in (('logP', P), ('logS', S)):
        n_gated, n_dense, worst, n_flat = _mono_counts(arr, LR, GATED_LOGRHO)
        ok &= check(f'{label}: {name} increases with T for logrho <= {GATED_LOGRHO:g}',
                    n_gated == 0, f'{n_gated} decreasing steps')
        if n_dense or n_flat:
            extra = f'; SCvH reference has {reference[name]}' if reference else ''
            info(f'{label}: {name} at logrho > {GATED_LOGRHO:g} (not gated)',
                 f'{n_dense} decreasing steps, worst {worst:+.5f} dex, '
                 f'{n_flat} flat (F8.5 ties){extra}')
    return ok


def reference_mono_counts():
    hp, P = F.read_table(REF_DIR / 'ptab.dat')
    _, S = F.read_table(REF_DIR / 'stab.dat')
    lr1, lt = F.rhomboid_grid(hp)
    LR = np.broadcast_to(lr1[:, None], lt.shape)
    out = {}
    for name, arr in (('logP', P), ('logS', S)):
        n_gated, n_dense, worst, _ = _mono_counts(arr, LR, GATED_LOGRHO)
        out[name] = f'{n_gated + n_dense} (worst {worst:+.5f})'
    return out


# --------------------------------------------------------------- S1 content
def check_content(out_dir):
    ok = True
    expect_z = {0.0: 0.0, 1.0: 0.017, 3.16: 0.051817, 5.0: 0.079588, 10.0: 0.147441}
    got = {f: M.zsolar_to_z(f) for f in expect_z}
    ok &= check('Z(f) mapping (Z/X scaling, Z_sun = 0.017)',
                all(abs(got[f] - v) < 5e-7 for f, v in expect_z.items()),
                ', '.join(f'{f:g}x -> {got[f]:.6f}' for f in sorted(got)))
    for name in sorted(out_dir.glob('ptab_*.dat')):
        stem = name.name[len('ptab_'):-len('.dat')]
        spath = out_dir / f'stab_{stem}.dat'
        hp, _ = F.read_table(name)
        hs, _ = F.read_table(spath)
        ok &= check(f'{stem}: ptab and stab headers identical', hp.matches(hs))
        f = float(stem.split('_Z')[1].replace('solar', ''))
        yp = float(stem.split('_Y')[1].split('_Z')[0])
        ok &= check(f'{stem}: header Y = Y\'(1-Z)',
                    abs(hp.y - yp * (1 - M.zsolar_to_z(f))) < 1e-8,
                    f'{hp.y:.8g}')
    # cd and cms must agree exactly where only the ideal model is used
    for f in (0.0, 3.16):
        stem = lambda e: f'{e}_Y0.25_Z{f:g}solar'
        pcd = out_dir / f'ptab_{stem("cd")}.dat'
        pcms = out_dir / f'ptab_{stem("cms")}.dat'
        if not (pcd.exists() and pcms.exists()):
            continue
        h, A = F.read_table(pcd)
        _, B = F.read_table(pcms)
        lr1, _ = F.rhomboid_grid(h)
        band = M.BLEND_RHO_Z[0] if f > 0 else M.BLEND_RHO[0]
        m = lr1 <= band
        ok &= check(f'Z={f:g}x: cd and cms identical where only the ideal model is used',
                    np.array_equal(A[m], B[m]), f'{m.sum()} rows (logrho <= {band})')
    return ok


# ------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(HERE), help='directory holding the tables')
    ap.add_argument('--quick', action='store_true',
                    help='skip the Fortran harness and the per-table thermodynamics')
    ap.add_argument('--sc-eos', default=None,
                    help='path to sc_eos.f for the Fortran reader gate '
                         '(default: the copy in the ORCHARD eos submodule)')
    args = ap.parse_args(argv)
    out_dir = Path(args.dir).resolve()
    tables = sorted(out_dir.glob('ptab_*.dat'))

    print(f'\n=== S0: format ({len(tables)} pairs + the two reference tables) ===')
    for p in [REF_DIR / 'ptab.dat', REF_DIR / 'stab.dat'] + tables:
        check_format(p)
    for p in tables:
        check_format(out_dir / p.name.replace('ptab_', 'stab_'))

    if not args.quick:
        print('\n=== S0: the real Fortran reader (gfortran on sc_eos.f) ===')
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            try:
                drv = build_fortran_harness(work, args.sc_eos)
            except RuntimeError as exc:
                check('gfortran harness builds', False, str(exc))
                drv = None
            if drv is None:
                sc_eos = Path(args.sc_eos) if args.sc_eos else SC_EOS_F
                why = ('gfortran is not installed' if not shutil.which('gfortran')
                       else f'{sc_eos} not found; pass --sc-eos PATH')
                info('Fortran reader gate skipped', why)
            else:
                check('gfortran harness builds', True)
                fortran_roundtrip('SCvH reference', REF_DIR / 'ptab.dat',
                                  REF_DIR / 'stab.dat', work, drv)
                for p in tables:
                    fortran_roundtrip(p.name[len('ptab_'):-len('.dat')], p,
                                      out_dir / p.name.replace('ptab_', 'stab_'),
                                      work, drv, n_random=4000)

    print('\n=== S1: grid convention and entropy unit ===')
    check_grid_and_units()

    print('\n=== S1: content ===')
    check_content(out_dir)

    if not args.quick:
        print('\n=== S2: thermodynamics and monotonicity ===')
        ref_mono = reference_mono_counts()
        for p in tables:
            stem = p.name[len('ptab_'):-len('.dat')]
            spath = out_dir / p.name.replace('ptab_', 'stab_')
            z_positive = float(stem.split('_Z')[1].replace('solar', '')) > 0
            check_thermo(stem, p, spath, z_positive)
            check_monotonicity(stem, p, spath, reference=ref_mono)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f'\n{len(_results) - n_fail}/{len(_results)} gates passed'
          + (f', {n_fail} FAILED' if n_fail else ''))
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
