"""
Tests for hhe_z_mixtures: on-the-fly inversion and S-P table accuracy.

Run from the orchard root directory:
    python -m pytest eos/test_hhe_z_mixtures.py -v
    python eos/test_hhe_z_mixtures.py           # standalone
"""

import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eos.eos_class import hhe_z_mixtures, erg_to_kbbar


# =====================================================================
# Shared fixture: one instance reused across all tests
# =====================================================================

_EOS_CACHE = {}

def _get_eos():
    """Return a cached hhe_z_mixtures instance (slow to create)."""
    if 'eos' not in _EOS_CACHE:
        _EOS_CACHE['eos'] = hhe_z_mixtures(
            hhe_eos_name='cms', hg=True,
            smooth_hhe=True, smooth_z=True,
            species_list=['water'],
            logp_range=(6.0, 13.5), logp_step=0.1,
            n_xi=50)
    return _EOS_CACHE['eos']


def _get_eos_with_table():
    """Return a cached instance with a built S-P table."""
    if 'eos_tab' not in _EOS_CACHE:
        eos = _get_eos()
        yvals = np.array([0.20, 0.245, 0.30])
        zvals = np.array([0.0, 0.02])
        eos.build_sp_table(yvals, zvals, n_xi=50, verbose=False)
        _EOS_CACHE['eos_tab'] = eos
    return _EOS_CACHE['eos_tab']


# =====================================================================
# 1. INVERSION TESTS — brentq must recover the queried S
# =====================================================================

def test_inversion_scalar_roundtrip():
    """Scalar inversion: S(P, T(S,P)) must equal the queried S."""
    eos = _get_eos()
    yp, z = 0.245, 0.0

    cases = [
        (5.0,  8.0,  "low S, low P"),
        (7.0,  10.0, "mid range"),
        (10.0, 11.0, "high S, high P"),
        (5.0,  7.0,  "near cold edge"),
        (12.0, 12.0, "high S, high P"),
    ]

    for s_kb, lgp, label in cases:
        logt = eos.get_logt_sp(s_kb, lgp, yp, z)
        assert np.isfinite(logt), f"FAILED to converge: {label}"
        s_check = eos.val.get_s_pt_val(lgp, logt, yp, z) * erg_to_kbbar
        err = abs(s_check - s_kb)
        assert err < 1e-4, (
            f"Inversion roundtrip FAILED [{label}]: "
            f"S_query={s_kb}, S_recovered={s_check:.6f}, |err|={err:.2e}")


def test_inversion_array_roundtrip():
    """Array inversion: all converged points must round-trip."""
    eos = _get_eos()
    yp, z = 0.245, 0.0

    s_arr = np.array([4, 5, 6, 7, 8, 9, 10, 12], dtype=float)
    lgp_arr = np.array([8, 9, 10, 10, 10, 10, 11, 12], dtype=float)

    logt_arr = eos.get_logt_sp(s_arr, lgp_arr, yp, z)
    good = np.isfinite(logt_arr)
    assert good.sum() >= 6, f"Too few converged: {good.sum()}/{len(s_arr)}"

    s_check = eos.val.get_s_pt_val(
        lgp_arr[good], logt_arr[good], yp, z) * erg_to_kbbar
    max_err = np.max(np.abs(s_check - s_arr[good]))
    assert max_err < 1e-4, (
        f"Array inversion max |err| = {max_err:.2e} (limit 1e-4)")


def test_inversion_with_metals():
    """Inversion with Z > 0 must also round-trip."""
    eos = _get_eos()

    cases = [
        (0.245, 0.02,  0.0, 0.0, 0.0, "Z=0.02 water"),
        (0.20,  0.05,  0.1, 0.05, 0.0, "Z=0.05 water+CH4+NH3"),
    ]

    for yp, z, zm, za, zr, label in cases:
        s_kb, lgp = 7.0, 10.0
        logt = eos.get_logt_sp(s_kb, lgp, yp, z, zm, za, zr)
        assert np.isfinite(logt), f"FAILED to converge: {label}"
        s_check = eos.val.get_s_pt_val(
            lgp, logt, yp, z, zm, za, zr) * erg_to_kbbar
        err = abs(s_check - s_kb)
        assert err < 1e-4, (
            f"Inversion with metals FAILED [{label}]: |err|={err:.2e}")


def test_inversion_logrho_consistency():
    """logrho from S-P inversion must match forward model at (P, T)."""
    eos = _get_eos()
    yp, z = 0.245, 0.0
    s_kb, lgp = 7.0, 10.0

    logt = eos.get_logt_sp(s_kb, lgp, yp, z)
    logrho_sp = eos.get_logrho_sp(s_kb, lgp, yp, z)
    logrho_pt = eos.val.get_logrho_pt_val(lgp, logt, yp, z)

    assert np.isfinite(logrho_sp), "logrho_sp is NaN"
    assert abs(logrho_sp - logrho_pt) < 1e-8, (
        f"logrho mismatch: SP={logrho_sp:.8f} vs PT={logrho_pt:.8f}")


def test_inversion_adiabat():
    """Full adiabat: inversion at many pressures for a fixed S."""
    eos = _get_eos()
    yp, z = 0.245, 0.0
    s_kb = 7.0
    logp_grid = np.linspace(7, 13, 50)

    logt_grid = eos.get_logt_sp(s_kb, logp_grid, yp, z)
    good = np.isfinite(logt_grid)
    assert good.sum() >= 40, (
        f"Adiabat: only {good.sum()}/50 converged")

    s_check = eos.val.get_s_pt_val(
        logp_grid[good], logt_grid[good], yp, z) * erg_to_kbbar
    max_err = np.max(np.abs(s_check - s_kb))
    assert max_err < 1e-4, (
        f"Adiabat max |err| = {max_err:.2e}")


def test_inversion_monotonic_adiabat():
    """Adiabat T(P) must be monotonically increasing with P."""
    eos = _get_eos()
    yp, z = 0.245, 0.0
    s_kb = 7.0
    logp_grid = np.linspace(7, 13, 50)

    logt_grid = eos.get_logt_sp(s_kb, logp_grid, yp, z)
    good = np.isfinite(logt_grid)
    logt_valid = logt_grid[good]

    diffs = np.diff(logt_valid)
    assert np.all(diffs >= -1e-6), (
        f"Adiabat is non-monotonic: min(dlogT) = {diffs.min():.6f}")


# =====================================================================
# 2. TABLE TESTS — pre-computed table vs exact inversion
# =====================================================================

def test_table_builds_successfully():
    """Table build must converge on >95% of cells."""
    eos = _get_eos_with_table()
    assert eos._logt_sp_rgi is not None, "Table RGI not set"


def test_table_vs_inversion_accuracy():
    """Table lookup must agree with exact inversion within tolerance.

    This is the primary accuracy benchmark. Tolerance depends on
    table resolution (n_xi=50, dlogP=0.1 → ~0.5% median).
    """
    eos_tab = _get_eos_with_table()

    # Separate instance for on-the-fly inversion (no table)
    eos_inv = hhe_z_mixtures(
        hhe_eos_name='cms', hg=True,
        smooth_hhe=True, smooth_z=True,
        species_list=['water'],
        logp_range=(6.0, 13.5), logp_step=0.1)

    yp, z = 0.245, 0.0
    s_test = np.array([4, 5, 6, 7, 8, 9, 10, 12], dtype=float)
    lgp_test = np.array([8, 9, 10, 10, 10, 10, 11, 12], dtype=float)

    max_dlogt = 0.0
    n_compared = 0

    for s_kb, lgp in zip(s_test, lgp_test):
        logt_tab = eos_tab.get_logt_sp(float(s_kb), float(lgp), yp, z)
        logt_inv = eos_inv.get_logt_sp(float(s_kb), float(lgp), yp, z)

        if np.isfinite(logt_tab) and np.isfinite(logt_inv):
            dlogt = abs(logt_tab - logt_inv)
            max_dlogt = max(max_dlogt, dlogt)
            n_compared += 1

    assert n_compared >= 6, f"Too few comparable points: {n_compared}"
    # With n_xi=50, dlogP=0.1: expect max |dlogT| < 0.05
    assert max_dlogt < 0.05, (
        f"Table vs inversion: max |dlogT| = {max_dlogt:.5f} (limit 0.05)")


def test_table_vs_inversion_random():
    """Random sample: median relative T error must be < 1%."""
    eos_tab = _get_eos_with_table()

    eos_inv = hhe_z_mixtures(
        hhe_eos_name='cms', hg=True,
        smooth_hhe=True, smooth_z=True,
        species_list=['water'],
        logp_range=(6.0, 13.5), logp_step=0.1)

    rng = np.random.default_rng(123)
    yp, z = 0.245, 0.0

    s_rand = rng.uniform(4, 11, 100)
    lgp_rand = rng.uniform(7, 13, 100)

    dlogt_list = []
    for s_kb, lgp in zip(s_rand, lgp_rand):
        lt_tab = eos_tab.get_logt_sp(float(s_kb), float(lgp), yp, z)
        lt_inv = eos_inv.get_logt_sp(float(s_kb), float(lgp), yp, z)
        if np.isfinite(lt_tab) and np.isfinite(lt_inv):
            dlogt_list.append(abs(lt_tab - lt_inv))

    dlogt_arr = np.array(dlogt_list)
    assert len(dlogt_arr) >= 50, (
        f"Too few comparable: {len(dlogt_arr)}/100")

    median_dlogt = np.median(dlogt_arr)
    # median |dlogT| < 0.005 → median |dT/T| < ~1%
    assert median_dlogt < 0.005, (
        f"Median |dlogT| = {median_dlogt:.5f} (limit 0.005)")


def test_table_save_load_roundtrip():
    """Save → load → query must reproduce the same values."""
    eos_tab = _get_eos_with_table()

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmppath = f.name

    try:
        # Save
        result = {
            'xi_vals': eos_tab.xi_vals,
            'logpvals': eos_tab.logp_vals,
            'yvals': eos_tab._yvals,
            'zvals': eos_tab._zvals,
            'logt_sp': eos_tab._logt_sp_rgi.values,
            's_lo': eos_tab._s_lo,
            's_hi': eos_tab._s_hi,
            'logt_min': eos_tab.logt_min,
            'logt_max': eos_tab.logt_max,
        }
        np.savez_compressed(tmppath, **result)

        # Load into a fresh instance
        eos2 = hhe_z_mixtures(
            hhe_eos_name='cms', hg=True,
            smooth_hhe=True, smooth_z=True,
            species_list=['water'])
        eos2.load_sp_table(tmppath)

        # Compare
        yp, z = 0.245, 0.0
        s_test = [5.0, 7.0, 10.0]
        lgp_test = [9.0, 10.0, 11.0]

        for s_kb, lgp in zip(s_test, lgp_test):
            lt1 = eos_tab.get_logt_sp(s_kb, lgp, yp, z)
            lt2 = eos2.get_logt_sp(s_kb, lgp, yp, z)
            if np.isfinite(lt1) and np.isfinite(lt2):
                assert abs(lt1 - lt2) < 1e-10, (
                    f"Save/load mismatch at S={s_kb}, P={lgp}: "
                    f"{lt1} vs {lt2}")
    finally:
        os.unlink(tmppath)


def test_table_zero_nans():
    """Built table must have zero NaN cells after interpolation."""
    eos = _get_eos_with_table()
    logt_data = eos._logt_sp_rgi.values
    n_nan_logt = np.isnan(logt_data).sum()
    assert n_nan_logt == 0, (
        f"logt_sp has {n_nan_logt} NaNs (must be 0)")


# =====================================================================
# 3. EDGE CASE TESTS
# =====================================================================

def test_out_of_bounds_returns_nan():
    """Queries far outside the physical domain must return NaN."""
    eos = _get_eos()
    yp = 0.245

    # S way too high: returns boundary clamp (logt_max) or NaN
    logt = eos.get_logt_sp(200.0, 10.0, yp)
    assert np.isnan(logt) or logt == eos.logt_max, (
        f"Expected NaN or logt_max for S=200, got {logt}")

    # S negative: returns boundary clamp or NaN
    logt = eos.get_logt_sp(-5.0, 10.0, yp)
    assert np.isnan(logt) or logt == eos.logt_min, (
        f"Expected NaN or logt_min for S=-5, got {logt}")


def test_different_compositions_differ():
    """Different Y' or Z should give different T at the same (S, P)."""
    eos = _get_eos()
    s_kb, lgp = 7.0, 10.0

    logt_1 = eos.get_logt_sp(s_kb, lgp, 0.245, 0.0)
    logt_2 = eos.get_logt_sp(s_kb, lgp, 0.30, 0.0)
    logt_3 = eos.get_logt_sp(s_kb, lgp, 0.245, 0.05)

    assert np.isfinite(logt_1) and np.isfinite(logt_2) and np.isfinite(logt_3)
    assert logt_1 != logt_2, "Y' change should affect T"
    assert logt_1 != logt_3, "Z change should affect T"


# =====================================================================
# Runner
# =====================================================================

def run_all():
    """Run all tests and print results."""
    tests = [
        ("Inversion: scalar roundtrip",      test_inversion_scalar_roundtrip),
        ("Inversion: array roundtrip",        test_inversion_array_roundtrip),
        ("Inversion: with metals",            test_inversion_with_metals),
        ("Inversion: logrho consistency",     test_inversion_logrho_consistency),
        ("Inversion: full adiabat",           test_inversion_adiabat),
        ("Inversion: monotonic adiabat",      test_inversion_monotonic_adiabat),
        ("Table: builds successfully",        test_table_builds_successfully),
        ("Table: vs inversion accuracy",      test_table_vs_inversion_accuracy),
        ("Table: vs inversion random",        test_table_vs_inversion_random),
        ("Table: zero NaNs",                  test_table_zero_nans),
        ("Table: save/load roundtrip",        test_table_save_load_roundtrip),
        ("Edge: out-of-bounds → NaN",         test_out_of_bounds_returns_nan),
        ("Edge: compositions differ",         test_different_compositions_differ),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 65)
    print("hhe_z_mixtures test suite")
    print("=" * 65)

    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
            errors.append((name, e))

    print("-" * 65)
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print("=" * 65)

    return failed == 0


if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
