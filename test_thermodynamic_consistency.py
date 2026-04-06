"""
Thermodynamic consistency tests for ORCHARD EOS tables and mixtures.

Tests the first law of thermodynamics and the Maxwell relation from
the Gibbs free energy.

First law:  dU = TdS - PdV = TdS + (P/rho^2)drho

  A) At constant P:  dU/dlogT|_P = T dS/dlogT|_P + (P/rho^2) drho/dlogT|_P
  B) At constant T:  dU/dlogP|_T = T dS/dlogP|_T + (P/rho^2) drho/dlogP|_T

Maxwell (from dG = VdP - SdT, cross-derivative equality):
  C)  (P/rho^2) drho/dlogT|_P = T dS/dlogP|_T

All derivatives are computed via centered finite differences on the
VAL on-the-fly forward model (no pre-computed tables).

Run standalone:
    python eos/test_thermodynamic_consistency.py

Or use the class directly:
    from eos.test_thermodynamic_consistency import ThermodynamicConsistency
    tc = ThermodynamicConsistency()
    tc.test_val_mixture('cd', y_prime=0.245, z=0.0)
    tc.plot_all_heatmaps()
"""

import numpy as np
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eos.eos_class import hhe_eos, z_eos, val_mixtures, hhe_z_mixtures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmasher as cmr


# =====================================================================
# ThermodynamicConsistency
# =====================================================================

class ThermodynamicConsistency:
    """Test thermodynamic consistency of ORCHARD EOS.

    Three relations are checked at each (logP, logT) grid point,
    using centered finite differences on linear S, U, rho.

    A) First law at constant P:
       dU/dlogT|_P = T * dS/dlogT|_P + (P/rho^2) * drho/dlogT|_P

    B) First law at constant T:
       dU/dlogP|_T = T * dS/dlogP|_T + (P/rho^2) * drho/dlogP|_T

    C) Maxwell relation (from dG = VdP - SdT, cross-deriv):
       (P/rho^2) * drho/dlogT|_P = T * dS/dlogP|_T
    """

    TEST_NAMES = ['first_law_constP', 'first_law_constT', 'maxwell']

    def __init__(self, fd_step=0.01, verbose=True,
                 lgp_range=(6.0, 14.0), lgp_step=0.5,
                 lgt_range=(2.5, 5.5), lgt_step=0.25):
        self.fd_step = fd_step
        self.verbose = verbose
        self.results = {}

        # Build test grid
        self.lgp_1d = np.arange(lgp_range[0], lgp_range[1] + 0.01, lgp_step)
        self.lgt_1d = np.arange(lgt_range[0], lgt_range[1] + 0.01, lgt_step)
        self.lgp_2d, self.lgt_2d = np.meshgrid(
            self.lgp_1d, self.lgt_1d, indexing='ij')
        self.lgp_flat = self.lgp_2d.ravel()
        self.lgt_flat = self.lgt_2d.ravel()
        self.grid_shape = self.lgp_2d.shape

        # EOS instance cache
        self._cache = {}

    # -----------------------------------------------------------------
    # Finite-difference helpers
    # -----------------------------------------------------------------

    def _diff_T(self, func, lgp, lgt):
        """d(func)/d(logT)|_P  via centered FD."""
        h = self.fd_step
        return (func(lgp, lgt + h) - func(lgp, lgt - h)) / (2.0 * h)

    def _diff_P(self, func, lgp, lgt):
        """d(func)/d(logP)|_T  via centered FD."""
        h = self.fd_step
        return (func(lgp + h, lgt) - func(lgp - h, lgt)) / (2.0 * h)

    # -----------------------------------------------------------------
    # Error computation
    # -----------------------------------------------------------------

    @staticmethod
    def _frac_error(lhs, rhs):
        """Symmetric fractional error, 0 where both sides are negligible."""
        denom = 0.5 * (np.abs(lhs) + np.abs(rhs))
        with np.errstate(invalid='ignore', divide='ignore'):
            err = np.abs(lhs - rhs) / (denom + 1e-30)
        err[denom < 1e-30] = 0.0
        return err

    @staticmethod
    def _compute_stats(errors):
        """Return summary statistics from an error array."""
        valid = np.isfinite(errors)
        e = errors[valid]
        if len(e) == 0:
            return {'median': np.nan, 'p95': np.nan, 'max': np.nan,
                    'n_valid': 0, 'n_total': len(errors)}
        return {
            'median': float(np.median(e)),
            'p95':    float(np.percentile(e, 95)),
            'max':    float(np.max(e)),
            'n_valid': int(np.sum(valid)),
            'n_total': len(errors),
        }

    # -----------------------------------------------------------------
    # Core: run three tests given S, rho, U callables
    # -----------------------------------------------------------------

    def _run_three_tests(self, S_func, rho_func, U_func, label):
        """Run tests A, B, C on the grid.

        Parameters
        ----------
        S_func   : callable(lgp, lgt) -> linear S  [erg/(g*K)]
        rho_func : callable(lgp, lgt) -> linear rho  [g/cm^3]
        U_func   : callable(lgp, lgt) -> linear U  [erg/g]
        label    : str

        Returns
        -------
        dict with keys 'first_law_constP', 'first_law_constT', 'maxwell',
        each containing 'errors_2d' (2-D array) and 'stats' (dict).
        """
        lgp = self.lgp_flat
        lgt = self.lgt_flat

        T   = 10.0**lgt
        P   = 10.0**lgp
        rho = rho_func(lgp, lgt)

        # Derivatives in logT (const P)
        dS_dlogT   = self._diff_T(S_func,   lgp, lgt)
        drho_dlogT = self._diff_T(rho_func, lgp, lgt)
        dU_dlogT   = self._diff_T(U_func,   lgp, lgt)

        # Derivatives in logP (const T)
        dS_dlogP   = self._diff_P(S_func,   lgp, lgt)
        drho_dlogP = self._diff_P(rho_func, lgp, lgt)
        dU_dlogP   = self._diff_P(U_func,   lgp, lgt)

        with np.errstate(invalid='ignore', divide='ignore'):
            P_over_rho2 = P / (rho**2)

        # --- Test A: first law at constant P ---
        #   dU = TdS + (P/rho^2)drho
        lhs_A = dU_dlogT
        rhs_A = T * dS_dlogT + P_over_rho2 * drho_dlogT
        err_A = self._frac_error(lhs_A, rhs_A)

        # --- Test B: first law at constant T ---
        lhs_B = dU_dlogP
        rhs_B = T * dS_dlogP + P_over_rho2 * drho_dlogP
        err_B = self._frac_error(lhs_B, rhs_B)

        # --- Test C: Maxwell relation ---
        #   From dG = VdP - SdT  =>  (dV/dT)|_P = -(dS/dP)|_T
        #   With V = 1/rho:  (P/rho^2) drho/dlogT|_P = T dS/dlogP|_T
        lhs_C = P_over_rho2 * drho_dlogT
        rhs_C = T * dS_dlogP
        err_C = self._frac_error(lhs_C, rhs_C)

        result = {}
        for name, err_flat in [('first_law_constP', err_A),
                                ('first_law_constT', err_B),
                                ('maxwell',          err_C)]:
            err_2d = err_flat.reshape(self.grid_shape)
            stats = self._compute_stats(err_flat)
            result[name] = {'errors_2d': err_2d, 'stats': stats}

            if self.verbose:
                s = stats
                print(f"  {label:40s} / {name:20s}  "
                      f"med={s['median']:.2e}  p95={s['p95']:.2e}  "
                      f"max={s['max']:.2e}  "
                      f"({s['n_valid']}/{s['n_total']} valid)")

        return result

    # =================================================================
    # Level 1: Pure species
    # =================================================================

    def test_pure_hhe(self, hhe_eos_name='cms'):
        """Test pure H and pure He tables for first-law consistency.

        Loads the raw (unsmoothed) CMS19 or CD21 tables via hhe_eos
        and converts log10(S), log10(rho), log10(U) to linear before
        applying the three consistency tests.  Tests H and He
        independently to isolate which table has larger violations.
        """
        key = f'hhe_eos_{hhe_eos_name}'
        if key not in self._cache:
            self._cache[key] = hhe_eos(hhe_eos=hhe_eos_name, smooth_hhe=False)
        eos = self._cache[key]

        results = {}
        for species, s_fn, rho_fn, u_fn in [
            ('H',  eos.get_s_h,  eos.get_logrho_h,  eos.get_logu_h),
            ('He', eos.get_s_he, eos.get_logrho_he, eos.get_logu_he),
        ]:
            label = f'pure_{hhe_eos_name}_{species}'
            if self.verbose:
                print(f"\n--- {label} ---")

            # Wrappers: convert log10 table outputs to linear
            def S_func(lgp, lgt, _fn=s_fn):
                return 10.0**_fn(lgp, lgt)

            def rho_func(lgp, lgt, _fn=rho_fn):
                return 10.0**_fn(lgp, lgt)

            def U_func(lgp, lgt, _fn=u_fn):
                return 10.0**_fn(lgp, lgt)

            results[species] = self._run_three_tests(
                S_func, rho_func, U_func, label)

        key_out = f'pure_hhe_{hhe_eos_name}'
        self.results[key_out] = results
        return results

    def test_pure_z(self, species='water'):
        """Test a pure Z species table for first-law consistency."""
        key = f'z_eos_{species}'
        if key not in self._cache:
            self._cache[key] = z_eos(species=species, smooth_z=False)
        z = self._cache[key]

        label = f'pure_z_{species}'
        if self.verbose:
            print(f"\n--- {label} ---")

        def S_func(lgp, lgt):
            return 10.0**z.get_logs_pt(lgp, lgt)

        def rho_func(lgp, lgt):
            return 10.0**z.get_logrho_pt(lgp, lgt)

        def U_func(lgp, lgt):
            return 10.0**z.get_logu_pt(lgp, lgt)

        result = self._run_three_tests(S_func, rho_func, U_func, label)
        self.results[label] = result
        return result

    # =================================================================
    # Level 2: H-He VAL mixtures (on-the-fly, no tables)
    # =================================================================

    def test_val_mixture(self, hhe_eos_name='cd', y_prime=0.245, z=0.0):
        """Test a val_mixtures composition for first-law consistency.

        Uses the on-the-fly VAL forward model (no pre-computed tables).
        The val_mixtures class mixes H-He (via hhe_eos) with Z species
        (via z_eos) using the volume addition law, with ideal + non-ideal
        entropy of mixing and P-T dependent mu_H.  S and U are returned
        in linear units; logrho is log10 and converted here.
        """
        key = f'val_{hhe_eos_name}'
        if key not in self._cache:
            self._cache[key] = val_mixtures(
                hhe_eos_name=hhe_eos_name, hg=True,
                smooth_hhe=False, smooth_z=False,
                species_list=['water'])
        val = self._cache[key]

        label = f'val_{hhe_eos_name}_yp{y_prime:.3f}_z{z:.3f}'
        if self.verbose:
            print(f"\n--- {label} ---")

        # val_mixtures: S and U are linear, logrho is log10
        def S_func(lgp, lgt):
            return val.get_s_pt_val(lgp, lgt, y_prime, z)

        def rho_func(lgp, lgt):
            return 10.0**val.get_logrho_pt_val(lgp, lgt, y_prime, z)

        def U_func(lgp, lgt):
            return val.get_u_pt_val(lgp, lgt, y_prime, z)

        result = self._run_three_tests(S_func, rho_func, U_func, label)
        self.results[label] = result
        return result

    # =================================================================
    # Level 3: H-He-Z mixtures (on-the-fly via val)
    # =================================================================

    def test_hhe_z_mixture(self, hhe_eos_name='cd', y_prime=0.245, z=0.02):
        """Test a full H-He-Z composition for first-law consistency.

        Uses the on-the-fly VAL forward model (pt_tab=False, inv_tab=False).
        """
        key = f'hhe_z_{hhe_eos_name}'
        if key not in self._cache:
            self._cache[key] = hhe_z_mixtures(
                hhe_eos_name=hhe_eos_name, hg=True,
                smooth_hhe=False, smooth_z=False,
                species_list=['water'], z_eos='water',
                pt_tab=False, inv_tab=False)
        mix = self._cache[key]

        label = f'hhe_z_{hhe_eos_name}_yp{y_prime:.3f}_z{z:.3f}'
        if self.verbose:
            print(f"\n--- {label} ---")

        def S_func(lgp, lgt):
            return mix.val.get_s_pt_val(lgp, lgt, y_prime, z)

        def rho_func(lgp, lgt):
            return 10.0**mix.val.get_logrho_pt_val(lgp, lgt, y_prime, z)

        def U_func(lgp, lgt):
            return mix.val.get_u_pt_val(lgp, lgt, y_prime, z)

        result = self._run_three_tests(S_func, rho_func, U_func, label)
        self.results[label] = result
        return result

    # =================================================================
    # run_all
    # =================================================================

    def run_all(self):
        """Run the full composition matrix and print a summary report."""
        print("=" * 90)
        print("Thermodynamic Consistency Tests")
        print("=" * 90)

        # Level 1: Pure species
        for name in ['cms', 'cd']:
            self.test_pure_hhe(name)
        self.test_pure_z('water')

        # Level 2: H-He mixtures
        for name in ['cms', 'cd']:
            for yp in [0.0, 0.245, 1.0]:
                self.test_val_mixture(name, y_prime=yp, z=0.0)

        # Level 3: H-He-Z mixtures
        self.test_hhe_z_mixture('cd', y_prime=0.245, z=0.02)
        self.test_hhe_z_mixture('cd', y_prime=0.245, z=0.50)

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print a compact summary table of all results."""
        print("\n")
        print("=" * 90)
        print("SUMMARY")
        print("=" * 90)
        hdr = f"{'Test':<55s} {'Median':>9s} {'P95':>9s} {'Max':>9s}"
        print(hdr)
        print("-" * 90)

        for group_label, group in self.results.items():
            if any(k in self.TEST_NAMES for k in group):
                for tname in self.TEST_NAMES:
                    if tname in group:
                        s = group[tname]['stats']
                        tag = f"{group_label} / {tname}"
                        print(f"  {tag:<53s} {s['median']:9.2e} "
                              f"{s['p95']:9.2e} {s['max']:9.2e}")
            else:
                for species, tests in group.items():
                    for tname in self.TEST_NAMES:
                        if tname in tests:
                            s = tests[tname]['stats']
                            tag = f"{group_label} / {species} / {tname}"
                            print(f"  {tag:<53s} {s['median']:9.2e} "
                                  f"{s['p95']:9.2e} {s['max']:9.2e}")
        print("=" * 90)

    # =================================================================
    # 2-D Heatmap plotting
    # =================================================================

    def plot_heatmaps(self, label, save_dir='plots'):
        """Plot 2-D heatmaps of log10(fractional error) for a test group."""
        os.makedirs(save_dir, exist_ok=True)
        group = self.results.get(label)
        if group is None:
            print(f"No results for '{label}'")
            return
        if any(k in self.TEST_NAMES for k in group):
            self._plot_one_heatmap(group, label, save_dir)
        else:
            for species, tests in group.items():
                self._plot_one_heatmap(tests, f"{label}_{species}", save_dir)

    def _plot_one_heatmap(self, tests, label, save_dir):
        """Plot a 3-panel heatmap figure for one composition.

        Each panel is independently normalized to [0, 1] by its own
        maximum, so every panel uses its full dynamic range.
        The per-panel max is shown in the title.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True,
                                 gridspec_kw={'wspace': 0.08,
                                              'right': 0.88})
        fig.suptitle(f'Thermodynamic Consistency: {label}',
                     fontsize=14, y=0.98)

        ims = []
        for ax, tname in zip(axes, self.TEST_NAMES):
            if tname not in tests:
                ax.set_visible(False)
                ims.append(None)
                continue
            err_2d = tests[tname]['errors_2d']
            finite = err_2d[np.isfinite(err_2d)]
            if len(finite) > 0:
                panel_p95 = np.percentile(finite, 95)
            else:
                panel_p95 = 1.0
            if panel_p95 == 0.0:
                panel_p95 = 1.0

            normed = np.clip(err_2d / panel_p95, 0.0, 1.0)

            im = ax.pcolormesh(
                self.lgp_1d, self.lgt_1d, normed.T,
                cmap=cmr.ember, vmin=0.0, vmax=1.0,
                shading='nearest')
            ims.append(im)
            ax.set_xlabel(r'$\log_{10}\,P$ [dyn cm$^{-2}$]')
            ax.set_title(f'{tname.replace("_", " ")}\n'
                         f'(p95 = {panel_p95:.2e})')

        axes[0].set_ylabel(r'$\log_{10}\,T$ [K]')

        # Place colorbar to the right of the last panel, outside
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
        last_im = next(im for im in reversed(ims) if im is not None)
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label('Fractional error (normalized to 95th pctl)')

        fname = os.path.join(save_dir,
                             f'thermodynamic_consistency_{label}.png')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"  Saved: {fname}")

    def plot_all_heatmaps(self, save_dir='plots'):
        """Plot heatmaps for every composition in self.results."""
        if self.verbose:
            print("\nGenerating heatmaps...")
        for label in self.results:
            self.plot_heatmaps(label, save_dir=save_dir)


# =====================================================================
# Standalone entry point
# =====================================================================

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    tc = ThermodynamicConsistency(
        fd_step=0.01, verbose=True,
        lgp_range=(6.0, 14.0), lgp_step=0.05,
        lgt_range=(2.5, 5.5),  lgt_step=0.025)

    # H-He-Z VAL mixtures (on-the-fly) with CMS19 and CD21
    print("=" * 90)
    print("Thermodynamic Consistency: H-He-Z VAL mixtures (on-the-fly)")
    print("=" * 90)

    for eos_name in ['cms', 'cd']:
        for z in [0.0, 0.017, 0.051]:
            tc.test_val_mixture(eos_name, y_prime=0.275, z=z)

    tc._print_summary()
    tc.plot_all_heatmaps()

    # Find worst error
    worst = 0.0
    for group_label, group in tc.results.items():
        if any(k in ThermodynamicConsistency.TEST_NAMES for k in group):
            items = group.items()
        else:
            items = []
            for sp_tests in group.values():
                items.extend(sp_tests.items())
        for tname, data in items:
            if isinstance(data, dict) and 'stats' in data:
                mx = data['stats']['max']
                if np.isfinite(mx) and mx > worst:
                    worst = mx

    print(f"\nWorst fractional error across all tests: {worst:.2e}")
