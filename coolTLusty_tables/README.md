# CoolTLusty EOS tables from the ORCHARD EOS

Drop-in replacements for the SCvH95 `ptab`/`stab` tables, built from
Chabrier & Debras (2021) (`cd`) and Chabrier et al. (2019) with the
Howard & Guillot (2023) corrections (`cms`), with pure-water metals
(revised AQUA) for the metal-enriched mixtures.

Same rhomboid, same grid convention, same file format as
`brianna_originals/{ptab,stab}.dat`, so nothing in CoolTLusty needs to change.

## The files

| File | Contents |
|---|---|
| `ptab_{cd,cms}_Y0.25_Z{0,1,3.16,5,10}solar.dat` | log10 P [dyn cm^-2] |
| `stab_{cd,cms}_Y0.25_Z{0,1,3.16,5,10}solar.dat` | log10 S (see units below) |
| `{cd,cms}_Y0.25_Z*solar.meta.json` | how each pair was built: EOS, Z, Y, blend parameters, git SHAs, checksums, cell counts |
| `heatmap_scvh_vs_cd_Z*solar.png` | SCvH and cd side by side, log P and log S, with the difference |
| `heatmap_cd_metallicity_series.png` | the pure H-He table, then what each metallicity adds |
| `compare_scvh_stats.txt`, `compare_scvh_*.png` | difference maps, isochores and the per-region statistics |
| `brianna_originals/` | your SCvH95 tables, included so the scripts run as shipped |

In the heatmaps, the two value panels in a row share one color scale so they can
be compared directly, the difference panel is centred on zero, and the scale for
it is set from logrho <= -2; denser than that every EOS disagrees and the panel
saturates.  Grey marks the SCvH fill cells.

`Y0.25` in the filename is **Y' = Y/(1-Z)**, the helium fraction of the H-He
sub-mixture, held fixed across the metallicity series.  The header carries the
**absolute** Y = Y'(1-Z), so it reads 0.250000 at Z = 0 and 0.213140 at 10x solar.

"N x solar" scales Z/X: `Z = f a/(1+f a)` with `a = Z_sun/(1-Z_sun)` and
Z_sun = 0.017 (Chen et al. 2023), giving Z = 0.017000, 0.051817, 0.079588 and
0.147441 for f = 1, 3.16, 5 and 10.

## Two conventions worth confirming

**Grid.** Row *i* sits at `logrho = R1 + i(R2-R1)/N` and column *j* at
`logT = alpha_i + j beta_i/100`, exactly as `LOOK` inverts them in `sc_eos.f`.
The header's upper edges (R2, and the two high-T bounds) are therefore *not*
grid nodes.  Fitting your `ptab.dat` to an ideal gas gives a constant mean
molecular weight to 1.5e-4 dex under this convention, versus 1.1e-2 dex if the
grid is built with `linspace` — which is how we pinned it down.

**Entropy unit.** Your `stab` is `log10(S * m_H / k_B)` with
m_H = 1.007825 amu, measured to rms 3e-5 dex over 8000 molecular cells and
confirmed independently by a Maxwell-relation slope ratio of 1.00784.  That is
+0.00339 dex above the literal `log10(S/RCON)` implied by
`RCON = 8.31434e7 = N_A k_B` in `sc_eos.f`.  These tables follow **your**
convention so they can stand in for the SCvH ones unchanged.  If CoolTLusty
really does apply `S = RCON * 10**stab`, then it has been carrying that 0.78%
offset all along; `--s-unit amu` regenerates the tables the other way.

Worth confirming on your side: that CoolTLusty's reader is still
`SETTABL`/`LOOK` (`SL(330,100)`, `FORMAT(10F8.5)`), and that `YHEA` from the
header is unused, as it is in `sc_eos.f`.

## How a table is built

Inside the domain of the H-He EOS, values come from the ORCHARD forward model:
density inverted for P at each (rho, T), and the entropy evaluated there.
Outside it — below about logrho = -6, and below logT = 2.25 where the raw
helium isotherms break down — values come from an analytic ideal-gas model:
H2 with a quantum rigid rotor (ortho/para in equilibrium) and harmonic
vibration, H2 <-> 2H in Saha equilibrium, He, and H2O vapor for the metals.
That model reproduces your SCvH table to 2.6e-6 dex in logP and 2.5e-5 dex in
log S where the gas is molecular, dissociation ridge included.

The two are joined by a smoothstep **in logrho only**, so the blended chi_T and
c_v stay exact convex combinations of the two models; a weight that varied with
P or T would corrupt grad_ad across dissociation.  Below the temperature anchor
the non-ideal excess is carried down at fixed rho by a first-order expansion
that satisfies the Maxwell relation exactly.

Roughly 60% of the rhomboid is below the EOS's native domain (logrho < -8 alone
is 47% of the cells), so this extension is doing real work — the alternative,
extrapolating the tables, is wrong by up to 2 dex in P at logrho = -15.

## How they compare with SCvH

From `compare_scvh_stats.txt` (pure H-He, Y' = 0.25, on identical nodes, with
SCvH's 360 fill cells masked):

| Region | median dlogP | median dlogS |
|---|---|---|
| ideal-gas region, molecular | +0.00000 | +0.00131 |
| ideal-gas region, dissociating | -0.00001 | +0.00002 |
| blend band | +0.00002 | +0.00217 |
| EOS source, logrho <= -2 | +0.00053 | +0.00270 |
| EOS source, logrho > -2 | +0.00009 | +0.00560 |

The small positive entropy offset is expected: ORCHARD's entropy of mixing
assumes atomic hydrogen even where hydrogen is molecular (+0.041 k_B/amu), and
we deliberately kept that convention so these tables stay consistent with
ORCHARD's interior models.  The ideal-gas extension carries the same
convention, so there is no step at the seam.

`cd` and `cms` are nearly identical on this domain (median |dlogP| = 0.00000),
as expected: the HG23 corrections matter at pressures well above an atmosphere.

Two things in the SCvH reference that these tables do not reproduce, by design:
a curvature kink in S at logT ~ 3.0 (30-40x the local background, visible as a
horizontal seam in `compare_scvh_maps.png`), and, at logrho > -2, 30 decreasing
P steps and 8 decreasing S steps along rows, the worst a 2.0 dex drop in S.
The new tables have 24 and 0 there.

## Regenerating

```bash
conda activate orchard_env
cd /Users/arevalo/orchard

# the ten pairs above (about 15 s)
python eos/coolTLusty_tables/make_cooltlusty_tables.py

# a different composition or domain
python eos/coolTLusty_tables/make_cooltlusty_tables.py --eos cms --yprime 0.275 \
    --zsolar 3.16 --logrho -8 0.5 --T-at-rhomin 100 9000 --T-at-rhomax 200 20000 --nrho 250

python eos/coolTLusty_tables/verify_tables.py     # quality gates; non-zero exit on failure
python eos/coolTLusty_tables/compare_scvh.py      # the comparison above
python eos/coolTLusty_tables/plot_heatmaps.py     # the heatmaps (--eos cms for the CMS set)
```

Only `make_cooltlusty_tables.py` needs ORCHARD itself.  Reading the tables,
verifying them and redrawing every figure works from this directory alone with
just numpy, scipy and matplotlib:

```bash
python verify_tables.py --dir . --sc-eos /path/to/sc_eos.f   # omit --sc-eos to skip that gate
python compare_scvh.py --dir .
python plot_heatmaps.py --dir .
```

`ctl_format.py` is a standalone reader for these files, with no ORCHARD imports:

```python
import ctl_format as F
header, logP = F.read_table('ptab_cd_Y0.25_Z0solar.dat')
logrho, logT = F.rhomboid_grid(header)      # node coordinates, edge-excluded
value = F.look(header, logP, -6.0, 3.0)     # the sc_eos.f LOOK interpolation
```

`verify_tables.py` compiles `SETTABL`/`LOOK` straight out of `sc_eos.f` with
gfortran, points it at the generated files, and checks the values it returns
against an independent Python port at ~33000 points per table (agreement is
~1e-14 dex, with identical off-table behavior).  It runs the same check on
`brianna_originals` first as a control.

## Known limitations

* **logrho > -2** (rho > 0.01 g cm^-3): CD, CMS and SCvH genuinely disagree by
  up to a few tenths of a dex, and all three are rough there.  Atmospheres
  never reach these densities.  Everything at logrho <= -2 is gated by the
  verification suite; denser cells are reported, not gated.
* **Water is treated as vapor** in the extension, with no condensation, even at
  50 K.  Inside the EOS domain, AQUA condenses it, which is the main reason the
  metal-rich tables carry a slightly larger thermodynamic inconsistency near
  the condensation line than the pure H-He ones.
* **Water dissociation above ~2500 K** and **hydrogen ionization** are neglected
  in the extension.  On this rhomboid the peak ionization fraction is 3e-5; the
  CLI warns if a requested domain pushes past 1e-3.
* A handful of cells per table (10-35) fall back to the saved rho-T table
  because the forward model has isolated dropouts at high density; they are
  counted in the metadata as `n_root_repaired`.
