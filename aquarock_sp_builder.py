"""Build S-P basis tables for ROCKWATER_INTERP_CORE_EOS f_rock variants.

Evaluates the class's OWN (bracket-fixed, cc1e785) forward/inversion on the
same (S, P) axes as the reference aquarock_core_eos_sp_frock0.50.npz and
writes f_rock-tagged npz files next to it, in the exact format
``_load_sp_table`` expects.  Pure speed optimization: table values are the
fixed bisection's own surfaces, so table-backed and on-the-fly paths agree
by construction (validated by the companion round-trip check).

Usage (repo root):
    python eos/aquarock_sp_builder.py 0.15 0.25 0.35 0.75 1.0
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eos import aquarock_core_eos as ac  # noqa: E402

REF = Path(__file__).parent / "aqua" / "aquarock" / \
    "aquarock_core_eos_sp_frock0.50.npz"


def build(f_rock):
    ref = np.load(REF)
    svals = np.asarray(ref["svals_sp"], dtype=float)   # k_B/baryon
    pvals = np.asarray(ref["pvals_sp"], dtype=float)   # GPa
    eos = ac.ROCKWATER_INTERP_CORE_EOS(f_rock=f_rock)
    nS, nP = svals.size, pvals.size
    T = np.empty((nS, nP)); RHO = np.empty((nS, nP)); U = np.empty((nS, nP))
    CP = np.empty((nS, nP)); CV = np.empty((nS, nP)); AL = np.empty((nS, nP))
    for i, s in enumerate(svals):
        Ti = np.asarray(eos.get_t_sp(np.full(nP, s), pvals)).reshape(-1)
        T[i] = Ti
        RHO[i] = np.asarray(eos.get_rho_pt(pvals, Ti)).reshape(-1)
        U[i] = np.asarray(eos.get_u_pt(pvals, Ti)).reshape(-1)
        CP[i] = np.asarray(eos.get_cp_pt(pvals, Ti)).reshape(-1)
        CV[i] = np.asarray(eos.get_cv_pt(pvals, Ti)).reshape(-1)
        AL[i] = np.asarray(eos.get_alpha_pt(pvals, Ti)).reshape(-1)
        if (i + 1) % 10 == 0:
            print(f"  f_rock {f_rock}: {i+1}/{nS} S rows", flush=True)
    out = REF.parent / f"aquarock_core_eos_sp_frock{f_rock:.2f}.npz"
    np.savez_compressed(out, svals_sp=svals, pvals_sp=pvals, t_grid_sp=T,
                        rho_grid_sp=RHO, u_grid_sp=U, cp_grid_sp=CP,
                        cv_grid_sp=CV, alpha_grid_sp=AL)
    print(f"wrote {out}  (T range {T.min():.0f}-{T.max():.0f} K)")


if __name__ == "__main__":
    for arg in (sys.argv[1:] or ["0.25"]):
        build(float(arg))
