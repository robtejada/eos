"""
Lightweight EOS data verification for ORCHARD.

Called at EOS initialization time in initial.py to give users a clear
error message (with download instructions) if the EOS data files are
missing or are unresolved Git LFS pointer stubs.
"""

import os

# Sentinel: v1.0 CD21+AQUA forward table (~981 MB when real data).
# Git LFS pointer stubs are ~130 bytes.
_SENTINEL_REL = os.path.join("cd", "cd_aqua_pt.npz")
_MIN_BYTES = 1_000_000  # 1 MB


def verify_eos_data(eos_dir=None):
    """
    Check that EOS data files are present and not LFS pointer stubs.

    Parameters
    ----------
    eos_dir : str, optional
        Path to the eos/ directory.  Defaults to the directory containing
        this file (i.e. the eos/ submodule root).

    Raises
    ------
    RuntimeError
        If the sentinel data file is missing or appears to be a Git LFS
        pointer stub, with instructions on how to download the data.
    """
    if eos_dir is None:
        eos_dir = os.path.dirname(os.path.abspath(__file__))

    sentinel = os.path.join(eos_dir, _SENTINEL_REL)

    if os.path.isfile(sentinel) and os.path.getsize(sentinel) > _MIN_BYTES:
        return  # data is present

    msg = (
        "\n"
        "============================================================\n"
        "  EOS data files not found\n"
        "============================================================\n"
        "\n"
        "The EOS data tables are missing or are unresolved Git LFS\n"
        "pointer stubs.  ORCHARD cannot run without them.\n"
        "\n"
        "To download the EOS data from Zenodo (~27 GB), run:\n"
        "\n"
        "    python setup_eos.py\n"
        "\n"
        "from the ORCHARD root directory.\n"
        "\n"
        "Alternatively, if you have Git LFS installed:\n"
        "\n"
        "    git lfs install && git lfs pull\n"
        "\n"
        "See README.md for full setup instructions.\n"
        "============================================================\n"
    )
    raise RuntimeError(msg)


def verify_endmember_data(zm, za, zr_values, hhe='cd', eos_dir=None):
    """
    Check the end-member P-T tables an ice/rock composition is blended from.

    Every composition cache is built from, and validated against, the
    end-members with non-zero weight at (``zm``, ``za``, each Z_r in
    ``zr_values``), so they must be present even when the composition's
    tables are already cached.

    Raises
    ------
    RuntimeError
        Naming each missing end-member file or Git LFS pointer stub.
    """
    if eos_dir is None:
        eos_dir = os.path.dirname(os.path.abspath(__file__))
    from eos import endmembers as _EM
    needed = set()
    for zr in zr_values:
        for name, w in _EM.weights(*_EM.canonical(zm, za, zr)).items():
            if w > 0.0:
                needed.add(name)
    bad = []
    for name in sorted(needed):
        path = _EM.endmember_path(name, hhe)
        if not (os.path.isfile(path) and os.path.getsize(path) > _MIN_BYTES):
            bad.append(path)
    if bad:
        raise RuntimeError(
            "\nEOS end-member tables missing (or Git LFS pointer stubs):\n  "
            + "\n  ".join(bad)
            + "\nThe v2.0 ice mixture (z_eos = ice_mixture, eos_version = 2.0)"
            " blends every composition from these files.  See "
            "eos/%s/endmembers/README.md.\n" % hhe)
