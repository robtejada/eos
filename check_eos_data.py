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
        "To download the EOS data from Zenodo (~9.9 GB), run:\n"
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
