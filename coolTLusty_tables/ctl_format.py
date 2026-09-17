"""CoolTLusty / SCvH ``ptab``-``stab`` table format: reader, writer, grid, lookup.

The format is defined by the Fortran that reads it, ``eos/scvh/eos/sc_eos.f``
(``SETTABL`` lines 140-170, ``LOOK`` lines 178-245):

* Header, list-directed ``READ(8,*) YHEA,INDEX,R1,R2,T1,T2,T12,T22``:
  helium mass fraction, number of density rows, log10 rho bounds, and the
  log10 T bounds at rho_min (T1, T2) and at rho_max (T12, T22).
  ``SETTABL`` reads the S file and then the P file into the *same*
  ``COMMON/TABLE/``, so the P header silently overwrites the S header --
  the two files must carry numerically identical headers.
* Body, ``FORMAT(10F8.5)`` into ``SL(330,100)``: exactly 100 temperature
  columns per density row, at most 330 rows, density-major, ten 8-character
  fields per line with no separators.  Negative values and values >= 10 fill
  the field completely (``-5.74316-5.72538``), so the only safe writer is
  ``''.join(f'{v:8.5f}')`` over values in [-9.99999, 99.99999].
* Grid, from ``LOOK``: row i sits at ``R1 + i*(R2-R1)/N`` and column j at
  ``alpha_i + j*beta_i/100``.  The upper edges R2 and T2/T22 are therefore
  *not* grid nodes.  Building the grid with ``np.linspace`` (as the old
  notebook did) misplaces every node by up to 0.03 dex.

Entropy convention: ``LOOK`` returns ``10**stab``, and ``STATO`` multiplies it
by ``RCON = 8.31434e7``.  Brianna's tables are per m_H rather than per amu; see
``S_UNIT_MH`` / ``S_UNIT_AMU`` below and the README.

This module deliberately imports nothing from orchard, so it stays usable as a
standalone reader for anyone who is handed the tables.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

N_T = 100          # SL(330,100): temperature columns, fixed by FORMAT(10F8.5)
MAX_ROWS = 330     # SL(330,100): density rows
PER_LINE = 10      # values per body line
FIELD = 8          # characters per F8.5 field

# F8.5 with two characters before the decimal point.
F8_MIN, F8_MAX = -9.99999, 99.99999

# Entropy units.  stab = log10(S_cgs / S_UNIT).
K_B = 1.380649e-16
AMU = 1.66053906660e-24
M_H = 1.00782503 * AMU
S_UNIT_MH = K_B / M_H        # Brianna's SCvH tables (measured, rms 3e-5 dex)
S_UNIT_AMU = K_B / AMU       # literal RCON = N_A k_B = 8.31434e7
S_UNITS = {'mH': S_UNIT_MH, 'amu': S_UNIT_AMU}


# ----------------------------------------------------------------------
# header
# ----------------------------------------------------------------------
class Header:
    """The eight header numbers, plus the exact line they were written as."""

    __slots__ = ('y', 'n_rho', 'r1', 'r2', 't1', 't2', 't12', 't22', 'line')

    def __init__(self, y, n_rho, r1, r2, t1, t2, t12, t22, line=None):
        self.y = float(y)
        self.n_rho = int(n_rho)
        self.r1, self.r2 = float(r1), float(r2)
        self.t1, self.t2 = float(t1), float(t2)
        self.t12, self.t22 = float(t12), float(t22)
        self.line = line if line is not None else self.format()

    @property
    def tokens(self):
        return (self.y, self.n_rho, self.r1, self.r2,
                self.t1, self.t2, self.t12, self.t22)

    def format(self, fmt=None):
        """One list-directed line, in the spacing Brianna's ptab.dat uses.

        Values are written with ``repr``, which round-trips a float exactly, so
        re-parsing the written header reproduces the grid bit for bit.
        """
        fmt = fmt or (lambda x: repr(float(x)))
        v = [fmt(x) for x in
             (self.y, self.r1, self.r2, self.t1, self.t2, self.t12, self.t22)]
        return (f'  {v[0]}   {self.n_rho:d}  {v[1]}  {v[2]}  '
                f'{v[3]} {v[4]}  {v[5]}  {v[6]}\n')

    def matches(self, other, tol=0.0):
        """Numeric equality, which is what SETTABL's shared COMMON needs."""
        return all(abs(a - b) <= tol for a, b in zip(self.tokens, other.tokens))

    def validate(self):
        if not 0.0 <= self.y <= 1.0:
            raise ValueError(f'header Y out of range: {self.y}')
        if not 4 <= self.n_rho <= MAX_ROWS:
            raise ValueError(f'header N must be in [4, {MAX_ROWS}], got {self.n_rho}')
        if not self.r1 < self.r2:
            raise ValueError(f'header needs R1 < R2, got {self.r1}, {self.r2}')
        if not (self.t1 < self.t2 and self.t12 < self.t22):
            raise ValueError('header needs T1 < T2 and T12 < T22')
        # beta = (t2-t1) + ((t22-t12)-(t2-t1))*f must stay positive on [0, 1]
        if min(self.t2 - self.t1, self.t22 - self.t12) <= 0:
            raise ValueError('header implies a non-positive temperature span')
        return self

    def __repr__(self):
        return (f'Header(Y={self.y:g}, N={self.n_rho}, logrho=[{self.r1:g}, {self.r2:g}], '
                f'logT@rhomin=[{self.t1:g}, {self.t2:g}], '
                f'logT@rhomax=[{self.t12:g}, {self.t22:g}])')


def parse_header(text_or_tokens):
    """Parse the first eight numeric tokens, as Fortran list-directed input does.

    Accepts a one-line or multi-line header, with or without leading '#'.
    """
    if isinstance(text_or_tokens, (list, tuple)):
        toks = [str(t) for t in text_or_tokens][:8]
        line = None
    else:
        toks, line, lines = [], None, text_or_tokens.splitlines(keepends=True)
        for raw in lines:
            if line is None:
                line = raw
            else:
                line += raw
            stripped = raw.lstrip().lstrip('#')
            toks.extend(stripped.split())
            if len(toks) >= 8:
                break
        toks = toks[:8]
    if len(toks) < 8:
        raise ValueError(f'header has {len(toks)} tokens, need 8')
    return Header(*toks, line=line).validate()


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
def rhomboid_grid(header):
    """Node coordinates, exactly as ``LOOK`` inverts them (sc_eos.f:194-205).

    Returns ``(logrho[N], logT[N, 100])``.  Note the upper edges R2, T2 and T22
    are *not* nodes: row N-1 is at R1 + (N-1)(R2-R1)/N.
    """
    h = header
    i = np.arange(h.n_rho, dtype=float)
    frac = i / h.n_rho                                  # DELTA/INDEX at a node
    logrho = h.r1 + frac * (h.r2 - h.r1)
    alpha = h.t1 + frac * (h.t12 - h.t1)
    beta = (h.t2 - h.t1) + ((h.t22 - h.t12) - (h.t2 - h.t1)) * frac
    j = np.arange(N_T, dtype=float)
    logt = alpha[:, None] + j[None, :] * beta[:, None] / N_T
    return logrho, logt


def effective_domain(header):
    """The (logrho, logT) box LOOK will actually interpolate in.

    ``LOOK`` rejects JR < 2, JR > N-1, JQ < 2 and JQ > 99, so the usable box is
    narrower than the header advertises: one row in from each end and one
    column in from the bottom, two from the top.
    """
    h = header
    drho = (h.r2 - h.r1) / h.n_rho
    lo_rho, hi_rho = h.r1 + drho, h.r1 + (h.n_rho - 1) * drho
    logrho, logt = rhomboid_grid(h)
    return {
        'logrho': (lo_rho, hi_rho),
        'logT_at_logrho_lo': (logt[1, 1], logt[1, 99]),
        'logT_at_logrho_hi': (logt[h.n_rho - 2, 1], logt[h.n_rho - 2, 99]),
    }


# ----------------------------------------------------------------------
# read / write
# ----------------------------------------------------------------------
def read_table(path):
    """Read a ptab/stab file.  Returns ``(Header, arr[N, 100])`` of log10 values."""
    text = Path(path).read_text()
    header = parse_header(text)
    body = text[len(header.line):]
    lines = body.splitlines()
    want = header.n_rho * (N_T // PER_LINE)
    if len(lines) != want:
        raise ValueError(f'{path}: expected {want} body lines, found {len(lines)}')
    vals = np.empty(header.n_rho * N_T, dtype=float)
    k = 0
    for ln, line in enumerate(lines):
        if len(line) != PER_LINE * FIELD:
            raise ValueError(f'{path}: body line {ln + 1} is {len(line)} chars, '
                             f'expected {PER_LINE * FIELD}')
        for c in range(0, PER_LINE * FIELD, FIELD):
            vals[k] = float(line[c:c + FIELD])
            k += 1
    return header, vals.reshape(header.n_rho, N_T)


def format_table(header, arr):
    """Render a table to the exact bytes ``SETTABL`` expects."""
    a = np.asarray(arr, dtype=float)
    if a.shape != (header.n_rho, N_T):
        raise ValueError(f'array shape {a.shape} does not match header '
                         f'({header.n_rho}, {N_T})')
    if not np.isfinite(a).all():
        bad = np.argwhere(~np.isfinite(a))
        raise ValueError(f'{len(bad)} non-finite value(s), first at '
                         f'row {bad[0][0]}, col {bad[0][1]}')
    lo, hi = a.min(), a.max()
    if lo < F8_MIN or hi > F8_MAX:
        raise ValueError(f'values [{lo:g}, {hi:g}] fall outside the F8.5 range '
                         f'[{F8_MIN}, {F8_MAX}]; the Fortran read would be corrupted')
    flat = a.ravel()
    out = [header.line]
    for start in range(0, flat.size, PER_LINE):
        chunk = flat[start:start + PER_LINE]
        line = ''.join(f'{v:{FIELD}.5f}' for v in chunk)
        if len(line) != len(chunk) * FIELD:
            raise ValueError(f'field width violation in line starting at {start}: {line!r}')
        out.append(line + '\n')
    return ''.join(out)


def write_table(path, header, arr):
    """Atomically write a ptab/stab file (temp file, then rename)."""
    text = format_table(header, arr)
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', newline='\n') as fh:
            fh.write(text)
        # mkstemp creates 0600; these tables get shared, so use the usual 0644
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# ----------------------------------------------------------------------
# lookup (port of LOOK)
# ----------------------------------------------------------------------
def look(header, arr, logrho, logt):
    """Vectorized port of ``LOOK`` (sc_eos.f:193-229), in log10 space.

    Returns the six-point bivariate interpolant (Abramowitz & Stegun p. 882)
    of ``arr`` at the requested points, and NaN wherever ``LOOK`` would report
    "off the table".  The Fortran then exponentiates; we stay in log10.
    """
    h = header
    lr = np.asarray(logrho, dtype=float)
    lt = np.asarray(logt, dtype=float)
    lr, lt = np.broadcast_arrays(lr, lt)

    f = (lr - h.r1) / (h.r2 - h.r1)
    alpha = h.t1 + f * (h.t12 - h.t1)
    beta = (h.t2 - h.t1) + ((h.t22 - h.t12) - (h.t2 - h.t1)) * f
    with np.errstate(divide='ignore', invalid='ignore'):
        ql = (lt - alpha) / beta
    delta = f * h.n_rho

    # IDINT truncates toward zero; np.trunc matches it (and differs from floor
    # for the negative values that occur just off the low edge).
    jr = 1 + np.trunc(delta)          # 1-based Fortran row index
    jq = 1 + np.trunc(100.0 * ql)     # 1-based Fortran column index
    ok = (jr >= 2) & (jr <= h.n_rho - 1) & (jq >= 2) & (jq <= 99)
    ok &= np.isfinite(delta) & np.isfinite(ql)

    jr_s = np.where(ok, jr, 2).astype(int)
    jq_s = np.where(ok, jq, 2).astype(int)
    p = delta - (jr_s - 1)
    q = 100.0 * ql - (jq_s - 1)
    p = np.where(ok, p, 0.0)
    q = np.where(ok, q, 0.0)

    i, j = jr_s - 1, jq_s - 1         # 0-based numpy indices
    out = (0.5 * q * (q - 1.0) * arr[i, j - 1]
           + 0.5 * p * (p - 1.0) * arr[i - 1, j]
           + (1.0 + p * q - p * p - q * q) * arr[i, j]
           + 0.5 * p * (p - 2.0 * q + 1.0) * arr[i + 1, j]
           + 0.5 * q * (q - 2.0 * p + 1.0) * arr[i, j + 1]
           + p * q * arr[i + 1, j + 1])
    return np.where(ok, out, np.nan)
