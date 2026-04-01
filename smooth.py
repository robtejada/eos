import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def hampel_filter_1d(arr, window=5, n_sigma=3.0):
    """Hampel filter: replace outliers with the local median.

    Scans a 1-D array and replaces points that deviate from the
    local median by more than n_sigma × MAD (median absolute
    deviation).  Uses linear interpolation from valid neighbors
    instead of the median, to preserve gradients.

    Parameters
    ----------
    arr : 1-D array
    window : int
        Half-width of the sliding window (total width = 2*window+1).
    n_sigma : float
        Threshold in units of MAD.

    Returns
    -------
    cleaned : 1-D array (copy)
    n_replaced : int, number of points replaced
    """
    arr = np.asarray(arr, dtype=float)
    cleaned = arr.copy()
    n = len(arr)
    flagged = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = arr[lo:hi]
        med = np.nanmedian(local)
        mad = 1.4826 * np.nanmedian(np.abs(local - med))
        if mad < 1e-30:
            continue
        if np.abs(arr[i] - med) > n_sigma * mad:
            flagged[i] = True

    # Interpolate flagged points from valid neighbors
    if flagged.any():
        good = ~flagged & np.isfinite(arr)
        if good.sum() >= 2:
            cleaned[flagged] = np.interp(
                np.where(flagged)[0],
                np.where(good)[0],
                arr[good])

    return cleaned, int(flagged.sum())


def hampel_filter_2d(grid, axis, window=5, n_sigma=3.0):
    """Apply Hampel filter along one axis of a 2-D grid.

    Parameters
    ----------
    grid : 2-D array, shape (nT, nP) or (nP, nT)
    axis : int, 0 or 1
        0 = filter along each column (varying T at fixed P)
        1 = filter along each row (varying P at fixed T)
    window, n_sigma : Hampel parameters

    Returns
    -------
    cleaned : 2-D array (copy)
    total_replaced : int
    """
    cleaned = grid.copy()
    total = 0
    if axis == 0:
        for j in range(grid.shape[1]):
            cleaned[:, j], n = hampel_filter_1d(grid[:, j], window, n_sigma)
            total += n
    else:
        for i in range(grid.shape[0]):
            cleaned[i, :], n = hampel_filter_1d(grid[i, :], window, n_sigma)
            total += n
    return cleaned, total


def get_polyfit(arr, deg=10):

# Identify discontinuity
    differences = np.diff(arr)
    threshold = 0.1  # Define a threshold to identify significant jumps
    discontinuity_index = np.where(np.abs(differences) > threshold)[0]

    # Compute polynomial fit excluding discontinuous region
    if discontinuity_index.size > 0:
        discontinuity_index = discontinuity_index[0]  # Take the first discontinuity index
        x = np.arange(len(arr))
        x_fit = np.delete(x, np.arange(discontinuity_index, discontinuity_index + 2))
        y_fit = np.delete(arr, np.arange(discontinuity_index, discontinuity_index + 2))
    else:
        x = np.arange(len(arr))
        x_fit = x
        y_fit = arr

    # Fit a polynomial
    poly_coeff = np.polyfit(x_fit, y_fit, deg)
    poly_fit = np.poly1d(poly_coeff)
    
    return poly_fit(x)

def joint_fit(arr, logrhoarr, deg=5, logrho_thresh1=-3, logrho_thresh2=0.0):
    
    # fit3 = get_polyfit(arr[logrhoarr > logrho_thresh2], deg=3)
    fit2 = get_polyfit(arr[(logrhoarr >= logrho_thresh1) & (logrhoarr <= logrho_thresh2)], deg=5)
    # fit1 = get_polyfit(arr[logrhoarr < logrho_thresh1], deg=deg)

    # return np.concatenate([fit1, fit2, fit3])

    #fit2 = get_polyfit(arr[logrhoarr >= logrho_thresh], deg=deg)
    fit1 = get_polyfit(arr[logrhoarr < logrho_thresh1], deg=deg)

    return np.concatenate([fit1, fit2, arr[logrhoarr > logrho_thresh2]])

# def gauss_smooth(data, base_sigma=5, base_window=10):
#     smoothed_data = np.zeros_like(data)
#     differences = np.abs(np.diff(data, prepend=data[0]))  # prepend to match the length
#     median_difference = np.median(differences)
    
#     for i in range(len(data)):
#         # Adjust sigma based on local difference information
#         local_sigma = base_sigma * (1 + (median_difference - differences[i]) / (median_difference + 1e-6))
        
#         # Determine the window size dynamically based on the position within the array
#         start_index = max(0, i - base_window // 2)
#         end_index = min(len(data), i + base_window // 2 + 1)
#         actual_window_size = end_index - start_index
        
#         # Generate the Gaussian kernel for the actual window size
#         x = np.linspace(-(actual_window_size // 2), actual_window_size // 2, actual_window_size)
#         gauss_kernel = np.exp(-0.5 * (x / local_sigma) ** 2)
#         gauss_kernel /= gauss_kernel.sum()  # Normalize the kernel
#         #print(actual_window_size)
        
#         # Apply the kernel to the data segment
#         smoothed_data[i] = np.dot(data[start_index:end_index], gauss_kernel)

#     return smoothed_data

def gauss_smooth(data, base_sigma=5, base_window=10, min_sigma=0.1, max_sigma=10):
    # Calculate differences and normalize
    differences = np.abs(np.diff(data, prepend=data[0]))
    median_difference = np.median(differences) + 1e-6  # Avoid division by zero
    normalized_diff = (median_difference - differences) / median_difference

    # Calculate local sigmas and clip to [min_sigma, max_sigma]
    local_sigmas = base_sigma * (1 + normalized_diff)
    local_sigmas = np.clip(local_sigmas, min_sigma, max_sigma)

    # Pre-compute Gaussian kernels for unique sigma values
    unique_sigmas = np.unique(local_sigmas)
    kernel_dict = {}
    for sigma in unique_sigmas:
        # Define window size based on sigma
        window_size = int(base_window * sigma / base_sigma)
        window_size = max(window_size, 3)  # Minimum window size
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        x = np.linspace(-window_size // 2, window_size // 2, window_size)
        gauss_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        gauss_kernel /= gauss_kernel.sum()
        kernel_dict[sigma] = gauss_kernel

    # Apply smoothing using the kernels
    smoothed_data = np.zeros_like(data)
    for i, (value, sigma) in enumerate(zip(data, local_sigmas)):
        kernel = kernel_dict[sigma]
        window_size = len(kernel)
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        data_segment = data[start_index:end_index]

        # Adjust kernel if we're at the edges
        if len(data_segment) != window_size:
            kernel = kernel[(window_size // 2 - (i - start_index)):(window_size // 2 + (end_index - i))]
            kernel /= kernel.sum()  # Renormalize

        smoothed_data[i] = np.dot(data_segment, kernel)

    return smoothed_data


# ===================================================================
#  HHE table smoothing routines
#  (validated via plots/eos_smooth_comparison.py)
# ===================================================================

def _find_interior_reference(row, logp_1d, ref_lo=6.0, ref_hi=16.0):
    """Return the median value and typical gradient in the reliable interior."""
    mask = (logp_1d >= ref_lo) & (logp_1d <= ref_hi)
    if mask.sum() < 4:
        return np.median(row), 0.0
    interior = row[mask]
    ref_val = np.median(interior)
    dp = np.diff(logp_1d[mask])
    dv = np.diff(interior)
    typical_grad = np.median(dv / dp)
    return ref_val, typical_grad


def detect_invalid_lowp(grid_2d, logp_1d, dev_thresh=2.0):
    """Detect low-P cells that diverge from the converged interior.

    For each isotherm, computes a reference value from the interior
    (logP 6-16) and scans from the interior toward low P.  Cells are
    flagged once the cumulative deviation from a linear extrapolation of
    the interior gradient exceeds `dev_thresh`.

    Returns a boolean mask (True = invalid).
    """
    n_t, n_p = grid_2d.shape
    invalid = np.zeros_like(grid_2d, dtype=bool)

    for ti in range(n_t):
        row = grid_2d[ti, :]
        ref_val, ref_grad = _find_interior_reference(row, logp_1d)

        int_start = np.searchsorted(logp_1d, 6.0)
        if int_start >= n_p:
            continue

        anchor_val = row[int_start]
        anchor_p = logp_1d[int_start]

        for pi in range(int_start - 1, -1, -1):
            expected = anchor_val + ref_grad * (logp_1d[pi] - anchor_p)
            deviation = abs(row[pi] - expected)
            if deviation > dev_thresh:
                invalid[ti, :pi + 1] = True
                break

    return invalid


def detect_invalid_highp(grids, logp_1d, logt_1d):
    """Detect the PPT floor region AND the post-floor corrupted zone.

    The PPT manifests as:
      - Density floor cells (logrho <= -8.5) at high P
      - Post-floor entropy/energy corruption (entropy jumps by >1 dex
        relative to pre-floor trend)

    Returns the index of the last clean cell for each isotherm (n_t,).
    A value of n_p means the full row is clean.
    """
    n_t, n_p = grids['logrho'].shape
    last_clean = np.full(n_t, n_p, dtype=int)

    for ti in range(n_t):
        rho_row = grids['logrho'][ti, :]
        s_row = grids['logs'][ti, :]

        floor_cells = np.where((rho_row <= -8.5) & (logp_1d > 15))[0]
        if len(floor_cells) > 0:
            first_floor = floor_cells[0]
            pre_floor = max(first_floor - 2, 0)

            if pre_floor >= 2:
                grad_before = s_row[pre_floor - 1] - s_row[pre_floor - 2]
                grad_at = s_row[pre_floor] - s_row[pre_floor - 1]
                if abs(grad_at - grad_before) > 0.1:
                    last_clean[ti] = pre_floor - 1
                else:
                    last_clean[ti] = first_floor - 1
            else:
                last_clean[ti] = first_floor - 1
            continue

        ref_val, ref_grad = _find_interior_reference(s_row, logp_1d)
        int_end = np.searchsorted(logp_1d, 16.0)
        anchor_val = s_row[min(int_end, n_p - 1)]
        anchor_p = logp_1d[min(int_end, n_p - 1)]

        for pi in range(int_end + 1, n_p):
            expected = anchor_val + ref_grad * (logp_1d[pi] - anchor_p)
            if abs(s_row[pi] - expected) > 2.0:
                last_clean[ti] = pi - 1
                break

    return last_clean


def extrapolate_from_boundary(grid_2d, logp_1d, invalid_mask=None,
                              last_clean_idx=None, side='left', n_anchor=5):
    """Replace invalid cells with linear extrapolation from the valid boundary.

    Parameters
    ----------
    side : 'left' for low-P, 'right' for high-P
    n_anchor : number of valid cells to fit the extrapolation slope
    """
    smoothed = grid_2d.copy()
    n_t, n_p = grid_2d.shape

    for ti in range(n_t):
        if side == 'left' and invalid_mask is not None:
            inv = invalid_mask[ti, :]
            if not inv.any():
                continue
            valid_idx = np.where(~inv)[0]
            if len(valid_idx) < n_anchor:
                continue
            anchor_slice = valid_idx[:n_anchor]
            p_anch = logp_1d[anchor_slice]
            v_anch = smoothed[ti, anchor_slice]
            slope, intercept = np.polyfit(p_anch, v_anch, 1)
            first_valid = valid_idx[0]
            for pi in range(first_valid - 1, -1, -1):
                smoothed[ti, pi] = slope * logp_1d[pi] + intercept

        elif side == 'right' and last_clean_idx is not None:
            lc = last_clean_idx[ti]
            if lc >= n_p:
                continue
            start = max(lc - n_anchor + 1, 0)
            anchor_slice = slice(start, lc + 1)
            p_anch = logp_1d[anchor_slice]
            v_anch = grid_2d[ti, anchor_slice]
            if len(p_anch) < 2:
                continue
            slope, intercept = np.polyfit(p_anch, v_anch, 1)
            for pi in range(lc + 1, n_p):
                smoothed[ti, pi] = slope * logp_1d[pi] + intercept

    return smoothed


def targeted_gaussian_smooth(grid_2d, logt_1d, logp_1d,
                             logt_range=(2.5, 3.5), logp_range=(10.0, 14.0),
                             sigma_t=2.0, sigma_p=3.0):
    """Apply 2D Gaussian smoothing in a targeted T-P region
    (the molecular dissociation / ionization zone).

    Uses a soft tanh blend so there are no sharp boundaries between
    the smoothed and original data.
    """
    smoothed_full = gaussian_filter(grid_2d.astype(float),
                                    sigma=[sigma_t, sigma_p],
                                    mode='nearest')

    logt_2d = logt_1d[:, np.newaxis]
    logp_2d = logp_1d[np.newaxis, :]

    mask_t = (0.5 * (1 + np.tanh((logt_2d - logt_range[0]) / 0.3)) *
              0.5 * (1 + np.tanh((logt_range[1] - logt_2d) / 0.3)))
    mask_p = (0.5 * (1 + np.tanh((logp_2d - logp_range[0]) / 1.0)) *
              0.5 * (1 + np.tanh((logp_range[1] - logp_2d) / 1.0)))
    mask = mask_t * mask_p

    return (1 - mask) * grid_2d + mask * smoothed_full


def smooth_lowp_2d(grid_2d, logp_1d, logp_thresh=8.0, blend_width=2.0,
                   sigma_t=2.0, sigma_p=2.0):
    """Apply 2D Gaussian smoothing in the low-P region to remove horizontal
    stripes caused by individual isotherms with anomalous values.

    The smoothing is strongest at the lowest pressures and blends smoothly
    into the original data as logP approaches logp_thresh.
    """
    smoothed_full = gaussian_filter(grid_2d.astype(float),
                                    sigma=[sigma_t, sigma_p],
                                    mode='nearest')

    logp_2d = logp_1d[np.newaxis, :]
    blend = 0.5 * (1.0 - np.tanh((logp_2d - logp_thresh) / blend_width))

    return (1.0 - blend) * grid_2d + blend * smoothed_full


def replace_outlier_isotherms(grid_2d, logt_1d, logp_1d,
                              neighbor_window=3, threshold=3.0):
    """Detect and replace entire isotherms with extreme outlier values.

    For each pressure column, computes a running median over the temperature
    axis.  Isotherms whose value deviates from the local median by more than
    `threshold` times the local MAD are replaced with linearly interpolated
    values from their valid neighbors.
    """
    result = grid_2d.copy()
    n_t, n_p = grid_2d.shape

    for pi in range(n_p):
        col = grid_2d[:, pi].copy()

        local_med = np.zeros(n_t)
        local_mad = np.zeros(n_t)
        for ti in range(n_t):
            lo = max(0, ti - neighbor_window)
            hi = min(n_t, ti + neighbor_window + 1)
            window = col[lo:hi]
            local_med[ti] = np.median(window)
            local_mad[ti] = np.median(np.abs(window - local_med[ti])) + 1e-10

        deviation = np.abs(col - local_med) / local_mad
        outlier = deviation > threshold

        if not outlier.any():
            continue

        valid = ~outlier
        if valid.sum() < 2:
            continue

        valid_idx = np.where(valid)[0]
        valid_vals = col[valid_idx]

        interp_vals = np.interp(np.arange(n_t), valid_idx, valid_vals)
        result[outlier, pi] = interp_vals[outlier]

    return result


def smooth_eos_table(grids, logt_1d, logp_1d):
    """Apply all smoothing strategies to a **per-component** EOS grid.

    This operates on a single component (e.g., one H-He table or one
    AQUA Z table) in the (T, P) plane.  It is called *before* the
    Volume Addition Law combines the components, so it only sees
    one species' raw table.  The goal is to remove table-construction
    artifacts (outlier isotherms, low-P extrapolation noise, PPT
    discontinuities, ionization/dissociation kinks) that would
    otherwise propagate into the combined EOS and its inversions.

    Note: This does NOT smooth the S bounds used by the xi mapping
    in ``hhe_z_mixtures``.  Those bounds are smoothed separately in
    ``compute_s_bounds_grid`` after evaluating the combined forward
    model, because the AQUA phase-transition entropy spikes that
    affect S_hi(P) only manifest in the VAL-combined entropy, not
    in the per-component tables.

    Parameters
    ----------
    grids : dict with keys 'logrho', 'logs', 'logu' (2D arrays)
    logt_1d, logp_1d : 1-D coordinate arrays

    Returns
    -------
    smoothed : dict with same keys, smoothed 2D arrays
    """
    smoothed = {}
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key] = grids[key].copy()

    # Step 0: Detect and replace outlier isotherms
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key] = replace_outlier_isotherms(
            smoothed[key], logt_1d, logp_1d,
            neighbor_window=3, threshold=3.0,
        )

    # Step 1: Low-P validity masking + extrapolation
    for key in ['logrho', 'logs', 'logu']:
        invalid_lowp = detect_invalid_lowp(smoothed[key], logp_1d, dev_thresh=1.0)
        smoothed[key] = extrapolate_from_boundary(
            smoothed[key], logp_1d, invalid_mask=invalid_lowp, side='left', n_anchor=5
        )

    # Step 2: Hampel filter along isotherms and isobars.
    # Removes isolated sharp glitches FIRST, so that the subsequent
    # Gaussian smoothing doesn't amplify outliers into broad bumps
    # that shift the adiabat.
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key], _ = hampel_filter_2d(
            smoothed[key], axis=1, window=5, n_sigma=3.0)
        smoothed[key], _ = hampel_filter_2d(
            smoothed[key], axis=0, window=5, n_sigma=3.0)

    # Step 3: 2D Gaussian smoothing in the low-P region.
    # Applied AFTER the Hampel filter, which already removed
    # sharp outliers that would otherwise get smeared into
    # broad bumps by the Gaussian.
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key] = smooth_lowp_2d(
            smoothed[key], logp_1d,
            logp_thresh=10.0, blend_width=2.0,
            sigma_t=3.0, sigma_p=3.0,
        )

    # Step 4: PPT floor + post-floor repair (high-P only)
    last_clean = detect_invalid_highp(grids, logp_1d, logt_1d)
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key] = extrapolate_from_boundary(
            smoothed[key], logp_1d, last_clean_idx=last_clean, side='right', n_anchor=10
        )

    # Step 5: Targeted 2D Gaussian in dissociation/ionization zone.
    for key in ['logrho', 'logs', 'logu']:
        smoothed[key] = targeted_gaussian_smooth(
            smoothed[key], logt_1d, logp_1d,
            logt_range=(2.0, 3.5), logp_range=(6.0, 16.0),
            sigma_t=5.0, sigma_p=3.0,
        )

    return smoothed
