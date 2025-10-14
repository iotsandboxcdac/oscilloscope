from typing import Dict, List
import numpy as np


def compute_waveform_metrics_single(sig: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    Robust measurement for a single-channel waveform.
    - Amplitude, pkpk, RMS are computed on the raw signal (no detrend).
    - Frequency: estimated from rising-edge mid-level crossings (interpolated).
    - Rise/Fall times: measured as median 10%-90% (and 90%-10%) times across transitions,
      using linear interpolation for sub-sample resolution.
    """
    metrics = {
        "frequency_hz": float('nan'),
        "amplitude_v": float('nan'),
        "rise_time_s": float('nan'),
        "fall_time_s": float('nan'),
        "pkpk_v": float('nan'),
        "rms_v": float('nan')
    }

    try:
        if sig is None:
            return metrics
        raw = np.asarray(sig, dtype=float)
        n = raw.size
        if n < 3 or sample_rate <= 0:
            return metrics

        # Basic amplitude statistics (on raw signal)
        vmin = float(np.min(raw))
        vmax = float(np.max(raw))
        pkpk = float(vmax - vmin)
        amplitude = float(vmax)  # user expects amplitude = peak value (not half pk-pk)
        rms = float(np.sqrt(np.mean(raw ** 2)))  # true RMS including DC

        # Frequency: use mid-level rising-edge crossings for robust estimate
        mid = vmin + 0.5 * pkpk
        crossings = []
        for i in range(n - 1):
            if raw[i] < mid and raw[i + 1] >= mid:
                # linear interpolation for fractional index
                denom = (raw[i + 1] - raw[i])
                frac = 0.0 if denom == 0 else (mid - raw[i]) / denom
                crossings.append(i + frac)

        freq_est = float('nan')
        if len(crossings) >= 2:
            diffs = np.diff(np.asarray(crossings, dtype=float))
            # reject outliers by comparing to median
            med = float(np.median(diffs))
            std = float(np.std(diffs))
            if std == 0:
                diffs_ok = diffs
            else:
                diffs_ok = diffs[np.abs(diffs - med) <= max(3 * std, 1e-12)]
                if diffs_ok.size == 0:
                    diffs_ok = diffs
            mean_period_samples = float(np.mean(diffs_ok))
            if mean_period_samples > 0:
                freq_est = 1.0 / (mean_period_samples / float(sample_rate))

        # Rise/fall times using 10% and 90% thresholds (based on raw vmin/vmax)
        low_level = vmin + 0.1 * pkpk
        high_level = vmin + 0.9 * pkpk

        def interp_cross_index(idx0: int, level: float) -> float:
            """Return fractional index where signal crosses 'level' between idx0 and idx0+1 (linear interp)."""
            if idx0 >= n - 1:
                return float(idx0)
            y0 = raw[idx0]; y1 = raw[idx0 + 1]
            denom = (y1 - y0)
            if denom == 0:
                return float(idx0)
            return float(idx0) + (level - y0) / denom

        # Collect rise times for each rising transition
        rise_times = []
        for cr in crossings:
            i_cross = int(np.floor(cr))
            # find previous index where value <= low_level
            start_idx = None
            for j in range(i_cross, -1, -1):
                if raw[j] <= low_level:
                    start_idx = j
                    break
            # find next index where value >= high_level
            end_idx = None
            for j in range(i_cross, n):
                if raw[j] >= high_level:
                    end_idx = j
                    break
            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                t_low = interp_cross_index(start_idx, low_level)
                # for the high crossing the crossing may happen between end_idx-1 and end_idx
                t_high = interp_cross_index(max(end_idx - 1, 0), high_level)
                rise_times.append((t_high - t_low) / float(sample_rate))

        # Falling transitions: mid-level falling crossings
        falling = []
        for i in range(n - 1):
            if raw[i] > mid and raw[i + 1] <= mid:
                denom = (raw[i + 1] - raw[i])
                frac = 0.0 if denom == 0 else (mid - raw[i]) / denom
                falling.append(i + frac)

        fall_times = []
        for cr in falling:
            i_cross = int(np.floor(cr))
            start_idx = None
            for j in range(i_cross, -1, -1):
                if raw[j] >= high_level:
                    start_idx = j
                    break
            end_idx = None
            for j in range(i_cross, n):
                if raw[j] <= low_level:
                    end_idx = j
                    break
            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                t_high = interp_cross_index(start_idx, high_level)
                t_low = interp_cross_index(max(end_idx - 1, 0), low_level)
                fall_times.append((t_low - t_high) / float(sample_rate))

        rise_time = float(np.nan) if len(rise_times) == 0 else float(np.median(rise_times))
        fall_time = float(np.nan) if len(fall_times) == 0 else float(np.median(fall_times))

        metrics["frequency_hz"] = freq_est
        metrics["amplitude_v"] = amplitude
        metrics["rise_time_s"] = rise_time
        metrics["fall_time_s"] = fall_time
        metrics["pkpk_v"] = pkpk
        metrics["rms_v"] = rms

    except Exception:
        # keep defensive — return NaNs on failure
        pass

    return metrics


def compute_metrics_per_channel(waveform: np.ndarray, sample_rate: float) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    if waveform is None:
        return results
    arr = np.asarray(waveform)
    if arr.ndim == 1:
        results.append(compute_waveform_metrics_single(arr, sample_rate))
    elif arr.ndim == 2:
        for ch in range(arr.shape[0]):
            results.append(compute_waveform_metrics_single(arr[ch], sample_rate))
    else:
        results.append(compute_waveform_metrics_single(arr.ravel(), sample_rate))
    return results
