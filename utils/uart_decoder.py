from typing import List, Tuple, Union, Optional
import numpy as np

def decode_uart_from_analog(
    sig: np.ndarray,
    sample_rate: float,
    threshold_v: float,
    baud: int,
    data_bits: int = 8,
    stop_bits: int = 1,
    max_bytes: int = 4096,
    hysteresis_percent: float = 0.05,
    sample_window_fraction: float = 0.35,
    min_stop_high_fraction: float = 0.6,
    parity: str = "none",              # allowed: "none", "even", "odd"
    require_parity_ok: bool = True,
    return_parity: bool = False,
    return_frames: bool = False,
    low_min: float = -0.3,
    low_max: float = 0.8,
    high_min: float = 2.0,
    high_max: float = 4.0,
    require_valid_levels: bool = True
) -> Union[
    Tuple[List[int], List[float], List[int]],
    Tuple[List[int], List[float], List[int], List[bool]],
    Tuple[List[int], List[float], List[int], List[dict]],
    Tuple[List[int], List[float], List[int], List[bool], List[dict]]
]:
    """
    Robust UART decoder with optional per-frame metadata for plotting.

    If return_frames=True, an extra return value `frames` is appended (a list of dicts).
    If return_parity=True, parity flags list is included (as before).

    Frame dict keys:
      - 'byte' : int
      - 'start_idx', 'end_idx' : sample indices
      - 'start_time', 'end_time' : seconds
      - 'bit_indices' : list[int]
      - 'bit_times' : list[float]
      - 'bit_levels' : list[int]
      - 'parity_ok' : Optional[bool] (None when parity=='none')
      - 'framing_error' : bool
      - 'level_error' : bool
    """
    decoded: List[int] = []
    sample_times: List[float] = []
    bit_levels: List[int] = []
    parity_flags: List[bool] = []
    frames: List[dict] = []

    try:
        arr = np.asarray(sig, dtype=float)
        n = arr.size
        if n == 0 or sample_rate <= 0 or baud <= 0:
            # return compatible empty results
            if return_parity and return_frames:
                return decoded, sample_times, bit_levels, parity_flags, frames
            if return_parity:
                return decoded, sample_times, bit_levels, parity_flags
            if return_frames:
                return decoded, sample_times, bit_levels, frames
            return decoded, sample_times, bit_levels

        samples_per_bit = float(sample_rate) / float(baud)
        # if extremely low samples/bit, decoding will be unreliable but we still proceed
        if samples_per_bit < 3:
            # still attempt decode but results may be unreliable
            pass

        # Estimate amplitude and compute hysteresis voltages
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        amp = max(1e-12, (vmax - vmin) / 2.0)
        hyst_v = abs(hysteresis_percent) * amp

        # Normalize parity choice: only none/even/odd supported
        parity_norm = (parity or "none").strip().lower()
        if parity_norm not in ("none", "even", "odd"):
            parity_norm = "none"

        high_th = threshold_v + hyst_v
        low_th = threshold_v - hyst_v

        # Schmitt trigger digitalization (stateful)
        digital = np.zeros(n, dtype=np.int8)
        state = 1 if arr[0] >= threshold_v else 0
        for i in range(n):
            v = arr[i]
            if state == 1:
                if v < low_th:
                    state = 0
            else:
                if v > high_th:
                    state = 1
            digital[i] = 1 if state == 1 else 0

        # Find start edges (1 -> 0) - falling edges indicate start bit
        falling = np.where((digital[:-1] == 1) & (digital[1:] == 0))[0] + 1
        if falling.size == 0:
            if return_parity and return_frames:
                return decoded, sample_times, bit_levels, parity_flags, frames
            if return_parity:
                return decoded, sample_times, bit_levels, parity_flags
            if return_frames:
                return decoded, sample_times, bit_levels, frames
            return decoded, sample_times, bit_levels

        win_radius = max(1, int(round(samples_per_bit * float(sample_window_fraction) / 2.0)))
        total_extra_bits = 0 if parity_norm == "none" else 1
        total_bits = 1 + int(data_bits) + int(stop_bits) + total_extra_bits
        frame_samples = int(round(total_bits * samples_per_bit))

        def window_vote(center_idx: int) -> int:
            """Return 1 if majority of samples in window are >= threshold_v, else 0."""
            a = max(0, center_idx - win_radius)
            b = min(n - 1, center_idx + win_radius)
            if a > b:
                a = b = center_idx
            seg = arr[a:b+1]
            count_high = np.count_nonzero(seg >= threshold_v)
            return 1 if count_high >= (seg.size / 2.0) else 0

        def window_fraction_high(center_idx: int) -> float:
            a = max(0, center_idx - win_radius)
            b = min(n - 1, center_idx + win_radius)
            seg = arr[a:b+1]
            return float(np.count_nonzero(seg >= threshold_v)) / float(seg.size)

        def get_window_min_max(center_idx: int) -> Tuple[float, float]:
            a = max(0, center_idx - win_radius)
            b = min(n - 1, center_idx + win_radius)
            seg = arr[a:b+1]
            return float(np.min(seg)), float(np.max(seg))

        next_search_idx = -1
        for idx in falling:
            if len(decoded) >= max_bytes:
                break
            if idx < next_search_idx:
                continue
            if (idx + frame_samples) >= n:
                # incomplete frame at end, skip
                continue

            # Start bit check
            start_center_f = idx + 0.5 * samples_per_bit
            start_idx = int(round(start_center_f))
            if start_idx < 0 or start_idx >= n:
                continue
            start_bit = window_vote(start_idx)
            if start_bit != 0:
                continue  # invalid start bit

            level_error = False

            # Check levels for start bit
            seg_min, seg_max = get_window_min_max(start_idx)
            if start_bit == 1:
                if seg_min < high_min or seg_max > high_max:
                    level_error = True
            else:
                if seg_min < low_min or seg_max > low_max:
                    level_error = True

            # First data center
            first_data_center = idx + 1.5 * samples_per_bit

            byte_val = 0
            local_times: List[float] = []
            local_bits: List[int] = []
            local_indices: List[int] = []
            valid = True

            # Sample data bits (LSB first) using window vote
            for k in range(data_bits):
                samp_f = first_data_center + k * samples_per_bit
                samp_idx = int(round(samp_f))
                if samp_idx < 0 or samp_idx >= n:
                    valid = False
                    break
                bit = window_vote(samp_idx)
                # Check levels for data bit
                seg_min, seg_max = get_window_min_max(samp_idx)
                if bit == 1:
                    if seg_min < high_min or seg_max > high_max:
                        level_error = True
                else:
                    if seg_min < low_min or seg_max > low_max:
                        level_error = True
                local_times.append(samp_idx / sample_rate)
                local_bits.append(int(bit))
                local_indices.append(samp_idx)
                byte_val |= (int(bit) & 0x1) << k

            if not valid:
                continue

            # Parity (if present)
            parity_ok: Optional[bool] = None
            if parity_norm != "none":
                parity_center_f = first_data_center + data_bits * samples_per_bit
                parity_idx = int(round(parity_center_f))
                if parity_idx < 0 or parity_idx >= n:
                    parity_ok = False
                else:
                    observed_p = window_vote(parity_idx)
                    # Check levels for parity bit
                    seg_min, seg_max = get_window_min_max(parity_idx)
                    if observed_p == 1:
                        if seg_min < high_min or seg_max > high_max:
                            level_error = True
                    else:
                        if seg_min < low_min or seg_max > low_max:
                            level_error = True
                    ones_count = sum(local_bits) % 2
                    if parity_norm == "even":
                        expected_p = ones_count
                    else:  # 'odd'
                        expected_p = 1 - ones_count
                    parity_ok = (observed_p == expected_p)
            if parity_ok is False and require_parity_ok:
                # preserve previous behavior (reject frame) when parity is required OK
                next_search_idx = idx + int(round(0.5 * samples_per_bit))
                continue

            # Stop-bit validation: compute fraction high for each stop bit.
            framing_error = False
            for s in range(stop_bits):
                stop_center_f = first_data_center + (data_bits + (1 if parity_norm != "none" else 0) + s) * samples_per_bit
                stop_idx = int(round(stop_center_f))
                if stop_idx < 0 or stop_idx >= n:
                    framing_error = True
                    break
                frac_high = window_fraction_high(stop_idx)
                if frac_high < float(min_stop_high_fraction):
                    framing_error = True
                # Check levels for stop bit
                stop_bit = window_vote(stop_idx)
                seg_min, seg_max = get_window_min_max(stop_idx)
                if stop_bit == 1:
                    if seg_min < high_min or seg_max > high_max:
                        level_error = True
                else:
                    if seg_min < low_min or seg_max > low_max:
                        level_error = True

            if level_error and require_valid_levels:
                next_search_idx = idx + int(round(0.5 * samples_per_bit))
                continue

            # Accept byte (even when framing_error=True or level_error=True)
            decoded.append(int(byte_val & 0xFF))
            sample_times.extend(local_times)
            bit_levels.extend(local_bits)
            parity_flags.append(parity_ok if parity_ok is not None else True)

            # Build frame metadata
            start_idx_frame = idx
            end_idx = int(round(first_data_center + (data_bits + (1 if parity_norm != "none" else 0) + stop_bits - 1) * samples_per_bit)) + win_radius
            start_idx_frame = max(0, start_idx_frame)
            end_idx = min(n - 1, end_idx)
            frame = {
                "byte": int(byte_val & 0xFF),
                "start_idx": int(start_idx_frame),
                "end_idx": int(end_idx),
                "start_time": float(start_idx_frame) / sample_rate,
                "end_time": float(end_idx) / sample_rate,
                "bit_indices": list(local_indices),
                "bit_times": list(local_times),
                "bit_levels": list(local_bits),
                "parity_ok": parity_ok,
                "framing_error": bool(framing_error),
                "level_error": bool(level_error),
            }
            frames.append(frame)

            # advance search index beyond this frame to avoid duplicates
            next_search_idx = idx + frame_samples

    except Exception:
        # keep robust: return what was decoded so far
        pass

    if return_parity and return_frames:
        return decoded, sample_times, bit_levels, parity_flags, frames
    if return_parity:
        return decoded, sample_times, bit_levels, parity_flags
    if return_frames:
        return decoded, sample_times, bit_levels, frames
    return decoded, sample_times, bit_levels