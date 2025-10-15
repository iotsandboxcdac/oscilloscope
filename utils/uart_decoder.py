from typing import List, Tuple, Union, Optional
import numpy as np

class Decoder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samplerate = None
        self.frame_start = -1
        self.frame_valid = True
        self.cur_frame_bit = 0
        self.startbit = -1
        self.cur_data_bit = 0
        self.datavalue = 0
        self.paritybit = -1
        self.stopbits = []
        self.startsample = -1
        self.databits = []
        self.break_start = None
        self.idle_start = None
        self.state = 'WAIT FOR START BIT'
        self.bit_levels = []

    def decode_uart_from_analog(
        self,
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
        parity: str = "none",
        require_parity_ok: bool = True,
        require_level_ok: bool = True,
        low_min: Optional[float] = None,
        low_max: Optional[float] = None,
        high_min: Optional[float] = None,
        high_max: Optional[float] = None,
        sample_point: float = 50.0,
        return_parity: bool = False,
        return_frames: bool = False
    ) -> Union[
        Tuple[List[int], List[float], List[int]],
        Tuple[List[int], List[float], List[int], List[bool]],
        Tuple[List[int], List[float], List[int], List[dict]],
        Tuple[List[int], List[float], List[int], List[bool], List[dict]]
    ]:
        """
        UART decoder for analog waveforms, combining libsigrokdecode state machine with analog voltage handling.

        Determines bit values using low_min, low_max, high_min, high_max within 0-3.3V range.
        Validates voltages and flags errors for signals above 4.0V for logic 1.
        Incorporates state machine for modularity, with break/idle detection.

        Frame dict keys:
          - 'byte': int
          - 'start_idx', 'end_idx': sample indices
          - 'start_time', 'end_time': seconds
          - 'bit_indices': list[int]
          - 'bit_times': list[float]
          - 'bit_levels': list[int]
          - 'parity_ok': Optional[bool]
          - 'framing_error': bool
          - 'level_ok': bool
          - 'break': bool
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
                return self._return_output(return_parity, return_frames, decoded, sample_times, bit_levels, parity_flags, frames)

            samples_per_bit = float(sample_rate) / float(baud)
            if samples_per_bit < 3:
                pass  # Proceed but results may be unreliable

            # Calculate frame length for break/idle detection
            total_bits = 1 + data_bits + (0 if parity == "none" else 1) + stop_bits
            frame_samples = total_bits * samples_per_bit
            frame_len_sample_count = int(ceil(frame_samples))
            break_min_sample_count = frame_len_sample_count

            # Estimate signal characteristics for dynamic thresholds
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            amp = max(1e-12, (vmax - vmin) / 2.0)
            hyst_v = abs(hysteresis_percent) * amp

            # Define level ranges dynamically if not provided, within 0-3.3V
            if low_min is None:
                low_min = max(0.0, threshold_v - hyst_v - 0.5 * threshold_v)
            if low_max is None:
                low_max = threshold_v - hyst_v
            if high_min is None:
                high_min = threshold_v + hyst_v
            if high_max is None:
                high_max = min(threshold_v + hyst_v + 1.5 * threshold_v, 3.3)

            # Normalize parity choice
            parity_norm = (parity or "none").strip().lower()
            if parity_norm not in ("none", "odd", "even", "zero", "one", "ignore"):
                parity_norm = "none"

            # Digitalization using Schmitt trigger
            digital = np.zeros(n, dtype=np.int8)
            state = 1 if arr[0] >= high_min else 0
            for i in range(n):
                v = arr[i]
                if state == 1:
                    if v < low_max:
                        state = 0
                else:
                    if v > high_min:
                        state = 1
                digital[i] = state

            win_radius = max(1, int(round(samples_per_bit * float(sample_window_fraction) / 2.0)))

            def window_vote(center_idx: int) -> int:
                a = max(0, center_idx - win_radius)
                b = min(n - 1, center_idx + win_radius)
                seg = arr[a:b+1]
                count_high = np.count_nonzero((seg >= high_min) & (seg <= high_max))
                count_low = np.count_nonzero((seg >= low_min) & (seg <= low_max))
                total_valid = count_high + count_low
                if total_valid < seg.size * 0.5:
                    return -1  # Invalid bit
                return 1 if count_high > count_low else 0

            def window_fraction_high(center_idx: int) -> float:
                a = max(0, center_idx - win_radius)
                b = min(n - 1, center_idx + win_radius)
                seg = arr[a:b+1]
                count_high = np.count_nonzero((seg >= high_min) & (seg <= high_max))
                return float(count_high) / float(seg.size)

            def get_sample_point(bitnum: int) -> int:
                perc = min(max(sample_point, 1.0), 99.0) / 100.0
                bitpos = (samples_per_bit - 1) * perc + self.frame_start + bitnum * samples_per_bit
                return int(round(bitpos))

            def validate_bit(center_idx: int, bit: int) -> bool:
                a = max(0, center_idx - win_radius)
                b = min(n - 1, center_idx + win_radius)
                seg = arr[a:b+1]
                min_v = float(np.nanmin(seg))
                max_v = float(np.nanmax(seg))
                if bit == 1:
                    return min_v >= high_min and max_v <= high_max and max_v <= 4.0
                elif bit == 0:
                    return min_v >= low_min and max_v <= low_max
                return False

            def handle_frame():
                ss = self.frame_start
                es = self.samplenum
                frame = {
                    "byte": int(self.datavalue & 0xFF),
                    "start_idx": int(ss),
                    "end_idx": int(es),
                    "start_time": float(ss) / sample_rate,
                    "end_time": float(es) / sample_rate,
                    "bit_indices": list(self.databits),
                    "bit_times": [idx / sample_rate for idx in self.databits],
                    "bit_levels": list(self.bit_levels),
                    "parity_ok": self.paritybit if self.parity_ok else None,
                    "framing_error": not self.frame_valid,
                    "level_ok": self.level_valid,
                    "break": False
                }
                frames.append(frame)
                decoded.append(int(self.datavalue & 0xFF))
                sample_times.extend([idx / sample_rate for idx in self.databits])
                bit_levels.extend(self.bit_levels)
                if self.paritybit is not None:
                    parity_flags.append(self.parity_ok)
                self.reset()

            def handle_break(ss: int, es: int):
                frame = {
                    "byte": 0,
                    "start_idx": int(ss),
                    "end_idx": int(es),
                    "start_time": float(ss) / sample_rate,
                    "end_time": float(es) / sample_rate,
                    "bit_indices": [],
                    "bit_times": [],
                    "bit_levels": [],
                    "parity_ok": None,
                    "framing_error": False,
                    "level_ok": True,
                    "break": True
                }
                frames.append(frame)
                self.state = 'WAIT FOR START BIT'
                self.break_start = None

            def handle_idle(ss: int, es: int):
                # Optional: Add idle frame if needed
                pass

            self.samplerate = sample_rate
            self.level_valid = True
            self.parity_ok = True
            self.samplenum = 0
            self.bit_levels = []

            while self.samplenum < n:
                if self.state == 'WAIT FOR START BIT':
                    # Look for falling edge in digital signal
                    if self.samplenum > 0 and digital[self.samplenum - 1] == 1 and digital[self.samplenum] == 0:
                        self.frame_start = self.samplenum
                        self.frame_valid = True
                        self.cur_frame_bit = 0
                        self.state = 'GET START BIT'
                    # Check for break condition
                    if digital[self.samplenum] == 0:
                        if self.break_start is None:
                            self.break_start = self.samplenum
                        elif self.samplenum - self.break_start >= break_min_sample_count:
                            handle_break(self.break_start, self.samplenum)
                    else:
                        self.break_start = None
                    # Check for idle condition
                    if digital[self.samplenum] == 1:
                        if self.idle_start is None:
                            self.idle_start = self.samplenum
                        elif self.samplenum - self.idle_start >= frame_len_sample_count:
                            handle_idle(self.idle_start, self.samplenum)
                            self.idle_start = self.samplenum
                    else:
                        self.idle_start = None
                elif self.state == 'GET START BIT':
                    sample_idx = get_sample_point(self.cur_frame_bit)
                    if sample_idx >= n:
                        self.state = 'WAIT FOR START BIT'
                        continue
                    self.samplenum = sample_idx
                    self.startbit = window_vote(self.samplenum)
                    self.bit_levels.append(self.startbit)
                    self.databits.append(self.samplenum)
                    self.cur_frame_bit += 1
                    if self.startbit != 0 or self.startbit == -1:
                        self.frame_valid = False
                        handle_frame()
                        self.state = 'WAIT FOR START BIT'
                        continue
                    if not validate_bit(self.samplenum, self.startbit):
                        self.level_valid = False
                    self.state = 'GET DATA BITS'
                elif self.state == 'GET DATA BITS':
                    sample_idx = get_sample_point(self.cur_frame_bit)
                    if sample_idx >= n:
                        self.state = 'WAIT FOR START BIT'
                        continue
                    self.samplenum = sample_idx
                    bit = window_vote(self.samplenum)
                    if bit == -1:
                        self.level_valid = False
                        bit = 0
                    self.bit_levels.append(bit)
                    self.databits.append(self.samplenum)
                    self.datavalue |= (bit << self.cur_data_bit)
                    self.cur_data_bit += 1
                    self.cur_frame_bit += 1
                    if not validate_bit(self.samplenum, bit):
                        self.level_valid = False
                    if self.cur_data_bit < data_bits:
                        continue
                    self.state = 'GET PARITY BIT' if parity_norm != "none" else 'GET STOP BITS'
                elif self.state == 'GET PARITY BIT':
                    sample_idx = get_sample_point(self.cur_frame_bit)
                    if sample_idx >= n:
                        self.state = 'WAIT FOR START BIT'
                        continue
                    self.samplenum = sample_idx
                    self.paritybit = window_vote(self.samplenum)
                    self.bit_levels.append(self.paritybit)
                    self.databits.append(self.samplenum)
                    self.cur_frame_bit += 1
                    if self.paritybit == -1:
                        self.level_valid = False
                        self.parity_ok = False
                    else:
                        ones_count = sum(1 for b in self.bit_levels[:-1] if b == 1) % 2  # Exclude parity itself for check
                        if parity_norm == "even":
                            expected_p = 0 if ones_count % 2 == 0 else 1
                        elif parity_norm == "odd":
                            expected_p = 1 if ones_count % 2 == 0 else 0
                        elif parity_norm == "zero":
                            expected_p = 0
                        elif parity_norm == "one":
                            expected_p = 1
                        else:
                            expected_p = self.paritybit
                        self.parity_ok = (self.paritybit == expected_p)
                    if not self.parity_ok and require_parity_ok:
                        handle_frame()
                        self.state = 'WAIT FOR START BIT'
                        continue
                    if not validate_bit(self.samplenum, self.paritybit):
                        self.level_valid = False
                    self.state = 'GET STOP BITS'
                elif self.state == 'GET STOP BITS':
                    sample_idx = get_sample_point(self.cur_frame_bit)
                    if sample_idx >= n:
                        self.state = 'WAIT FOR START BIT'
                        continue
                    self.samplenum = sample_idx
                    bit = window_vote(self.samplenum)
                    if bit == -1:
                        bit = 0
                        self.level_valid = False
                    self.bit_levels.append(bit)
                    self.databits.append(self.samplenum)
                    self.stopbits.append(bit)
                    self.cur_frame_bit += 1
                    if bit != 1:
                        self.frame_valid = False
                    if not validate_bit(self.samplenum, bit):
                        self.level_valid = False
                    if len(self.stopbits) < stop_bits:
                        continue
                    if not self.level_valid and require_level_ok:
                        handle_frame()
                        self.state = 'WAIT FOR START BIT'
                        continue
                    handle_frame()
                    self.state = 'WAIT FOR START BIT'
                self.samplenum += 1

        except Exception:
            pass

        return self._return_output(return_parity, return_frames, decoded, sample_times, bit_levels, parity_flags, frames)

    def _return_output(self, return_parity, return_frames, decoded, sample_times, bit_levels, parity_flags, frames):
        if return_parity and return_frames:
            return decoded, sample_times, bit_levels, parity_flags, frames
        if return_parity:
            return decoded, sample_times, bit_levels, parity_flags
        if return_frames:
            return decoded, sample_times, bit_levels, frames
        return decoded, sample_times, bit_levels