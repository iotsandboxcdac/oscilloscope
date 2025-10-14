#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pprint
import time
import math
import sys
from datetime import timedelta

try:
    import niscope
    HAS_NISCOPE = True
except Exception:
    HAS_NISCOPE = False

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

pp = pprint.PrettyPrinter(indent=2, width=120)

def _build_measure_map():
    if not HAS_NISCOPE:
        return {}
    M = {
        "rms": niscope.ScalarMeasurement.VOLTAGE_RMS,
        "pkpk": niscope.ScalarMeasurement.VOLTAGE_PEAK_TO_PEAK,
        "amplitude": niscope.ScalarMeasurement.AMPLITUDE,
        "freq": niscope.ScalarMeasurement.FREQUENCY,
        "rise_time": niscope.ScalarMeasurement.RISE_TIME,
        "fall_time": niscope.ScalarMeasurement.FALL_TIME,
        "max": niscope.ScalarMeasurement.VOLTAGE_MAX,
        "min": niscope.ScalarMeasurement.VOLTAGE_MIN,
        "high": niscope.ScalarMeasurement.VOLTAGE_HIGH,
        "low": niscope.ScalarMeasurement.VOLTAGE_LOW,
        "avg": niscope.ScalarMeasurement.VOLTAGE_AVERAGE,
        "voltage_cycle_rms": niscope.ScalarMeasurement.VOLTAGE_CYCLE_RMS,
        "width_pos": niscope.ScalarMeasurement.WIDTH_POS,
        "width_neg": niscope.ScalarMeasurement.WIDTH_NEG,
        "duty_pos": niscope.ScalarMeasurement.DUTY_CYCLE_POS,
        "duty_neg": niscope.ScalarMeasurement.DUTY_CYCLE_NEG,
    }
    return M

MEAS_MAP = _build_measure_map()

def run_with_hardware(args):
    """Open session, configure, initiate, and fetch scalar + array measurements (corrected)."""
    resource = args.resource
    requested_channel_str = args.channels
    sample_rate_hz = args.sample_khz * 1e3
    fetch_time_s = args.fetch_s
    total_samples = max(2, int(round(sample_rate_hz * fetch_time_s)))
    timeout_td = timedelta(seconds=max(1.0, fetch_time_s + 2.0))

    print(f"Opening NI-SCOPE session {resource}, channels={requested_channel_str}, "
          f"sample_rate={sample_rate_hz:.0f} Hz, samples={total_samples}")
    # open session
    with niscope.Session(resource_name=resource) as session:
        try:
            available = session.get_channel_names()
            print("Device reports channel names:", available)
        except Exception:
            try:
                cc = int(getattr(session, "channel_count", 0))
                available = [str(i) for i in range(cc)]
                print("Device channel_count suggests channels:", available)
            except Exception:
                available = []
        ch_list = []
        for token in str(requested_channel_str).split(","):
            t = token.strip()
            if t == "":
                continue
            ch_list.append(t)

        if not ch_list:
            ch_list = [available[0]] if available else ["0"]

        valid_ch_list = []
        for ch in ch_list:
            if available and ch not in available:
                if available and ch.isdigit():
                    idx = int(ch)
                    if 0 <= idx < len(available):
                        print(f"Mapping requested channel '{ch}' --> device channel '{available[idx]}'")
                        valid_ch_list.append(available[idx])
                    else:
                        print(f"Requested channel '{ch}' not present; skipping")
                else:
                    print(f"Requested channel '{ch}' not present on device; skipping")
            else:
                valid_ch_list.append(ch)
        if not valid_ch_list:
            raise RuntimeError(f"No valid channels to fetch (requested {ch_list}, device has {available})")
        print("Will use channels:", valid_ch_list)

        for ch in valid_ch_list:
            session.channels[ch].configure_vertical(range=args.voltage_range, coupling=niscope.VerticalCoupling.DC)

        # horizontal timing
        session.configure_horizontal_timing(
            min_sample_rate=sample_rate_hz,
            min_num_pts=total_samples,
            ref_position=50.0,
            num_records=1,
            enforce_realtime=True,
        )
        print("Session configured. Initiating acquisition...")

        # configure trigger mode
        if args.trigger_mode.lower() == "immediate":
            session.configure_trigger_immediate()
        elif args.trigger_mode.lower() == "edge":
            src = args.trigger_source or valid_ch_list[0]
            level = float(args.trigger_level)
            slope = niscope.TriggerSlope.POSITIVE if args.trigger_edge.lower() == "rising" else niscope.TriggerSlope.NEGATIVE
            try:
                session.configure_trigger_edge(src, level, niscope.TriggerCoupling.DC, slope=slope)
            except TypeError:
                session.configure_trigger_edge(src, level, niscope.TriggerCoupling.DC)

        results = {"scalar": {}, "array": {}}

        # Initiate acquisition and fetch measurements
        with session.initiate():
            for mname in args.measure:
                if mname not in MEAS_MAP:
                    print(f"Skipping unknown measurement: {mname}")
                    continue
                enum_val = MEAS_MAP[mname]
                per_channel_vals = {}
                for ch in valid_ch_list:
                    try:
                        val = session.channels[ch].fetch_measurement_stats(scalar_meas_function=enum_val, timeout=timeout_td)
                        per_channel_vals[ch] = val
                        print(f"Measurement {mname} @ ch {ch}: {val}")
                    except Exception as ex:
                        per_channel_vals[ch] = f"ERROR: {ex}"
                        print(f"Failed to fetch measurement {mname} for channel {ch}: {ex}")
                results["scalar"][mname] = per_channel_vals

            if args.fft:
                for ch in valid_ch_list:
                    try:
                        arr = session.channels[ch].fetch_array_measurement(
                            array_meas_function=niscope.ArrayMeasurement.FFT_AMP_SPECTRUM_VOLTS_RMS,
                            meas_wfm_size=total_samples,
                            timeout=timeout_td,
                        )
                        results["array"][ch] = np.asarray(arr)
                        print(f"Fetched FFT amplitude spectrum array for channel {ch}, shape: {np.shape(arr)}")
                    except Exception as ex:
                        results["array"][ch] = f"ERROR: {ex}"
                        print(f"Failed to fetch array measurement (FFT) for channel {ch}: {ex}")

        return results

    # FFT example
    if args.fft:
        s = wf[0, :]
        n = len(s)
        nfft = 1 << (int(np.ceil(np.log2(n))))
        S = np.fft.rfft(s * np.hanning(n), n=nfft) / n
        amp = np.abs(S) * np.sqrt(2.0)
        freqs = np.fft.rfftfreq(nfft, dt)
        if HAS_MPL:
            plt.figure()
            plt.plot(freqs, amp)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (approx, V RMS)")
            plt.title("Simulated FFT amplitude")
            plt.grid(True)
            plt.show()
        else:
            print("[SIM] FFT done (not plotted; matplotlib not available)")

def main():
    p = argparse.ArgumentParser(description="NI-SCOPE measurement fetch example (PXIe-5111 friendly).")
    p.add_argument("--resource", default="PXI1Slot2", help="NI resource name (e.g. PXI1Slot2 or Dev1)")
    p.add_argument("--channels", default="0", help="channel list string (e.g. '0' or '0,1')")
    p.add_argument("--sample-khz", type=float, default=1000.0, help="sample rate in kHz (UI friendly)")
    p.add_argument("--fetch-s", type=float, default=0.001, help="fetch/acquisition length in seconds")
    p.add_argument("--voltage-range", type=float, default=5.0, help="vertical range (V pk-pk) to configure")
    p.add_argument("--trigger-mode", choices=["Immediate", "Edge"], default="Immediate", help="trigger mode")
    p.add_argument("--trigger-edge", choices=["Rising", "Falling"], default="Rising", help="edge slope for edge trigger")
    p.add_argument("--trigger-level", type=float, default=0.0, help="trigger level (V) when using Edge")
    p.add_argument("--fft", action="store_true", help="also fetch FFT amplitude spectrum (array measurement)")
    p.add_argument("--simulate", action="store_true", help="simulate waveform (no hardware)")
    p.add_argument("--measure", nargs="+", default=["rms", "pkpk", "freq", "amplitude"],
                   help="scalar measurements to fetch (choices such as: {})".format(", ".join(sorted(MEAS_MAP.keys()))))
    args = p.parse_args()

    print("niscope available:", HAS_NISCOPE)

    for m in args.measure:
        if m not in MEAS_MAP:
            print(f"Warning: requested measurement '{m}' not in supported map; skipping it.")
    try:
        results = run_with_hardware(args)
        print("\n=== Results ===")
        pp.pprint(results)
        if args.fft and "fft_amp_spectrum" in results and HAS_MPL:
            arr = results["fft_amp_spectrum"]
            arr = np.asarray(arr)
            if arr.ndim > 1:
                y = arr[0]
            else:
                y = arr
            sample_rate_hz = args.sample_khz * 1e3
            nfft = len(y) * 2 if (len(y) > 1) else len(y)
            freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate_hz)[:len(y)]
            plt.figure()
            plt.plot(freqs, y)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (V RMS)")
            plt.title("Device FFT amplitude (first channel)")
            plt.grid(True)
            plt.show()
    except Exception as ex:
        print("Error during NI operation:", ex, file=sys.stderr)
        raise

if __name__ == "__main__":
    main()

# python niscope_measure_example.py --resource PXI1Slot2 --channels 0 --sample-khz 1000 --fetch-s 0.001 --measure rms pkpk freq amplitude rise_time fall_time --fft