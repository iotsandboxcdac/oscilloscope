import time
import math
import numpy as np
import datetime
from PySide6 import QtCore
from scipy.signal import sawtooth
from niscope import ScalarMeasurement

from .oscillo_model import OscilloModel

try:
    import niscope
    NISCOPE_AVAILABLE = True
except Exception:
    NISCOPE_AVAILABLE = False

MEAS_MAP = {
    "frequency_hz": ScalarMeasurement.FREQUENCY,
    "amplitude_v": ScalarMeasurement.AMPLITUDE,
    "rise_time_s": ScalarMeasurement.RISE_TIME,
    "fall_time_s": ScalarMeasurement.FALL_TIME,
    "pkpk_v": ScalarMeasurement.VOLTAGE_PEAK_TO_PEAK,
    "rms_v": ScalarMeasurement.VOLTAGE_RMS,
}

class NiScopeWorker(QtCore.QThread):
    data_ready = QtCore.Signal(object)  # dict {waveform, timestamp}
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, model: OscilloModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._running = False
        self._single_shot = False

    def start_continuous(self):
        self._single_shot = False
        self._running = True
        if not self.isRunning():
            self.start()

    def start_single(self):
        self._single_shot = True
        self._running = True
        if not self.isRunning():
            self.start()

    def stop(self):
        self._running = False

    def run(self):
        try:
            total_samples = int(max(1, round(self.model.fetch_time * self.model.sample_rate)))
        except Exception as e:
            self.error.emit(f"Invalid acquisition parameters: {e}")
            return

        self.status.emit("Worker started")
        if self.model.simulate or not NISCOPE_AVAILABLE:
            self._run_simulator(total_samples)
        else:
            self._run_niscope(total_samples)
        self.status.emit("Worker stopped")

    def _run_simulator(self, total_samples: int):
        self.status.emit("Simulator mode ON")
        t = np.arange(total_samples) / self.model.sample_rate
        base_freq = 1000.0
        ch_list = [ch.strip() for ch in self.model.channels.split(",") if ch.strip()]
        ch_count = max(1, len(ch_list))
        phase_offsets = [i * 0.5 * math.pi for i in range(ch_count)]

        while self._running:
            if self.model.resource_name == 'sawtooth':
                data = np.array([
                    0.9 * self.model.voltage_range * 
                    (2 * ((base_freq * (1 + 0.02 * i)) * t + ph / (2 * math.pi) - np.floor(0.5 + (base_freq * (1 + 0.02 * i)) * t + ph / (2 * math.pi)))) +
                    0.02 * np.random.randn(total_samples)
                    for i, ph in enumerate(phase_offsets)
                ])
            else:
                data = np.array([
                    0.9 * self.model.voltage_range *
                    np.sin(2 * math.pi * (base_freq * (1 + 0.02 * i)) * t + ph) +
                    0.02 * np.random.randn(total_samples)
                    for i, ph in enumerate(phase_offsets)
                ])
            waveform = data[0] if data.shape[0] == 1 else data
            ts = time.time()
            metrics_list = None
            self.data_ready.emit({"waveform": waveform, "timestamp": ts, "metrics": metrics_list})
            if self._single_shot:
                self._running = False
                break
            time.sleep(max(0.02, self.model.fetch_time))

    def _run_niscope(self, total_samples: int):
    # Lazy import for type access and local utilities
        import niscope
        try:
            self.status.emit(f"Opening session: {self.model.resource_name}")
            with niscope.Session(resource_name=self.model.resource_name) as session:
                try:
                    session.channels[self.model.channels].configure_vertical(
                        range=self.model.voltage_range,
                        coupling=niscope.VerticalCoupling.DC,
                        offset=getattr(self.model, "vertical_offset", 0.0)
                    )
                except TypeError:
                    session.channels[self.model.channels].configure_vertical(
                        range=self.model.voltage_range,
                        coupling=niscope.VerticalCoupling.DC
                    )

                session.configure_horizontal_timing(
                    min_sample_rate=self.model.sample_rate,
                    min_num_pts=total_samples,
                    ref_position=50.0,
                    num_records=1,
                    enforce_realtime=True
                )
                print(self.model.sample_rate)
                self.status.emit("Session configured")

                # Channel list & count
                ch_list = [ch.strip() for ch in self.model.channels.split(",") if ch.strip()]
                ch_count = len(ch_list) if ch_list else 1

                # Configure trigger
                try:
                    trig_mode = getattr(self.model, "trigger_mode", "Immediate") or "Immediate"
                    trig_mode_l = trig_mode.lower()
                    if trig_mode_l == "immediate":
                        session.configure_trigger_immediate()
                        self.status.emit("Trigger: Immediate")

                    elif trig_mode_l == "edge":
                        source = getattr(self.model, "trigger_source", "") or (ch_list[0] if ch_list else "0")
                        level = float(getattr(self.model, "trigger_level", 0.0))
                        direction = getattr(self.model, "trigger_edge_direction", "Rising").lower()
                        slope = niscope.TriggerSlope.POSITIVE if direction == "rising" else niscope.TriggerSlope.NEGATIVE

                        holdoff_val = float(getattr(self.model, "trigger_holdoff", 0.0))
                        delay_val = float(getattr(self.model, "trigger_delay", 0.0))
                        holdoff_td = datetime.timedelta(seconds=holdoff_val)
                        delay_td = datetime.timedelta(seconds=delay_val)

                        try:
                            session.configure_trigger_edge(source, level, niscope.TriggerCoupling.DC,
                                                        slope=slope, holdoff=holdoff_td, delay=delay_td)
                        except TypeError:
                            try:
                                session.configure_trigger_edge(source, level, niscope.TriggerCoupling.DC, slope, holdoff_td, delay_td)
                            except Exception:
                                session.configure_trigger_edge(source, level, niscope.TriggerCoupling.DC)

                        self.status.emit(f"Trigger: Edge src={source} lvl={level} slope={direction} holdoff={holdoff_val}s delay={delay_val}s")

                    # elif trig_mode_l == "software":
                    #     # Configure for software trigger; consumer may call send_software_trigger_edge() later
                    #     try:
                    #         session.configure_trigger_software()
                    #         self.status.emit("Trigger: Software (arm only)")
                    #     except Exception:
                    #         # if not supported, fall back to immediate to avoid hanging
                    #         session.configure_trigger_immediate()
                    #         self.status.emit("Trigger: Software not supported, fallback to Immediate")

                    # else:
                    #     # Unknown mode -> fallback to immediate
                    #     session.configure_trigger_immediate()
                    #     self.status.emit("Trigger: Unknown mode fallback to Immediate")
                except Exception as ex:
                    self.status.emit(f"Trigger config failed (continuing): {ex}")

                fetch_timeout_val = float(getattr(self.model, "trigger_timeout", 5.0))
                fetch_timeout = datetime.timedelta(seconds=fetch_timeout_val)

                # Main acquisition loop
                while self._running:
                    try:
                        with session.initiate():
                            waveform = np.empty((ch_count, total_samples), dtype=np.float64)
                            for i, ch in enumerate(ch_list):
                                session.channels[ch].fetch_into(waveform[i], num_records=1, timeout=fetch_timeout)

                            # ---- Fetch hardware measurements per channel ----
                            metrics_list = []
                            for ch in ch_list:
                                ch_metrics = {}
                                for key, enum in MEAS_MAP.items():
                                    try:
                                        stats = session.channels[ch].fetch_measurement_stats(
                                            scalar_meas_function=enum, timeout=fetch_timeout
                                        )
                                        if isinstance(stats, list) and len(stats) >0:
                                            ch_metrics[key] = float(stats[0].result)
                                        else:
                                            ch_metrics[key] = float("nan")
                                    except Exception as ex:
                                        ch_metrics[key] = float("nan")
                                        self.status.emit(f"Measurement {key} failed: {ex}")
                                metrics_list.append(ch_metrics)

                        ts = time.time()
                        self.data_ready.emit({
                            "waveform": waveform,
                            "timestamp": ts,
                            "metrics": metrics_list
                        })
                        if self._single_shot:
                            self._running = False
                            break
                        time.sleep(max(0.01, self.model.fetch_time))

                    except Exception as ex:
                        # Handle NI timeout and recover: don't crash the thread on a trigger timeout
                        errmsg = str(ex)
                        lower = errmsg.lower()
                        if "maximum time exceeded" in lower or "timeout" in lower or "-1074126845" in errmsg:
                            # timeout waiting for trigger; abort and continue (or break if single-shot)
                            try:
                                session.abort()
                            except Exception:
                                pass
                            self.status.emit("Trigger wait timed out — no trigger received")
                            if self._single_shot:
                                self._running = False
                                break
                            else:
                                time.sleep(0.1)
                                continue
                        else:
                            # Other errors: send as error and stop loop
                            self.error.emit(f"NI-SCOPE error: {ex}")
                            break

        except Exception as ex:
            self.error.emit(f"NI-SCOPE error: {ex}")

