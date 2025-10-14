# oscillo_controller.py
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
from PySide6 import QtCore, QtWidgets

from model.oscillo_model import OscilloModel
from model.worker import NiScopeWorker
# from utils.metrics import compute_metrics_per_channel


class OscilloController:
    def __init__(self, model: OscilloModel, view, niscope_available: bool = False):
        self.model = model
        self.view = view
        self.niscope_available = niscope_available
        self.worker: Optional[NiScopeWorker] = None

        # Wire toolbar actions
        # toggle_start_action is checkable: on -> start continuous, off -> stop
        self.view.toggle_start_action.triggered.connect(self._on_toggle_start)
        self.view.single_action.triggered.connect(self.start_single)

        # Update param measurement table headers based on current selections
        selected = [c for c in ["0", "1"]
                    if (c == "0" and self.view.controls.ch0_cb.isChecked())
                    or (c == "1" and self.view.controls.ch1_cb.isChecked())]
        if not selected:
            selected = ["0"]
        self.view.update_param_table_headers(selected)

        # Show niscope availability
        if self.niscope_available:
            self.view.log("NI-SCOPE module available.")
        else:
            self.view.log("NI-SCOPE module NOT available. Using Simulation mode.")
            self.model.simulate = True
            self.view.controls.simulate_cb.setChecked(True)

    def _on_toggle_start(self, checked: bool):
        if checked:
            self.view.log("Start requested")
            self.start_continuous()
        else:
            self.view.log("Stop requested")
            self.stop()

    def _update_model_from_ui(self):
        self.model.resource_name = self.view.controls.resource_edit.text().strip()
        chs = []
        if self.view.controls.ch0_cb.isChecked():
            chs.append("0")
        if self.view.controls.ch1_cb.isChecked():
            chs.append("1")
        self.model.channels = ",".join(chs) if chs else ""

        self.model.voltage_range = float(self.view.controls.voltage_spin.value())

        # Sample rate: UI uses kHz; convert to Hz internally
        try:
            sr_khz_text = self.view.controls.sample_rate_edit.text().strip()
            if sr_khz_text == "":
                raise ValueError("empty sample rate")
            sr_khz = float(sr_khz_text)
            self.model.sample_rate = sr_khz * 1e3
        except Exception:
            # restore display if parse failed
            self.view.controls.sample_rate_edit.setText(str(int(self.model.sample_rate / 1e3)))

        # Fetch time: parse from line edit (seconds)
        try:
            ft_text = self.view.controls.fetch_time_edit.text().strip()
            if ft_text == "":
                raise ValueError("empty fetch time")
            self.model.fetch_time = float(ft_text)
        except Exception:
            self.view.controls.fetch_time_edit.setText(str(self.model.fetch_time))

        self.model.simulate = bool(self.view.controls.simulate_cb.isChecked())

        if chs:
            self.view.update_param_table_headers(chs)
        else:
            self.view.update_param_table_headers(["0"])

        # ---- trigger settings ----
        try:
            tcfg = self.view.trigger.get_config()
            self.model.trigger_mode = tcfg.get("mode", "Immediate")
            self.model.trigger_edge_direction = tcfg.get("edge_direction", "Rising")
            self.model.trigger_level = float(tcfg.get("level", 0.0))
            src = tcfg.get("source", "") or (chs[0] if chs else "0")
            self.model.trigger_source = src
            self.model.trigger_holdoff = float(tcfg.get("holdoff", 0.0))
            self.model.trigger_delay = float(tcfg.get("delay", 0.0))
        except Exception:
            self.view.log("Warning: could not parse trigger setting from UI, using defaults")

    def start_worker(self, single_shot: bool = False):
        # stop old worker if running
        if self.worker and self.worker.isRunning():
            self.stop()

        self._update_model_from_ui()
        self.worker = NiScopeWorker(self.model)
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.status.connect(self.on_status)
        self.worker.error.connect(self.on_error)

        # make sure curves visibility matches channel selection
        self.view.plot.curve_items['0'].setVisible(self.view.controls.ch0_cb.isChecked())
        self.view.plot.curve_items['1'].setVisible(self.view.controls.ch1_cb.isChecked())

        if single_shot:
            self.worker.start_single()
        else:
            self.worker.start_continuous()
        self.view.log("Worker started (thread)")

    def start_continuous(self):
        self.start_worker(single_shot=False)

    def start_single(self):
        self.view.log("Single capture requested")
        self.start_worker(single_shot=True)

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.view.log("Stopped worker")
        else:
            self.view.log("No worker to stop")
        try:
            if getattr(self.view, "toggle_start_action", None) and self.view.toggle_start_action.isChecked():
                self.view.toggle_start_action.setChecked(False)
        except Exception:
            pass

    @QtCore.Slot(object)
    def on_data_ready(self, payload):
        try:
            wf = payload.get("waveform")
            ts = payload.get("timestamp", time.time())
            self.model.last_waveform = wf
            self.model.last_timestamp = ts

            arr = np.asarray(wf)
            requested_chs = [ch.strip() for ch in self.model.channels.split(",") if ch.strip() != ""]
            if arr.ndim == 1:
                ch_list = requested_chs if requested_chs else ["0"]
                channel_arrays = [arr]
            elif arr.ndim == 2:
                ch_list = requested_chs if requested_chs else [str(i) for i in range(arr.shape[0])]
                if len(ch_list) < arr.shape[0]:
                    ch_list = [str(i) for i in range(arr.shape[0])]
                channel_arrays = [arr[i] for i in range(arr.shape[0])]
            else:
                channel_arrays = [arr.ravel()]
                ch_list = requested_chs if requested_chs else ["0"]

            # Update plots
            try:
                for idx, ch_arr in enumerate(channel_arrays):
                    ch_name = ch_list[idx] if idx < len(ch_list) else str(idx)
                    key = ch_name
                    if key not in self.view.plot.curve_items:
                        continue
                    self.view.update_curve(key, ch_arr)
            except Exception as ex:
                self.view.log(f"Plot update error: {ex}")

            # metrics
            try:
                metrics_list = payload.get("metrics", None)
                if metrics_list is None:
                    # metrics_list = compute_metrics_per_channel(wf, self.model.sample_rate)
                    metrics_list = [ {k: float("nan") for k in [
                    "frequency_hz","amplitude_v","rise_time_s","fall_time_s","pkpk_v","rms_v"
                ]} for _ in ch_list ]

                # if not metrics_list:
                #     ch_list = [str(i) for i in range(1)]
                # else:
                #     if not ch_list:
                #         ch_list = [str(i) for i in range(len(metrics_list))]

                # append params row
                row_items = [datetime.fromtimestamp(ts).isoformat(sep=' ')]
                for metrics in metrics_list:
                    def f(k):
                        v = metrics.get(k, float('nan'))
                        try:
                            return f"{float(v):.6g}"
                        except Exception:
                            return "NaN"
                    row_items += [
                        f("frequency_hz"),
                        f("amplitude_v"),
                        f("rise_time_s"),
                        f("fall_time_s"),
                        f("pkpk_v"),
                        f("rms_v"),
                    ]

                expected_cols = 1 + len(metrics_list) * 6
                if self.view.param_table.columnCount() != expected_cols:
                    ch_names = ch_list if ch_list else [str(i) for i in range(len(metrics_list))]
                    self.view.update_param_table_headers(ch_names)

                self.view.append_param_row(row_items)

                # update measurements table
                try:
                    if hasattr(self.view, "update_measurement_table"):
                        self.view.update_measurement_table(metrics_list, ch_list)
                except Exception as ex:
                    self.view.log(f"Could not update measurement table: {ex}")

            except Exception as ex:
                self.view.log(f"Metrics computation/table error: {ex}")

        except Exception as ex:
            self.view.log(f"on_data_ready top-level error: {ex}")

    @QtCore.Slot(str)
    def on_status(self, text: str):
        self.view.log(f"STATUS: {text}")

    @QtCore.Slot(str)
    def on_error(self, text: str):
        self.view.log(f"ERROR: {text}")
        QtWidgets.QMessageBox.critical(self.view, "Worker error", text)
