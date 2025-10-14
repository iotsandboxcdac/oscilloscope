# main_window.py
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QFont, QIcon
from utils.uart_decoder import decode_uart_from_analog
from PySide6.QtCore import QSize, Qt

# -------------------- Controls Panel --------------------
class ControlsPanel(QtWidgets.QFrame):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Acquisition Controls</b>"))
        layout.addWidget(QtWidgets.QLabel("Resource:"))
        self.resource_edit = QtWidgets.QLineEdit(getattr(self.model, "resource_name", ""))
        layout.addWidget(self.resource_edit)

        layout.addWidget(QtWidgets.QLabel("Channels to acquire:"))
        chs_layout = QtWidgets.QHBoxLayout()
        self.ch0_cb = QtWidgets.QCheckBox("CH0")
        self.ch1_cb = QtWidgets.QCheckBox("CH1")
        chs = [s.strip() for s in getattr(self.model, "channels", "").split(",") if s.strip() != ""]
        self.ch0_cb.setChecked("0" in chs)
        self.ch1_cb.setChecked("1" in chs)
        chs_layout.addWidget(self.ch0_cb)
        chs_layout.addWidget(self.ch1_cb)
        layout.addLayout(chs_layout)

        layout.addWidget(QtWidgets.QLabel("Voltage range (V):"))
        self.voltage_spin = QtWidgets.QDoubleSpinBox()
        self.voltage_spin.setRange(0.01, 20.0)
        self.voltage_spin.setValue(getattr(self.model, "voltage_range", 5.0))
        layout.addWidget(self.voltage_spin)

        layout.addWidget(QtWidgets.QLabel("Sample rate (KHz):"))
        sr_khz = int(getattr(self.model, "sample_rate", 1e6) / 1e3)
        self.sample_rate_edit = QtWidgets.QLineEdit(str(sr_khz))
        # self.sample_rate_edit.setToolTip("Enter sample rate in kiloHertz (kHz)")
        layout.addWidget(self.sample_rate_edit)

        layout.addWidget(QtWidgets.QLabel("Fetch time (s):"))
        self.fetch_time_edit = QtWidgets.QLineEdit(str(getattr(self.model, "fetch_time", 0.001)))
        self.fetch_time_edit.setToolTip("Enter fetch/acquisition time in seconds (e.g. 0.001).")
        layout.addWidget(self.fetch_time_edit)

        self.simulate_cb = QtWidgets.QCheckBox("Simulation mode (no NI hardware)")
        self.simulate_cb.setChecked(getattr(self.model, "simulate", False))
        layout.addWidget(self.simulate_cb)

        # Buttons: Clear Display and Reset model defaults
        btn_row = QtWidgets.QHBoxLayout()
        self.clear_display_btn = QtWidgets.QPushButton("Clear")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        btn_row.addWidget(self.clear_display_btn)
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)

class UARTPanel(QtWidgets.QFrame):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.uart_scatter = pg.ScatterPlotItem(size=6, brush='r')
        self._scatter_added_to_plot = False
        self.uart_frame_items = []
        self._frames_added_to_plot = False
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>UART Decoder</b>"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Channel:"))
        self.uart_channel_combo = QtWidgets.QComboBox()
        self.uart_channel_combo.addItems(["0", "1"])
        row.addWidget(self.uart_channel_combo)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Baud:"))
        self.uart_baud = QtWidgets.QLineEdit("9600")
        self.uart_baud.setMaximumWidth(125)
        row.addWidget(self.uart_baud)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Data bits:"))
        self.uart_data_bits = QtWidgets.QSpinBox()
        self.uart_data_bits.setRange(5, 9)
        self.uart_data_bits.setValue(8)
        row.addWidget(self.uart_data_bits)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Stop bits:"))
        self.uart_stop_bits = QtWidgets.QSpinBox()
        self.uart_stop_bits.setRange(1, 2)
        self.uart_stop_bits.setValue(1)
        row.addWidget(self.uart_stop_bits)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Threshold (V):"))
        self.uart_threshold = QtWidgets.QDoubleSpinBox()
        self.uart_threshold.setRange(-10.0, 10.0)
        self.uart_threshold.setValue(0.0)
        row.addWidget(self.uart_threshold)
        layout.addLayout(row)

        row.addWidget(QtWidgets.QLabel("Hysteresis %:"))
        self.uart_hyst = QtWidgets.QDoubleSpinBox()
        self.uart_hyst.setRange(0.0, 0.5)
        self.uart_hyst.setSingleStep(0.01)
        self.uart_hyst.setValue(0.05)
        row.addWidget(self.uart_hyst)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Parity:"))
        self.uart_parity = QtWidgets.QComboBox()
        self.uart_parity.addItems(["none", "even", "odd"])
        row.addWidget(self.uart_parity)
        layout.addLayout(row)

        btn_row = QtWidgets.QHBoxLayout()
        self.decode_btn = QtWidgets.QPushButton("Decode UART")
        self.clear_overlays_btn = QtWidgets.QPushButton("Clear Overlays")
        self.clear_overlays_btn.setToolTip("Remove UART overlays/annotations from waveform")
        btn_row.addWidget(self.decode_btn)
        btn_row.addWidget(self.clear_overlays_btn)
        layout.addLayout(btn_row)

        layout.addWidget(QtWidgets.QLabel("Decoded Hex:"))
        self.uart_hex = QtWidgets.QTextEdit()
        self.uart_hex.setReadOnly(True)
        layout.addWidget(self.uart_hex)

        layout.addWidget(QtWidgets.QLabel("Decoded ASCII:"))
        self.uart_ascii = QtWidgets.QTextEdit()
        self.uart_ascii.setReadOnly(True)
        layout.addWidget(self.uart_ascii)

    def _find_owner_with_plot(self):
        """Walk the parent chain to find the main window (object that has .plot)."""
        w = self
        # parent(), parentWidget() may return dock widget or other containers
        while w is not None:
            if hasattr(w, "plot") and getattr(w.plot, "plot_widget", None) is not None:
                return w
            # try parentWidget (works for QWidget), fallback to parent()
            w = getattr(w, "parentWidget", lambda: None)() or getattr(w, "parent", lambda: None)()
        return None

    def on_decode_uart(self):
        """Called by UI. decode waveform, display decoded strings and overlay markers."""
        data = self.model.last_waveform
        if data is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No waveform to decode.")
            return

        ch_sel = int(self.uart_channel_combo.currentText())
        arr = np.asarray(data)
        if arr.ndim == 2:
            if ch_sel < arr.shape[0]:
                sig = arr[ch_sel]
            else:
                QtWidgets.QMessageBox.warning(self, "Channel error", "Selected channel not present.")
                return
        else:
            sig = arr
            if ch_sel != 0:
                owner = self._find_owner_with_plot()
                if owner and hasattr(owner, "log"):
                    owner.log("Single-channel capture: using CH0 for UART decode.")

        try:
            baud = int(self.uart_baud.text().strip())
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Baud error", "Invalid baud rate.")
            return

        data_bits = int(self.uart_data_bits.value())
        stop_bits = int(self.uart_stop_bits.value())
        threshold = float(self.uart_threshold.value())
        sr = self.model.sample_rate
        hysteresis_percent = float(self.uart_hyst.value())
        parity_mode = str(self.uart_parity.currentText())

        # call decoder (unchanged)
        decoded, samp_times, bit_levels, parity_flags, frames= decode_uart_from_analog(sig, sr, threshold, baud, data_bits=data_bits, stop_bits=stop_bits, hysteresis_percent=hysteresis_percent, sample_window_fraction=0.35, min_stop_high_fraction=0.6, parity=parity_mode, require_parity_ok=False, return_parity=True, return_frames=True)

        # --- basic sanitization of decoded bytes (avoid showing nonsensical huge/negative numbers) ---
        clean_decoded = []
        for b in decoded:
            try:
                ib = int(b) & 0xFF
            except Exception:
                continue
            # filter obviously unreasonable values (non-byte)
            if 0 <= ib <= 0xFF:
                clean_decoded.append(ib)
        decoded = clean_decoded

        hex_str_parts = []
        ascii_str_parts = []
        for i, b in enumerate(decoded):
            tag = ""
            if i < len(frames):
                fr = frames[i]
                if fr.get("parity_ok") is False:
                    tag += "[PERR]"   # parity error
                if fr.get("framing_error"):
                    tag += "[FERR]"   # framing error
            hex_str_parts.append(f"{b:02X}{tag}")
            ascii_str_parts.append(chr(b) if 32 <= b < 127 else ".")

        hex_str = " ".join(hex_str_parts)
        ascii_str = "".join(ascii_str_parts)

        self.uart_hex.setPlainText(hex_str)
        self.uart_ascii.setPlainText(ascii_str)

        owner = self._find_owner_with_plot()
        if owner and hasattr(owner, "log"):
            owner.log(f"UART decode: {len(decoded)} bytes")
            for i, fr in enumerate(frames):
                if fr.get("parity_ok") is False:
                    owner.log(f"Parity error at byte {i}: 0x{fr['byte']:02X}")
                if fr.get("framing_error"):
                    owner.log(f"Framing error at byte {i}: 0x{fr['byte']:02X}")

        # overlay markers once (previous code called this twice)
        bit_width_s = 1.0 / float(baud) if baud > 0 else None
        self._overlay_uart_markers(samp_times, bit_levels, frames=frames, bit_width_s=bit_width_s)

    def _overlay_uart_markers(self, sample_times, bit_levels, frames=None, bit_width_s: Optional[float]=None):
        """Plot scatter markers for each sampled bit level (green=1, red=0)."""
        # clear previous points
        try:
            self.uart_scatter.clear()
        except Exception:
            pass

        if not sample_times:
            return

        owner = self._find_owner_with_plot()
        plot_widget = None
        if owner is not None and hasattr(owner, "plot"):
            try:
                plot_widget = owner.plot.plot_widget
            except Exception:
                plot_widget = None
        elif hasattr(self, "window") and self.window() is not None:
            # fallback
            try:
                plot_widget = getattr(self.window().plot, "plot_widget", None)
            except Exception:
                plot_widget = None

        try:
            if plot_widget is not None:
                for it in getattr(self, "uart_frame_items", []) + getattr(self, "uart_text_items", []):
                    try:
                        plot_widget.removeItem(it)
                    except Exception:
                        # also try scene removal
                        try:
                            it.setParentItem(None)
                        except Exception:
                            pass
        except Exception:
            pass
        self.uart_frame_items = []
        self.uart_text_items = []
        self._frames_added_to_plot = False

        if not sample_times:
            return

        # get y-range from model's last waveform safely
        arr = np.asarray(self.model.last_waveform) if self.model.last_waveform is not None else np.array([0.0])
        try:
            ymax = float(np.nanmax(arr))
            ymin = float(np.nanmin(arr))
        except Exception:
            ymax, ymin = 1.0, 0.0
        height = ymax - ymin if ymax != ymin else 1.0

        # x-mode: "time" or "samples" (match the plot)
        x_mode = "samples"
        if owner is not None and hasattr(owner, "plot"):
            x_mode = getattr(owner.plot, "_x_mode", "samples")

        # Show scatter points (existing behavior)
        spots = []
        for t, bit in zip(sample_times, bit_levels):
            if x_mode == "time":
                x = float(t)
            else:
                try:
                    x = int(round(float(t) * float(self.model.sample_rate)))
                except Exception:
                    x = int(round(float(t) if isinstance(t, (int, float)) else 0))
            y = ymax + (0.03 if bit else 0.01) * height
            spots.append({'pos': (x, y), 'brush': 'g' if bit else 'r', 'size': 6})
        # add scatter to plot once
        if plot_widget is not None:
            try:
                if not getattr(self, "_scatter_added_to_plot", False):
                    plot_widget.addItem(self.uart_scatter)
                    self._scatter_added_to_plot = True
            except Exception:
                pass
        try:
            self.uart_scatter.addPoints(spots)
        except Exception:
            pass

        # If frames available, draw vertical shaded regions for each bit (start/data/parity/stop)
        if frames and plot_widget is not None:
            for fi, fr in enumerate(frames):
                # per-frame bit times or indices
                bit_times = fr.get("bit_times", None)
                bit_indices = fr.get("bit_indices", None)

                # if bit_times is present, prefer it; else convert indices to times using sample_rate
                use_times = []
                if bit_times and len(bit_times) > 0:
                    use_times = list(bit_times)
                elif bit_indices and len(bit_indices) > 0:
                    use_times = [float(idx) / float(self.model.sample_rate) for idx in bit_indices]
                else:
                    # no per-bit timing info; skip frame plotting
                    continue

                # assume each entry in use_times is center of the bit; compute left/right using bit_width_s
                # if bit_width_s None, best-effort: use distance to next bit center (or 1/baud if available)
                bw = float(bit_width_s) if bit_width_s else None

                for bi, center_t in enumerate(use_times):
                    # compute left/right
                    if bw is None:
                        # attempt neighbor spacing
                        if bi + 1 < len(use_times):
                            bw = abs(use_times[bi+1] - center_t)
                        elif bi - 1 >= 0:
                            bw = abs(center_t - use_times[bi-1])
                        else:
                            bw = 0.0  # fallback
                    left = center_t - 0.5 * bw
                    right = center_t + 0.5 * bw

                    if x_mode != "time":
                        # convert to sample indices for x-mode
                        left_x = left * float(self.model.sample_rate)
                        right_x = right * float(self.model.sample_rate)
                        x0 = left_x; x1 = right_x
                    else:
                        x0 = left; x1 = right

                    # choose brush/pen depending on bit type and value
                    # Determine bit value if available (data bits are in 'bit_levels' in frame)
                    bit_val = None
                    if "bit_levels" in fr and bi < len(fr["bit_levels"]):
                        bit_val = fr["bit_levels"][bi]
                    # start bit: usually first bit (we assume bit 0 in frame refers to first data bit: so we can't rely on that)
                    # We will color: start (blue), data-high (green), data-low (gray), parity (cyan), stop (yellow)
                    is_parity_bit = False
                    is_stop_bit = False
                    # Estimate positions: frames were constructed as data_bits then parity(if any) then stop_bits.
                    # Compute indices: start bit is not in local_indices; our frame bit list contains data bit centers only.
                    # So to draw start/parity/stop, use frame start_time & bit_width to place regions around expected centers.
                    # Draw data bits using bit_times; draw start region centered at (first data center - bw)
                    # draw stop bits after last data+parity.
                    # Data bit rectangle:
                    brush = pg.mkBrush(150, 150, 150, 50)  # default gray for unknown
                    pen = pg.mkPen(None)
                    label_text = ""
                    # If this center corresponds to a data bit
                    label_text = f"{bi}:?"
                    if "bit_levels" in fr and bi < len(fr["bit_levels"]):
                        val = fr["bit_levels"][bi]
                        label_text = f"b{bi}={val}"
                        if val:
                            brush = pg.mkBrush(0, 180, 0, 80)  # green-ish for logic 1
                        else:
                            brush = pg.mkBrush(120, 120, 120, 60)  # gray for logic 0

                    # Create vertical region (LinearRegionItem) spanning x0..x1
                    try:
                        region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, brush=brush)
                        region.setZValue(800)
                        region.setMovable(False)
                        plot_widget.addItem(region)
                        self.uart_frame_items.append(region)
                    except Exception:
                        # fallback to a thin line if region fails
                        try:
                            ln = pg.InfiniteLine(pos=x0, angle=90, movable=False, pen=pg.mkPen('w'))
                            plot_widget.addItem(ln)
                            self.uart_frame_items.append(ln)
                        except Exception:
                            pass

                    # Label the bit near the top of the waveform
                    try:
                        mid_x = (x0 + x1) / 2.0
                        # choose display x coordinate consistent with x_mode
                        tx = mid_x
                        txt = pg.TextItem(label_text, anchor=(0.5, 0.0))
                        txt.setZValue(900)
                        # place text up above waveform
                        if x_mode == "time":
                            txt.setPos(tx, ymax + 0.02 * height)
                        else:
                            txt.setPos(tx, ymax + 0.02 * height)
                        plot_widget.addItem(txt)
                        self.uart_text_items.append(txt)
                    except Exception:
                        pass

                # Additionally: draw start region and stop regions relative to data bit centers
                try:
                    # compute first data center and last center
                    first_center = use_times[0]
                    last_center = use_times[-1]
                    # start center approx one bit width before first data center
                    if bit_width_s:
                        start_left = (first_center - 0.5 * bit_width_s) - 0.5 * bit_width_s
                        start_right = (first_center - 0.5 * bit_width_s) + 0.5 * bit_width_s
                    else:
                        start_left = first_center - 1.5 * bw
                        start_right = first_center - 0.5 * bw
                    if x_mode != "time":
                        sx = start_left * float(self.model.sample_rate); ex = start_right * float(self.model.sample_rate)
                    else:
                        sx = start_left; ex = start_right
                    sbrush = pg.mkBrush(0, 0, 200, 60)  # blue for start bit
                    start_region = pg.LinearRegionItem(values=(sx, ex), orientation=pg.LinearRegionItem.Vertical, brush=sbrush)
                    start_region.setMovable(False); start_region.setZValue(800)
                    plot_widget.addItem(start_region)
                    self.uart_frame_items.append(start_region)
                    # label start
                    stext = pg.TextItem("START", anchor=(0.5, 0.0))
                    stext.setPos((sx+ex)/2.0, ymax + 0.02 * height)
                    plot_widget.addItem(stext)
                    self.uart_text_items.append(stext)
                except Exception:
                    pass

                # stop bits: number of stop bits in frame may be >1; try to compute positions after last_center
                try:
                    stop_bits = fr.get("bit_times_stop_bits", None)
                    # If decoder doesn't provide explicit stop bit centers, compute them after last_center
                    n_stop = int((1 if fr.get("framing_error") is False else 1))  # default 1
                    # Better: estimate using frame metadata: compute expected stop positions using bit_width_s
                    if bit_width_s:
                        # stop bit centers begin at last_center + 1*bit_width_s (if parity absent)
                        parity_present = (fr.get("parity_ok") is not None and fr.get("parity_ok") is not True) or (parity_mode := None)
                        # We can't reliably infer parity presence here; simpler: draw a stop region after last_center + 0.5*bw
                        st_left = last_center + 0.5 * (bit_width_s)
                        st_right = last_center + 1.5 * (bit_width_s)
                        if x_mode != "time":
                            st0 = st_left * float(self.model.sample_rate); st1 = st_right * float(self.model.sample_rate)
                        else:
                            st0 = st_left; st1 = st_right
                        sbrush2 = pg.mkBrush(220, 200, 40, 40)  # yellow-ish for stop bits
                        stop_region = pg.LinearRegionItem(values=(st0, st1), orientation=pg.LinearRegionItem.Vertical, brush=sbrush2)
                        stop_region.setMovable(False); stop_region.setZValue(800)
                        plot_widget.addItem(stop_region)
                        self.uart_frame_items.append(stop_region)
                        # label stop
                        stop_text = pg.TextItem("STOP", anchor=(0.5, 0.0))
                        stop_text.setPos((st0+st1)/2.0, ymax + 0.02 * height)
                        plot_widget.addItem(stop_text)
                        self.uart_text_items.append(stop_text)
                except Exception:
                    pass

                # Finally draw error badges for this frame if any
                try:
                    badge_msgs = []
                    if fr.get("parity_ok") is False:
                        badge_msgs.append("PARITY")
                    if fr.get("framing_error"):
                        badge_msgs.append("FRAMING")
                    if badge_msgs:
                        badge_text = "|".join(badge_msgs)
                        # place at the right end of the frame
                        # find frame start/end in x coords
                        frame_start = fr.get("start_time", None)
                        frame_end = fr.get("end_time", None)
                        if frame_start is not None and frame_end is not None:
                            if x_mode != "time":
                                bx = frame_end * float(self.model.sample_rate)
                            else:
                                bx = frame_end
                        else:
                            # fallback near last_center
                            bx = (x0 + x1) / 2.0
                        btxt = pg.TextItem(badge_text, color=(200, 20, 20), anchor=(1.0, 0.0))
                        btxt.setZValue(1200)
                        btxt.setPos(bx, ymax + 0.06 * height)
                        plot_widget.addItem(btxt)
                        self.uart_text_items.append(btxt)
                except Exception:
                    pass

            # done frames loop
            self._frames_added_to_plot = True

    def clear_uart_overlays(self):
        """Remove UART scatter points, frame rectangles and text badges from the plot."""
        try:
            self.uart_scatter.clear()
        except Exception:
            pass

        owner = self._find_owner_with_plot()
        plot_widget = None
        if owner is not None and hasattr(owner, "plot"):
            try:
                plot_widget = owner.plot.plot_widget
            except Exception:
                plot_widget = None
        else:
            # fallback to top-level window's plot if present
            try:
                plot_widget = getattr(self.window().plot, "plot_widget", None)
            except Exception:
                plot_widget = None

        try:
            for item in getattr(self, "uart_frame_items", []) + getattr(self, "uart_text_items", []):
                try:
                    if plot_widget is not None:
                        plot_widget.removeItem(item)
                    else:
                        # try a gentle detach if the above fails
                        try:
                            item.setParentItem(None)
                        except Exception:
                            pass
                except Exception:
                    # best-effort removal; ignore failures
                    try:
                        item.setParentItem(None)
                    except Exception:
                        pass
        except Exception:
            pass

        # reset tracking lists/flags
        self.uart_frame_items = []
        self.uart_text_items = []
        self._scatter_added_to_plot = False
        self._frames_added_to_plot = False

# -------------------- Trigger Panel --------------------
class TriggerPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Trigger</b>"))

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Immediate", "Edge"])
        layout.addWidget(self.mode_combo)

        edge_row = QtWidgets.QHBoxLayout()
        self.edge_dir_combo = QtWidgets.QComboBox()
        self.edge_dir_combo.addItems(["Rising", "Falling"])
        edge_row.addWidget(QtWidgets.QLabel("Edge:"))
        edge_row.addWidget(self.edge_dir_combo)
        layout.addLayout(edge_row)

        level_row = QtWidgets.QHBoxLayout()
        self.level_spin = QtWidgets.QDoubleSpinBox()
        self.level_spin.setRange(-1e6, 1e6)
        self.level_spin.setDecimals(6)
        self.level_spin.setValue(0.0)
        level_row.addWidget(QtWidgets.QLabel("Level (V):"))
        level_row.addWidget(self.level_spin)
        layout.addLayout(level_row)

        src_row = QtWidgets.QHBoxLayout()
        self.source_edit = QtWidgets.QLineEdit()
        self.source_edit.setPlaceholderText("eg: 0, 1, PFI0")
        self.source_edit.setToolTip("Trigger source (channel number or external terminal). Example: '0' or 'PFI0'")
        src_row.addWidget(QtWidgets.QLabel("Source:"))
        src_row.addWidget(self.source_edit)
        layout.addLayout(src_row)

        hd_row = QtWidgets.QHBoxLayout()
        self.holdoff_spin = QtWidgets.QDoubleSpinBox()
        self.holdoff_spin.setRange(0.0, 10.0)
        self.holdoff_spin.setDecimals(6)
        self.holdoff_spin.setSingleStep(0.001)
        self.holdoff_spin.setValue(0.0)
        self.holdoff_spin.setSuffix(" s")
        self.holdoff_spin.setToolTip("Holdoff time (seconds) between triggers")

        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 10.0)
        self.delay_spin.setDecimals(6)
        self.delay_spin.setSingleStep(0.001)
        self.delay_spin.setValue(0.0)
        self.delay_spin.setSuffix(" s")
        self.delay_spin.setToolTip("Delay (seconds) after trigger before reference event")

        hd_row.addWidget(QtWidgets.QLabel("Holdoff:"))
        hd_row.addWidget(self.holdoff_spin)
        hd_row.addWidget(QtWidgets.QLabel("Delay:"))
        hd_row.addWidget(self.delay_spin)
        layout.addLayout(hd_row)
        layout.addStretch(1)

        self._on_mode_changed(self.mode_combo.currentText())
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

    def _on_mode_changed(self, text):
        is_edge = text == "Edge"
        self.edge_dir_combo.setEnabled(is_edge)
        self.level_spin.setEnabled(is_edge)

    def get_config(self) -> dict:
        return {
            "mode": self.mode_combo.currentText(),
            "edge_direction": self.edge_dir_combo.currentText(),
            "level": float(self.level_spin.value()),
            "source": self.source_edit.text().strip(),
            "holdoff": float(self.holdoff_spin.value()),
            "delay": float(self.delay_spin.value()),
        }

# -------------------- Plot Panel (fixed cursor colors, reset on Off) --------------------
class PlotPanel(QtWidgets.QFrame):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._build_ui()
        self._create_cursors_and_handles()
        self._cursor_sync_in_progress = False

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Waveform</b>")
        hl.addWidget(title)
        hl.addStretch(1)

        # Cursor mode controls
        hl.addWidget(QtWidgets.QLabel("Cursor:"))
        self.cursor_mode = QtWidgets.QComboBox()
        self.cursor_mode.addItems(["Off", "Vertical", "Horizontal", "Both"])
        hl.addWidget(self.cursor_mode)

        hl.addStretch(1)
        layout.addLayout(hl)

        # plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', "Voltage", units='V')
        self.plot_widget.setLabel('bottom', "Sample")
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget, stretch=1)

        # curves
        self.curve_items: Dict[str, pg.PlotDataItem] = {}
        pen0 = pg.mkPen(color=(200, 40, 40), width=1.5)
        pen1 = pg.mkPen(color=(40, 120, 220), width=1.5)
        self.curve_items['0'] = self.plot_widget.plot([], pen=pen0, name='CH0')
        self.curve_items['1'] = self.plot_widget.plot([], pen=pen1, name='CH1')

        # crosshair (mouse)
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(width=1, style=QtCore.Qt.DotLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(width=1, style=QtCore.Qt.DotLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # control row
        ctrls = QtWidgets.QHBoxLayout()
        self.autoscale_btn = QtWidgets.QPushButton("Autoscale Y")
        self.save_image_btn = QtWidgets.QPushButton("Save Image")
        self.export_csv_btn = QtWidgets.QPushButton("Export CSV")
        ctrls.addWidget(self.autoscale_btn)
        ctrls.addWidget(self.save_image_btn)
        ctrls.addWidget(self.export_csv_btn)
        ctrls.addStretch(1)

        # mouse readout
        self.mouse_readout = QtWidgets.QLabel("x=-- , y=--")
        font = QFont()
        font.setStyleHint(QFont.TypeWriter)
        self.mouse_readout.setFont(font)
        self.mouse_readout.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self.mouse_readout.setFixedHeight(22)
        ctrls.addWidget(self.mouse_readout)

        # cursor delta readout
        self.cursor_readout = QtWidgets.QLabel("")
        self.cursor_readout.setFixedHeight(22)
        self.cursor_readout.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        ctrls.addWidget(self.cursor_readout)

        layout.addLayout(ctrls)
        self.cursor_mode.currentTextChanged.connect(self._on_cursor_mode_changed)

        # fixed cursor colors (no picker)
        self._v_color = (255, 165, 0)  # orange
        self._h_color = (0, 200, 0)    # green

        # --- wheel-event for horizontal/time zoom by default ---
        try:
            vb = self.plot_widget.getViewBox()
            import types
            def _custom_wheel_event(self, ev, axis=None):
                try:
                    # angleDelta().y() is standard; fallback to older ev.delta()
                    delta = int(ev.angleDelta().y()) if hasattr(ev, "angleDelta") else int(ev.delta())
                except Exception:
                    ev.ignore()
                    return

                if delta == 0:
                    ev.ignore()
                    return

                # 120 is one notch; steps may be fractional if high-res wheel
                steps = delta / 120.0
                zoom_step = 0.9                   # per-notch multiplicative factor (tweakable)
                zoom_factor = float(zoom_step) ** steps

                # Default: plain wheel => X-axis zoom only
                modifiers = ev.modifiers()
                sx = zoom_factor
                sy = 1.0
                if modifiers & Qt.ShiftModifier:
                    # Shift => Y-axis only
                    sx = 1.0
                    sy = zoom_factor
                elif modifiers & Qt.ControlModifier:
                    # Ctrl => uniform (both axes)
                    sx = zoom_factor
                    sy = zoom_factor

                # center the zoom at the mouse cursor position in data coords
                try:
                    center = self.mapSceneToView(ev.scenePos())
                except Exception:
                    center = None

                try:
                    if center is not None:
                        self.scaleBy((sx, sy), center=center)
                    else:
                        self.scaleBy((sx, sy))
                    ev.accept()
                except Exception:
                    ev.ignore()

            vb.wheelEvent = types.MethodType(_custom_wheel_event, vb)
        except Exception:
            pass


    def _create_cursors_and_handles(self):
        if not hasattr(self, "v_c1") or self.v_c1 is None:
            self.v_c1 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color=self._v_color, width=2))
        if not hasattr(self, "v_c2") or self.v_c2 is None:
            self.v_c2 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color=self._v_color, width=2))
        if not hasattr(self, "h_c1") or self.h_c1 is None:
            self.h_c1 = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color=self._h_color, width=2))
        if not hasattr(self, "h_c2") or self.h_c2 is None:
            self.h_c2 = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color=self._h_color, width=2))

        # Ensure each item is added to the plot scene only once and connected once
        for c in (self.v_c1, self.v_c2, self.h_c1, self.h_c2):
            # make invisible by default
            try:
                c.setVisible(False)
            except Exception:
                pass

            # addItem only if item not already in a scene
            try:
                if getattr(c, "scene", lambda: None)() is None:
                    # add with ignoreBounds True so they don't change autoscale
                    self.plot_widget.addItem(c, ignoreBounds=True)
            except Exception:
                # fallback: try addItem directly
                try:
                    self.plot_widget.addItem(c)
                except Exception:
                    pass

            # connect signal only once
            if not getattr(c, "_sig_connected", False):
                try:
                    c.sigPositionChanged.connect(self._on_line_moved)
                    c._sig_connected = True
                except Exception:
                    pass

            # ensure they draw above plot traces
            try:
                c.setZValue(1000)
            except Exception:
                pass

        # mark handles not placed until we know the view rect
        self._initialized_handles = False

    def _place_handles_near_top(self):
        vb = self.plot_widget.plotItem.vb
        try:
            view_rect = vb.viewRect()
        except Exception:
            view_rect = None

        if view_rect is None:
            # view rect not ready yet; schedule placement shortly (one-shot)
            QtCore.QTimer.singleShot(50, self._place_handles_near_top)
            return

        xmin = view_rect.left(); xmax = view_rect.right()
        ymin = view_rect.top(); ymax = view_rect.bottom()
        x_span = xmax - xmin if (xmax - xmin) != 0 else 1.0
        y_span = ymax - ymin if (ymax - ymin) != 0 else 1.0

        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        # small offsets so cursors are not exactly overlapping
        x_off = max(x_span * 0.05, max(1e-6, x_span * 1e-6))
        y_off = max(y_span * 0.05, max(1e-6, y_span * 1e-6))

        # set sensible starting positions safely (ignore any exception)
        try:
            # ensure the lines are added before setting position
            for c in (self.v_c1, self.v_c2, self.h_c1, self.h_c2):
                if getattr(c, "scene", lambda: None)() is None:
                    try:
                        self.plot_widget.addItem(c, ignoreBounds=True)
                    except Exception:
                        pass

            self.v_c1.setPos(center_x - x_off)
            self.v_c2.setPos(center_x + x_off)
            self.h_c1.setPos(center_y - y_off)
            self.h_c2.setPos(center_y + y_off)
        except Exception:
            # fallback: simple numeric set without offset
            try:
                self.v_c1.setPos(center_x)
                self.v_c2.setPos(center_x + 1.0)
                self.h_c1.setPos(center_y)
                self.h_c2.setPos(center_y + 1.0)
            except Exception:
                pass

        self._initialized_handles = True

    def _on_line_moved(self):
        try:
           if not self._initialized_handles:
               self._place_handles_near_top()
        except Exception:
           pass

        self._update_cursor_readout()
    
    def _on_cursor_mode_changed(self, mode):
        # Reset behavior when turned Off
        if not all(hasattr(self, a) for a in ("v_c1", "v_c2", "h_c1", "h_c2")):
            self._create_cursors_and_handles()

        # If turning Off: hide all and reset initialization flag
        if mode == "Off":
            for c in (self.v_c1, self.v_c2, self.h_c1, self.h_c2):
                try:
                    c.setVisible(False)
                except Exception:
                    pass
            self._initialized_handles = False
            # schedule a placement when turned on next time
            QtCore.QTimer.singleShot(50, self._place_handles_near_top)
            self._update_cursor_readout()
            return

        # Ensure handles are placed (if not yet)
        if not self._initialized_handles:
            self._place_handles_near_top()

        # Ensure all required items are added to the scene (defensive)
        for c in (self.v_c1, self.v_c2, self.h_c1, self.h_c2):
            try:
                if getattr(c, "scene", lambda: None)() is None:
                    self.plot_widget.addItem(c, ignoreBounds=True)
            except Exception:
                pass

        # Show/hide according to mode
        try:
            if mode == "Vertical":
                self.v_c1.setVisible(True); self.v_c2.setVisible(True)
                self.h_c1.setVisible(False); self.h_c2.setVisible(False)
            elif mode == "Horizontal":
                self.h_c1.setVisible(True); self.h_c2.setVisible(True)
                self.v_c1.setVisible(False); self.v_c2.setVisible(False)
            elif mode == "Both":
                for c in (self.v_c1, self.v_c2, self.h_c1, self.h_c2):
                    c.setVisible(True)
        except Exception:
            pass

        # Safety: if the two vertical cursors are exactly equal (overlap), nudge v_c2 a bit
        try:
            if (mode in ("Vertical", "Both")) and self.v_c1.isVisible() and self.v_c2.isVisible():
                x1 = float(self.v_c1.value())
                x2 = float(self.v_c2.value())
                if abs(x2 - x1) < 1e-12:
                    # use view span to determine a small nudge
                    vb = self.plot_widget.plotItem.vb
                    vr = vb.viewRect()
                    if vr is not None:
                        xspan = max(1.0, vr.right() - vr.left())
                        self.v_c2.setPos(x1 + max(1e-6, xspan * 0.02))
                    else:
                        self.v_c2.setPos(x1 + 1.0)
        except Exception:
            pass

        self._update_cursor_readout()
    
    def _format_time_value(self, seconds: float) -> str:
        """Format seconds into s / ms / µs / ns with a compact number."""
        if not np.isfinite(seconds):
            return "NaN"
        s = abs(seconds)
        if s >= 1.0:
            return f"{seconds:.6g} s"
        if s >= 1e-3:
            return f"{seconds * 1e3:.6g} ms"
        if s >= 1e-6:
            return f"{seconds * 1e6:.6g} µs"
        return f"{seconds * 1e9:.6g} ns"

    def _update_cursor_readout(self):
        mode = self.cursor_mode.currentText()
        try:
            if mode == "Vertical":
                if not (self.v_c1.isVisible() and self.v_c2.isVisible()):
                    self.cursor_readout.setText("")
                    return
                x1 = float(self.v_c1.value()); x2 = float(self.v_c2.value())
                dx = abs(x2 - x1)
                if getattr(self, "_x_mode", "samples") == "time":
                    self.cursor_readout.setText(f"ΔX = {self._format_time_value(dx)}")
                else:
                    self.cursor_readout.setText(f"ΔX = {dx:.0f} samples")
            elif mode == "Horizontal":
                if not (self.h_c1.isVisible() and self.h_c2.isVisible()):
                    self.cursor_readout.setText("")
                    return
                y1 = float(self.h_c1.value()); y2 = float(self.h_c2.value())
                dy = abs(y2 - y1)
                self.cursor_readout.setText(f"ΔY = {dy:.6g} V")
            elif mode == "Both":
                x1 = float(self.v_c1.value()); x2 = float(self.v_c2.value())
                y1 = float(self.h_c1.value()); y2 = float(self.h_c2.value())
                dx = abs(x2 - x1); dy = abs(y2 - y1)
                if getattr(self, "_x_mode", "samples") == "time":
                    dx_str = self._format_time_value(dx)
                else:
                    dx_str = f"{dx:.0f} samples"
                self.cursor_readout.setText(f"ΔX={dx_str}, ΔY={dy:.6g} V")
            else:
                self.cursor_readout.setText("")
        except Exception:
            self.cursor_readout.setText("")

    def _on_mouse_moved(self, pos):
        try:
            vb = self.plot_widget.plotItem.vb
            if not vb.sceneBoundingRect().contains(pos):
                return
            mouse_point = vb.mapSceneToView(pos)
            x = float(mouse_point.x())
            y = float(mouse_point.y())
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            if getattr(self, "_x_mode", "samples") == "time":
                x_str = self._format_time_value(x)
                self.mouse_readout.setText(f"x={x_str} s, y={y:.6g} V")
            else:
                self.mouse_readout.setText(f"sample={x:.0f}, y={y:.6g} V")
        except Exception:
            pass

    # Public helpers used by main window / controller to set X-mode and update curves
    def set_x_mode(self, mode: str):
        self._x_mode = mode
        if mode == "time":
            self.plot_widget.setLabel('bottom', "Time", units='s')
        else:
            self.plot_widget.setLabel('bottom', "Samples")

    def update_curve(self, channel: str, y: np.ndarray):
        if channel not in self.curve_items:
            return
        y = np.asarray(y)
        try:
            if getattr(self, "_x_mode", "samples") == "time":
                x = np.arange(y.size) / float(self.model.sample_rate)
            else:
                x = np.arange(y.size)
            self.curve_items[channel].setData(x, y)
            if x.size:
                self.plot_widget.setXRange(float(np.min(x)), float(np.max(x)))
        except Exception:
            pass

    def autoscale_y(self):
        data = getattr(self.model, "last_waveform", None)
        if data is None:
            return
        arr = np.asarray(data)
        try:
            ymin = float(np.nanmin(arr))
            ymax = float(np.nanmax(arr))
            margin = max(1e-6, 0.15 * (ymax - ymin))
            self.plot_widget.setYRange(ymin - margin, ymax + margin)
        except Exception:
            pass

    def save_plot_image(self, fname: Optional[str] = None):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save plot image", "waveform.png",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        pixmap = self.plot_widget.grab()
        pixmap.save(fname)

# -------------------- Measurements Panel --------------------
class MeasurementsPanel(QtWidgets.QFrame):
    METRIC_DISPLAY = {
        "amplitude_v": "Amplitude (V)",
        "pkpk_v": "Peak-to-peak (V)",
        "frequency_hz": "Frequency (Hz)",
        "rise_time_s": "Rise (s)",
        "fall_time_s": "Fall (s)",
        "rms_v": "RMS (V)",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.measure_table_config: Dict[str, List[str]] = {}
        self.measure_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._measure_row_map: Dict[Tuple[str, str], int] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("<b>Measurements</b>"))
        header.addStretch(1)
        self.addremove_btn = QtWidgets.QPushButton("Add/Remove")
        header.addWidget(self.addremove_btn)
        layout.addLayout(header)

        self.measure_table = QtWidgets.QTableWidget(0, 7)
        self.measure_table.setHorizontalHeaderLabels(
            ["Channel", "Measurement", "Value", "Mean", "Minimum", "Maximum", "Count"]
        )
        self.measure_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.measure_table.verticalHeader().setVisible(False)
        header_h = self.measure_table.horizontalHeader()
        header_h.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.measure_table.setColumnWidth(0, 80)
        self.measure_table.setColumnWidth(1, 120)
        self.measure_table.setColumnWidth(2, 110)
        self.measure_table.setColumnWidth(3, 110)
        self.measure_table.setColumnWidth(4, 110)
        self.measure_table.setColumnWidth(5, 110)
        self.measure_table.setColumnWidth(6, 70)
        layout.addWidget(self.measure_table)

        # Connect Add/Remove (single connection)
        self.addremove_btn.clicked.connect(self.open_measure_config_dialog)

    def open_measure_config_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add/Remove Measurements")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel("Select channels and measurements to show:"))
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        ch0 = QtWidgets.QCheckBox("CH0")
        ch1 = QtWidgets.QCheckBox("CH1")
        ch0.setChecked("0" in self.measure_table_config)
        ch1.setChecked("1" in self.measure_table_config)
        grid.addWidget(QtWidgets.QLabel("<b>Channels</b>"), 0, 0)
        grid.addWidget(ch0, 0, 1)
        grid.addWidget(ch1, 0, 2)

        grid.addWidget(QtWidgets.QLabel("<b>Measurements</b>"), 1, 0)
        metric_cbs = {}
        r = 2
        for key, disp in self.METRIC_DISPLAY.items():
            cb = QtWidgets.QCheckBox(disp)
            cb.setChecked(key in (m for ch in self.measure_table_config.values() for m in ch))
            metric_cbs[key] = cb
            grid.addWidget(cb, r, 0, 1, 3)
            r += 1

        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("OK"); cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch(1); btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        selected_chs = []
        if ch0.isChecked(): selected_chs.append("0")
        if ch1.isChecked(): selected_chs.append("1")
        selected_metrics = [k for k, cb in metric_cbs.items() if cb.isChecked()]

        new_config = {ch: list(selected_metrics) for ch in selected_chs}
        self.measure_table_config = new_config
        allowed_keys = set((ch, m) for ch in self.measure_table_config for m in self.measure_table_config[ch])
        for k in list(self.measure_stats.keys()):
            if k not in allowed_keys:
                self.measure_stats.pop(k, None)
        self._rebuild_rows()

    def _rebuild_rows(self):
        self.measure_table.setRowCount(0)
        self._measure_row_map.clear()
        channels = sorted(self.measure_table_config.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for ch in channels:
            for metric_key in self.measure_table_config[ch]:
                r = self.measure_table.rowCount()
                self.measure_table.insertRow(r)
                item_ch = QtWidgets.QTableWidgetItem(f"CH{ch}")
                item_ch.setFlags(item_ch.flags() ^ QtCore.Qt.ItemIsEditable)
                self.measure_table.setItem(r, 0, item_ch)
                item_m = QtWidgets.QTableWidgetItem(self.METRIC_DISPLAY.get(metric_key, metric_key))
                item_m.setFlags(item_m.flags() ^ QtCore.Qt.ItemIsEditable)
                self.measure_table.setItem(r, 1, item_m)
                self.measure_table.setItem(r, 2, QtWidgets.QTableWidgetItem("--"))
                self.measure_table.setItem(r, 3, QtWidgets.QTableWidgetItem("--"))
                self.measure_table.setItem(r, 4, QtWidgets.QTableWidgetItem("--"))
                self.measure_table.setItem(r, 5, QtWidgets.QTableWidgetItem("--"))
                self.measure_table.setItem(r, 6, QtWidgets.QTableWidgetItem("0"))
                self._measure_row_map[(ch, metric_key)] = r

    def update_measurements(self, metrics_list: List[Dict[str, float]], ch_list: List[str]):
        if not self.measure_table_config:
            return
        try:
            for idx, metrics in enumerate(metrics_list):
                ch = ch_list[idx] if idx < len(ch_list) else str(idx)
                keys = self.measure_table_config.get(ch, [])
                for k in keys:
                    v = metrics.get(k, float("nan"))
                    try:
                        val = float(v)
                    except Exception:
                        val = float("nan")
                    key = (ch, k)
                    st = self.measure_stats.get(key)
                    if st is None:
                        st = {'last': float('nan'), 'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                        self.measure_stats[key] = st
                    if np.isfinite(val):
                        st['last'] = val
                        st['sum'] += val
                        st['count'] += 1
                        if val < st['min']: st['min'] = val
                        if val > st['max']: st['max'] = val
                    else:
                        st['last'] = float('nan')
                    mean = st['sum'] / st['count'] if st['count'] > 0 else float('nan')
                    minimum = st['min'] if st['min'] != float('inf') else float('nan')
                    maximum = st['max'] if st['max'] != float('-inf') else float('nan')
                    row = self._measure_row_map.get(key)
                    if row is not None:
                        self.measure_table.item(row, 2).setText(f"{st['last']:.6g}" if np.isfinite(st['last']) else "NaN")
                        self.measure_table.item(row, 3).setText(f"{mean:.6g}" if np.isfinite(mean) else "NaN")
                        self.measure_table.item(row, 4).setText(f"{minimum:.6g}" if np.isfinite(minimum) else "NaN")
                        self.measure_table.item(row, 5).setText(f"{maximum:.6g}" if np.isfinite(maximum) else "NaN")
                        self.measure_table.item(row, 6).setText(str(st['count']))
        except Exception:
            pass

    def clear(self):
        self.measure_stats.clear()
        self._measure_row_map.clear()
        self.measure_table.setRowCount(0)
        self.measure_table_config.clear()


# -------------------- Log Panel --------------------
class LogPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Event Log</b>"))
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumWidth(260)
        layout.addWidget(self.log_view, stretch=1)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.clear_log_btn = QtWidgets.QPushButton("Clear Log")
        btn_row.addWidget(self.clear_log_btn)
        layout.addLayout(btn_row)


# -------------------- Main Window --------------------
class OscilloMainWindow(QtWidgets.QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.setWindowTitle("Oscilloscope Waveform Analyzer")
        self.resize(1280, 860)
        self.setStyleSheet(self.load_osci_stylesheet())
        pg.setConfigOptions(antialias=True)

        # Components
        self.controls = ControlsPanel(self.model)
        self.trigger = TriggerPanel()
        self.plot = PlotPanel(self.model)
        self.measurements = MeasurementsPanel()
        self.logpanel = LogPanel()
        self.uart_panel = UARTPanel(self.model)

        self._compose_ui()
        self._build_toolbar()
        self._set_toolbar_style_and_sizes() 
        self._build_menu_view()
        self._wire_defaults()

    def _make_dock(self, widget: QtWidgets.QWidget, title: str, area=QtCore.Qt.LeftDockWidgetArea, allowed=QtCore.Qt.AllDockWidgetAreas) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title.replace(" ", "_"))
        dock.setWidget(widget)
        dock.setAllowedAreas(allowed)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(area, dock)
        return dock

    def _compose_ui(self):
        # Central area: vertical splitter with Plot on top, Measurements middle, Params bottom
        central_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        central_splitter.setContentsMargins(6, 6, 6, 6)
        central_splitter.setHandleWidth(8)

        # Plot container
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot, stretch=1)

        # Param table container (history) below measurements
        params_container = QtWidgets.QWidget()
        params_layout = QtWidgets.QVBoxLayout(params_container)
        params_layout.setContentsMargins(6, 4, 6, 6)
        params_layout.addWidget(QtWidgets.QLabel("Captured Parameters (history)"))
        if not hasattr(self, "param_table") or self.param_table is None:
            self.param_table = QtWidgets.QTableWidget(0, 1)
            self.param_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.param_table.horizontalHeader().setStretchLastSection(True)
        params_layout.addWidget(self.param_table, stretch=1)

        # Add containers to central splitter (plot gets most space)
        central_splitter.addWidget(plot_container)
        central_splitter.addWidget(params_container)
        central_splitter.setStretchFactor(0, 4)
        central_splitter.setStretchFactor(1, 1)

        # Set splitter as central widget
        self.setCentralWidget(central_splitter)

        # Left docks: Controls and Trigger (tabified)
        self.controls_dock = self._make_dock(self.controls, "Controls", QtCore.Qt.LeftDockWidgetArea)
        self.trigger_dock  = self._make_dock(self.trigger,  "Trigger",  QtCore.Qt.LeftDockWidgetArea)
        # try:
        #     self.tabifyDockWidget(self.controls_dock, self.trigger_dock)
        #     self.controls_dock.raise_()
        # except Exception:
        #     pass

        self.measure_dock = self._make_dock(self.measurements, "Measurements", QtCore.Qt.RightDockWidgetArea)
        self.log_dock = self._make_dock(self.logpanel, "Event Log", QtCore.Qt.RightDockWidgetArea)
        # try:
        #     self.tabifyDockWidget(self.measure_dock, self.log_dock)
        #     self.measure_dock.raise_()
        # except Exception:
        #     pass

        # Minimum sizes to avoid accidental collapse and ensure responsive behavior
        try:
            self.controls.setMinimumWidth(240)
            self.trigger.setMinimumWidth(240)
            self.measurements.setMinimumHeight(300)
            self.param_table.setMinimumHeight(120)
            self.logpanel.setMinimumHeight(160)
        except Exception:
            pass

        self.uart_dock = self._make_dock(self.uart_panel, "UART Decoder", QtCore.Qt.RightDockWidgetArea)

        # Ensure docks are visible initially
        self.controls_dock.show()
        self.trigger_dock.show()
        self.measure_dock.show()
        self.log_dock.show()
        self.uart_dock.show()

    def _build_toolbar(self):
        self.toolbar = self.addToolBar("Transport")
        # toggle Start/Stop (checkable)
        self.toggle_start_action = QAction(QIcon(), "Start", self)
        self.toggle_start_action.setCheckable(True)
        self.toggle_start_action.toggled.connect(self._on_toggle_text)
        self.toolbar.addAction(self.toggle_start_action)

        # single
        self.single_action = QAction(QIcon(), "Single", self)
        self.toolbar.addAction(self.single_action)

        # clear display
        self.clear_display_action = QAction(QIcon(), "Clear Display", self)
        self.toolbar.addAction(self.clear_display_action)
        self.toolbar.addSeparator()

    def _set_toolbar_style_and_sizes(self):
        self.toolbar.setIconSize(QSize(36, 28))
        self.toolbar.setStyleSheet("""
            QToolButton { font-size: 13px; padding: 8px 12px; min-width: 90px; min-height: 36px; }
            QToolButton:checked { background: #efefef; }
        """)
        def adjust_button_sizes():
            for act in self.toolbar.actions():
                widget = self.toolbar.widgetForAction(act)
                if widget is not None:
                    widget.setMinimumSize(90, 36)
        QtCore.QTimer.singleShot(20, adjust_button_sizes)

    def _build_menu_view(self):
        menubar = self.menuBar()
        view_menu = menubar.addMenu("&View")

        # X-axis mode actions (kept as before)
        group = QActionGroup(self)
        group.setExclusive(True)
        act_samples = QAction("Voltage vs Samples", self, checkable=True)
        act_time = QAction("Voltage vs Time", self, checkable=True)
        act_time.setChecked(True)
        group.addAction(act_samples); group.addAction(act_time)
        view_menu.addAction(act_samples); view_menu.addAction(act_time)
        act_samples.triggered.connect(lambda _: self.plot.set_x_mode("samples"))
        act_time.triggered.connect(lambda _: self.plot.set_x_mode("time"))

        try:
            self.plot.set_x_mode("time")
        except Exception:
            pass

        view_menu.addSeparator()

        # Panels submenu to show/hide docks
        panels_menu = view_menu.addMenu("Panels")
        panels_menu.setToolTipsVisible(True)

        # Utility to create a checkable action bound to a dock's visibility
        def _make_panel_action(title: str, dock_widget: QtWidgets.QDockWidget):
            action = QAction(title, self, checkable=True)
            action.setChecked(bool(dock_widget.isVisible()))
            action.setToolTip(f"Show or hide the '{title}' panel")
            # When action toggled, set dock visibility
            action.triggered.connect(lambda checked, d=dock_widget: d.setVisible(checked))
            # Keep action checked state in sync when the dock's visibility changes
            try:
                dock_widget.visibilityChanged.connect(action.setChecked)
            except Exception:
                # If signal not available, ignore (some Qt versions may differ)
                pass
            panels_menu.addAction(action)
            return action

        # Only create actions for dock widgets (not for measurements which is central now)
        if hasattr(self, "controls_dock"):
            _make_panel_action("Controls", self.controls_dock)
        if hasattr(self, "trigger_dock"):
            _make_panel_action("Trigger", self.trigger_dock)
        if hasattr(self, "measure_dock"):
            _make_panel_action("Measurements", self.measure_dock)
        if hasattr(self, "log_dock"):
            _make_panel_action("Event Log", self.log_dock)
        if hasattr(self, "uart_dock"):
            _make_panel_action("UART Decoder", self.uart_dock)

        # Add a small helper action to restore default layout if needed
        restore_action = QAction("Restore all panel", self)
        restore_action.setToolTip("Restore or show all panels")
        def _restore_layout():
            # show all docks and re-tabify left docks
            try:
                if hasattr(self, "controls_dock"): self.controls_dock.show()
                if hasattr(self, "trigger_dock"): self.trigger_dock.show()
                if hasattr(self, "log_dock"): self.log_dock.show()
                if hasattr(self, "measure_dock"): self.measure_dock.show()
                if hasattr(self, "controls_dock") and hasattr(self, "trigger_dock"):
                    try:
                        # self.tabifyDockWidget(self.controls_dock, self.trigger_dock)
                        self.controls_dock.raise_()
                        self.trigger_dock.raise_()
                    except Exception:
                        pass
                if hasattr(self, "measure_dock") and hasattr(self, "log_dock"):
                    try:
                        # self.tabifyDockWidget(self.measure_dock, self.log_dock)
                        self.measure_dock.raise_()
                        self.log_dock.raise_()
                    except Exception:
                        pass
            except Exception:
                pass
        restore_action.triggered.connect(_restore_layout)
        panels_menu.addSeparator()
        panels_menu.addAction(restore_action)


    def _wire_defaults(self):
        self.clear_display_action.triggered.connect(self.clear_plot)
        self.controls.clear_display_btn.clicked.connect(self.clear_plot)
        self.controls.reset_btn.clicked.connect(self.reset_ui)
        self.logpanel.clear_log_btn.clicked.connect(self.clear_log)
        self.plot.autoscale_btn.clicked.connect(self.plot.autoscale_y)
        self.plot.save_image_btn.clicked.connect(self.plot.save_plot_image)
        self.plot.export_csv_btn.clicked.connect(self.export_csv)
        self.uart_panel.decode_btn.clicked.connect(self.uart_panel.on_decode_uart)
        self.uart_panel.clear_overlays_btn.clicked.connect(self.uart_panel.clear_uart_overlays)

    def _on_toggle_text(self, checked: bool):
        self.toggle_start_action.setText("Stop" if checked else "Start")
        self.toggle_start_action.setToolTip("Stop" if checked else "Start")

    def update_param_table_headers(self, channels: List[str]):
        if not channels:
            channels = ["0"]
        cols = 1 + len(channels) * 6
        self.param_table.setColumnCount(cols)
        headers = ["Timestamp"]
        for ch in channels:
            headers += [
                f"CH{ch} Freq(Hz)",
                f"CH{ch} Amp(V)",
                f"CH{ch} Rise(s)",
                f"CH{ch} Fall(s)",
                f"CH{ch} PkPk(V)",
                f"CH{ch} RMS(V)",
            ]
        self.param_table.setHorizontalHeaderLabels(headers)
        self.param_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)

    def update_measurement_table(self, metrics_list: List[Dict[str, float]], ch_list: List[str]):
        self.measurements.update_measurements(metrics_list, ch_list)

    def update_curve(self, channel: str, y: np.ndarray):
        self.plot.update_curve(channel, y)

    def clear_plot(self):
        try:
            for item in self.plot.curve_items.values():
                item.setData([], [])
                item.setVisible(False)
            self.plot.plot_widget.enableAutoRange(True)
            self.model.last_waveform = None
            self.model.last_timestamp = None
            self.param_table.setRowCount(0)
            self.measurements.clear()
            self.log("Display cleared")
            if getattr(self, "toggle_start_action", None) and self.toggle_start_action.isChecked():
                self.toggle_start_action.setChecked(False)
            try:
                if hasattr(self, "uart_panel") and callable(getattr(self.uart_panel, "clear_uart_overlays", None)):
                    self.uart_panel.clear_uart_overlays()
            except Exception:
                pass
        except Exception as ex:
            self.log(f"Clear plot error: {ex}")

    def clear_log(self):
        self.logpanel.log_view.clear()
        self.log("Event log cleared")

    def reset_ui(self):
        try:
            self.controls.resource_edit.setText(self.model.resource_name)
            chs = [s.strip() for s in (self.model.channels or "").split(",") if s.strip()]
            self.controls.ch0_cb.setChecked("0" in chs)
            self.controls.ch1_cb.setChecked("1" in chs)
            self.controls.voltage_spin.setValue(self.model.voltage_range)
            self.controls.sample_rate_edit.setText(str(int(self.model.sample_rate / 1e3)))
            self.controls.fetch_time_edit.setText(str(self.model.fetch_time))
            self.controls.simulate_cb.setChecked(self.model.simulate)
            self.plot.plot_widget.enableAutoRange(True)
            self.plot.plot_widget.setYRange(-self.controls.voltage_spin.maximum(), self.controls.voltage_spin.maximum())
            self.log("UI reset to model defaults")
            self.clear_plot()
        except Exception as ex:
            self.log(f"Reset UI error: {ex}")

    def set_x_mode(self, mode: str):
        if callable(getattr(self.plot, "set_x_mode", None)):
            self.plot.set_x_mode(mode)

    def log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logpanel.log_view.append(f"[{ts}] {text}")

    def append_param_row(self, items: List[str]):
        r = self.param_table.rowCount()
        self.param_table.insertRow(r)
        for c, v in enumerate(items):
            it = QtWidgets.QTableWidgetItem(str(v))
            self.param_table.setItem(r, c, it)
        self.param_table.scrollToBottom()

    def load_osci_stylesheet(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'osci_style.qss')
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {path}")
            return ""

    def export_csv(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export waveform and params CSV",
            f"waveform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not fname:
            return
        data = self.model.last_waveform
        timestamp = self.model.last_timestamp or time.time()
        sample_rate = self.model.sample_rate
        if data is None:
            QtWidgets.QMessageBox.warning(self, "No data", "No waveform available to export.")
            return
        try:
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"# Generated: {datetime.fromtimestamp(timestamp).isoformat()}"])
                writer.writerow([f"# Resource: {self.model.resource_name}"])
                writer.writerow([f"# Channels: {self.model.channels}"])
                writer.writerow([f"# Sample rate (Hz): {sample_rate}"])
                writer.writerow([])
                arr = np.asarray(data)
                if arr.ndim == 1:
                    writer.writerow(["time_s" if getattr(self.plot, "_x_mode", "samples") == "time" else "sample_index", "voltage_V"])
                    if getattr(self.plot, "_x_mode", "samples") == "time":
                        t = np.arange(arr.size) / sample_rate
                        for ti, v in zip(t, arr):
                            writer.writerow([f"{ti:.9f}", f"{v:.9e}"])
                    else:
                        t = np.arange(arr.size)
                        for ti, v in zip(t, arr):
                            writer.writerow([f"{ti}", f"{v:.9e}"])
                elif arr.ndim == 2:
                    ch_count, npts = arr.shape
                    header = ["time_s" if getattr(self.plot, "_x_mode", "samples") == "time" else "sample_index"] + [f"ch{c}_V" for c in range(ch_count)]
                    writer.writerow(header)
                    if getattr(self.plot, "_x_mode", "samples") == "time":
                        t = np.arange(npts) / sample_rate
                    else:
                        t = np.arange(npts)
                    for i in range(npts):
                        row = [f"{t[i]:.9f}" if getattr(self.plot, "_x_mode", "samples") == "time" else f"{int(t[i])}"] + [f"{arr[ch, i]:.9e}" for ch in range(ch_count)]
                        writer.writerow(row)
                else:
                    writer.writerow(["# Unrecognized waveform format"])
                writer.writerow([])
                writer.writerow(["# Parameters history (most recent last)"])
                headers = [
                    self.param_table.horizontalHeaderItem(c).text()
                    if self.param_table.horizontalHeaderItem(c) else f"col{c}"
                    for c in range(self.param_table.columnCount())
                ]
                writer.writerow(headers)
                for r in range(self.param_table.rowCount()):
                    row = []
                    for c in range(self.param_table.columnCount()):
                        item = self.param_table.item(r, c)
                        row.append(item.text() if item is not None else "")
                    writer.writerow(row)
            self.log(f"Exported CSV: {fname}")
        except Exception as ex:
            self.log(f"Export failed: {ex}")
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not write file: {ex}")
