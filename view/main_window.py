from typing import List, Tuple, Optional
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from utils.uart_decoder import Decoder  # Changed import to use Decoder class

class UARTPanel(QtWidgets.QFrame):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.decoder = Decoder()  # Instantiate Decoder
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

        row = QtWidgets.QHBoxLayout()
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

        # Optional: Add sample point control
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Sample Point (%):"))
        self.uart_sample_point = QtWidgets.QDoubleSpinBox()
        self.uart_sample_point.setRange(1.0, 99.0)
        self.uart_sample_point.setValue(50.0)
        row.addWidget(self.uart_sample_point)
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
        w = self
        while w is not None:
            if hasattr(w, "plot") and getattr(w.plot, "plot_widget", None) is not None:
                return w
            w = getattr(w, "parentWidget", lambda: None)() or getattr(w, "parent", lambda: None)()
        return None

    def on_decode_uart(self):
        """Called by UI. Decode waveform, display decoded strings and overlay markers."""
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
        sample_point = float(self.uart_sample_point.value())

        # Call decoder using the Decoder instance
        decoded, samp_times, bit_levels, parity_flags, frames = self.decoder.decode_uart_from_analog(
            sig, sr, threshold, baud,
            data_bits=data_bits,
            stop_bits=stop_bits,
            hysteresis_percent=hysteresis_percent,
            sample_window_fraction=0.35,
            min_stop_high_fraction=0.6,
            parity=parity_mode,
            require_parity_ok=False,
            require_level_ok=True,  # Enforce voltage validation
            sample_point=sample_point,
            return_parity=True,
            return_frames=True
        )

        # Sanitize decoded bytes
        clean_decoded = []
        for b in decoded:
            try:
                ib = int(b) & 0xFF
            except Exception:
                continue
            if 0 <= ib <= 0xFF:
                clean_decoded.append(ib)
        decoded = clean_decoded

        # Process frames for display, including break conditions
        hex_str_parts = []
        ascii_str_parts = []
        for i, fr in enumerate(frames):
            if fr.get("break", False):
                hex_str_parts.append("[BREAK]")
                ascii_str_parts.append("[BREAK]")
                continue
            b = fr.get("byte", 0)
            tag = ""
            if not fr.get("level_ok", True):
                tag += "[VERR]"
            if fr.get("parity_ok") is False:
                tag += "[PERR]"
            if fr.get("framing_error", False):
                tag += "[FERR]"
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
                if fr.get("break", False):
                    owner.log(f"Break condition at {fr['start_time']:.6f}s to {fr['end_time']:.6f}s")
                if not fr.get("level_ok", True):
                    owner.log(f"Voltage error at byte {i}: 0x{fr.get('byte', 0):02X}")
                if fr.get("parity_ok") is False:
                    owner.log(f"Parity error at byte {i}: 0x{fr.get('byte', 0):02X}")
                if fr.get("framing_error"):
                    owner.log(f"Framing error at byte {i}: 0x{fr.get('byte', 0):02X}")

        bit_width_s = 1.0 / float(baud) if baud > 0 else None
        self._overlay_uart_markers(samp_times, bit_levels, frames=frames, bit_width_s=bit_width_s)

    def _overlay_uart_markers(self, sample_times, bit_levels, frames=None, bit_width_s: Optional[float]=None):
        """Plot scatter markers and frame regions for UART bits, including break conditions."""
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

        arr = np.asarray(self.model.last_waveform) if self.model.last_waveform is not None else np.array([0.0])
        try:
            ymax = float(np.nanmax(arr))
            ymin = float(np.nanmin(arr))
        except Exception:
            ymax, ymin = 1.0, 0.0
        height = ymax - ymin if ymax != ymin else 1.0

        x_mode = "samples"
        if owner is not None and hasattr(owner, "plot"):
            x_mode = getattr(owner.plot, "_x_mode", "samples")

        # Scatter points for bit centers
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
        if plot_widget is not None:
            try:
                if not self._scatter_added_to_plot:
                    plot_widget.addItem(self.uart_scatter)
                    self._scatter_added_to_plot = True
            except Exception:
                pass
        try:
            self.uart_scatter.addPoints(spots)
        except Exception:
            pass

        # Draw frame regions
        if frames and plot_widget is not None:
            for fi, fr in enumerate(frames):
                if fr.get("break", False):
                    # Draw break region (purple)
                    start_t = fr.get("start_time", 0.0)
                    end_t = fr.get("end_time", 0.0)
                    if x_mode != "time":
                        x0 = start_t * float(self.model.sample_rate)
                        x1 = end_t * float(self.model.sample_rate)
                    else:
                        x0, x1 = start_t, end_t
                    brush = pg.mkBrush(128, 0, 128, 80)  # Purple for break
                    region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, brush=brush)
                    region.setMovable(False)
                    region.setZValue(800)
                    plot_widget.addItem(region)
                    self.uart_frame_items.append(region)
                    txt = pg.TextItem("BREAK", anchor=(0.5, 0.0))
                    txt.setPos((x0 + x1) / 2.0, ymax + 0.02 * height)
                    plot_widget.addItem(txt)
                    self.uart_text_items.append(txt)
                    continue

                bit_times = fr.get("bit_times", [])
                bit_indices = fr.get("bit_indices", [])
                bit_levels = fr.get("bit_levels", [])

                use_times = bit_times if bit_times else [float(idx) / float(self.model.sample_rate) for idx in bit_indices]
                if not use_times:
                    continue

                # Draw start bit (before first data bit)
                if bit_width_s:
                    start_center = use_times[0] - bit_width_s
                    start_left = start_center - 0.5 * bit_width_s
                    start_right = start_center + 0.5 * bit_width_s
                    if x_mode != "time":
                        x0 = start_left * float(self.model.sample_rate)
                        x1 = start_right * float(self.model.sample_rate)
                    else:
                        x0, x1 = start_left, start_right
                    brush = pg.mkBrush(0, 0, 200, 60)  # Blue for start
                    region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, brush=brush)
                    region.setMovable(False)
                    region.setZValue(800)
                    plot_widget.addItem(region)
                    self.uart_frame_items.append(region)
                    txt = pg.TextItem("START", anchor=(0.5, 0.0))
                    txt.setPos((x0 + x1) / 2.0, ymax + 0.02 * height)
                    plot_widget.addItem(txt)
                    self.uart_text_items.append(txt)

                # Draw data and parity bits
                for bi, center_t in enumerate(use_times):
                    left = center_t - 0.5 * bit_width_s if bit_width_s else center_t - 0.5 * (use_times[bi + 1] - center_t) if bi + 1 < len(use_times) else center_t - 0.5 * bit_width_s
                    right = center_t + 0.5 * bit_width_s if bit_width_s else center_t + 0.5 * (center_t - use_times[bi - 1]) if bi - 1 >= 0 else center_t + 0.5 * bit_width_s
                    if x_mode != "time":
                        x0 = left * float(self.model.sample_rate)
                        x1 = right * float(self.model.sample_rate)
                    else:
                        x0, x1 = left, right

                    is_parity_bit = (bi == len(use_times) - 1 and fr.get("parity_ok") is not None)
                    brush = pg.mkBrush(120, 120, 120, 60)  # Gray for data 0
                    label_text = f"b{bi}={bit_levels[bi]}" if bi < len(bit_levels) else f"b{bi}=?"
                    if is_parity_bit:
                        brush = pg.mkBrush(0, 255, 255, 60)  # Cyan for parity
                        label_text = "PARITY"
                    elif bi < len(bit_levels) and bit_levels[bi]:
                        brush = pg.mkBrush(0, 180, 0, 80)  # Green for data 1
                    region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, brush=brush)
                    region.setMovable(False)
                    region.setZValue(800)
                    plot_widget.addItem(region)
                    self.uart_frame_items.append(region)
                    txt = pg.TextItem(label_text, anchor=(0.5, 0.0))
                    txt.setPos((x0 + x1) / 2.0, ymax + 0.02 * height)
                    plot_widget.addItem(txt)
                    self.uart_text_items.append(txt)

                # Draw stop bits
                if bit_width_s:
                    last_center = use_times[-1]
                    stop_start = last_center + (0.5 * bit_width_s if fr.get("parity_ok") is None else 1.5 * bit_width_s)
                    for i in range(stop_bits):
                        stop_center = stop_start + i * bit_width_s
                        stop_left = stop_center - 0.5 * bit_width_s
                        stop_right = stop_center + 0.5 * bit_width_s
                        if x_mode != "time":
                            x0 = stop_left * float(self.model.sample_rate)
                            x1 = stop_right * float(self.model.sample_rate)
                        else:
                            x0, x1 = stop_left, stop_right
                        brush = pg.mkBrush(220, 200, 40, 40)  # Yellow for stop
                        region = pg.LinearRegionItem(values=(x0, x1), orientation=pg.LinearRegionItem.Vertical, brush=brush)
                        region.setMovable(False)
                        region.setZValue(800)
                        plot_widget.addItem(region)
                        self.uart_frame_items.append(region)
                        txt = pg.TextItem(f"STOP{i+1}", anchor=(0.5, 0.0))
                        txt.setPos((x0 + x1) / 2.0, ymax + 0.02 * height)
                        plot_widget.addItem(txt)
                        self.uart_text_items.append(txt)

                # Draw error badges
                badge_msgs = []
                if not fr.get("level_ok", True):
                    badge_msgs.append("VERR")
                if fr.get("parity_ok") is False:
                    badge_msgs.append("PARITY")
                if fr.get("framing_error"):
                    badge_msgs.append("FRAMING")
                if badge_msgs:
                    badge_text = "|".join(badge_msgs)
                    frame_end = fr.get("end_time", use_times[-1] + bit_width_s * stop_bits)
                    bx = frame_end * float(self.model.sample_rate) if x_mode != "time" else frame_end
                    btxt = pg.TextItem(badge_text, color=(200, 20, 20), anchor=(1.0, 0.0))
                    btxt.setZValue(1200)
                    btxt.setPos(bx, ymax + 0.06 * height)
                    plot_widget.addItem(btxt)
                    self.uart_text_items.append(btxt)

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
                        try:
                            item.setParentItem(None)
                        except Exception:
                            pass
                except Exception:
                    try:
                        item.setParentItem(None)
                    except Exception:
                        pass
        except Exception:
            pass

        self.uart_frame_items = []
        self.uart_text_items = []
        self._scatter_added_to_plot = False
        self._frames_added_to_plot = False