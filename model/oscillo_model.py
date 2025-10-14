from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class OscilloModel:
    resource_name: str = "PXI1Slot2"
    channels: str = "0"                 # e.g., "0" or "0,1"
    voltage_range: float = 5.0
    volts_per_division: float = 1.0
    volts_per_div: float = 1.0
    sample_rate: float = 1e6            # Hz
    fetch_time: float = 0.001           # seconds
    simulate: bool = False
    last_waveform: Optional[np.ndarray] = field(default=None, repr=False)
    last_timestamp: Optional[float] = None

    trigger_mode: str = "Immediate"          # "Immediate" or "Edge"
    trigger_edge_direction: str = "Rising"   # "Rising" or "Falling" (used when Edge)
    trigger_level: float = 0.0               # volts
    trigger_source: str = ""                 # channel or source name (defaults to first channel)
    trigger_holdoff: float = 0.0             # seconds
    trigger_delay: float = 0.0               # seconds
