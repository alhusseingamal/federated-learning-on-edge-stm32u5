"""Backend API for the B-U585 HAR dashboard GUI.

The GUI only ever talks to the `Backend` interface below. Each concrete
transport implementation lives in its own module next to this one:

    backend.py            <- this file: the API (never changes)
    uart_backend.py       <- pyserial transport
    udp_backend.py        <- socket transport (wifi, added later)
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class Backend(ABC):
    """Abstract transport to a board."""

    def __init__(self) -> None:
        self.connected: bool = False

        # Raw / unrecognised lines and command acknowledgements (e.g. "[board] ok: start")
        self.on_message: Optional[Callable[[str], None]] = None

        # Fired on every "PRED <cls> <p0> <p1> <p2> <p3> <dyn> <gyro>" line.
        # cls: predicted class index, probs: list of 4 floats summing to ~1,
        # dyn: preprocessed dynamic-acceleration magnitude (g), gyro: mean gyro magnitude (dps)
        self.on_prediction: Optional[Callable[[int, List[float], float, float], None]] = None

        # Fired on every "TRAINED <cls> <loss> <dyn> <gyro>" line, i.e. after
        # a one-shot corrective SGD step the board just performed.
        self.on_training: Optional[Callable[[int, float, float, float], None]] = None

    @abstractmethod
    def connect(self) -> None:
        """Open the link (serial port, socket, ...) and set `connected`."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link, stop any running data stream, clear `connected`."""

    @abstractmethod
    def send(self, text: str) -> None:
        """Transmit one text command to the board."""
