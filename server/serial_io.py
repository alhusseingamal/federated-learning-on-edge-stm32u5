"""pyserial helpers for talking to a board's UART command bridge.

Mirrors protocol.recv_exact()'s guarantee - a single ser.read(n) call is
NOT guaranteed to return all n bytes even with timeout=None on some
platforms/drivers, so every fixed-size read goes through read_exact().
"""

import time


class SerialLineReader:
    """Buffered reader that can hand back either newline-terminated ASCII
    lines OR a raw fixed-size binary blob from the same underlying stream -
    needed because GET_WEIGHTS/SET_WEIGHTS mix ASCII command/ack lines with
    a raw 224-byte binary payload on the same UART link."""

    def __init__(self, ser):
        self.ser = ser
        self._buf = b""

    def read_line(self, timeout: float = 5.0) -> str:
        """Block until a newline-terminated line is available, or raise
        TimeoutError. Returns the line with \\r\\n stripped."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no line received within {timeout}s (buffer so far: {self._buf!r})")
            chunk = self.ser.read(max(1, self.ser.in_waiting or 1))
            if chunk:
                self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", errors="replace").rstrip("\r")

    def read_exact(self, n: int, timeout: float = 10.0) -> bytes:
        """Block until exactly n raw bytes are available (first drawing
        from anything already buffered past a previously-read line), or
        raise TimeoutError."""
        deadline = time.monotonic() + timeout
        while len(self._buf) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"only {len(self._buf)} of {n} bytes received within {timeout}s"
                )
            chunk = self.ser.read(max(1, self.ser.in_waiting or 1))
            if chunk:
                self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data
