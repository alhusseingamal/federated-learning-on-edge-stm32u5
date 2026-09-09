import serial
from nicegui import ui
from backend import Backend


class UartBackend(Backend):
    """Backend implementation that talks to the B-U585 over a serial port."""

    def __init__(self, port: str = "/dev/ttyACM0") -> None:
        super().__init__()
        self._port = port
        self._ser: serial.Serial | None = None
        self._poll_timer = ui.timer(0.1, self._poll, active=False)
        # String buffer to hold partial lines across polling ticks
        self._buffer = ""

    def connect(self) -> None:
        try:
            self._ser = serial.Serial(self._port, 115200, timeout=0)
        except serial.SerialException as e:
            ui.notify(f"Cannot open {self._port}: {e}", type="negative")
            return

        self._buffer = ""  # Clear buffer on connect
        self._poll_timer.activate()
        self.connected = True

        if self.on_message:
            self.on_message(f"[link] connected to {self._port}")

    def disconnect(self) -> None:
        self._poll_timer.deactivate()
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self.connected = False

        if self.on_message:
            self.on_message("[link] disconnected")

    def send(self, text: str) -> None:
        if not self.connected or not self._ser:
            ui.notify("Not connected", type="warning")
            return
        self._ser.write((text.strip() + "\n").encode())

    def _poll(self) -> None:
        if not self._ser:
            return

        # 1. Read all currently available bytes into the string buffer
        while self._ser.in_waiting:
            self._buffer += self._ser.read(self._ser.in_waiting).decode(errors="ignore")

        # 2. Extract and process only complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            if line.startswith("PRED ") and self.on_prediction:
                self._handle_pred(line)
            elif line.startswith("TRAINED ") and self.on_training:
                self._handle_trained(line)
            elif self.on_message:
                # command acks ("[board] ok: ..."), errors, and any other
                # diagnostic prints the firmware still emits land here.
                self.on_message(line)

    def _handle_pred(self, line: str) -> None:
        # "PRED <cls> <p0> <p1> <p2> <p3> <dyn> <gyro>" -> 8 whitespace tokens
        parts = line.split()
        if len(parts) != 8:
            if self.on_message:
                self.on_message(line)  # malformed - surface raw rather than drop silently
            return
        try:
            cls = int(parts[1])
            probs = [float(p) for p in parts[2:6]]
            dyn = float(parts[6])
            gyro = float(parts[7])
        except ValueError:
            if self.on_message:
                self.on_message(line)
            return
        self.on_prediction(cls, probs, dyn, gyro)

    def _handle_trained(self, line: str) -> None:
        # "TRAINED <cls> <loss> <dyn> <gyro>" -> 5 whitespace tokens
        parts = line.split()
        if len(parts) != 5:
            if self.on_message:
                self.on_message(line)
            return
        try:
            cls = int(parts[1])
            loss = float(parts[2])
            dyn = float(parts[3])
            gyro = float(parts[4])
        except ValueError:
            if self.on_message:
                self.on_message(line)
            return
        self.on_training(cls, loss, dyn, gyro)
