"""Bridges ONE real STM32 board to server.py over TCP, using its UART
GET_WEIGHTS/SET_WEIGHTS commands - a stand-in for the NetX Duo TCP client
that doesn't exist on the MCU yet. Run one instance per board.

Per round:
    1. Receive the current global weights from the FL server (TCP).
    2. Push them into the board via `setweights` + 224 raw bytes (UART).
    3. Pause so you can locally train the board for a bit (via the GUI's
       l0-l3 buttons, or --train-seconds for an unattended fixed pause).
    4. Pull the board's now-locally-adapted weights back via `getweights`
       (UART).
    5. Send those weights to the FL server to complete the round (TCP).

Usage (one instance per board):
    python3 bridge.py --serial-port /dev/ttyACM0 --tcp-host 192.168.1.10 --tcp-port 9999
    python3 bridge.py --serial-port /dev/ttyACM1 --tcp-host 192.168.1.10 --tcp-port 9999

Requires the firmware's `getweights`/`setweights` UART commands (see the
firmware snippets discussed alongside this script) - it will hang waiting
for an ack line if the board doesn't understand these commands yet.
"""

import argparse
import socket
import struct
import sys

import serial

from protocol import PAYLOAD_SIZE, HANDSHAKE_FORMAT, HANDSHAKE_SIZE, recv_exact, send_all
from serial_io import SerialLineReader


def push_weights_to_board(reader: SerialLineReader, ser: serial.Serial, payload: bytes) -> None:
    ser.write(b"setweights\n")
    ack = reader.read_line(timeout=5.0)
    if "send 224 bytes" not in ack:
        raise RuntimeError(f"unexpected response to setweights: {ack!r}")
    ser.write(payload)
    confirm = reader.read_line(timeout=5.0)
    if "weights updated" not in confirm:
        raise RuntimeError(f"board did not confirm weights update: {confirm!r}")


def pull_weights_from_board(reader: SerialLineReader, ser: serial.Serial, verbose: bool = True) -> bytes:
    ser.write(b"getweights\n")
    # Other lines (PRED/TRAINED/log output) may legitimately arrive first -
    # skip past them until the binary-payload marker shows up.
    while True:
        line = reader.read_line(timeout=5.0)
        if line == "WEIGHTS_DUMP":
            break
        if verbose and line:
            print(f"    (board) {line}")
    return reader.read_exact(PAYLOAD_SIZE, timeout=5.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--serial-port', required=True, help='e.g. /dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--tcp-host', default='127.0.0.1')
    ap.add_argument('--tcp-port', type=int, default=9999)
    ap.add_argument('--train-seconds', type=float, default=None,
                     help='if set, pause exactly this long each round instead of waiting for Enter')
    args = ap.parse_args()

    ser = serial.Serial(args.serial_port, args.baud, timeout=0.1)
    reader = SerialLineReader(ser)
    print(f"[bridge] opened {args.serial_port} @ {args.baud} baud")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.tcp_host, args.tcp_port))
    print(f"[bridge] connected to FL server at {args.tcp_host}:{args.tcp_port}")

    num_rounds = struct.unpack(HANDSHAKE_FORMAT, recv_exact(sock, HANDSHAKE_SIZE))[0]
    print(f"[bridge] server requested {num_rounds} round(s)")

    for r in range(num_rounds):
        global_payload = recv_exact(sock, PAYLOAD_SIZE)
        print(f"[bridge] round {r + 1}/{num_rounds}: received global weights, pushing to board...")
        push_weights_to_board(reader, ser, global_payload)
        print(f"[bridge] round {r + 1}/{num_rounds}: board updated")

        if args.train_seconds is not None:
            print(f"[bridge] training for {args.train_seconds:.0f}s (use the GUI's label buttons now)...")
            import time
            time.sleep(args.train_seconds)
        else:
            input(f"[bridge] round {r + 1}/{num_rounds}: train the board now (GUI label buttons), "
                  f"then press Enter to pull its weights back...")

        local_payload = pull_weights_from_board(reader, ser)
        print(f"[bridge] round {r + 1}/{num_rounds}: pulled updated weights, sending to server...")
        send_all(sock, local_payload)

    sock.close()
    ser.close()
    print("[bridge] session complete")


if __name__ == '__main__':
    try:
        main()
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        print(f"[bridge] FATAL: {e}", file=sys.stderr)
        sys.exit(1)
