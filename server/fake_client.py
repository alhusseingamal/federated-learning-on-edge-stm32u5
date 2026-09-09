"""Simulates one STM32 FL client, for testing server.py without hardware.

This speaks the exact same wire protocol a real board would (see
protocol.py) - it's useful both as a test harness and as a way to exercise
the server before your NetX Duo firmware side is ready.

Usage:
    python3 fake_client.py --port 9999 --client-id A --seed 1
"""

import argparse
import socket
import struct

import numpy as np

from protocol import PAYLOAD_SIZE, HANDSHAKE_FORMAT, HANDSHAKE_SIZE, NUM_WEIGHTS, recv_exact, send_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9999)
    ap.add_argument('--client-id', default='fake')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--step', type=float, default=0.01,
                     help='deterministic per-round "local training" perturbation scale')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[{args.client_id}] connected to {args.host}:{args.port}")

    num_rounds = struct.unpack(HANDSHAKE_FORMAT, recv_exact(sock, HANDSHAKE_SIZE))[0]
    print(f"[{args.client_id}] server requested {num_rounds} round(s)")

    for r in range(num_rounds):
        global_payload = recv_exact(sock, PAYLOAD_SIZE)
        global_vec = np.frombuffer(global_payload, dtype='<f4')

        # Stand-in for on-device SGD: a small deterministic nudge, reproducible via --seed
        nudge = rng.normal(loc=0.0, scale=args.step, size=NUM_WEIGHTS).astype('<f4')
        updated_vec = (global_vec + nudge).astype('<f4')

        print(f"[{args.client_id}] round {r + 1}/{num_rounds}: "
              f"recv global (norm={np.linalg.norm(global_vec):.4f}) -> "
              f"send update (norm={np.linalg.norm(updated_vec):.4f})")
        send_all(sock, updated_vec.tobytes())

    sock.close()
    print(f"[{args.client_id}] session complete, disconnected")


if __name__ == '__main__':
    try:
        main()
    except ConnectionError as e:
        # Expected if the server aborted the session (e.g. a peer client
        # dropped mid-round) - not a bug in this script.
        print(f"session ended early: {e}")

