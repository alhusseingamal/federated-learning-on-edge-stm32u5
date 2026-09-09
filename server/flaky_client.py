"""Test utility: connects, completes round 1 normally, then disconnects
mid-round-2 without sending its update - simulating a real WiFi drop.

Use this against server.py to confirm the server aborts the session
cleanly instead of hanging when a client disappears mid-round:

    python3 server.py --port 9999 --num-clients 2 --num-rounds 5 &
    python3 fake_client.py --port 9999 --client-id GOOD &
    python3 flaky_client.py --port 9999

Expect the server log to show a "connection lost mid-round" error for
this client, a "round aborted" error for the other, and a clean exit -
not a hang.
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
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    num_rounds = struct.unpack(HANDSHAKE_FORMAT, recv_exact(sock, HANDSHAKE_SIZE))[0]
    print(f"[flaky] server requested {num_rounds} round(s)")

    if num_rounds < 2:
        print("[flaky] needs --num-rounds >= 2 on the server to demonstrate a mid-round drop")

    # Round 1: behave normally
    recv_exact(sock, PAYLOAD_SIZE)
    send_all(sock, np.zeros(NUM_WEIGHTS, dtype='<f4').tobytes())
    print("[flaky] round 1 submitted normally")

    # Round 2: receive the broadcast, then vanish without responding or closing cleanly
    recv_exact(sock, PAYLOAD_SIZE)
    print("[flaky] received round 2 broadcast - dropping the connection now, no response")
    sock.close()


if __name__ == '__main__':
    main()
