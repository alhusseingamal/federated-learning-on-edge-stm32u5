"""Federated-learning aggregation server for the STM32U585 HAR project.

Protocol, byte layout, and framing are defined in protocol.py - read that
file first if you're implementing the matching NetX Duo client on the MCU.

Session shape:
    1. Wait for exactly --num-clients TCP connections.
    2. Send each client a 4-byte handshake: the number of rounds this
       session will run.
    3. For each of --num-rounds rounds, in lockstep across all clients:
         server -> client : current global weights (224 bytes)
         client -> server : that client's locally-updated weights (224 bytes)
       Once ALL clients have submitted for this round, FedAvg them into the
       new global model, checkpoint it to disk, and log round stats.
    4. After the final round, close all connections and exit.

Usage:
    python3 server.py --num-clients 2 --num-rounds 20 \\
        --init-weights initial_weights.bin --checkpoint-dir ./checkpoints

--init-weights should normally be a raw 224-byte dump pulled from one of
the boards via its UART `GET_WEIGHTS` command (see the HAR firmware's
command parser) - that way round 0 broadcasts the SAME baseline weights
the boards were originally flashed with, rather than an arbitrary
all-zero head. If omitted, the server starts from an all-zero global head
and logs a warning; round-1 predictions made from that state are
meaningless, but the round-1 aggregate (once real client updates come in)
is not.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from protocol import (
    PAYLOAD_SIZE,
    HANDSHAKE_FORMAT,
    HANDSHAKE_SIZE,
    recv_exact,
    send_all,
    pack_weights,
    unpack_weights,
)
from aggregate import fedavg, payload_norm, payload_delta_norm


def load_initial_weights(path: str | None, logger: logging.Logger) -> bytes:
    if path is None:
        logger.warning(
            "No --init-weights given; starting from an all-zero global head. "
            "Round-0 broadcasts will not match either board's on-flash "
            "baseline - pull one via the board's GET_WEIGHTS UART command "
            "and pass it with --init-weights for a faithful round 0."
        )
        return bytes(PAYLOAD_SIZE)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--init-weights path does not exist: {p}")

    if p.suffix == '.npz':
        data = np.load(p)
        payload = pack_weights(data['W'], data['b'])
    else:
        payload = p.read_bytes()
        if len(payload) != PAYLOAD_SIZE:
            raise ValueError(
                f"{p} is {len(payload)} bytes, expected exactly {PAYLOAD_SIZE} "
                f"(56 float32 = 224 bytes). Is this really a raw GET_WEIGHTS dump?"
            )
    logger.info(f"loaded initial global weights from {p} (norm={payload_norm(payload):.4f})")
    return payload


class FLServer:
    def __init__(self, host: str, port: int, num_clients: int, num_rounds: int,
                 initial_weights: bytes, checkpoint_dir: Path, logger: logging.Logger):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

        self._current_global = initial_weights
        self.round_idx = 0

        self._pending_updates: dict[str, bytes] = {}
        self._pending_lock = threading.Lock()
        self.barrier = threading.Barrier(num_clients, action=self._aggregate_round)

        self._listen_sock: socket.socket | None = None
        self._client_socks: list[socket.socket] = []

    # -- aggregation -------------------------------------------------------

    def _aggregate_round(self) -> None:
        """Barrier action: runs exactly once per round, on one of the client
        threads, guaranteed to complete before any client thread proceeds
        past barrier.wait(). Safe to mutate self._current_global here with
        no extra locking - see the verified Barrier semantics in the
        surrounding conversation / test script."""
        with self._pending_lock:
            client_ids = sorted(self._pending_updates)
            payloads = [self._pending_updates[cid] for cid in client_ids]
            per_client_norms = {cid: payload_norm(p) for cid, p in self._pending_updates.items()}
            self._pending_updates.clear()

        new_global = fedavg(payloads)
        delta = payload_delta_norm(self._current_global, new_global)

        self.round_idx += 1
        self._save_checkpoint(self.round_idx, new_global)

        norms_str = ", ".join(f"{cid}={n:.4f}" for cid, n in per_client_norms.items())
        self.logger.info(
            f"round {self.round_idx}/{self.num_rounds} complete | "
            f"client update norms: [{norms_str}] | "
            f"global change (L2): {delta:.6f} | "
            f"checkpoint: global_round_{self.round_idx:04d}.bin"
        )

        self._current_global = new_global

    def _save_checkpoint(self, round_idx: int, payload: bytes) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        bin_path = self.checkpoint_dir / f"global_round_{round_idx:04d}.bin"
        bin_path.write_bytes(payload)

        latest_path = self.checkpoint_dir / "global_latest.bin"
        latest_path.write_bytes(payload)

        W, b = unpack_weights(payload)
        npz_path = self.checkpoint_dir / f"global_round_{round_idx:04d}.npz"
        np.savez(npz_path, W=np.array(W, dtype=np.float32), b=np.array(b, dtype=np.float32))

    # -- networking ----------------------------------------------------------

    def _client_round_loop(self, client_id: str, sock: socket.socket) -> None:
        try:
            for _ in range(self.num_rounds):
                send_all(sock, self._current_global)
                update = recv_exact(sock, PAYLOAD_SIZE)
                with self._pending_lock:
                    self._pending_updates[client_id] = update
                self.barrier.wait()
            try:
                send_all(sock, self._current_global)
                self.logger.info(f"{client_id}: final global model sent")
            except OSError as e:
                self.logger.error(f"{client_id}: failed to send final model ({e})")
                
        except threading.BrokenBarrierError:
            self.logger.error(f"{client_id}: round aborted (another client dropped)")
        except (ConnectionError, OSError) as e:
            self.logger.error(f"{client_id}: connection lost mid-round ({e})")
            self.barrier.abort()  # release every other waiting client instead of hanging forever
        finally:
            sock.close()
            self.logger.info(f"{client_id}: connection closed")

    def serve(self) -> None:
        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_sock.bind((self.host, self.port))
        self._listen_sock.listen(self.num_clients)
        self.logger.info(
            f"listening on {self.host}:{self.port}, "
            f"waiting for {self.num_clients} client(s)..."
        )

        threads = []
        accepted = 0
        while accepted < self.num_clients:
            conn, addr = self._listen_sock.accept()
            client_id = f"client{accepted}@{addr[0]}:{addr[1]}"
            try:
                send_all(conn, struct.pack(HANDSHAKE_FORMAT, self.num_rounds))
            except OSError as e:
                # Peer connected then vanished before the handshake finished
                # (e.g. a stray probe, or a real drop mid-connect) - don't
                # let this take down the whole accept loop; just keep
                # waiting for a working client in its place.
                self.logger.warning(f"{client_id}: handshake failed ({e}), still waiting for a client here")
                conn.close()
                continue

            self._client_socks.append(conn)
            self.logger.info(f"{client_id} connected ({accepted + 1}/{self.num_clients})")
            t = threading.Thread(target=self._client_round_loop, args=(client_id, conn), daemon=True)
            threads.append(t)
            accepted += 1

        self.logger.info(f"all clients connected - starting {self.num_rounds} FL round(s)")
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        self.logger.info(
            f"FL session complete: {self.round_idx}/{self.num_rounds} round(s) in {elapsed:.1f}s. "
            f"Final model: {self.checkpoint_dir / 'global_latest.bin'}"
        )
        self._listen_sock.close()

    def shutdown(self) -> None:
        for s in self._client_socks:
            try:
                s.close()
            except OSError:
                pass
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='0.0.0.0', help='interface to bind (default: all interfaces)')
    ap.add_argument('--port', type=int, default=9999)
    ap.add_argument('--num-clients', '-k', type=int, default=2, help='number of MCU clients to wait for (default: 2)')
    ap.add_argument('--num-rounds', '-r', type=int, default=20, help='number of FL rounds to run (default: 20)')
    ap.add_argument('--init-weights', default=None,
                     help='path to a 224-byte raw GET_WEIGHTS dump, or a .npz with W/b arrays')
    ap.add_argument('--checkpoint-dir', default='./checkpoints')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()

    if args.num_clients < 1:
        ap.error("--num-clients must be at least 1")
    if args.num_rounds < 1:
        ap.error("--num-rounds must be at least 1")

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('fl_server')

    initial_weights = load_initial_weights(args.init_weights, logger)

    server = FLServer(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        initial_weights=initial_weights,
        checkpoint_dir=Path(args.checkpoint_dir),
        logger=logger,
    )

    try:
        server.serve()
    except KeyboardInterrupt:
        logger.warning("interrupted - shutting down")
        server.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()