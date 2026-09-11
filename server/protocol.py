"""Wire protocol between the FL server and each STM32U585 client.

Payload layout matches `trainable_head_export_weights()` /
`trainable_head_import_weights()` in trainable_head.c EXACTLY, so the MCU
side can fill/consume this buffer with a single memcpy - no repacking
needed on-device:

    void trainable_head_export_weights(float *flat_buffer) {
        memcpy(flat_buffer, W, sizeof(W));                                   // 52 floats, W[4][13] row-major
        memcpy(flat_buffer + (HEAD_NUM_CLASSES*HEAD_TOTAL_FEATURES), b, sizeof(b));  // 4 floats
    }

Byte layout (little-endian float32 - Cortex-M33 native order, so no
byte-swapping is required on the MCU in either direction):

    bytes[0   : 208] -> W, row-major, shape (HEAD_NUM_CLASSES, HEAD_TOTAL_FEATURES) = (4, 13)
    bytes[208 : 224] -> b, shape (HEAD_NUM_CLASSES,) = (4,)

Session handshake (sent once, immediately after TCP accept):
    server -> client : 4 bytes, little-endian uint32 = NUM_ROUNDS for this session

Per round (repeated NUM_ROUNDS times, over the same persistent connection):
    server -> client : 224 bytes, current global weights payload
    client -> server : 224 bytes, this client's locally-updated weights payload

There is no explicit STOP message. The round count sent in the handshake is
the sole termination signal, so the MCU client never needs to parse a
variable-length or tagged message - just N fixed-size round-trips and then
close the socket.
"""

import struct

HEAD_NUM_CLASSES = 4
HEAD_TOTAL_FEATURES = 13

NUM_W = HEAD_NUM_CLASSES * HEAD_TOTAL_FEATURES
NUM_B = HEAD_NUM_CLASSES
NUM_WEIGHTS = NUM_W + NUM_B

PAYLOAD_FORMAT = f'<{NUM_WEIGHTS}f'
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)

HANDSHAKE_FORMAT = '<I'
HANDSHAKE_SIZE = struct.calcsize(HANDSHAKE_FORMAT)


def pack_weights(W, b) -> bytes:
    """W: HEAD_NUM_CLASSES rows of HEAD_TOTAL_FEATURES floats each (nested or flat).
    b: HEAD_NUM_CLASSES floats.
    Returns the exact 224-byte wire payload."""
    flat_w = []
    for row in W:
        if hasattr(row, '__iter__'):
            flat_w.extend(float(x) for x in row)
        else:
            flat_w.append(float(row))
    if len(flat_w) != NUM_W:
        raise ValueError(f"expected {NUM_W} W values, got {len(flat_w)}")
    flat_b = [float(x) for x in b]
    if len(flat_b) != NUM_B:
        raise ValueError(f"expected {NUM_B} b values, got {len(flat_b)}")
    return struct.pack(PAYLOAD_FORMAT, *flat_w, *flat_b)


def unpack_weights(payload: bytes):
    """Returns (W, b): W as HEAD_NUM_CLASSES lists of HEAD_TOTAL_FEATURES floats,
    b as a list of HEAD_NUM_CLASSES floats."""
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"expected {PAYLOAD_SIZE}-byte payload, got {len(payload)}")
    values = struct.unpack(PAYLOAD_FORMAT, payload)
    flat_w, flat_b = values[:NUM_W], values[NUM_W:]
    W = [list(flat_w[i * HEAD_TOTAL_FEATURES:(i + 1) * HEAD_TOTAL_FEATURES])
         for i in range(HEAD_NUM_CLASSES)]
    b = list(flat_b)
    return W, b


def recv_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a socket, or raise ConnectionError if the
    peer closes early. TCP is a byte stream with no message boundaries - a
    single recv() is not guaranteed to return all requested bytes - so every
    fixed-size read in this protocol MUST go through this helper rather than
    a bare sock.recv(n)."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"connection closed with {remaining} of {n} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def send_all(sock, data: bytes) -> None:
    sock.sendall(data)
