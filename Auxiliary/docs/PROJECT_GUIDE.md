[**This is an AI-refined version of a manually written project write up document and draft notes.**]
# On-Device Federated Learning for Human Activity Recognition — Project Guide

**Two STM32U585 boards, ThreadX + NetX Duo, on-device SGD, and a lockstep FedAvg server — from a bare CubeMX project to a working two-board federated session.**

This guide is written so that someone with the same hardware (two B-U585I-IOT02A
Discovery Kits) can reproduce the entire system: firmware, AI pipeline,
networking, host software, and the operational recipe for running a real
federated learning session. It also preserves the debugging history in
detail, because several of the failures encountered here are easy to hit
again and expensive to re-diagnose from scratch.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware & Software Prerequisites](#2-hardware--software-prerequisites)
3. [Repository Layout](#3-repository-layout)
4. [Part A — Firmware Project Setup (CubeMX)](#4-part-a--firmware-project-setup-cubemx)
5. [Part B — The HAR Model Pipeline (ST Edge AI)](#5-part-b--the-har-model-pipeline-st-edge-ai)
6. [Part C — On-Device Application Logic](#6-part-c--on-device-application-logic)
7. [Part D — Wi-Fi & Federated Learning Networking](#7-part-d--wi-fi--federated-learning-networking)
8. [Part E — Host-Side Software](#8-part-e--host-side-software)
9. [End-to-End: Running a Full Session](#9-end-to-end-running-a-full-session)
10. [Command Reference](#10-command-reference)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Extending the Project](#12-extending-the-project)
13. [Appendix A: Wire Protocol Byte Layout](#appendix-a-wire-protocol-byte-layout)
14. [Appendix B: Full UART Command Set](#appendix-b-full-uart-command-set)
15. [Appendix C: Live Telemetry Line Formats](#appendix-c-live-telemetry-line-formats)

---

## 1. Project Overview

Two STM32U585 boards each run a hybrid HAR model: a frozen, quantized
convolutional backbone (deployed via ST Edge AI) produces 12-dimensional
embeddings from a 48-sample accelerometer window, which are combined with a
derived gyroscope-magnitude feature into a 13-dimensional vector fed to a
small **trainable linear head** (4 classes × 13 features + 4 biases = 56
parameters, 224 bytes). The head is trained **on-device**, in place, using
closed-form softmax cross-entropy SGD — no autodiff, no stored activations,
just a fixed sequence of dot products triggered by human-in-the-loop gesture
labels sent over UART from a browser dashboard.

Each board also runs a NetX Duo TCP client that periodically exchanges this
224-byte parameter vector with a Python aggregation server over Wi-Fi,
running a lockstep Federated Averaging (FedAvg) protocol: the server
broadcasts the current global model, each board trains locally for as long
as the operator wants, then uploads its locally-adapted weights; once every
connected board has submitted, the server averages them into the new global
model and the cycle repeats.

The system was built incrementally, and most of the interesting engineering
is in the failure modes that showed up along the way — TCP fragmentation
over an SPI-bridged Wi-Fi link, USB device renumbering, a missing NVIC
interrupt enable, and more. [Section 11](#11-troubleshooting-guide)
documents all of them; do not skip it.

---

## 2. Hardware & Software Prerequisites

### Hardware
- 2× STMicroelectronics **B-U585I-IOT02A** Discovery Kits (STM32U585AII6,
  Cortex-M33, 2 MB flash, 786 KB SRAM)
  - Onboard ISM330DHCX 6-axis IMU (accelerometer + gyroscope), used at 26 Hz
    over I2C2
  - Onboard MXCHIP EMW3080B Wi-Fi module, bridged over SPI2
  - Onboard ST-LINK V3 (SWD programmer + CDC-ACM virtual COM port)
- A host Linux laptop with a Wi-Fi adapter capable of AP (hotspot) mode on
  the 2.4 GHz band — required because the EMW3080B does not support 5 GHz

### Software
- STM32CubeMX (for regenerating peripheral init code, if you need to change
  pin/clock configuration)
- STMCube Firmware: Drivers, BSP, and Components from STM32Cube_FW_U5_V1.8.0
- STM32CubeProgrammer CLI (`STM32_Programmer_CLI`) and `picocom` (or any
  serial terminal) for flashing and monitoring
- A CMake + GCC ARM embedded toolchain (the project builds via CMake
  presets, not raw CubeIDE builds)
- ST Edge AI Core CLI (`stedgeai`), version 4.0 in this build
- Python 3.9+ on the host, with `numpy`, `nicegui`, and `pyserial`. The
  dashboard is run via [`uv`](https://docs.astral.sh/uv/); the FL server can
  be run with plain `python3`.
- `nmcli` (NetworkManager) and `nft`/`iptables` on the host for hotspot and
  firewall configuration

---

## 3. Repository Layout

The project keeps the firmware and host-side Python tools in the same
repository. The same firmware is flashed to both boards:

```
fl/
├── Core/
│   ├── Inc/trainable_head.h       # Trainable linear head interface
│   └── Src/
│       ├── main.c                # Peripheral init, UART RX priming
│       ├── app_threadx.c         # Sensor thread, command parser, mutexes
│       └── trainable_head.c      # On-device linear head + SGD
├── AZURE_RTOS/App/               # ThreadX application setup
├── NetXDuo/App/app_netxduo.c     # FL TCP client (App_SNTP_Thread_Entry)
├── AI_Runtime/                   # ST Edge AI models and runtime headers
│   ├── st_ign_wl_48.keras        # Original pretrained model
│   └── Inc/
├── HAR/
│   ├── AI/                       # Generated full-model C code
│   └── AI_EMBED/                 # Generated frozen-backbone C code with final head removed
├── Auxiliary/
│   ├── slice_model.py            # Cuts the Keras model at the dense layer
│   ├── generate_baseline.py      # Extracts baseline head weights
│   └── EMW3080update_*.bin      # Wi-Fi module firmware update image
├── build/Debug/                 # CMake build output, including the ELF
├── Drivers/, Middlewares/        # STM32, ThreadX, and NetX Duo dependencies
├── CMakeLists.txt, CMakePresets.json
├── fl.ioc                       # STM32CubeMX project configuration
├── server/
│   ├── protocol.py              # Wire format + socket helpers
│   ├── aggregate.py             # FedAvg math
│   ├── server.py                # FL aggregation server
│   ├── fake_client.py           # Simulated board, for testing without hardware
│   ├── flaky_client.py          # Simulated dropped-connection test
│   ├── bridge.py / serial_io.py # UART<->TCP bridge (superseded, see §8.3)
│   └── checkpoints/              # Saved global models per round
└── dashboard/
    ├── main.py                  # NiceGUI dashboard
    ├── backend.py               # Abstract transport interface
    └── uart_backend.py          # pyserial transport
```

---

## 4. Part A — Firmware Project Setup (CubeMX)

This project is layered on top of ST's **`Nx_SNTP_Client`** lab example
rather than built from an empty NetX Duo project — that example already
provides the MXCHIP driver bindings, DHCP-gated thread startup, and packet
pool setup, all of which are fragile to hand-configure from scratch. Start
there, then port in the peripherals and application logic below.

### 4.1 Peripheral Configuration

| Peripheral | Setting |
|---|---|
| **I2C2** | Auto-mapped to PH4 (SCL) / PH5 (SDA) on this board when I2C2 is enabled — no manual pin assignment needed. Set **Fast Mode (400 kHz)** under I2C2 → Configuration → Parameter Settings. |
| **USART1** | Default configuration (115200-8-N-1), used for both the dashboard link and `printf` redirection. |
| **Timebase source** | System Core → SYS → set **Timebase Source to TIM6**, *not* SysTick. This is easy to miss and important: ThreadX needs SysTick for its own scheduler tick, and leaving the default timebase on SysTick creates a conflict. |
| **ICACHE** | Leave at CubeMX defaults. |
| **ThreadX** | Enable **"Create ThreadX Application Thread"**. Note that preemption threshold, time slice, and auto-start are **not exposed in the CubeMX GUI** — you set the real values by hand-editing the generated `tx_thread_create()` call afterward, if needed. |

### 4.2 Build & Flash Workflow

The project builds via CMake presets:

```bash
cmake --preset Debug
cmake --build --preset Debug
```

List connected boards (each has a unique ST-LINK serial number):

```bash
STM32_Programmer_CLI -l
```
```
Board Name  : B-U585I-IOT02A
ST-LINK SN: 003E00473432511430343838
Port: ttyACM1
...
Board Name  : B-U585I-IOT02A
ST-LINK SN: 0028003E3432511430343838
Port: ttyACM0
...
```

Flash a **specific** board by its serial number — with two boards attached,
an unqualified `-c port=SWD` flash targets whichever probe the tool happens
to enumerate first, which is not something to rely on:

```bash
STM32_Programmer_CLI -c port=SWD sn=<SERIAL_NUMBER> -w build/Debug/<project>.elf -s
```

**Use a hardware reset, not a software reset**, when flashing — see
[§11.9](#119-board-needs-two-flashes-to-connect-to-wifi) for why this
matters specifically on this board:

```bash
STM32_Programmer_CLI -c port=SWD mode=UR reset=HWrst sn=<SERIAL_NUMBER> -w build/Debug/<project>.elf -s
```

Monitor serial output:

```bash
picocom -b 115200 /dev/ttyACM0
```

---

## 5. Part B — The HAR Model Pipeline (ST Edge AI)

### 5.1 Starting Model

The project starts from a pretrained WISDM-trained HAR model, using the
48-sample-window variant, from ST's public model zoo:

```
https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/human_activity_recognition/st_ign/ST_pretrainedmodel_public_dataset/WISDM/st_ign_wl_48/st_ign_wl_48.keras
```
(ST also publishes a 24-sample-window variant at the sibling
`st_ign_wl_24/st_ign_wl_24.keras` path, used earlier in this project before
switching to the 48-sample window for more temporal context per
prediction.)

`stedgeai` natively accepts Keras (`.keras`/`.h5`), TensorFlow Lite
(`.tflite`), or ONNX. PyTorch/scikit-learn/MATLAB models are **not**
accepted directly — export to ONNX first (e.g. `torch.onnx.export()`).

### 5.2 Runtime Library Setup

Create an `AI_Runtime/` directory and populate it with:
- the `.keras` model above,
- the prebuilt Cortex-M33 runtime library, e.g.
  `<STEdgeAI-install>/4.0/Middlewares/ST/AI/Lib/GCC/ARMCortexM33/NetworkRuntime1201_CM33_GCC.a`,
- the corresponding `AI/Inc` headers from the same install.

### 5.3 Analyze Before You Commit

Check the model's resource footprint against the MCU's actual budget (2 MB
flash / 786 KB SRAM) before generating code:

```bash
stedgeai analyze -m AI_Runtime/st_ign_wl_48.keras --target stm32u5
```
This reports MACC/inference, weight (flash) size, and required activation
(RAM) size.

To see what operations a given format supports on this target:
```bash
stedgeai supported-ops -t onnx
```

### 5.4 Generate the Full-Model C Code (baseline reference)

```bash
stedgeai generate -m AI_Runtime/st_ign_wl_48.keras --target stm32u5 -o HAR/AI
```
The generated `network.h` defines the `STAI_NETWORK_*` macros your
application code uses to size input/output buffers and activation memory.
Note that every `stai_*` API takes **pointer arrays**, even for a single
tensor — `stai_ptr inputs[1]` is correct, not a bare pointer.

### 5.5 Slice the Model for the Split Architecture

The whole point of this project's architecture is that only a small linear
head is trained on-device — the convolutional backbone stays frozen. Cut
the Keras model at its dense layer (`Auxiliary/slice_model.py` in this
repo) to produce an embedding-only ONNX export, then generate C code for
*that*:

```bash
python3 Auxiliary/slice_model.py
stedgeai generate --model AI_Runtime/st_ign_wl_48_embed.onnx --target stm32u5 --optimization balanced --name embed_network --output HAR/AI_EMBED
```

### 5.6 Extract the Pretrained Head as a Baseline

`Auxiliary/generate_baseline.py` extracts the original dense layer's
weights into the `BASELINE_W_INIT[4][13]` / `BASELINE_B_INIT[4]` C arrays
used by `trainable_head_init()`, and also emits a raw 224-byte
`checkpoints/baseline_weights.bin` you can hand to the FL server via
`--init-weights` so round 0 broadcasts the same baseline the boards were
actually flashed with, rather than an all-zero head.

---

## 6. Part C — On-Device Application Logic

### 6.1 Firmware Boot Flow

```
main.c: main() -> MX_ThreadX_Init()
app_threadx.c: MX_ThreadX_Init() -> tx_kernel_enter()
  (tx_api.h: tx_kernel_enter is _tx_initialize_kernel_enter)
tx_initialize_kernel_enter.c: _tx_initialize_kernel_enter() -> tx_application_define()
app_azure_rtos.c: tx_application_define()
  -> App_ThreadX_Init()   (app_threadx.c: starts the sensor/command/logger threads)
  -> MX_NetXDuo_Init()    (app_netxduo.c: starts the NetX Duo / FL threads)
app_netxduo.c: MX_NetXDuo_Init() -> App_SNTP_Thread_Entry() (repurposed as the FL client)
```

`main()` never returns from `MX_ThreadX_Init()` — anything after that call
in `main()` is dead code once the scheduler takes over.

### 6.2 ThreadX Threads

| Thread | Purpose | Priority |
|---|---|---|
| `tx_sensor_entry` | 26 Hz I2C sampling, windowing, backbone inference, head forward/train step | `TX_APP_THREAD_PRIO` (CubeMX-configured, high) |
| `logger_thread` | Drains a message queue and prints to UART | 15 |
| `press_thread` | Pressure sensor polling (unused by the HAR pipeline itself) | 15 |
| `command_thread` | Parses UART commands, woken by a semaphore from the RX ISR | 15 |
| `AppMainThread` | DHCP startup gate, resumes the FL thread once an IP is assigned | `DEFAULT_MAIN_PRIORITY` |
| `App_SNTP_Thread_Entry` (FL client) | NetX Duo TCP client: handshake, per-round weight exchange | `DEFAULT_PRIORITY` |

Synchronization: `i2c_mutex` (shared I2C2 bus access), `uart_mutex` (all
UART transmits, including from ISR-adjacent contexts, to prevent
interleaved output), `head_mutex` (guards the trainable head's `W`/`b`
against concurrent inference and training access — created and owned
internally by `trainable_head_init()`, and already locked/unlocked inside
`trainable_head_forward()`, `trainable_head_train_step()`,
`trainable_head_export_weights()`, and `trainable_head_import_weights()`,
so callers never need to lock it manually), `command_sem` (RX ISR → command
thread handoff), `fl_round_done_sem` (gates the FL thread until the
operator signals a round is done).

### 6.3 Sensor Pipeline

Each 26 Hz tick: read accelerometer + gyroscope over I2C2 (mutex-protected),
convert acceleration from milli-g to **m/s²** (`(raw/1000.0f) * 9.8f` — see
[§11.1](#111-model-locked-onto-stationary-regardless-of-real-motion) for why
the `*9.8f` matters), accumulate a 48-sample window with 50% overlap
(`WINDOW_SIZE 48`, `OVERLAP 24` — confirmed values, matching the
`st_ign_wl_48` backbone's expected input; these replaced `WINDOW_SIZE 24`
/ `OVERLAP 12` from the earlier `st_ign_wl_24` backbone), run
`HAR_PreprocessWindow` (gravity removal + Rodrigues rotation) and the
frozen backbone, and append a derived gyroscope-magnitude feature (mean
over the window, scaled by `/250.0f`) to form the 13-value feature vector.
Depending on `is_training_mode`, this feature vector either
runs through `trainable_head_forward()` for a live prediction, or
`trainable_head_train_step()` for one corrective SGD step against the
operator's current label — after which `is_training_mode` auto-clears back
to inference (each labeled correction is a one-shot step on the *next*
completed window, not continuous training).

### 6.4 The Trainable Head

A linear layer, `y = W·x + b`, with `W` a 4×13 matrix and `b` a 4-vector,
softmax over 4 classes
(`Jogging=0, Stationary=1, Stairs=2, Walking=3`). Loss is softmax
cross-entropy; because the head is linear, the gradient has closed form
`∂L/∂y_k = p̂_k − t_k`, so each training step
is a fixed sequence of dot products and outer-product updates — no
autodiff graph, no stored activations. `trainable_head_export_weights()` /
`import_weights()` flatten `W` (row-major) followed by `b` into a 56-float,
224-byte buffer — this exact layout is the FL wire payload (see
[Appendix A](#appendix-a-wire-protocol-byte-layout)).

Feature/class counts live in `trainable_head.h` as
`HEAD_NUM_CLASSES` (4), `HEAD_CNN_FEATURES` (12), `NUM_NEW_FEATURES` (1),
`HEAD_TOTAL_FEATURES` = `(HEAD_CNN_FEATURES + NUM_NEW_FEATURES)` = 13, and
`HEAD_FLAT_SIZE` = `(HEAD_NUM_CLASSES * HEAD_TOTAL_FEATURES + HEAD_NUM_CLASSES)`
= 56 — note that `HEAD_TOTAL_FEATURES` and `HEAD_FLAT_SIZE` are both
already correctly parenthesized in the actual header, avoiding a common C
macro pitfall where an unparenthesized multi-term macro silently
miscomputes once it's multiplied by something else elsewhere. The same
56-float/224-byte quantity is re-exposed under different names in
different files — `FL_PAYLOAD_FLOATS`/`FL_PAYLOAD_BYTES` in
`app_threadx.c`, `FL_WEIGHTS_BYTES` in `app_netxduo.c` — all of them
ultimately just aliasing `HEAD_FLAT_SIZE`, not independent values that
could drift apart.

### 6.5 UART Command Protocol

All commands are lowercase, space-free ASCII lines (the RX ISR silently
strips spaces, so multi-word commands aren't possible without further
firmware changes). See [Appendix B](#appendix-b-full-uart-command-set) for
the full table. Two commands exchange **raw binary**, not text —
`getweights` responds with a `WEIGHTS_DUMP\r\n` marker line followed
immediately by 224 raw bytes, and `setweights` expects exactly 224 raw
bytes right after its ack line (though as of this writing `setweights`
only buffers those bytes rather than applying them — see
[§11.12](#1112-setweights-receives-bytes-but-never-applies-them-confirmed-currently-unresolved)).
**Never send either through the NiceGUI
dashboard** — its parser splits the incoming stream on `\n`, and a raw
float's bit pattern can easily contain a `0x0A` byte, corrupting both the
binary payload and every line parsed afterward. Use a dedicated script or
raw terminal instead.

---

## 7. Part D — Wi-Fi & Federated Learning Networking

### 7.1 Host Hotspot Setup

Identify your Wi-Fi interface:
```bash
ip link
# e.g. wlp0s20f3
```

Create a dormant AP profile. The **2.4 GHz band restriction is mandatory**
— the EMW3080B does not support 5 GHz:
```bash
nmcli con add type wifi ifname wlp0s20f3 \
  con-name lab-hotspot autoconnect no \
  ssid "STM32-Lab" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk CoolPass \
  ipv4.method shared \
  ipv4.addresses 192.168.42.1/24
```
`ipv4.method shared` also starts a local DHCP server automatically, so
boards get an address in `192.168.42.0/24` with no extra configuration.

Open a firewall hole for DHCP (adjust for `nftables` vs `iptables`
depending on your distribution's default backend):
```bash
sudo nft insert rule ip filter INPUT  iifname "wlp0s20f3" udp dport 67 accept
sudo nft insert rule ip filter OUTPUT oifname "wlp0s20f3" udp dport 68 accept
```

Activate / deactivate / test:
```bash
nmcli con up lab-hotspot
nmcli con down lab-hotspot
ping 192.168.42.1        # the host itself
ping 192.168.42.167      # a board, once connected
```
Note that activating the hotspot switches your Wi-Fi adapter into AP
("Master") mode — your OS will show you as disconnected from the internet
because most adapters cannot be an AP and a client simultaneously. This is
expected, not a fault.

### 7.2 EMW3080 Module Firmware (one-time, per board)

The EMW3080 is a **separate chip** with its own firmware, independent of
whatever you flash onto the STM32 — this only needs doing once per board
(or again if you need to update it):
1. Put the module into its update mode by pressing the board's blue button,
   then type `flash` and press Enter in the serial terminal.
2. Transfer `EMW3080update_B-U585I-IOT02A-RevC_V2.3.4_SPI.bin` to the board.
3. Wait for the update to complete (a few seconds).

### 7.3 The FL TCP Client Thread

Repurposed from the lab's `App_SNTP_Thread_Entry`. On boot: create a TCP
socket (with the 1536-byte receive window from §7.3 below already applied
at creation), bind, connect to the server's IP:port, receive the 4-byte
little-endian round-count handshake, then for each round: receive 224
bytes of global weights (with a very generous timeout —
`NX_IP_PERIODIC_RATE * 10000` in the current firmware — since there's no
way to know in advance how long the operator will take to finish labeling
gestures before the *next* round's data even starts flowing; this receive
is really "wait for global weights," not "wait a bounded amount of time"),
`trainable_head_import_weights()`, block on `fl_round_done_sem` until the
operator sends `done` over UART, export the locally-trained weights, and
send them back. After the *last* round completes, there's a distinct final
step: the client does one more `fl_tcp_receive_exact()` call (with a much
shorter timeout, `NX_IP_PERIODIC_RATE * 5`, since the server sends this
immediately after the last round's aggregation — there's nothing to wait
on here) for the fully-aggregated final model, applies it via
`trainable_head_import_weights()` if it arrives, and logs a non-fatal
warning (not a `break`/abort) if it doesn't — the session is already
complete either way at that point, so a missed final broadcast just means
this board's in-SRAM head stays at its last locally-trained state rather
than the aggregated one.

The single most important correctness detail here is **`fl_tcp_receive_exact`**
— see [§11.4](#114-boards-hang-non-deterministically-during-fl-rounds).
A single `nx_tcp_socket_receive()` call is not guaranteed to return an
entire 224-byte message; this helper loops, accumulating bytes via
`nx_packet_data_extract_offset()` across as many receive calls as it takes:

```c
static UINT fl_tcp_receive_exact(NX_TCP_SOCKET *socket_ptr, UCHAR *dest,
                                  ULONG n_bytes, ULONG wait_option)
{
  ULONG total_received = 0;
  NX_PACKET *packet_ptr;
  UINT status;
  ULONG bytes_copied;

  while (total_received < n_bytes)
  {
    status = nx_tcp_socket_receive(socket_ptr, &packet_ptr, wait_option);
    if (status != NX_SUCCESS)
    {
      printf("[fl_tcp_receive_exact] nx_tcp_socket_receive failed: 0x%02X\r\n", status);
      return status;
    }

    status = nx_packet_data_extract_offset(packet_ptr, 0, dest + total_received,
                                            n_bytes - total_received, &bytes_copied);
    nx_packet_release(packet_ptr);

    if (status != NX_SUCCESS)
    {
      printf("[fl_tcp_receive_exact] extract_offset failed: 0x%02X\r\n", status);
      return status;
    }
    if (bytes_copied == 0)
    {
      /* guard against an infinite loop if 0 bytes ever comes back */
      return NX_NOT_SUCCESSFUL;
    }

    total_received += bytes_copied;
    printf("[FL] recv chunk: %lu bytes (total %lu/%lu)\r\n", bytes_copied, total_received, n_bytes);
  }
  return NX_SUCCESS;
}
```
(the real function also prints a `"0 bytes extracted! Continuing wait..."`
line immediately before that `return NX_NOT_SUCCESSFUL` — the message text
is stale/misleading since the code actually returns rather than
continuing; harmless, but worth a one-line fix if you're already in there)

Use this for **every** fixed-size receive on this socket, including the
4-byte handshake — not just the 224-byte payload.

---

## 8. Part E — Host-Side Software

### 8.1 FL Aggregation Server (`server.py`)

A threaded TCP server: waits for exactly `--num-clients` connections, sends
each a 4-byte round-count handshake, then runs `--num-rounds` lockstep
rounds using a `threading.Barrier` to synchronize all connected clients
before each FedAvg aggregation step. Saves a checkpoint (raw `.bin` +
inspectable `.npz`) after every round. On every accepted connection,
`TCP_NODELAY` avoids Nagle-induced coalescing delays on the small per-round
messages, `SO_KEEPALIVE` is enabled, and `SO_LINGER` is set to a 5-second
timeout so buffered data isn't dropped on close. After the final round, the
server sends one additional broadcast of the fully-aggregated model and
sleeps for 1 second before closing that connection — giving the board time
to actually read the final model off the wire before the socket goes away,
rather than racing a `FIN`/`RST` against it.

```bash
# Single board, starting from an all-zero head
python3 server.py --num-clients 1 --num-rounds 2

# Single board, starting from the extracted pretrained baseline
python3 server.py --num-clients 1 --num-rounds 2 --init-weights checkpoints/baseline_weights.bin

# Both boards, resuming from the last saved global checkpoint
python3 server.py --num-clients 2 --num-rounds 2 --init-weights checkpoints/global_latest.bin
```

### 8.2 Dashboard (`main.py` / `uart_backend.py`)

A NiceGUI web app per board: live sensor plot, classification display with
per-class confidence bars, Start/Stop/LED controls, four labeling buttons
(one per activity class), a **"Finish Round"** button that sends `done`
over UART to release the FL thread's semaphore, and a raw log/command box
for anything else. Because each dashboard instance is its own NiceGUI web
server, running two at once requires distinct web ports:

```bash
uv run main.py /dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_<SN1>-if02 8080
uv run main.py /dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_<SN2>-if02 8081
```

### 8.3 Bridge / Test Utilities (development scaffolding)

`fake_client.py` and `flaky_client.py` simulate a board's TCP-side
behavior (including a mid-round disconnect) so the server can be exercised
without any hardware attached. `bridge.py` + `serial_io.py` were an interim
tool that relayed a real board's `GET_WEIGHTS`/`SET_WEIGHTS` UART exchange
to the TCP server *before* the NetX Duo FL client thread existed on-device
— now that `App_SNTP_Thread_Entry` talks to the server directly over
Wi-Fi, these are no longer needed for normal operation, but remain useful
as an isolated way to test one board's UART weight exchange without
involving the network stack at all, if a future bug needs narrowing down
to "UART side" vs "Wi-Fi side."

---

## 9. End-to-End: Running a Full Session

Order matters — see [§11.6](#116-fl-connect-failed-0x38) for why the server
must be listening *before* the boards boot.

1. **Bring up the hotspot** on the host: `nmcli con up lab-hotspot`.
2. **Start the FL server first**, before touching the boards:
   ```bash
   python3 server.py --num-clients 2 --num-rounds 2 --init-weights checkpoints/baseline_weights.bin
   ```
3. **Start both dashboards**, using persistent `/dev/serial/by-id/...` paths
   (never bare `/dev/ttyACMx` — see
   [§11.5](#115-boards-swap-identities--dashboard-shows-no-data)):
   ```bash
   uv run main.py /dev/serial/by-id/usb-...SN1...-if02 8080
   uv run main.py /dev/serial/by-id/usb-...SN2...-if02 8081
   ```
4. **Reset both boards** (physical black RESET button, or less preferrably a reflash) so
   they boot cleanly and connect to the already-listening server. **Note: sometimes, you need to try this more than one time for successful connection.**
5. Watch both dashboards' logs for `FL connected to server` and `FL server
   requested N round(s)`.
6. For each round: use the label buttons on each dashboard to correct the
   live prediction against the physical gesture you're performing on that
   board, for as long as you want that round's local training to run, then
   click **Finish Round**.
7. Once the server logs the round complete on both clients, it broadcasts
   the new global model automatically and the next round begins — repeat
   step 6.
8. After the final round, the server sends the fully-aggregated final
   model once more; both dashboards should show it applied, and both
   connections close cleanly.

---

## 10. Command Reference

| Task | Command |
|---|---|
| List connected boards | `STM32_Programmer_CLI -l` |
| Build | `cmake --preset Debug && cmake --build --preset Debug` |
| Flash (hardware reset) | `STM32_Programmer_CLI -c port=SWD mode=UR reset=HWrst sn=<SN> -w build/Debug/<proj>.elf -s` |
| Serial monitor | `picocom -b 115200 /dev/ttyACM0` |
| Find persistent serial paths | `ls -l /dev/serial/by-id/` |
| Start hotspot | `nmcli con up lab-hotspot` |
| Stop hotspot | `nmcli con down lab-hotspot` |
| Start FL server (fresh) | `python3 server.py --num-clients 2 --num-rounds N` |
| Start FL server (from baseline) | `python3 server.py --num-clients 2 --num-rounds N --init-weights checkpoints/baseline_weights.bin` |
| Start dashboard | `uv run main.py <serial-path> <web-port>` |
| Raw UDP connectivity test | `echo "hello udp" \| nc -u -4 -w1 <board-ip> 6000` |

---

## 11. Troubleshooting Guide

### 11.1 Model locked onto "Stationary" regardless of real motion
**Cause:** the accelerometer read path was switched from raw ADC counts
(manually scaled to m/s² including a `*9.8f` g→m/s² conversion) to the
calibrated `BSP_MOTION_SENSOR_GetAxes()` API (returns milli-g), and only
the `/1000.0f` (mg→g) step was kept — silently dropping the model's input
magnitude by a factor of ~9.8×, far below what the frozen backbone was
trained on.
**Fix:** `(raw / 1000.0f) * 9.8f`, converting all the way to m/s².

### 11.2 UART commands have zero effect (start/stop/l0-3/etc. all silent)
**Cause:** `HAL_NVIC_EnableIRQ(USART1_IRQn)` was never actually called, and
`stm32u5xx_it.c` had no `USART1_IRQHandler` routing into
`HAL_UART_IRQHandler()` — likely a CubeMX/TrustZone code-generation gap.
TX (via blocking `HAL_UART_Transmit`) worked fine and masked the problem,
since it doesn't depend on interrupts at all; only RX was actually broken.
**Fix:** manually add both, in `USER CODE` regions so they survive
regeneration:
```c
/* main.c, before HAL_UART_Receive_IT priming */
HAL_NVIC_SetPriority(USART1_IRQn, 5, 0);
HAL_NVIC_EnableIRQ(USART1_IRQn);
```
```c
/* stm32u5xx_it.c */
void USART1_IRQHandler(void) { HAL_UART_IRQHandler(&huart1); }
```
**How to confirm this specific cause:** temporarily echo every received
byte straight back inside the RX ISR. If nothing echoes even while typing
into the terminal, the ISR is never firing — confirming an NVIC/vector
problem rather than a bug in the command parser itself.

### 11.3 Testing `getweights`/`setweights` through the dashboard corrupts everything
**Cause:** these commands exchange raw binary; the dashboard's parser
splits on `\n`, and a coincidental `0x0A` byte inside a float's bit pattern
desyncs the parser for every line afterward, not just that one response.
**Fix:** never use the dashboard for these two commands. Use a raw serial
terminal or a small dedicated script instead.

### 11.4 Boards hang non-deterministically during FL rounds
**Cause 1 (fragmentation):** a single `nx_tcp_socket_receive()` call was
assumed to return an entire 224-byte payload. TCP gives no message-boundary
guarantee; over an SPI-bridged Wi-Fi link, a payload can legitimately
arrive split across multiple receives. **Fix:** `fl_tcp_receive_exact()`
(§7.3), used for every fixed-size receive on the socket.
**Cause 2 (dangling pointer, found during code review after the first
fix):** leftover code from before the `fl_tcp_receive_exact()` fix kept
calling `nx_packet_data_extract_offset(recv_packet, ...)` /
`nx_packet_release(recv_packet)` on a packet pointer that the new helper
function never actually set — `recv_packet` was stale or uninitialized,
risking packet-pool corruption or a double-release. **Fix:** delete the
leftover extract/release calls entirely once `fl_tcp_receive_exact()`
already populated the destination buffer; delete the now-unused
`recv_packet` variable too, so a future edit can't reintroduce a read of it.
**Diagnostic tip:** the current `fl_tcp_receive_exact()` already logs
`[FL] recv chunk: N bytes (total X/Y)` on every chunk it extracts (plus a
distinct error print on each failure path) — if a 224-byte transfer ever
takes more than one such line, that's direct confirmation fragmentation is
really happening on your link, independent of whichever other bug is
currently suspected. No need to add this instrumentation yourself; just
watch for it in `picocom`.

### 11.5 Boards swap identities / dashboard shows no data
**Cause:** `/dev/ttyACM0` / `/dev/ttyACM1` are assigned by USB enumeration
order, which is **not stable** across a board reset or brownout (e.g. from
peak Wi-Fi TX current on both boards simultaneously). A board can silently
swap device nodes, leaving a dashboard reading a dead handle or the wrong
board entirely.
**Fix:** bind dashboards to the persistent, hardware-serial-derived paths
instead:
```bash
ls -l /dev/serial/by-id/
# usb-STMicroelectronics_ST-LINK_V3_<SN1>-if02 -> ../../ttyACM0
# usb-STMicroelectronics_ST-LINK_V3_<SN2>-if02 -> ../../ttyACM1
uv run main.py /dev/serial/by-id/usb-STMicroelectronics_ST-LINK_V3_<SN1>-if02 8080
```

### 11.6 `FL connect failed (0x38)`
**Cause:** `0x38` is `NX_NOT_CONNECTED`. The board attempted its TCP
handshake before `server.py` was listening, and the FL client thread has
no retry logic — it gives up immediately on a failed connect.
**Fix:** always start `server.py` *before* resetting or powering on the
boards (see the ordering in [§9](#9-end-to-end-running-a-full-session)).

### 11.7 Only one board logs output at a time / one board goes silent
This is usually **not** one bug but a combination of §11.5 (device
renumbering after a reset/brownout) and a dashboard's pyserial read thread
blocking forever on a now-dead file handle after the underlying device
disappears. Binding by `/dev/serial/by-id/` (§11.5) resolves the
renumbering half; if a dashboard still goes silent after a board-side
reset, restart that specific dashboard instance rather than assuming the
firmware hung.

### 11.8 `[Errno 98] Address already in use` running a second dashboard
**Cause:** NiceGUI defaults to web port 8080 for every instance.
**Fix:** accept a web-port CLI argument and pass distinct ports:
```python
web_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
...
ui.run(title=f"HAR Dashboard ({serial_port})", port=web_port, reload=False)
```

### 11.9 Board needs two flashes to connect to Wi-Fi
**Cause:** `STM32_Programmer_CLI ... -s` performs a **software** reset,
which only resets the STM32 core — the EMW3080 module (a separate chip on
SPI) is not power-cycled. If the STM32 resets mid-handshake with the
module, the SPI state machine is left desynced, and the first boot after
flashing fails to associate; a second flash (or reset) happens to catch the
module already idle.
**Fix:** use a **hardware** reset instead, which resets both chips
together — either the physical black RESET button, or:
```bash
STM32_Programmer_CLI -c port=SWD mode=UR reset=HWrst sn=<SN> -w build/Debug/<proj>.elf -s
```

### 11.10 GUI never shows any `PRED`/`TRAINED` lines at all
Confirm, in order: (1) the dashboard shows "Connected", not just that you
clicked Connect; (2) `start` was sent (streaming may be paused); (3) the
correct `/dev/serial/by-id/...` path was used for *this* board, not the
other one (§11.5).

### 11.11 Weight synchronization looks wrong after aggregation
Pull each board's live weights via `getweights` (§6.5, never through the
dashboard — §11.3) and diff the raw 224-byte payload against the server's
`checkpoints/global_round_XXXX.bin` for that round. A mismatch narrows the
problem to either the board's weight-import path or the server's FedAvg
math — the two are independently testable (the Python side has its own
unit-testable `fedavg()` function). Note the caveat in §11.12 below before
relying on `setweights` for this — as currently wired, it doesn't actually
load anything into the head.

### 11.12 `setweights` receives bytes but never applies them (confirmed, currently unresolved)
Tracing the current firmware directly (not from a report — read straight
from the real `app_threadx.c`): `process_command()`'s `setweights` branch
correctly arms `awaiting_weights_payload`, and the RX ISR correctly
accumulates the next 224 incoming bytes into `rx_weights_buffer` and then
calls `tx_semaphore_put(&weights_ready_sem)`. But:
- **`weights_ready_sem` is never created** — there's no
  `tx_semaphore_create(&weights_ready_sem, ...)` anywhere in
  `App_ThreadX_Init`. Every other semaphore in this file is created there;
  this one was declared but never wired up.
- **Nothing ever waits on it.** There is no `weights_thread` (or any other
  consumer) calling `tx_semaphore_get(&weights_ready_sem, ...)` and then
  `trainable_head_import_weights(rx_weights_buffer)` — that consumer
  existed in an earlier version of this file as a commented-out
  `weights_thread_entry`, and has since been removed entirely, not just
  left disabled.

**Net effect:** sending `setweights` today receives your 224 bytes into a
buffer and then does nothing further with them — the board's actual head
weights are untouched. This does **not** affect the FL round loop, which
never uses this UART path at all: `App_SNTP_Thread_Entry` calls
`trainable_head_import_weights()` directly in C on every round. It only
matters if you're relying on `setweights` for manual testing (e.g. a
`getweights` → `setweights` round-trip check, or reviving the old
`bridge.py` workflow from earlier in this project).

**To actually finish this feature:** add the missing
`tx_semaphore_create(&weights_ready_sem, "Weights SEM", 0)` in
`App_ThreadX_Init`, and either create a small dedicated thread or extend
`command_thread_entry` to also wait on it and call
`trainable_head_import_weights((float *)rx_weights_buffer)` when it fires.

---

## 12. Extending the Project

- **Differential privacy:** add calibrated noise to the exported update in
  `trainable_head_export_weights()` before transmission.
- **Payload compression:** the 224-byte payload is small already, but
  quantizing it (e.g. to int8 with a shared scale) would cut round-trip
  bandwidth further if scaling to many more clients or a slower link.
- **Asynchronous / partial participation:** the current server uses a
  `threading.Barrier`, which requires every registered client to submit
  before any round advances. Scaling past a handful of clients, or
  tolerating slow/dropped clients gracefully, means moving to a
  timeout-based or asynchronous aggregation scheme instead.
- **On-device retry logic:** the FL client thread currently gives up
  permanently on a failed connect (§11.6) rather than retrying — worth
  adding if boards need to survive the server starting late or restarting
  mid-deployment.
- **NetX Duo-native labeling:** the current design routes gesture labels
  over UART from a laptop-tethered dashboard; a longer-term goal from the
  original design notes is decoupling labeling from any wired connection
  entirely (e.g. a phone app talking to the board directly over Wi-Fi).

---

## Appendix A: Wire Protocol Byte Layout

The 224-byte FL payload is exactly what
`trainable_head_export_weights()`/`import_weights()` read and write — no
repacking needed on either end:

| Bytes | Content |
|---|---|
| 0–207 | `W`, row-major, shape (4 classes, 13 features) = 52 × float32 |
| 208–223 | `b`, shape (4,) = 4 × float32 |

All values are **little-endian float32**, matching the Cortex-M33's native
byte order — the MCU never needs to byte-swap in either direction.

Session handshake (once, immediately after TCP accept):
`server → client`: 4 bytes, little-endian `uint32_t` = total round count.

Per round: `server → client`: 224-byte global weights. `client → server`:
224-byte locally-updated weights. After the final round, the server sends
one additional 224-byte broadcast of the fully-aggregated model before
closing the connection.

## Appendix B: Full UART Command Set

| Command | Effect |
|---|---|
| `start` | Resume streaming sensor/prediction log lines |
| `stop` | Pause streaming (board keeps running, just stops printing) |
| `led` | Toggle the onboard LED (basic liveness check) |
| `infer` | Force inference mode, cancelling any pending one-shot training request |
| `l0` / `l1` / `l2` / `l3` | Queue one corrective SGD step against the given class label (`0=Jogging, 1=Stationary, 2=Stairs, 3=Walking`) on the next completed sensor window, then auto-revert to inference |
| `getweights` | Board replies with `WEIGHTS_DUMP\r\n` followed immediately by 224 raw bytes (current head weights) — **not** through the dashboard, see §11.3 |
| `setweights` | Board replies with an ack, then expects exactly 224 raw bytes next, which it buffers — **as currently wired, these bytes are never actually applied to the head; see §11.12** — and **not** through the dashboard either way |
| `done` | Signals the FL client thread that this round's local training is finished; releases `fl_round_done_sem` so the board exports and transmits its updated weights |

## Appendix C: Live Telemetry Line Formats

These are the plain-ASCII lines the sensor thread prints during normal
operation (distinct from the binary FL payload in Appendix A) — this is
what the dashboard's `uart_backend.py` actually parses out of the serial
stream:

```
PRED <class> <p0> <p1> <p2> <p3> <dyn> <gyro>
TRAINED <class> <loss> <dyn> <gyro>
```
- `<class>`: predicted/trained-on class index (0–3)
- `<p0..p3>`: softmax probabilities for each class, `%.3f`
- `<dyn>`: peak preprocessed dynamic acceleration magnitude for that window, in g, `%.2f`
- `<gyro>`: mean gyroscope rotational magnitude for that window, in dps, `%.1f`
- `<loss>` (TRAINED only): cross-entropy loss for that corrective step, `%.4f`

Example: `PRED 1 0.021 0.912 0.045 0.022 0.34 12.5` — predicted Stationary
at 91.2% confidence.

