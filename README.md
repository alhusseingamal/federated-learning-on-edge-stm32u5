# On-Device Federated Learning for Human Activity Recognition

Two STM32U585 microcontrollers train a shared activity-recognition model
collaboratively over Wi-Fi — no cloud, no raw sensor data ever leaving
either device. Only a 224-byte parameter update is exchanged per round.

**Status:** research prototype. Core FL loop (local training → Wi-Fi
exchange → FedAvg → redistribution) is validated end-to-end on physical
hardware across multiple rounds. See [Known Limitations](#known-limitations)
before treating this as production-ready.

## Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Known Limitations](#known-limitations)
- [Acknowledgments](#acknowledgments)

## Overview

Each board runs a hybrid model: a frozen, quantized convolutional backbone
(`st_ign_wl_48`, from [ST's model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo))
turns a 48-sample accelerometer window into a 12-dimensional embedding,
combined with a gyroscope-magnitude feature into a 13-dimensional vector
fed to a small **trainable linear head** — 4 classes
(`Jogging`, `Stationary`, `Stairs`, `Walking`) × 13 features + 4 biases,
56 parameters total.

The head is trained **on the device**, in place, via closed-form softmax
cross-entropy SGD — no autodiff, no stored activations — triggered by
human-in-the-loop gesture labels sent from a browser dashboard over UART.
Each board periodically exchanges its 224-byte parameter vector with a
Python aggregation server over a persistent TCP socket (Azure RTOS
ThreadX + NetX Duo on the MCU side), running a lockstep Federated
Averaging protocol.

## Architecture

![Project Architecture](Auxiliary/docs/Architecture.png)  

## Repository Structure

```
firmware/
├── Core/Src/main.c                # Peripheral init, UART RX priming
├── AZURE_RTOS/App/app_threadx.c   # Sensor thread, command parser, mutexes
├── NetXDuo/App/app_netxduo.c      # FL TCP client thread
├── HAR/trainable_head.c/.h        # On-device linear head + SGD
└── Auxiliary/                     # Model slicing / baseline-extraction scripts

host/
├── server/
│   ├── protocol.py                # Wire format + socket helpers
│   ├── aggregate.py                # FedAvg math
│   ├── server.py                   # FL aggregation server
│   ├── fake_client.py              # Simulated board, for testing without hardware
│   └── checkpoints/                # Saved global models per round
└── dashboard/
    ├── main.py                     # NiceGUI dashboard
    └── uart_backend.py             # pyserial transport
```

## Hardware Requirements

- 2× STMicroelectronics **B-U585I-IOT02A** Discovery Kits (onboard
  ISM330DHCX IMU, MXCHIP EMW3080B Wi-Fi module, ST-LINK V3)
- A host machine with a Wi-Fi adapter capable of 2.4 GHz AP (hotspot) mode

## Quick Start

Full setup (CubeMX configuration, ST Edge AI pipeline, EMW3080 firmware,
build/flash workflow) is in **[PROJECT_GUIDE.md](Auxiliary/docs/PROJECT_GUIDE.md)** — this
is the condensed happy path once everything's already built and flashed.

```bash
# 1. Bring up the host hotspot
nmcli con up lab-hotspot

# 2. Start the FL server BEFORE touching the boards
python3 server.py --num-clients 2 --num-rounds 2 --init-weights checkpoints/baseline_weights.bin

# 3. Start both dashboards, using persistent serial paths
uv run main.py /dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_<SN1>-if02 8080
uv run main.py /dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_<SN2>-if02 8081

# 4. Reset both boards to trigger connection to server. (physical button or a reflash)
```
Then use each dashboard's label buttons to correct live predictions
against the gesture you're performing, click **Finish Round** when done,
and repeat once the server broadcasts the next round's global model.

## Documentation

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — full reproduction guide:
  CubeMX peripheral setup, the ST Edge AI model pipeline, the complete
  wire protocol, every UART command, and a symptom-indexed troubleshooting
  section covering every failure mode hit while building this (TCP
  fragmentation over the SPI-bridged Wi-Fi link, USB device renumbering,
  a missing NVIC interrupt enable, and more).
- **`Edge_Native_Federated_Learning__Synchronized_On_Device_Training_and_Real_Time_Inference_on_Dual_ARM_Cortex_M33_Microcontrollers.pdf`** — a short IEEE-format writeup of the system
  architecture, training formulation, and empirical convergence results,
  if you want a citable, condensed technical summary instead of the full guide.

## Known Limitations

- **Synchronous aggregation:** the server uses a blocking barrier — every
  registered client must submit before any round advances. One dropped
  connection stalls the whole session rather than degrading gracefully to
  the remaining `K-1` clients.
- **Frozen backbone ceiling:** since only the linear head adapts, classes
  with overlapping latent-space projections at the embedding stage (e.g.
  Walking vs. Stairs) can't be fully separated no matter how much the head
  is corrected. Improving this means fine-tuning or replacing the backbone
  itself, not just training longer on-device.
- **`setweights` is currently non-functional** — it buffers an incoming
  224-byte UART payload but nothing applies it to the head (the intended
  consumer thread was removed at some point). Doesn't affect the FL loop
  itself, which loads weights directly in C, not over UART. See
  `PROJECT_GUIDE.md` §11.12 for the exact fix.
- **Wired labeling:** gesture labels are sent over a USB-tethered UART
  connection, so live labeling during real walking/stairs motion is
  constrained by cable length, not just Wi-Fi range.
- **No client-side retry:** if a board's FL thread fails to connect (e.g.
  the server wasn't listening yet), it doesn't retry — it gives up for
  that boot cycle.

## Acknowledgments

- Backbone model: [ST's STM32 AI model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo),
  trained on the [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) activity-recognition dataset.
- Networking baseline adapted from ST's `Nx_SNTP_Client` Azure RTOS example.
- FedAvg: McMahan et al., *"Communication-Efficient Learning of Deep
  Networks from Decentralized Data"*, AISTATS 2017.
