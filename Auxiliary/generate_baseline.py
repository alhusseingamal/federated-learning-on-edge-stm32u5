## Note: this hardcodes the values from Core/Inc/head_weights_init.h
## Later: extend it to read from it rather than hardcode here (useful in case head_weights_init.h changes)
import struct

# 12 pretrained CNN feature weights per class (from head_weights_init.h)
BASELINE_W_INIT = [
    [ +0.41144618, -0.27634409, -0.41994816, -0.30799478, -0.25349864, +0.60604203,
      +0.41643548, -0.26220688, -0.40759113, -0.54590893, -0.52807373, +0.34769195 ],
    [ -0.29802987, +0.57246256, +0.20034041, +0.29542601, +1.05006313, +0.30830652,
      -0.45093498, +0.39254665, +0.41561195, -0.77681679, +0.63434869, +0.44714406 ],
    [ +0.35085478, +0.30025128, +0.17852283, +0.27557778, -0.22677901, +0.55973315,
      -0.20593944, +0.37063166, -0.39014873, -0.00643534, -0.47656161, -0.23919868 ],
    [ -0.64062786, -0.27995038, +0.18473996, +0.29065681, -0.22112809, -0.55255896,
      +0.42830521, +0.38183263, -0.38785842, +0.03242418, +0.52118963, -0.28830707 ],
]

# Baseline biases per class
BASELINE_B_INIT = [-0.32412970, -0.89685374, +0.36461982, +0.38826057]

flat_floats = []

# Exactly mirrors trainable_head_export_weights() memory layout:
# 1. Pack W[4][13] (12 pretrained features + 1 zero for Gyro)
for c in range(4):
    flat_floats.extend(BASELINE_W_INIT[c])
    flat_floats.append(0.0)  # index 12: FEAT_GYRO initialized to 0.0f
    # flat_floats.append(0.0)  # index 13: FEAT_PRESS initialized to 0.0f

# 2. Pack b[4]
flat_floats.extend(BASELINE_B_INIT)

# 3. Pack into 56 little-endian 32-bit single-precision floats (224 bytes)
binary_payload = struct.pack(f"<{len(flat_floats)}f", *flat_floats)

with open("baseline_weights.bin", "wb") as f:
    f.write(binary_payload)

print(f"Successfully generated baseline_weights.bin: {len(binary_payload)} bytes ({len(flat_floats)} floats)")