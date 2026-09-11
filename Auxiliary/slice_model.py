import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tf2onnx

model_path = "AI_Runtime/st_ign_wl_48.keras"
onnx_path = "AI_Runtime/st_ign_wl_48_embed.onnx"
header_path = "Core/Inc/head_weights_init.h"

# 1. Load trained model
model = keras.models.load_model(model_path)

# 2. Extract backbone ending at 'dense' (12-dim embedding)
dense_layer = model.get_layer("dense")
embed_model = keras.Model(inputs=model.inputs, outputs=dense_layer.output)

# 3. Wrap in tf.function with FIXED batch dimension (1, 24, 3, 1) for MCU static allocation
input_spec = (tf.TensorSpec(shape=(1, 48, 3, 1), dtype=tf.float32, name="input_tensor"),)

@tf.function(input_signature=input_spec)
def serve_model(x):
    return embed_model(x)

# Pass the tf.function directly (tf2onnx calls .get_concrete_function internally)
model_proto, _ = tf2onnx.convert.from_function(
    function=serve_model,
    input_signature=input_spec,
    opset=13,
    output_path=onnx_path
)
print(f"[+] Exported backbone to ONNX: {onnx_path}")

# 4. Extract dense_1 parameters for linear head initialization
dense_1 = model.get_layer("dense_1")
weights, bias = dense_1.get_weights()
# Keras weight matrix is (12, 4) -> Transpose to (4, 12) for C array [class][feature]
W_init = weights.T

os.makedirs(os.path.dirname(header_path), exist_ok=True)
with open(header_path, "w") as f:
    f.write("#ifndef HEAD_WEIGHTS_INIT_H\n#define HEAD_WEIGHTS_INIT_H\n\n")
    f.write("#define PRETRAINED_NUM_CLASSES  4\n")
    f.write("#define PRETRAINED_NUM_FEATURES 12\n\n")
    
    f.write("// Baseline weights: [class][feature]\n")
    f.write("static const float BASELINE_W_INIT[PRETRAINED_NUM_CLASSES][PRETRAINED_NUM_FEATURES] = {\n")
    for row in W_init:
        f.write("    { " + ", ".join(f"{val:+.8f}f" for val in row) + " },\n")
    f.write("};\n\n")
    
    f.write("// Baseline biases: [class]\n")
    f.write("static const float BASELINE_B_INIT[PRETRAINED_NUM_CLASSES] = {\n    ")
    f.write(", ".join(f"{val:+.8f}f" for val in bias) + "\n};\n\n")
    f.write("#endif // HEAD_WEIGHTS_INIT_H\n")

print(f"[+] Exported baseline head parameters to: {header_path}")