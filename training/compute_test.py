import os
import platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_efficient_kan import KANLinear

MODEL_PATH = "../model_testing/final_one_step_fcst_wide.keras"

model = load_model(
    MODEL_PATH,
    custom_objects={"KANLinear": KANLinear}
)

def count_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

def count_non_trainable_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))

print("Model summary")
model.summary()

print("\nLightweightness")
print("-" * 40)
print(f"Input shape:              {model.input_shape}")
print(f"Output shape:             {model.output_shape}")
print(f"Total layers:             {len(model.layers)}")
print(f"Total parameters:         {model.count_params():,}")
print(f"Trainable parameters:     {count_trainable_params(model):,}")
print(f"Non-trainable parameters: {count_non_trainable_params(model):,}")
print(f"Model size:               {os.path.getsize(MODEL_PATH) / (1024 ** 2):.2f} MB")

print("\nCompute currently detected")
print("-" * 40)
print(f"Platform: {platform.platform()}")
print(f"TensorFlow: {tf.__version__}")
print(f"CPUs: {tf.config.list_physical_devices('CPU')}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")