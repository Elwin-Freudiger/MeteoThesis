import json
import zipfile
from pathlib import Path

model_path = Path("../model_testing/classifier_mlp.keras")

with zipfile.ZipFile(model_path, "r") as z:
    for name in z.namelist():
        if "config.json" in name:
            with z.open(name) as f:
                config = json.load(f)

def find_quantization(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "quant" in k.lower():
                print(f"FOUND: {path}.{k}")
            find_quantization(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_quantization(v, f"{path}[{i}]")

find_quantization(config)
