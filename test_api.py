import torch
from pathlib import Path
from src.model import build_baseline_model
from src.gateway import load_gateway_model
from src.near_ood import load_feature_bank

print("1. build_baseline_model")
model = build_baseline_model()

print("2. load_gateway_model")
gw_model = load_gateway_model("cuda")

print("3. load_feature_bank")
feature_bank = load_feature_bank(Path("models"), "cuda")

print("Done!")
