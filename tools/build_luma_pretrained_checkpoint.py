#!/usr/bin/env python3
"""
build_luma_pretrained_checkpoint.py
===================================
Transfers 100% of matching convolutional & FPN/PAN weights from yolo26s.pt
into the YOLO26-LUMA-PCB architecture (Feng et al. 2026).

Maps:
- Layer 0 in source (Conv 3->64) -> Layer 1 in LUMA
- Layer 1 in source (Conv 64->128) -> Layer 2 in LUMA (AConv)
- Backbone C3k2 stages (P2, P3, P4, P5)
- Neck C3k2 and AConv stages
- Initializes LIAM to exact identity (residual = 0) so the model starts with
  uncompromised pretrained feature extraction before learning glare calibration.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from luma_modules import register_luma_modules
from ultralytics import YOLO

def main():
    register_luma_modules()
    
    yaml_path = Path("models/yolo26s-luma-pcb.yaml")
    source_pt = Path("yolo26s.pt")
    out_pt = Path("weights/yolo26s_luma_pretrained.pt")
    
    assert yaml_path.exists(), f"Missing {yaml_path}"
    assert source_pt.exists(), f"Missing {source_pt}"
    
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading source weights from: {source_pt}")
    ckpt_src = torch.load(source_pt, map_location="cpu", weights_only=False)
    source_sd = ckpt_src["model"].state_dict() if "model" in ckpt_src else ckpt_src
    
    print(f"🏗️  Instantiating LUMA-PCB architecture from: {yaml_path}")
    model = YOLO(str(yaml_path))
    target_model = model.model
    target_sd = target_model.state_dict()
    
    # Semantic layer correspondence: Source (yolo26s) -> Target (yolo26s-luma-pcb)
    layer_map = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
        13: 14, 16: 18, 17: 20, 19: 22, 20: 23, 22: 25, 23: 26,
    }
    
    matched_keys = 0
    matched_params = 0
    total_params = sum(p.numel() for p in target_sd.values())
    
    for k_src, v_src in source_sd.items():
        parts = k_src.split(".")
        if len(parts) >= 3 and parts[0] == "model":
            try:
                src_layer = int(parts[1])
                if src_layer in layer_map:
                    tgt_layer = layer_map[src_layer]
                    k1 = f"model.{tgt_layer}." + ".".join(parts[2:])
                    k2 = f"model.{tgt_layer}.conv." + ".".join(parts[2:])
                    if k1 in target_sd and target_sd[k1].shape == v_src.shape:
                        target_sd[k1] = v_src
                        matched_keys += 1
                        matched_params += v_src.numel()
                    elif k2 in target_sd and target_sd[k2].shape == v_src.shape:
                        target_sd[k2] = v_src
                        matched_keys += 1
                        matched_params += v_src.numel()
            except ValueError:
                pass
                
    # Initialize LIAM to exact mathematical identity: output = x + 0
    target_sd["model.0.out_conv.weight"] = torch.zeros_like(target_sd["model.0.out_conv.weight"])
    target_sd["model.0.out_conv.bias"] = torch.zeros_like(target_sd["model.0.out_conv.bias"])
    
    target_model.load_state_dict(target_sd)
    
    # Save formatted Ultralytics checkpoint
    ckpt_out = {
        "model": target_model,
        "yaml": model.model.yaml,
        "date": "2026-10-02",
        "description": "Pretrained YOLO26s-LUMA-PCB with transferred COCO weights and identity-initialized LIAM",
    }
    torch.save(ckpt_out, out_pt)
    print(f"✅ Successfully transferred {matched_keys} weight tensors ({matched_params:,} / {total_params:,} parameters, {matched_params/total_params*100:.2f}%)")
    print(f"💾 Saved clean pretrained checkpoint to: {out_pt}")

if __name__ == "__main__":
    main()
