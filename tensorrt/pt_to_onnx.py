#!/usr/bin/env python3
"""
Export a PyTorch .pt model to ONNX suitable for TensorRT conversion.

Usage examples:
  # Generic torch script model
  python tools/tensorrt/pt_to_onnx.py --pt weights/best.pt --onnx weights/best.onnx --input-shape 1 3 640 640

  # For Ultralytics YOLOv8 models (if 'ultralytics' package installed):
  python tools/tensorrt/pt_to_onnx.py --pt yolov8n.pt --onnx yolov8n.onnx --input-shape 1 3 640 640 --ultralytics

Notes:
 - ONNX opset default 14, can be changed with --opset
 - For dynamic batch size add --dynamic
 - This script attempts to detect an Ultralytics YOLO model and use its internal module when requested
"""
import argparse
import sys
import torch
import os


def export_torch_module(torch_module, onnx_path, input_shape, opset, dynamic):
    torch_module.eval()
    dummy = torch.randn(*input_shape)
    input_names = ["images"]
    output_names = ["output"]
    dynamic_axes = None
    if dynamic:
        # allow dynamic batch dimension
        dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}

    print(f"Exporting to ONNX: {onnx_path} (opset={opset}, dynamic={dynamic})")
    with torch.no_grad():
        torch.onnx.export(
            torch_module,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )
    print("Export complete.")


def try_load_ultralytics(pt_path):
    try:
        from ultralytics import YOLO

        print("Ultralytics detected: using YOLO(...) loader")
        return YOLO(pt_path).model
    except Exception as e:
        print(f"Ultralytics not available or failed to load model: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="Path to .pt model file")
    p.add_argument("--onnx", required=True, help="Output .onnx path")
    p.add_argument("--input-shape", nargs=4, type=int, metavar=("B","C","H","W"),
                   default=[1,3,640,640], help="Input shape (B C H W)")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    p.add_argument("--dynamic", action="store_true", help="Export with dynamic batch dimension")
    p.add_argument("--ultralytics", action="store_true", help="Try to load as Ultralytics YOLO model")
    args = p.parse_args()

    pt_path = args.pt
    onnx_path = args.onnx
    input_shape = tuple(args.input_shape)

    if not os.path.exists(pt_path):
        print(f"Error: .pt file not found: {pt_path}")
        sys.exit(2)

    # Try ultralytics loader if asked
    model_module = None
    if args.ultralytics:
        model_module = try_load_ultralytics(pt_path)

    if model_module is None:
        # Generic torch load
        try:
            print("Attempting generic torch.load(...) to get nn.Module")
            loaded = torch.load(pt_path, map_location="cpu")
            # Many .pt are state_dicts, or scripted modules, handle both
            if isinstance(loaded, dict) and "model" in loaded:
                # common pattern: {'model': nn.Module}
                possible = loaded["model"]
                if hasattr(possible, "eval"):
                    model_module = possible
            elif hasattr(loaded, "eval"):
                model_module = loaded
            else:
                # If it's a state dict, user must provide code to recreate module
                print("Loaded object is not an nn.Module. If this is a state_dict, you must re-create the model architecture in code and load the state_dict before exporting.")
                sys.exit(3)
        except Exception as e:
            print(f"Failed to load .pt via torch.load: {e}")
            sys.exit(4)

    export_torch_module(model_module, onnx_path, input_shape, args.opset, args.dynamic)


if __name__ == "__main__":
    main()
