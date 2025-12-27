#!/usr/bin/env bash
# Simple helper script to run trtexec (TensorRT) to build a TensorRT engine from an ONNX file.
# Run this on Jetson or a machine with TensorRT installed.
# Usage:
#   bash onnx_to_trt.sh /path/to/model.onnx /path/to/model.engine --fp16 --workspace=2048

set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 model.onnx model.engine [trtexec options]"
  exit 1
fi

ONNX="$1"
ENGINE="$2"
shift 2

TRTEXEC=$(which trtexec || true)
if [ -z "$TRTEXEC" ]; then
  echo "trtexec not found in PATH. On Jetson, ensure TensorRT is installed and trtexec is available."
  exit 2
fi

echo "Using trtexec: $TRTEXEC"
echo "Converting ONNX -> TensorRT engine"

# Common options:
# --onnx=<file> --saveEngine=<file> --fp16 --memPoolSize=workspace:<MiB>
# Pass extra options after the engine path.

# Translate legacy flags for newer TensorRT versions (e.g., TRT >= 10.3 removed --workspace/--explicitBatch).
translated_args=()
while (( "$#" )); do
  case "$1" in
    --workspace=*)
      size="${1#--workspace=}"
      translated_args+=("--memPoolSize=workspace:${size}")
      ;;
    --workspace)
      # Accept "--workspace 4096" style.
      if [ -n "${2-}" ]; then
        translated_args+=("--memPoolSize=workspace:${2}")
        shift
      else
        echo "Missing value for --workspace"
        exit 1
      fi
      ;;
    --explicitBatch)
      # Deprecated/removed; ignore for explicit ONNX models.
      ;;
    *)
      translated_args+=("$1")
      ;;
  esac
  shift
done

echo "$TRTEXEC --onnx=$ONNX --saveEngine=$ENGINE ${translated_args[*]}"
exec $TRTEXEC --onnx="$ONNX" --saveEngine="$ENGINE" "${translated_args[@]}"

# Example to build fp16 engine with 4GB workspace:
# bash onnx_to_trt.sh yolov8n.onnx yolov8n.engine --fp16 --memPoolSize=workspace:4096
