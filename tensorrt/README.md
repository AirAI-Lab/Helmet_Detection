TensorRT conversion & C++ inference helper
=========================================

This folder contains a minimal end-to-end example to convert a PyTorch .pt model to a TensorRT engine
and run inference in C++ on Ubuntu / Jetson devices.

Files
- pt_to_onnx.py: export PyTorch (.pt) to ONNX. Supports Ultralytics YOLO when --ultralytics is provided.
- onnx_to_trt.sh: wrapper around trtexec to build a TensorRT engine (.engine) from an ONNX file.
- trt_infer.cpp: minimal C++ example showing how to load a serialized engine and run inference.
- CMakeLists.txt: basic CMake to build the example (you may need to set include/lib paths for TensorRT).

Quick steps (on Jetson / Ubuntu with TensorRT installed)

1) Export .pt -> ONNX

   # example for a 640x640 input
   python3 tools/tensorrt/pt_to_onnx.py --pt /path/to/best.pt --onnx /path/to/best.onnx --input-shape 1 3 640 640 --ultralytics

2) Convert ONNX -> TensorRT engine with trtexec (on Jetson)

   # build FP16 engine with 4GB workspace
   bash tools/tensorrt/onnx_to_trt.sh /path/to/best.onnx /path/to/best.engine --fp16 --workspace=4096

   Notes:
   - On Jetson, trtexec is usually available after installing TensorRT via JetPack.
   - For INT8 you must provide a calibration cache or implement a calibration step.

3) Build C++ example

   mkdir -p build && cd build
   cmake ..
   make -j

   If cmake fails to find TensorRT, set CMAKE_LIBRARY_PATH and CMAKE_INCLUDE_PATH or edit CMakeLists to point to the TensorRT include/lib dirs.

4) Run inference

   ./trt_infer /path/to/best.engine /path/to/image.jpg 640 640

Batch/video helper (`trt_batch_infer`) usage

```
./tensorrt/trt_batch_infer <engine.trt> <in_frames_or_video> <out_frames_dir> <input_w> <input_h> <names.txt> [--conf 0.25] [--out-video path] [--log-level 0|1|2]
```

- `--log-level`: control verbosity. `0` = errors only, `1` = info (default), `2` = debug.
- Example: process a video and write MP4 (auto-select codec):

```
mkdir -p out_frames_batch
./tensorrt/trt_batch_infer ./best.engine _helmet_maker/helmet_video.mp4 out_frames_batch 640 640 tensorrt/names.txt --conf 0.25 --out-video trt_result_video_autocodec.mp4 --log-level 1
```

5) Adaptation notes

   - The C++ example is minimal and assumes a single input and single output; for YOLO-like models you must parse outputs into boxes/scores/cls and run NMS.
   - Adjust output buffer size to match your model's outputs. Query binding dims via engine->getBindingDimensions(bindingIndex).
   - For better performance on Jetson, use --fp16 when building the engine and ensure your model supports fp16.

Troubleshooting
- If trtexec reports unsupported ops in the ONNX, try upgrading ONNX opset (use --opset in pt_to_onnx.py) or modify/export model with fewer custom ops.
- If the engine build fails with memory errors, increase --workspace parameter.
- For Ultralytics YOLO exports, you may prefer using the Ultralytics export tools (they have onnx export utilities); this script is a lightweight helper.

Security & licenses
- TensorRT is NVIDIA software; please follow NVIDIA's licensing when using trtexec and TensorRT libraries.

编译：
```bash
g++ tensorrt/trt_batch_infer.cpp -o tensorrt/trt_batch_infer -std=c++17 -I/usr/include/opencv4 -I/usr/local/cuda/include -L/usr/local/lib -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lnvinfer -lnvinfer_plugin -lcudart
```
运行：
- 视频输入+输出
```bash
./tensorrt/trt_batch_infer ./best.engine _helmet_maker/helmet_video.mp4 out_frames_batch 640 640 tensorrt/names.txt --conf 0.25 --out-video trt_result_video_autocodec.mp4

mkdir -p out_frames_batch && ./tensorrt/trt_batch_infer ./best.engine ../_helmet_maker/helmet_video.mp4 out_frames_batch 640 640 tensorrt/names.txt --conf 0.25 --out-video trt_result_video_autocodec.mp4
#第二段的区别在于先建立图片文件夹再执行程序
```

- 实时rtsp输入 + rtmp输出
```bash
./tensorrt/trt_batch_infer ./best.engine "rtsp://admin:123456@192.168.2.108:554/h265/ch1/main/av_stream"   out_rtsp 640 640 tensorrt/names.txt   --conf 0.25 --log-level 1 --alarm-dir out_rtsp/alarms   --rtmp rtmp://202.96.165.88/live/allen_9_1209 --out-fps 15
```
> SRS公网服务器
`http://202.96.165.88:1985/console/ng_index.html#/streams`

- 单张图片输入+输出
```bash
```bash
./tensorrt/trt_batch_infer ./best.engine "test_photo.png" out_images 640 640 tensorrt/names.txt --conf 0.25 --log-level 1 --alarm-dir out_images/alarms
```
- 图片文件夹输入+输出
```bash
./tensorrt/trt_batch_infer     ./best.engine     "input_photos/"     out_images     640 640     tensorrt/names.txt     --conf 0.25     --log-level 1     --alarm-dir out_images/alarms     --img-fps 10
```
- rtmp输入+输出
```bash
mkdir -p rtmp_test && \
./tensorrt/trt_batch_infer \
    ./best.engine \
    "rtmp://202.96.165.88/live/Allen_9" \
    rtmp_test \
    640 640 \
    tensorrt/names.txt \
    --conf 0.25 \
    --log-level 2 \
    --rtmp "rtmp://202.96.165.88/live/Allen_detected_output" \
    --duration 0
    #duration 为 0 则循环
```