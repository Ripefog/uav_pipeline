# uav_pipeline — UAV Edge AI Pipeline (FPT UAV AI Hackathon 2026)

Single-frame streaming onboard pipeline covering the three competition pillars:

- **Phát hiện (Detect)** — object detection, 4 backends (torch/onnx/openvino/trt)
- **Theo dấu (Track)** — faithful port of [`pratap424/visdrone_mot`](https://github.com/pratap424/visdrone_mot) (CMC + ByteTrack + EMAT + interpolation)
- **Bám đuổi (Follow)** — PID target-following + mock/MAVLink/ROS2 controllers

Plus optional license-plate **OCR**.

### ✅ Self-contained — clone and run

This repository is a single self-contained package, `uav_pipeline/`. It needs
**no sibling folders**:

- The YOLO helpers (NMS, letterbox, model defs) are **vendored** under
  `uav_pipeline/_vendor/`.
- The **model weights** + class names ship under `uav_pipeline/weights/`
  (YOLO `.onnx`/`.xml`/`.bin`, `plate_ocr.keras`, `names.yaml`).

```bash
git clone https://github.com/Ripefog/uav_full_pipeline.git
cd uav_full_pipeline
pip install -r uav_pipeline/requirements.txt        # + tensorflow-cpu fast-plate-ocr for OCR
python -m uav_pipeline.scripts.validate_pipeline     # smoke-test (Track + Follow, no model)
python -m uav_pipeline.scripts.run_pipeline \
  -c uav_pipeline/configs/windows_onnx.yaml --source path/to/video.mp4
```

Full architecture, configuration, and deployment (Jetson/TensorRT, wiring a real
drone via MAVLink/ROS2) are documented in
**[`uav_pipeline/README.md`](uav_pipeline/README.md)**.
