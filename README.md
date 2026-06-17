# uav_pipeline — UAV Edge AI Pipeline

Single-frame streaming onboard pipeline for **FPT UAV AI Hackathon 2026**,
covering the three competition pillars:

- **Phát hiện (Detect)** — object detection, 4 backends (torch/onnx/openvino/trt)
- **Theo dấu (Track)** — faithful port of [`pratap424/visdrone_mot`](https://github.com/pratap424/visdrone_mot) (CMC + ByteTrack + EMAT + interpolation)
- **Bám đuổi (Follow)** — PID target-following + mock/MAVLink/ROS2 controllers

Plus optional license-plate **OCR**.

> ### ⚠️ This repo contains ONLY `uav_pipeline/`
>
> It is the **orchestration + tracking + follow + OCR** layer. It **imports**
> two sibling packages verbatim and does **not** vendor them (they hold the
> detection/OCR backends and the model weights):
>
> - `eval_yolo/` — detection (preprocess, NMS, model loaders, `weights/`)
> - `eval_ocr/` — plate-OCR (fast-plate-ocr Keras CCT, `weights/plate_ocr.keras`)
>
> **To run**, place this folder as a **sibling** of `eval_yolo/` and `eval_ocr/`:
>
> ```
> some_code_root/
> ├── uav_pipeline/      ← this repo
> ├── eval_yolo/         ← (sibling) + weights/
> └── eval_ocr/          ← (sibling) + weights/
> ```
>
> Then from `some_code_root/`: `pip install -r uav_pipeline/requirements.txt`.

Full architecture, quickstart, and config docs live in
**[`uav_pipeline/README.md`](uav_pipeline/README.md)**.

## Smoke-test (no weights needed — exercises Track + Follow)

```bash
python -m uav_pipeline.scripts.validate_pipeline
```

## Run the full pipeline

```bash
python -m uav_pipeline.scripts.run_pipeline \
  -c uav_pipeline/configs/windows_onnx.yaml \
  --source path/to/video.mp4
```
