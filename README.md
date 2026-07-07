# UAV Edge Pipeline — FPT UAV AI Hackathon 2026

A single-frame streaming AI pipeline that runs **onboard a UAV** and covers the
three competition pillars:

| Pillar (VN)           | Module      | What it does                                                                                                                                                                                        |
| --------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phát hiện** | `detect/` | Real-time object detection — vendored YOLO backends (torch/onnx/openvino) + a TensorRT backend.                                                                                                    |
| **Theo dấu**   | `track/`  | Multi-object tracking with occlusion handling —**faithful port of [`pratap424/visdrone_mot`](https://github.com/pratap424/visdrone_mot)** (CMC + ByteTrack 2-stage + EMAT + interpolation). |
| **Bám đuổi** | `follow/` | Keeps the target framed and emits UAV commands via a PID controller (gimbal/body rates). Mock controller by default; MAVLink/ROS2 stubs ready to wire.                                              |

Plus: license-plate **OCR** (`ocr/`, vendored), streaming **sources**
(video/webcam/image-dir/GStreamer), and **sinks** (annotated HUD video,
telemetry JSONL/CSV, command log).

**Self-contained:** the YOLO helpers it needs (NMS, letterbox, model defs) are
vendored under `_vendor/`, and the model **weights** + class names ship in
`weights/`. It has **no dependency on any sibling `eval_yolo` / `eval_ocr`
folder** — clone and run.

---

## Architecture

### Per-frame data flow (the three pillars)

`pipeline.process_frame()` runs the following chain for **each frame** in a
single-frame streaming loop (it does not load the whole video into RAM, unlike
the original YOLO infer scripts):

```
┌────────────┐
│ FrameSource│  video / webcam / image_dir / gstreamer (RTSP)
│ (sources/) │──► (FrameMeta idx,ts,shape_hw,  frame BGR)
└────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ ① PHÁT HIỆN  detect/                                          │
│   preprocess(letterbox) → backend._infer → postprocess(NMS)  │
│   └─ 4 backends: openvino · onnx · torch · trt (config-chosen)│
│   UnifiedDetector.detect(frame)         → List[Detection]     │
│   UnifiedDetector.detect_plates(frame)  → List[Detection] opt │
└──────────────────────────────────────────────────────────────┘
        │ detections                                   plates ─┐
        ▼                                                      │
┌──────────────────────────────────────────────────────────────┐
│ ② THEO DẤU  track/   (faithful port of visdrone_mot)          │
│   DroneByteTracker.update(frame, detections)                  │
│     CMC(ORB+affine) → apply → EMAT → predict → greedy×2      │
│   → List[Track]  (id, bbox, cls, name, velocity, trajectory)  │
└──────────────────────────────────────────────────────────────┘
        │ tracks                                       plates ─┤
        ▼                                                      ▼
┌──────────────────────────────────────────────────────────────┐
│ ③a OCR (optional)  ocr/plate_ocr.py  (vendored)              │
│   crop_mode: plate_detection ─┐  or  vehicle_lower_third      │
│   PlateOCR.recognize → PlateVoter.majority → Track.plate_text │
└──────────────────────────────────────────────────────────────┘
        │ tracks (now carrying plate_text)
        ▼
┌──────────────────────────────────────────────────────────────┐
│ ③ BÁM ĐUỔI  follow/                                           │
│   TargetSelector (locked_id → preferred → score·√area)        │
│   errors (ex, ey, escale) → 3×PID (yaw, pitch, forward)       │
│   FollowController.step → (FollowState, Command)              │
│   └─ deadzone + coast lost_recovery_frames, then brake        │
└──────────────────────────────────────────────────────────────┘
        │ Command{yaw_rate, pitch_rate, forward_vel, vertical_vel, …}
        ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│ controllers/        │     │ sinks/                           │
│  mock (default)     │     │  video      HUD: box+ID+plate+   │
│  mavlink (stub)     │     │             reticle+FPS+motion    │
│  ros2    (stub)     │     │  telemetry  JSONL/frame + CSV     │
└─────────────────────┘     │  control_log Command JSONL       │
                            └──────────────────────────────────┘
```

### Code provenance — vendored vs ported vs new

```
VENDORED (copied into _vendor/ + weights/)   PORT FAITHFUL              NEW (the differentiator)
─────────────────────────────────────────     ──────────────────────    ──────────────────────
  utils.util  (NMS×2, make_anchors)           ← pratap424/visdrone_mot   follow/   (selector · pid · controller)
  nets.{nn,nn_v26}  (torch model defs)        │ track/camera_motion.py   controllers/ (mock + stubs)
  letterbox (preprocess.py, inlined)          │ track/drone_tracker.py   sinks/    (HUD · telemetry · control_log)
  _load_model (torch_backend, inlined)        │   (ByteTrack)            pipeline.py (orchestrator)
  weights/ (onnx/xml/bin + keras + names)     │   constant-velocity,     sources/  (4 source types)
  decode_plate (plate_ocr.py, inlined)        │   greedy IoU, EMAT,      scripts/export_tensorrt (TRT)
                                               └─ NO Kalman/Hungarian/   detect/backends/trt (Jetson)
                                                   neural ReID
```

Everything the pipeline needs to run is inside this package — no external
`eval_yolo` / `eval_ocr` directories are required at runtime.

### Inside the tracker (the Theo dấu pillar)

```
DroneByteTracker.update(frame, detections):
  ┌─ CMC  camera_motion.estimate(frame)
  │    ORB(1000) → BFMatcher(NORM_HAMMING, k=2) → Lowe ratio 0.75
  │    → estimateAffinePartial2D(RANSAC 5.0, downscale 0.5)
  │    → warp_matrix + motion_severity{translation, rotation, scale, severity∈[0,1]}
  ├─ _apply_cmc:  warp every track's predicted bbox + trajectory
  ├─ EMAT:  adaptive_iou  = iou  · (1 − 0.5·severity)        ← relax thresholds
  │         adaptive_high = high · (1 − 0.35·severity)         under heavy motion
  ├─ predict:  track.bbox += velocity                         ← constant-velocity, NO Kalman
  ├─ split detections: conf ≥ adaptive_high  /  low
  ├─ Stage1  associate(tracks, high)   ─ greedy argmax IoU, NO Hungarian
  ├─ Stage2  associate(remaining, low) ─ iou·0.8
  ├─ mark_missed (age++) → spawn new tracks → kill age > max_age(50)
  └─ return confirmed (age==0 & total_visible ≥ min_hits=3)
```

Re-identification (the "nhận diện lại" requirement) is achieved through
**CMC + high `max_age` + track interpolation** — exactly how the reference repo
does it. No neural ReID is added.

### Backend / environment matrix

| Target            | Detector                                | OCR                       | Controller     | Status                 |
| ----------------- | --------------------------------------- | ------------------------- | -------------- | ---------------------- |
| Windows dev (x86) | `onnx` / `openvino`                 | TF/Keras (CPU)            | mock           | ✅ verified onnx + OCR |
| Jetson Orin       | `trt` (FP16, via `export_tensorrt`) | TF/Keras (CPU, throttled) | mavlink / ros2 | Jetson-only            |

---

## Layout

```
uav_pipeline/           ← repo root == the package (git clone produces this folder)
├── __init__.py         # makes the repo root the importable `uav_pipeline` package
├── contracts.py        # Detection / Track / Command / FrameMeta / FollowState / FrameContext
├── config.py           # typed config from one YAML
├── pipeline.py         # single-frame loop: detect → track → ocr → follow → sinks
├── _paths.py           # puts _vendor on sys.path; resolves weight paths; exposes PKG_DIR
├── _vendor/            # vendored YOLO helpers (no external dependency)
│   ├── utils/util.py   #   non_max_suppression[_v26], make_anchors, wh2xy, …
│   └── nets/{nn,nn_v26}.py  # torch model defs (for the .pt torch backend)
├── weights/            # bundled models + class names (ships with the repo)
│   ├── best_yolov26n_qat_int8_static.{onnx,xml,bin}
│   ├── yolov26n_qat_plate_int8.{onnx,xml,bin}
│   ├── plate_ocr.keras + plate_config.yaml
│   └── names.yaml      # class-id → name (0:pedestrian … 9:motor)
├── sources/            # FrameSource ABC + video/webcam/image_dir/gstreamer
├── detect/             # preprocess (letterbox) / postprocess (NMS) + backends + UnifiedDetector
│   └── backends/       #   openvino | onnx | torch | trt
├── track/              # camera_motion.py (CMC+EMAT) + drone_tracker.py (ByteTrack port)
├── ocr/                # plate_ocr.py (fast-plate-ocr, lazy)
├── follow/             # pid / selector / controller + controllers/{mock,mavlink,ros2}
├── sinks/              # HUD video / telemetry / control_log
├── scripts/            # run_pipeline / export_tensorrt / validate_pipeline
└── configs/            # default / windows_onnx / windows_openvino / jetson_trt
```

---

## Quickstart

> `git clone` produces a folder named `uav_pipeline/` — that folder **is** the
> package (the repo is flat: no extra wrapper folder). Run the commands below
> from that folder's **parent** (the directory you cloned into — *not* from
> inside `uav_pipeline/`), so that `import uav_pipeline` resolves. Install deps
> first: `pip install -r uav_pipeline/requirements.txt` (at least `numpy`,
> `opencv-python`, `pyyaml`, `torch`, `torchvision`). The weights ship in
> `uav_pipeline/weights/`, so no other folders are needed.

### 1. Smoke-test the core (no model, no weights — tests Track + Follow)

```bash
python -m uav_pipeline.scripts.validate_pipeline
```

Creates synthetic moving objects, asserts stable track IDs, near-zero motion
severity on a static frame, and correct PID sign conventions.

### 2. Run the full pipeline on Windows (OpenVINO CPU)

```bash
python -m uav_pipeline.scripts.run_pipeline \
  -c uav_pipeline/configs/windows_openvino.yaml \
  --source path/to/your_video.mp4
```

Produces `output/pipeline.mp4` (HUD: boxes, track IDs, plate text, FPS, motion
severity bar, follow reticle), `output/telemetry.jsonl` + `telemetry_summary.csv`,
and `output/commands.jsonl`.

Point at an image folder instead with `--source-type image_dir --source path/to/frames`.

### 3. Deploy on Jetson Orin (TensorRT FP16 + GStreamer)

```bash
# a) export the engine (FP16) from the bundled ONNX
python -m uav_pipeline.scripts.export_tensorrt \
  --onnx uav_pipeline/weights/best_yolov26n_qat_int8_static.onnx \
  --engine uav_pipeline/weights/best_yolov26n_qat_int8_static.engine --imgsz 640 --fp16

# b) run with the Jetson config (edit the GStreamer pipeline string first)
python -m uav_pipeline.scripts.run_pipeline -c uav_pipeline/configs/jetson_trt.yaml
```

---

## Configuration

One YAML drives everything (see `configs/default.yaml` for every key with
comments). Highlights:

- **`detector.backend`** — `openvino` (Win/x86 default), `onnx`, `torch`,
  `trt` (Jetson). `classes_of_interest: []` keeps all classes; list ids to filter.
- **`tracker`** — the visdrone_mot defaults (`high_conf=0.4`, `low_conf=0.15`,
  `iou=0.3`, `max_age=50`, `min_hits=3`). `emat: true` relaxes thresholds under
  heavy camera motion. `cmc.enabled: false` disables compensation for ablation.
  `same_class_gate: true` prevents IDs crossing class boundaries in dense scenes.
- **`follow`** — `preferred_classes` restricts the target; `target_area_norm`
  sets desired keep-distance (drives the forward PID); PID sign convention:
  `yaw>0` pan right, `pitch>0` tilt down, `forward<0` back up.
- **`controller.backend`** — `mock` (default, safe), `mavlink`, `ros2` (stubs).
- **`ocr.enabled`** — adds TF/Keras; off by default. **`ocr.device`** defaults to
  `gpu` (auto-falls-back to CPU where TF has no GPU). See below.

### Enable OCR (license plates)

1. The weights already ship in `weights/` (`plate_ocr.keras`, `plate_config.yaml`).
   Install the OCR deps: `pip install tensorflow fast-plate-ocr` (use
   `tensorflow-cpu` on native Windows — TF ≥2.11 has no GPU there, so it's leaner).
2. In the config: `ocr.enabled: true`. Default `crop_mode: vehicle_lower_third`
   reads the bottom third of each vehicle track.
3. (Optional) `ocr.plate_detector.enabled: true` + `crop_mode: plate_detection`
   to use the dedicated plate model (`yolov26n_qat_plate_int8`).
4. **Device:** `ocr.device: gpu` (the default) uses the GPU where TF supports it
   (Linux/WSL2 with CUDA 12 + cuDNN 9, or Jetson with NVIDIA's NVTF). It
   silently runs on CPU on native Windows (TF ≥2.11) or Jetson without NVTF —
   set `ocr.device: cpu` to force CPU / suppress the GPU-probe log.

---

## Wiring a real drone (Follow → actuation)

The pipeline never flies anything by default. To command a real UAV:

- **MAVLink** (`follow/controllers/mavlink.py`): implement the OFFBOARD loop —
  connection/heartbeat on `controller.mavlink.connection`, arming + OFFBOARD
  mode entry, then map `Command` → `SET_ATTITUDE_TARGET` (yaw/pitch rates,
  deg→rad) and `SET_POSITION_TARGET_LOCAL_NED` (forward/vertical vel). Needs
  `pymavlink`. The file documents the exact mapping.
- **ROS2** (`follow/controllers/ros2.py`): publish `geometry_msgs/Twist` on
  `cmd_vel_topic` (`linear.x=forward`, `linear.z=vertical`, `angular.z=yaw`) and
  a gimbal command on `gimbal_topic`.

**Safety:** gains tuned on video are not gains for a real airframe, and
yaw/pitch drive gimbal vs. body rates depending on platform. Validate on
hardware with the mock controller first; arming/heartbeat are out of band.

---

## Adding competition classes (fire, debris, …)

The pipeline is model-agnostic. Retrain the detector with the extra classes,
point `detector.primary.names_yaml` at the new `args.yaml`, and everything
(tracking, follow, OCR vehicle filter, HUD) picks up the new names.

---

## How tracking was ported

`track/camera_motion.py` and `track/drone_tracker.py` reproduce visdrone_mot's
algorithm verbatim (ORB+affine CMC at half-res with RANSAC, constant-velocity
prediction — **no Kalman**, greedy IoU association — **no Hungarian**, EMAT
adaptive thresholds, linear track interpolation). The only deliberate changes:
the tracker consumes our `Detection` contract, `Track` is multi-class aware
(`cls`/`name`/`plate_text`), and an optional `same_class_gate` exists for dense
multi-class scenes (off by default = original behavior).

## Performance notes

- Jetson Orin Nano 8GB reference: YOLO @640 TRT FP16 ≈ 37 FPS without CMC,
  ≈ 15 FPS with CMC (ORB matching is the cost). Lower `cmc.n_features` or set
  `cmc.enabled: false` for max throughput.
- OCR (TF) is CPU-only on Jetson; it's throttled (`ocr.every_n_frames`) and
  only runs on confirmed vehicle tracks.
