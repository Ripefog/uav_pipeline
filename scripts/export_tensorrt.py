"""export_tensorrt — build a serialized TensorRT engine from an ONNX model.

Targets TensorRT 10 (JetPack 6 / Orin). Produces a static-shape engine (fixed
batch + imgsz — the TRT backend assumes static shapes). FP16 by default;
``--no-fp16`` for full precision (FP32). INT8 only with a calibration
directory (the single biggest accuracy risk — supply your own representative
crops).

``--imgsz`` takes one value (square) or two (``H W``, e.g. a non-square
YOLOX export). ``--batch`` bakes a fixed batch size (e.g. 16, matching
``detector.batch`` in the config) into the optimization profile.

Prereq: an ONNX model (opset 12, fixed imgsz). A ready one ships in
``weights/best_yolov26n_qat_int8_static.onnx``; export your own from a ``.pt``
with your training stack if needed.

Then:
    python -m uav_pipeline.scripts.export_tensorrt \\
        --onnx uav_pipeline/weights/best_yolov26n_qat_int8_static.onnx \\
        --engine uav_pipeline/weights/best_yolov26n_qat_int8_static.engine \\
        --imgsz 640 --fp16

    # YOLOX-X: non-square imgsz, fixed batch=16, full precision
    python -m uav_pipeline.scripts.export_tensorrt \\
        --onnx uav_pipeline/weights/best_yoloxx.onnx \\
        --engine uav_pipeline/weights/best_yoloxx.engine \\
        --imgsz 736 1280 --batch 16 --no-fp16
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)


class _EntropyCalibrator:
    """Minimal IInt8EntropyCalibrator2 feeding random-ish image crops.

    CALIBRATION DATA IS THE BIGGEST INT8 ACCURACY RISK. Replace the dummy
    batches with ~500 real crops from your target distribution (gimbal frames)
    via ``--calib-dir``. The dummy generator is only so the flag path is wired.
    """

    def __init__(self, calib_dir, imgsz, cache="uav_pipeline_int8.cache"):
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        import tensorrt as trt

        self._np = np
        self.h, self.w = imgsz
        self.cache = cache
        self.batch_size = 1
        self.n_samples = 500

        # Prefer real images; fall back to synthetic noise if dir empty/missing.
        self._files = []
        if calib_dir and os.path.isdir(calib_dir):
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            self._files = [os.path.join(calib_dir, f)
                           for f in os.listdir(calib_dir) if f.lower().endswith(exts)]
        self._idx = 0

        self._device = cuda.mem_alloc(self.batch_size * 3 * self.h * self.w * 4)
        self.trt = trt

    def _read_sample(self):
        import cv2
        np = self._np
        if self._idx < len(self._files):
            img = cv2.imread(self._files[self._idx])
            self._idx += 1
            if img is None:
                img = np.zeros((self.h, self.w, 3), np.uint8)
        else:
            img = np.random.randint(0, 256, (self.h, self.w, 3), np.uint8)
        img = cv2.resize(img, (self.w, self.h))
        sample = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(sample[None, ...])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        import pycuda.driver as cuda
        if self._idx >= self.n_samples and self._idx >= len(self._files):
            return None
        batch = self._read_sample()
        cuda.memcpy_htod(self._device, batch)
        return [int(self._device)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            with open(self.cache, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as f:
            f.write(cache)


def build_engine(onnx, engine, imgsz, batch, fp16, int8, calib_dir, workspace_gb):
    import tensorrt as trt

    h, w = imgsz
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10 (Jetson/JetPack 6) requires the EXPLICIT_BATCH flag; TRT 11+
    # dropped it (explicit batch is the only mode, flags default to 0).
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
             if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") else 0)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _EntropyCalibrator(calib_dir, imgsz)
        print("[export] INT8 enabled — using calibrator (use real crops for accuracy)")

    # Force a static shape (the TRT backend assumes static shapes): fixed
    # batch (e.g. 16, matching detector.batch) at the fixed H/W.
    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    shape = (batch, 3, h, w)
    profile.set_shape(in_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    print(f"[export] building engine: {onnx} -> {engine} "
          f"(imgsz={h}x{w}, batch={batch}, fp16={fp16}, int8={int8}, ws={workspace_gb}GB)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None — check TRT logs.")
    os.makedirs(os.path.dirname(engine) or ".", exist_ok=True)
    with open(engine, "wb") as f:
        f.write(serialized)
    print(f"[export] done -> {engine} ({serialized.nbytes / 1e6:.1f} MB)")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_signature(onnx, imgsz, batch, fp16, int8):
    import tensorrt as trt

    return {
        "schema": 1,
        "onnx": os.path.realpath(onnx),
        "onnx_sha256": _sha256(onnx),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "batch": int(batch),
        "fp16": bool(fp16),
        "int8": bool(int8),
        "tensorrt": trt.__version__,
    }


def ensure_detector_engine(detector_cfg, log=print):
    """Build a configured TensorRT plan when its portable source changed.

    The engine and its ``.build.json`` manifest are written atomically. A file
    lock prevents two ROS launches from compiling the same plan concurrently.
    Returns ``True`` when a new plan was built.
    """
    policy = detector_cfg.trt_build
    if detector_cfg.backend != "trt" or not policy.enabled:
        return False

    from .._paths import resolve

    onnx = resolve(detector_cfg.primary.onnx)
    engine = resolve(detector_cfg.primary.trt)
    if not os.path.isfile(onnx):
        raise FileNotFoundError(f"TensorRT source ONNX not found: {onnx}")

    raw_size = detector_cfg.imgsz
    if isinstance(raw_size, int):
        imgsz = (raw_size, raw_size)
    elif len(raw_size) == 2:
        imgsz = (int(raw_size[0]), int(raw_size[1]))
    else:
        raise ValueError("detector.imgsz must be an int or [H, W]")

    os.makedirs(os.path.dirname(engine) or ".", exist_ok=True)
    manifest = engine + ".build.json"
    lock_path = engine + ".lock"
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        signature = _build_signature(
            onnx, imgsz, detector_cfg.batch, detector_cfg.fp16, policy.int8)
        current = None
        if os.path.isfile(engine) and os.path.isfile(manifest):
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except (OSError, ValueError):
                current = None
        if current == signature:
            log(f"[export] TensorRT engine is current: {engine}")
            return False

        log(f"[export] TensorRT engine missing/stale; compiling {onnx}")
        fd, temp_engine = tempfile.mkstemp(
            prefix=os.path.basename(engine) + ".", suffix=".tmp",
            dir=os.path.dirname(engine) or ".")
        os.close(fd)
        temp_manifest = temp_engine + ".json"
        try:
            build_engine(
                onnx, temp_engine, imgsz, detector_cfg.batch,
                detector_cfg.fp16, policy.int8, policy.calib_dir,
                policy.workspace_gb)
            with open(temp_manifest, "w", encoding="utf-8") as f:
                json.dump(signature, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_engine, engine)
            os.replace(temp_manifest, manifest)
        finally:
            for path in (temp_engine, temp_manifest):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        log(f"[export] TensorRT engine ready: {engine}")
        return True


def main():
    ap = argparse.ArgumentParser(description="Export ONNX -> TensorRT engine")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640],
                    help="Square: --imgsz 640. Non-square: --imgsz H W (e.g. 736 1280).")
    ap.add_argument("--batch", type=int, default=1,
                    help="Fixed batch size baked into the engine (e.g. 16, matching detector.batch).")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default="", help="Dir of calibration images for INT8")
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    args = ap.parse_args()

    if len(args.imgsz) == 1:
        imgsz = (args.imgsz[0], args.imgsz[0])
    elif len(args.imgsz) == 2:
        imgsz = (args.imgsz[0], args.imgsz[1])
    else:
        raise ValueError("--imgsz takes 1 (square) or 2 (H W) values")

    build_engine(args.onnx, args.engine, imgsz, args.batch,
                 args.fp16, args.int8, args.calib_dir, args.workspace_gb)


if __name__ == "__main__":
    main()
