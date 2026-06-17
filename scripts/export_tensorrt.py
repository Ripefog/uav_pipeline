"""export_tensorrt — build a serialized TensorRT engine from an ONNX model.

Targets TensorRT 10 (JetPack 6 / Orin). Produces a static batch=1 engine at the
detector's fixed input size. FP16 by default; INT8 only with a calibration
directory (the single biggest accuracy risk — supply your own representative
crops).

Prereq: an ONNX model (opset 12, fixed imgsz). A ready one ships in
``weights/best_yolov26n_qat_int8_static.onnx``; export your own from a ``.pt``
with your training stack if needed.

Then:
    python -m uav_pipeline.scripts.export_tensorrt \\
        --onnx uav_pipeline/weights/best_yolov26n_qat_int8_static.onnx \\
        --engine uav_pipeline/weights/best_yolov26n_qat_int8_static.engine \\
        --imgsz 640 --fp16
"""
import argparse
import os
import sys

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
        self.imgsz = imgsz
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

        self._device = cuda.mem_alloc(self.batch_size * 3 * imgsz * imgsz * 4)
        self.trt = trt

    def _read_sample(self):
        import cv2
        np = self._np
        if self._idx < len(self._files):
            img = cv2.imread(self._files[self._idx])
            self._idx += 1
            if img is None:
                img = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        else:
            img = np.random.randint(0, 256, (self.imgsz, self.imgsz, 3), np.uint8)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
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


def build_engine(onnx, engine, imgsz, fp16, int8, calib_dir, workspace_gb):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
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

    # Force static batch=1 (the TRT backend assumes static shapes).
    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    shape = (1, 3, imgsz, imgsz)
    profile.set_shape(in_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    print(f"[export] building engine: {onnx} -> {engine} "
          f"(imgsz={imgsz}, fp16={fp16}, int8={int8}, ws={workspace_gb}GB)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None — check TRT logs.")
    os.makedirs(os.path.dirname(engine) or ".", exist_ok=True)
    with open(engine, "wb") as f:
        f.write(serialized)
    print(f"[export] done -> {engine} ({len(serialized) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Export ONNX -> TensorRT engine")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default="", help="Dir of calibration images for INT8")
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    args = ap.parse_args()

    build_engine(args.onnx, args.engine, args.imgsz,
                 args.fp16, args.int8, args.calib_dir, args.workspace_gb)


if __name__ == "__main__":
    main()
