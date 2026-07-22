"""TensorRT backend (Jetson Orin, static shape).

Loads a serialized ``.engine`` (built by ``scripts/export_tensorrt.py``).
Fixed-batch engines (e.g. batch=16, matching ``detector.batch``) are
supported the same way as ``OpenVINOBackend``: a single frame is tiled to
fill the batch and the result is sliced back down. Targets the TensorRT 10
tensor-address API (``set_tensor_address`` + ``execute_async_v3``) as used by
JetPack 6, with a fallback to the TRT 8 binding-index API
(``execute_async_v2``).

Cannot be exercised on a Windows dev box (no TRT/pycuda) — it is imported
lazily via ``build_backend('trt')`` only on Jetson.
"""
import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox, letterbox_yolox, to_chw_bgr_float, to_chw_rgb_float


def _volume(shape):
    v = 1
    for d in shape:
        v *= int(d) if d >= 0 else 1
    return v


class TensorRTBackend(DetectorBackend):
    backend_name = "trt"

    def __init__(self, engine_path, imgsz=640, device="", fp16=True, preprocess="ultralytics"):
        super().__init__(engine_path, imgsz, device, fp16, preprocess)
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates the primary CUDA context)

        self._trt = trt
        self._cuda = cuda
        self.fp16 = fp16

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Resolve tensor I/O. TRT 10 iterates tensor names; TRT 8 uses bindings.
        self._trt10 = hasattr(self.context, "execute_async_v3")
        self._setup_io()

    # ------------------------------------------------------------------ #
    def _setup_io(self):
        trt = self._trt
        cuda = self._cuda
        ctx = self.context
        # Fixed shape baked into the engine's optimization profile (batch,3,H,W)
        # — read it back rather than assuming batch=1, so a batch=16 engine
        # (matching detector.batch) works the same way as OpenVINOBackend.
        if self._trt10:
            names = list(self.engine)  # tensor names
            self.input_name = None
            for n in names:
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                    self.input_name = n
                    break
            in_shape = tuple(self.engine.get_tensor_profile_shape(self.input_name, 0)[1])
            self._batch = in_shape[0]
            ctx.set_input_shape(self.input_name, in_shape)
            self._buf = {}
            self._dev = {}
            for n in names:
                shape = tuple(ctx.get_tensor_shape(n))
                dtype = trt.nptype(self.engine.get_tensor_dtype(n))
                sz = _volume(shape)
                self._buf[n] = cuda.pagelocked_empty(sz, dtype).reshape(shape)
                self._dev[n] = cuda.mem_alloc(sz * np.dtype(dtype).itemsize)
                ctx.set_tensor_address(n, int(self._dev[n]))
        else:  # TRT 8 binding API
            self.bindings = []
            self._buf = {}
            self._dev = []
            self.input_name = None
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                shape = tuple(self.engine.get_binding_shape(i))
                if is_input:
                    self.input_name = name
                    shape = tuple(self.engine.get_profile_shape(0, i)[1])
                    self._batch = shape[0]
                    ctx.set_binding_shape(i, shape)
                else:
                    shape = tuple(ctx.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                sz = _volume(shape)
                self._buf[name] = cuda.pagelocked_empty(sz, dtype).reshape(shape)
                self._dev.append(cuda.mem_alloc(sz * np.dtype(dtype).itemsize))
                self.bindings.append(int(self._dev[-1]))

    @property
    def _output_name(self):
        trt = self._trt
        if self._trt10:
            for n in self.engine:
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                    return n
        else:
            for i in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(i):
                    return self.engine.get_binding_name(i)
        raise RuntimeError("No output tensor found in engine")

    # ------------------------------------------------------------------ #
    def _preprocess(self, img0):
        if self.preprocess == "yolox":
            img_lb, ratio, pad = letterbox_yolox(img0, self.imgsz)
            sample = to_chw_bgr_float(img_lb, half=self.fp16)  # fp16/fp32, BGR, 0-255
        else:
            img_lb, ratio, pad = letterbox(img0, self.imgsz)
            sample = to_chw_rgb_float(img_lb, half=self.fp16)  # fp16/fp32 /255
        return sample[None, ...], ratio, pad                    # [1,3,H,W]

    def _infer(self, model_input):
        cuda = self._cuda
        ctx = self.context
        in_name = self.input_name

        n = model_input.shape[0]
        if self._batch > n:                        # pad single frame up to fixed batch
            pad = np.repeat(model_input[:1], self._batch - n, axis=0)
            model_input = np.concatenate([model_input, pad], axis=0)
        np.copyto(self._buf[in_name], model_input)

        # pycuda's Stream.handle is a callable method on some versions, a
        # plain int attribute on others (2026.1+) — support both.
        stream_handle = self.stream.handle() if callable(self.stream.handle) else self.stream.handle

        if self._trt10:
            cuda.memcpy_htod_async(self._dev[in_name], self._buf[in_name], self.stream)
            ctx.execute_async_v3(stream_handle)
            out_name = self._output_name
            cuda.memcpy_dtoh_async(self._buf[out_name], self._dev[out_name], self.stream)
        else:
            in_idx = self.engine.get_binding_index(in_name)
            cuda.memcpy_htod_async(self._dev[in_idx], self._buf[in_name], self.stream)
            ctx.execute_async_v2(self.bindings, stream_handle)
            out_name = self._output_name
            out_idx = self.engine.get_binding_index(out_name)
            cuda.memcpy_dtoh_async(self._dev[out_idx], self._buf[out_name], self.stream)
        self.stream.synchronize()
        return self._buf[self._output_name][:n]     # drop the padded-frame rows
