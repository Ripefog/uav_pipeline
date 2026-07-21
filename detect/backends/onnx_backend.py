"""ONNX Runtime backend."""
import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox, letterbox_yolox, to_chw_bgr_float, to_chw_rgb_float


class ONNXBackend(DetectorBackend):
    backend_name = "onnx"

    def __init__(self, model_path, imgsz=640, device="", fp16=False, preprocess="ultralytics"):
        super().__init__(model_path, imgsz, device, fp16, preprocess)
        dev = (device or "").lower()
        if dev.startswith("cuda"):
            # onnxruntime-gpu (CUDA 13 build) cần lib CUDA/cuDNN nạp sẵn trong
            # process; import torch trước sẽ dlopen libcudart.so.13 để ORT thấy
            # mà không phải set LD_LIBRARY_PATH thủ công.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
        import onnxruntime as ort

        if dev.startswith("cuda"):
            # ORT không parse index từ "cuda:1"; phải truyền device_id riêng.
            device_id = int(dev.split(":", 1)[1]) if ":" in dev else 0
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3   # 0=Verbose..2=Warning,3=Error → ẩn warning Memcpy
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.device = dev or "cpu"

    def _preprocess(self, img0):
        if self.preprocess == "yolox":
            img_lb, ratio, pad = letterbox_yolox(img0, self.imgsz)
            sample = to_chw_bgr_float(img_lb, half=False)     # float32, BGR, 0-255
        else:
            img_lb, ratio, pad = letterbox(img0, self.imgsz)
            sample = to_chw_rgb_float(img_lb, half=False)     # float32 /255
        return sample[None, ...], ratio, pad                  # [1,3,H,W]

    def _infer(self, model_input):
        return self.sess.run(None, {self.in_name: model_input})[0]
