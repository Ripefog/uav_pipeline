"""ONNX Runtime backend."""
import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox, to_chw_rgb_float


class ONNXBackend(DetectorBackend):
    backend_name = "onnx"

    def __init__(self, model_path, imgsz=640, device="", fp16=False):
        super().__init__(model_path, imgsz, device, fp16)
        import onnxruntime as ort

        dev = (device or "").lower()
        if dev.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.device = dev or "cpu"

    def _preprocess(self, img0):
        img_lb, ratio, pad = letterbox(img0, self.imgsz)
        sample = to_chw_rgb_float(img_lb, half=False)        # float32 /255
        return sample[None, ...], ratio, pad                  # [1,3,H,W]

    def _infer(self, model_input):
        return self.sess.run(None, {self.in_name: model_input})[0]
