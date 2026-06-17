"""OpenVINO IR backend (default for Windows CPU dev; also runs on Jetson GPU)."""
import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox, to_chw_rgb_float


class OpenVINOBackend(DetectorBackend):
    backend_name = "openvino"

    def __init__(self, model_path, imgsz=640, device="", fp16=False):
        super().__init__(model_path, imgsz, device, fp16)
        import openvino as ov

        self._ov = ov
        self.core = ov.Core()
        xml = model_path if model_path.endswith(".xml") else model_path + ".xml"
        dev = (device or "CPU").upper()
        available = self.core.available_devices
        if dev not in available:
            dev = "CPU"
        self.device = dev
        self.compiled = self.core.compile_model(xml, dev)
        self.req = self.compiled.create_infer_request()

    def _preprocess(self, img0):
        img_lb, ratio, pad = letterbox(img0, self.imgsz)
        sample = to_chw_rgb_float(img_lb, half=False)        # float32 /255
        return sample[None, ...], ratio, pad                  # [1,3,H,W]

    def _infer(self, model_input):
        self.req.set_input_tensors([self._ov.Tensor(model_input)])
        self.req.infer()
        return self.req.get_output_tensor(0).data
