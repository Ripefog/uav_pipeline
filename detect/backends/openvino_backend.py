"""OpenVINO IR backend (default for Windows CPU dev; also runs on Jetson GPU)."""
import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox, letterbox_yolox, to_chw_bgr_float, to_chw_rgb_float


class OpenVINOBackend(DetectorBackend):
    backend_name = "openvino"

    def __init__(self, model_path, imgsz=640, device="", fp16=False, preprocess="ultralytics"):
        super().__init__(model_path, imgsz, device, fp16, preprocess)
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
        # Fixed-batch IR (e.g. batch=16): tile the single frame to fill the batch
        # and slice the first result back out.
        try:
            self._batch = int(self.compiled.input(0).get_partial_shape()[0].get_length())
        except Exception:
            self._batch = 1
        if self._batch < 1:
            self._batch = 1

    def _preprocess(self, img0):
        if self.preprocess == "yolox":
            img_lb, ratio, pad = letterbox_yolox(img0, self.imgsz)
            sample = to_chw_bgr_float(img_lb, half=False)     # float32, BGR, 0-255
        else:
            img_lb, ratio, pad = letterbox(img0, self.imgsz)
            sample = to_chw_rgb_float(img_lb, half=False)     # float32 /255
        return sample[None, ...], ratio, pad                  # [1,3,H,W]

    def _infer(self, model_input):
        n = model_input.shape[0]
        if self._batch > n:                       # pad single frame up to fixed batch
            pad = np.repeat(model_input[:1], self._batch - n, axis=0)
            model_input = np.concatenate([model_input, pad], axis=0)
        self.req.set_input_tensors([self._ov.Tensor(np.ascontiguousarray(model_input))])
        self.req.infer()
        out = self.req.get_output_tensor(0).data
        return out[:n]                            # keep only the real frame(s)
