"""PyTorch backend — reuses ``eval_yolo.infer_torch._load_model``.

That loader handles the HybridYOLO/yolov26 checkpoint whose state-dict was
pickled against ``nets.nn`` but must unpickle against ``nets.nn_v26``
(via a custom ``Unpickler.find_class`` remap). We import it under a stdout
redirect only to silence the one diagnostic print at module import.
"""
import contextlib
import io

import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox


def _load_torch_loader():
    """Import infer_torch._load_model once, silencing its import-time print."""
    with contextlib.redirect_stdout(io.StringIO()):
        from infer_torch import _load_model  # noqa: WPS433 (eval_yolo on path)
    return _load_model


class TorchBackend(DetectorBackend):
    backend_name = "torch"

    def __init__(self, model_path, imgsz=640, device="", fp16=False):
        super().__init__(model_path, imgsz, device, fp16)
        import torch
        self._torch = torch
        dev = (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = dev
        load = _load_torch_loader()
        self.model = load(model_path, dev)

    def _preprocess(self, img0):
        # NOTE: division + dtype happen in _infer (matches infer_torch) so the
        # half-precision path on CUDA is honored.
        img_lb, ratio, pad = letterbox(img0, self.imgsz)
        sample = np.ascontiguousarray(img_lb.transpose((2, 0, 1))[::-1])
        return sample, ratio, pad

    def _infer(self, model_input):
        torch = self._torch
        t = torch.from_numpy(model_input).to(self.device)
        t = (t.half() if self.device == "cuda" else t.float()) / 255.0
        with torch.no_grad():
            return self.model(t)
