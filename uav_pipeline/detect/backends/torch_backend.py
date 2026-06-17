"""PyTorch backend (self-contained).

Loads a HybridYOLO/yolov26 ``.pt`` checkpoint. The state-dict was pickled
against ``nets.nn`` but must unpickle against ``nets.nn_v26`` (the v26 model
defs vendored under ``_vendor/nets/``). We handle that with a custom
``Unpickler.find_class`` remap — the same logic as the original infer script,
now inlined here so there is no dependency on a sibling ``eval_yolo`` folder.
"""
import pickle
import types

import numpy as np

from ..base import DetectorBackend
from ..preprocess import letterbox


def _load_model(model_path, device):
    """Load a checkpoint, falling back to the v26 unpickler for yolov26 models."""
    import torch

    def _extract(ckpt):
        m = ckpt["model"].float().fuse()
        return m.half().eval() if device == "cuda" else m.eval()

    try:
        return _extract(torch.load(model_path, map_location=device, weights_only=False))
    except AttributeError:
        # yolov26 checkpoint: remap nets.nn -> nets.nn_v26 during unpickling.
        import nets.nn_v26 as _nn_v26  # resolves to _vendor/nets via _paths

        class _V26Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "nets.nn" and hasattr(_nn_v26, name):
                    return getattr(_nn_v26, name)
                return super().find_class(module, name)

        _pkl = types.ModuleType("_v26_pkl")
        _pkl.Unpickler = _V26Unpickler
        for _a in ("dump", "dumps", "load", "loads",
                   "PicklingError", "UnpicklingError", "HIGHEST_PROTOCOL"):
            if hasattr(pickle, _a):
                setattr(_pkl, _a, getattr(pickle, _a))
        return _extract(torch.load(
            model_path, map_location=device, weights_only=False, pickle_module=_pkl))


class TorchBackend(DetectorBackend):
    backend_name = "torch"

    def __init__(self, model_path, imgsz=640, device="", fp16=False):
        super().__init__(model_path, imgsz, device, fp16)
        import torch
        self._torch = torch
        dev = (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = dev
        self.model = _load_model(model_path, dev)

    def _preprocess(self, img0):
        # NOTE: division + dtype happen in _infer (matches the original) so the
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
