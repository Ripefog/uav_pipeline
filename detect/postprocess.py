"""Shared postprocessing — NMS (auto-select v26 / DFL / YOLOX head) + inverse mapping.

Uses the vendored ``utils.util.non_max_suppression*`` (verbatim copies of the
original YOLO helpers, under ``_vendor/utils/util.py``) so numerics are
byte-identical to the standalone infer scripts. The head format is
auto-detected from shape: last-dim == 6 => HybridYOLO/v26 ``[batch, k, 6]``
head; last-dim < second-to-last (channels-last, with a separate objectness
channel) => decoded YOLOX head ``[batch, anchors, 4+1+nc]``; else DFL/v8
``[batch, 4+nc, anchors]`` (channels-first, no objectness).
"""
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except ImportError:  # detection backends need torch for NMS; core pipeline does not
    torch = None

from utils import util  # noqa: E402  (vendored via uav_pipeline._paths._vendor)

from ..contracts import Detection


def _run_nms(out, conf: float, iou: float):
    if out.shape[-1] == 6:
        return util.non_max_suppression_v26(out, conf, iou)
    if out.shape[-1] < out.shape[-2]:
        return util.non_max_suppression_yolox(out, conf, iou)
    return util.non_max_suppression(out, confidence_threshold=conf, iou_threshold=iou)


def postprocess(raw_output, conf: float, iou: float) -> np.ndarray:
    """Run auto-selected NMS on a single-batch raw model output.

    Returns ``[N, 6]`` = ``[x1, y1, x2, y2, score, cls]`` in **letterbox-input**
    pixel coordinates (caller maps back to original-image coords).
    """
    if torch is None:
        raise RuntimeError("torch is required for NMS postprocessing")
    if isinstance(raw_output, torch.Tensor):
        out = raw_output
    else:
        out = torch.from_numpy(np.asarray(raw_output))

    dets = _run_nms(out, conf, iou)

    d = dets[0]
    if d is None or len(d) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return d.detach().cpu().numpy().astype(np.float32)


def postprocess_batch(raw_output, conf: float, iou: float) -> List[np.ndarray]:
    """Like ``postprocess`` but keeps every image of the batch.

    Returns a list of ``[N_i, 6]`` arrays, one per input frame (same order).
    """
    if torch is None:
        raise RuntimeError("torch is required for NMS postprocessing")
    out = raw_output if isinstance(raw_output, torch.Tensor) \
        else torch.from_numpy(np.asarray(raw_output))

    dets = _run_nms(out, conf, iou)

    results: List[np.ndarray] = []
    for d in dets:
        if d is None or len(d) == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
        else:
            results.append(d.detach().cpu().numpy().astype(np.float32))
    return results


def to_detections(
    dets: np.ndarray,
    ratio: tuple,
    pad: tuple,
    names: Dict[int, str],
    img_hw: Tuple[int, int],
    classes_of_interest: Tuple[int, ...] = (),
) -> List[Detection]:
    """Map letterbox-coord detections back to original-image coords.

    Inverse map mirrors ``infer_onnx.py``: ``x = (x - pad_w) / r``.
    """
    H, W = img_hw
    rx, ry = float(ratio[0]), float(ratio[1])
    pad_w, pad_h = float(pad[0]), float(pad[1])
    out: List[Detection] = []
    for x1, y1, x2, y2, score, cls in dets:
        cls = int(cls)
        if classes_of_interest and cls not in classes_of_interest:
            continue
        X1 = max(0.0, min(W - 1.0, (float(x1) - pad_w) / rx))
        Y1 = max(0.0, min(H - 1.0, (float(y1) - pad_h) / ry))
        X2 = max(0.0, min(W - 1.0, (float(x2) - pad_w) / rx))
        Y2 = max(0.0, min(H - 1.0, (float(y2) - pad_h) / ry))
        if X2 <= X1 or Y2 <= Y1:
            continue
        out.append(Detection(
            x1=X1, y1=Y1, x2=X2, y2=Y2,
            score=float(score), cls=cls,
            name=names.get(cls, str(cls)),
        ))
    return out
