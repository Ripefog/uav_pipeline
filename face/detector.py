"""SCRFD face detector — anchor-based ONNX model.

Standalone (own pre/postprocess): SCRFD's stride/anchor output format differs
from the YOLO boxes ``detect/backends`` + ``detect/postprocess.py`` handle, so
it cannot reuse that pipeline.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
DEFAULT_MODEL = os.path.join(_WEIGHTS_DIR, "scrfd_2.5g_bnkps.onnx")


class FaceDetector:
    _INPUT_SIZE = (640, 640)   # (width, height)
    _STRIDES = [8, 16, 32]
    _NUM_ANCHORS = 2

    def __init__(self, model_path: str = DEFAULT_MODEL, conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4, device: str = "cpu", num_threads: int = 2):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        providers = (["CPUExecutionProvider"] if device == "cpu"
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        opts = ort.SessionOptions()
        # Keep this low: it runs alongside the primary YOLO detector + tracker
        # + OCR on the same CPU (esp. on Jetson) — 4 threads here oversubscribes.
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._anchor_cache = {}

    def _anchor_centers(self, feat_h, feat_w, stride):
        key = (feat_h, feat_w, stride)
        if key not in self._anchor_cache:
            # mgrid[::-1] → (x_grid, y_grid); centers at 0, stride, 2*stride, ...
            centers = (
                np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
                .astype(np.float32)
                .reshape(-1, 2)
                * stride
            )
            # repeat num_anchors times per location
            self._anchor_cache[key] = np.repeat(centers, self._NUM_ANCHORS, axis=0)
        return self._anchor_cache[key]

    def preprocess(self, img):
        # Fused resize + BGR->RGB + NCHW + (x-127.5)/128 in one pass (vs. 4
        # separate array ops) — cheaper per-frame than the naive chain.
        return cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 128.0, size=self._INPUT_SIZE,
            mean=(127.5, 127.5, 127.5), swapRB=True)

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            inter = (
                np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
                * np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
            )
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.nms_threshold]
        return keep

    def detect(self, img):
        """
        Input : BGR numpy array
        Output: list of {"box": [x1, y1, x2, y2], "score": float, "crop": face_img}
        """
        orig_h, orig_w = img.shape[:2]
        iw, ih = self._INPUT_SIZE
        outputs = self.session.run(None, {self.input_name: self.preprocess(img)})

        boxes_all, scores_all = [], []
        for i, stride in enumerate(self._STRIDES):
            feat_h, feat_w = ih // stride, iw // stride
            scores = outputs[i].flatten()
            # bbox raw predictions (l,t,r,b) in anchor units → scale by stride → pixel space
            bboxes = outputs[i + len(self._STRIDES)].reshape(-1, 4) * stride
            centers = self._anchor_centers(feat_h, feat_w, stride)

            decoded = np.stack([
                centers[:, 0] - bboxes[:, 0],
                centers[:, 1] - bboxes[:, 1],
                centers[:, 0] + bboxes[:, 2],
                centers[:, 1] + bboxes[:, 3],
            ], axis=1)

            mask = scores >= self.conf_threshold
            boxes_all.append(decoded[mask])
            scores_all.append(scores[mask])

        if not any(len(b) for b in boxes_all):
            return []

        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)

        # Scale boxes from 640×640 input space back to original image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * (orig_w / iw), 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * (orig_h / ih), 0, orig_h)

        results = []
        for idx in self._nms(boxes, scores):
            x1, y1, x2, y2 = boxes[idx].astype(int)
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box (int truncation on a tiny/edge detection)
            results.append({
                "box": [x1, y1, x2, y2],
                "score": float(scores[idx]),
                "crop": img[y1:y2, x1:x2],
            })
        return results
