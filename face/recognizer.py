"""MobileFaceNet embedding + turbovec-backed gallery + per-track majority vote.

``turbovec`` (quantized ANN index) is imported lazily inside ``FaceDB`` so
importing this module doesn't require it unless the gallery is actually used.
"""
import json
import os
from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(os.path.dirname(_DIR), "weights")
DEFAULT_MODEL = os.path.join(_WEIGHTS_DIR, "mobilefacenet.onnx")
DEFAULT_INDEX = os.path.join(_DIR, "examples", "face-data", "face_index")


class FaceRecognizer:
    """Lazy ONNX MobileFaceNet — BGR face crop -> L2-normalized 512-dim embedding."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu", num_threads: int = 2):
        providers = (["CPUExecutionProvider"] if device == "cpu"
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        opts = ort.SessionOptions()
        # Keep this low: runs once per detected face, alongside the primary
        # YOLO detector + tracker + OCR on the same CPU (esp. on Jetson).
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face_crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Input: BGR face crop (any size). Output: L2-normalized 512-dim float32 vector,
        or None if the crop is empty (mirrors ``ocr.PlateOCR.recognize``'s guard)."""
        if face_crop is None or face_crop.size == 0:
            return None
        tensor = cv2.dnn.blobFromImage(
            face_crop, scalefactor=1.0 / 128.0, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True)
        embedding = self.session.run(None, {self.input_name: tensor})[0][0]
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


class FaceDB:
    """Local vector store over the enrolled-face gallery (turbovec IdMapIndex)."""

    def __init__(self, dim: int = 512, bit_width: int = 4):
        from turbovec import IdMapIndex
        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        self._meta: dict = {}
        self._next_id: int = 0

    def add(self, vector: np.ndarray, person_name: str, image_file: str):
        arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        ids = np.array([self._next_id], dtype=np.uint64)
        self._index.add_with_ids(arr, ids)
        self._meta[self._next_id] = {
            "person_name": person_name,
            "image_file": image_file,
        }
        self._next_id += 1

    def search(self, vector: np.ndarray, k: int = 1) -> list:
        """Returns list of (score, person_name, image_file), best first."""
        arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self._index.search(arr, k)
        results = []
        for score, uid in zip(scores[0], ids[0]):
            meta = self._meta[int(uid)]
            results.append((float(score), meta["person_name"], meta["image_file"]))
        return results

    def save(self, base_path: str = DEFAULT_INDEX):
        self._index.write(f"{base_path}.tvim")
        payload = {
            "next_id": self._next_id,
            "meta": {str(k): v for k, v in self._meta.items()},
        }
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, base_path: str = DEFAULT_INDEX) -> "FaceDB":
        from turbovec import IdMapIndex
        obj = cls.__new__(cls)
        obj._index = IdMapIndex.load(f"{base_path}.tvim")
        with open(f"{base_path}.json", encoding="utf-8") as f:
            payload = json.load(f)
        obj._next_id = payload["next_id"]
        obj._meta = {int(k): v for k, v in payload["meta"].items()}
        return obj


class FaceVoter:
    """Per-track majority vote over a sliding window of identity readings
    (mirrors ``ocr.PlateVoter``)."""

    def __init__(self, window: int = 5):
        self.window = max(1, window)
        self._buf = defaultdict(lambda: deque(maxlen=self.window))

    def add(self, track_id: int, person_name: Optional[str]):
        if person_name:
            self._buf[track_id].append(person_name)

    def majority(self, track_id: int) -> Optional[str]:
        d = self._buf.get(track_id)
        if not d:
            return None
        return Counter(d).most_common(1)[0][0]

    def drop(self, track_id: int):
        self._buf.pop(track_id, None)
