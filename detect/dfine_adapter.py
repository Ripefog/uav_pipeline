"""D-FINE-cpp adapter returning the pipeline's shared Detection contract.

D-FINE owns preprocessing, TensorRT inference, and decode. Unlike the YOLO
backends, its native runtime already returns final boxes in source-image
coordinates, so those results must not pass through the shared YOLO NMS path.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..contracts import Detection


class DfineDetectorAdapter:
    """Thin adapter over the optional ``dfine-cpp`` Python bindings."""

    def __init__(
        self,
        engine_path: str,
        meta_path: Optional[str] = None,
        *,
        threshold: float = 0.25,
        classes_of_interest: Iterable[int] = (),
    ):
        try:
            from dfine import Detector
        except ImportError as exc:
            raise ImportError(
                "D-FINE backend requires the dfine-cpp Python bindings. "
                "Add /path/to/dfine-cpp/python to PYTHONPATH and set "
                "DFINE_LIBRARY=/path/to/dfine-cpp/build/libdfine.so."
            ) from exc

        self._detector = Detector(
            engine_path,
            meta_path,
            threshold=threshold,
            is_bgr=True,
        )
        self._classes = frozenset(int(value) for value in classes_of_interest)

    @property
    def max_batch(self) -> int:
        return int(self._detector.max_batch)

    def _convert(self, detections: Sequence) -> List[Detection]:
        converted: List[Detection] = []
        for item in detections:
            class_id = int(item.class_id)
            if self._classes and class_id not in self._classes:
                continue
            box = item.box
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            converted.append(Detection(
                x1=float(box.x1),
                y1=float(box.y1),
                x2=float(box.x2),
                y2=float(box.y2),
                score=float(item.score),
                cls=class_id,
                name=str(item.class_name),
            ))
        return converted

    def detect(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Detection]:
        output = self._detector.detect(
            frame,
            threshold=threshold,
            is_bgr=True,
        )
        return self._convert(output)

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[List[Detection]]:
        if not frames:
            return []
        if len(frames) > self.max_batch:
            raise ValueError(
                f"D-FINE batch {len(frames)} exceeds engine max batch "
                f"{self.max_batch}"
            )
        output = self._detector.detect_batch(
            frames,
            threshold=threshold,
            is_bgr=True,
        )
        return [self._convert(items) for items in output]

    def close(self) -> None:
        self._detector.close()

