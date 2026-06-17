"""Drone-aware ByteTrack — faithful port of ``pratap424/visdrone_mot``
``src/drone_tracker.py``.

Two-stage ByteTrack with Camera Motion Compensation + EMAT (Ego-Motion-Adaptive
Tracking). Ported nearly verbatim; the only intentional adaptations are:

  * consumes ``List[Detection]`` (our contract) instead of the repo's detection dicts;
  * ``Track`` lives in ``contracts`` and is multi-class aware (cls/name/plate_text);
  * optional ``same_class_gate`` (default OFF = pure-IoU repo behavior) to keep
    IDs from crossing class boundaries in dense multi-class scenes;
  * ``emat`` toggle (default ON).

Algorithm is otherwise identical: constant-velocity prediction (no Kalman),
greedy IoU association (no Hungarian), EMAT-adaptive IoU/high-conf thresholds,
linear track interpolation for short occlusion gaps.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import TrackerCfg
from ..contracts import Detection, Track
from .camera_motion import CameraMotionCompensator


class DroneByteTracker:
    def __init__(self, cfg: TrackerCfg):
        self.high_conf_threshold = cfg.high_conf
        self.low_conf_threshold = cfg.low_conf
        self.iou_threshold = cfg.iou
        self.max_age = cfg.max_age
        self.min_hits = cfg.min_hits
        self.emat = cfg.emat
        self.same_class_gate = cfg.same_class_gate

        self.cmc = (CameraMotionCompensator(
            method=cfg.cmc.method, downscale=cfg.cmc.downscale,
            n_features=cfg.cmc.n_features, match_ratio=cfg.cmc.match_ratio,
            ransac_thresh=cfg.cmc.ransac_thresh, min_matches=cfg.cmc.min_matches,
        ) if cfg.cmc.enabled else None)

        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        self._track_history = defaultdict(dict)   # {track_id: {frame_id: bbox}}

        self.timing: List[float] = []
        self.id_switches = 0
        self.last_motion: Dict = {"severity": 0.0, "translation_px": 0.0,
                                  "rotation_deg": 0.0, "scale_change": 1.0}

    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        import time
        t0 = time.perf_counter()
        self.frame_count += 1

        # ---- Step 1: Camera Motion Compensation ----
        warp_matrix = None
        if self.cmc is not None:
            warp_matrix = self.cmc.estimate(frame)
            if warp_matrix is not None and len(self.tracks) > 0:
                self._apply_cmc(warp_matrix, frame.shape)

        # ---- EMAT: adapt IoU / high-conf to motion severity ----
        if self.cmc is not None and self.emat and warp_matrix is not None:
            motion = self.cmc.motion_severity(warp_matrix)
            severity = motion["severity"]
            adaptive_iou = self.iou_threshold * (1.0 - 0.5 * severity)
            adaptive_high = self.high_conf_threshold * (1.0 - 0.35 * severity)
            self.last_motion = motion
        else:
            if self.cmc is not None:
                self.last_motion = self.cmc.motion_severity(warp_matrix)
            adaptive_iou = self.iou_threshold
            adaptive_high = self.high_conf_threshold

        # ---- Step 2: Predict (constant velocity) ----
        predicted = [t.predict() for t in self.tracks]

        # ---- Step 3: Split detections high/low ----
        n = len(detections)
        det_bboxes = np.array([d.as_xyxy() for d in detections], dtype=np.float32) \
            if n else np.empty((0, 4))
        det_confs = np.array([d.score for d in detections], dtype=np.float32) if n else np.empty(0)
        det_cls = np.array([d.cls for d in detections], dtype=np.int32) if n else np.empty(0, dtype=np.int32)

        high_mask = det_confs >= adaptive_high
        low_mask = (det_confs >= self.low_conf_threshold) & (~high_mask)
        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]
        track_cls = np.array([t.cls for t in self.tracks], dtype=np.int32) \
            if self.tracks else np.empty(0, dtype=np.int32)

        # ---- Step 4: First association (high-conf) ----
        track_bboxes = np.array(predicted) if predicted else np.empty((0, 4))
        matched1, un_tr1, un_det1 = self._associate(
            track_bboxes, det_bboxes[high_idx], adaptive_iou,
            track_cls=track_cls, det_cls=det_cls[high_idx])
        for t_local, d_local in matched1:
            d_orig = int(high_idx[d_local])
            det = detections[d_orig]
            self.tracks[t_local].update(det.as_xyxy(), det.score, self.frame_count,
                                        cls=det.cls, name=det.name)

        # ---- Step 5: Second association (low-conf -> remaining tracks) ----
        rem = list(un_tr1)
        rem_bboxes = track_bboxes[rem] if rem else np.empty((0, 4))
        rem_cls = track_cls[rem] if rem else np.empty(0, dtype=np.int32)
        matched2, un_tr2, un_det2 = self._associate(
            rem_bboxes, det_bboxes[low_idx], self.iou_threshold * 0.8,
            track_cls=rem_cls, det_cls=det_cls[low_idx])
        for t_local, d_local in matched2:
            t_orig = rem[t_local]
            d_orig = int(low_idx[d_local])
            det = detections[d_orig]
            self.tracks[t_orig].update(det.as_xyxy(), det.score, self.frame_count,
                                       cls=det.cls, name=det.name)

        # ---- Step 6: Mark unmatched tracks missed ----
        matched_tracks = set(t for t, _ in matched1)
        matched_tracks.update(rem[t] for t, _ in matched2)
        for t_idx in range(len(self.tracks)):
            if t_idx not in matched_tracks:
                self.tracks[t_idx].mark_missed()

        # ---- Step 7: New tracks from unmatched high-conf detections ----
        for d_local in un_det1:
            d_orig = int(high_idx[d_local])
            det = detections[d_orig]
            self.tracks.append(Track(
                track_id=self.next_id, bbox=det.as_xyxy(), confidence=det.score,
                frame_id=self.frame_count, cls=det.cls, name=det.name))
            self.next_id += 1

        # ---- Step 8: Remove dead tracks ----
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # ---- Step 9: Return confirmed tracks ----
        active = [t for t in self.tracks
                  if t.age == 0 and t.total_visible >= self.min_hits]

        self.timing.append((time.perf_counter() - t0) * 1000.0)
        return active

    # ------------------------------------------------------------------ #
    def _apply_cmc(self, warp_matrix: np.ndarray, frame_shape: Tuple):
        if not self.tracks:
            return
        bboxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        warped = self.cmc.warp_bboxes(bboxes, warp_matrix, frame_shape)
        for i, track in enumerate(self.tracks):
            track.bbox = warped[i]
            if track.trajectory:
                pts = np.array(track.trajectory, dtype=np.float32)
                warped_pts = self.cmc.warp_points(pts, warp_matrix)
                track.trajectory = [tuple(p) for p in warped_pts]

    def _associate(self, track_bboxes, det_bboxes, iou_threshold,
                   track_cls=None, det_cls=None):
        """Hungarian-free greedy IoU association (ByteTrack style)."""
        if len(track_bboxes) == 0 or len(det_bboxes) == 0:
            return ([], list(range(len(track_bboxes))), list(range(len(det_bboxes))))

        iou_matrix = self._iou_matrix(track_bboxes, det_bboxes)
        if self.same_class_gate and track_cls is not None and det_cls is not None \
                and len(track_cls) and len(det_cls):
            same = (track_cls[:, None] == det_cls[None, :])
            iou_matrix = np.where(same, iou_matrix, 0.0)

        matched, matched_t, matched_d = [], set(), set()
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
            t_idx, d_idx = (int(v) for v in np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape))
            matched.append((t_idx, d_idx))
            matched_t.add(t_idx)
            matched_d.add(d_idx)
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        un_tr = [i for i in range(len(track_bboxes)) if i not in matched_t]
        un_det = [i for i in range(len(det_bboxes)) if i not in matched_d]
        return matched, un_tr, un_det

    @staticmethod
    def _iou_matrix(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Vectorized IoU matrix (N tracks x M dets) via broadcasting."""
        a = b1[:, np.newaxis, :]   # (N,1,4)
        b = b2[np.newaxis, :, :]   # (1,M,4)
        ix1 = np.maximum(a[..., 0], b[..., 0])
        iy1 = np.maximum(a[..., 1], b[..., 1])
        ix2 = np.minimum(a[..., 2], b[..., 2])
        iy2 = np.minimum(a[..., 3], b[..., 3])
        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        inter = iw * ih
        area1 = np.maximum(0, (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))
        area2 = np.maximum(0, (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    @property
    def avg_tracking_ms(self) -> float:
        if not self.timing:
            return 0.0
        recent = self.timing[-100:]
        return sum(recent) / len(recent)

    @property
    def n_active_tracks(self) -> int:
        return sum(1 for t in self.tracks if t.age == 0)

    def get_stats(self) -> Dict:
        return {
            "total_tracks_created": self.next_id - 1,
            "active_tracks": self.n_active_tracks,
            "lost_tracks": sum(1 for t in self.tracks if t.age > 0),
            "avg_tracking_ms": round(self.avg_tracking_ms, 2),
            "cmc_enabled": self.cmc is not None,
            "cmc_success_rate": (self.cmc.success_rate if self.cmc else "N/A"),
            "frame_count": self.frame_count,
        }

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.timing.clear()
        self._track_history = defaultdict(dict)
        if self.cmc:
            self.cmc.reset()

    # ------------------------------------------------------------------ #
    # post-processing interpolation (verbatim algorithm)
    # ------------------------------------------------------------------ #
    def record_for_interpolation(self, frame_id: int, active_tracks: List[Track]):
        for track in active_tracks:
            self._track_history[track.track_id][frame_id] = track.bbox.copy()

    def interpolate_tracks(self, max_gap: int = 5) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
        """Linear interpolation to fill short (<=max_gap) occlusion gaps."""
        interpolated = defaultdict(list)
        n_filled = 0
        for track_id, frame_bboxes in self._track_history.items():
            if len(frame_bboxes) < 2:
                continue
            frames = sorted(frame_bboxes.keys())
            for i in range(len(frames) - 1):
                f_start, f_end = frames[i], frames[i + 1]
                gap = f_end - f_start - 1
                if gap < 1 or gap > max_gap:
                    continue
                b_start, b_end = frame_bboxes[f_start], frame_bboxes[f_end]
                for f in range(f_start + 1, f_end):
                    alpha = (f - f_start) / (f_end - f_start)
                    interp = b_start * (1 - alpha) + b_end * alpha
                    interpolated[f].append((track_id, interp, 0.5))
                    n_filled += 1
        return dict(interpolated)
