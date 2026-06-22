"""Pipeline orchestrator — single-frame streaming loop.

    for each frame:
        detect -> (plates) -> track -> (ocr) -> follow.step -> controller.send -> sinks.write

Owns timing (EMA FPS), graceful shutdown, and final stats. Differs from the
the original YOLO infer scripts in that it streams one frame at a time instead of
loading the whole video into RAM.
"""
import time
from typing import List, Optional

import numpy as np

from .config import Config
from ._paths import resolve
from .contracts import Detection, FrameContext, FrameMeta
from .detect.wrapper import UnifiedDetector
from .follow import FollowController, make_controller
from .ocr import PlateOCR, PlateVoter, lower_third_crop
from .sinks import ControlLogSink, HUDAnnotatedSink, Sink, TelemetrySink
from .sources import make_source
from .track import DroneByteTracker


class Pipeline:
    def __init__(self, config: Config):
        self.cfg = config

        # ---- source ----
        self.source = make_source(config.source)
        if config.sinks.video.fps <= 0:
            config.sinks.video.fps = self.source.fps or 30.0

        # ---- detect ----
        self.detector = UnifiedDetector(config.detector)
        if config.ocr.enabled and config.ocr.plate_detector.enabled:
            self.detector.enable_plate(config.ocr.plate_detector)

        # ---- ocr (lazy) ----
        self.plate_ocr: Optional[PlateOCR] = None
        self.plate_voter: Optional[PlateVoter] = None
        if config.ocr.enabled:
            if config.ocr.keras_model and config.ocr.plate_config:
                try:
                    self.plate_ocr = PlateOCR(resolve(config.ocr.keras_model),
                                              resolve(config.ocr.plate_config),
                                              device=config.ocr.device)
                    self.plate_voter = PlateVoter(config.ocr.vote_window)
                except Exception as e:  # pragma: no cover
                    print(f"[pipeline] OCR disabled (load failed): {e}")
            else:
                print("[pipeline] OCR enabled but ocr.keras_model / ocr.plate_config "
                      "not set — skipping OCR.")

        # ---- track / follow / control ----
        self.tracker = DroneByteTracker(config.tracker)
        self.follower = FollowController(config.follow)
        self.controller = make_controller(config.controller)

        # ---- sinks ----
        self.sinks: List[Sink] = []
        if config.sinks.video.enabled:
            self.sinks.append(HUDAnnotatedSink(
                config.sinks.video, target_area_norm=config.follow.target_area_norm))
        if config.sinks.telemetry.enabled:
            self.sinks.append(TelemetrySink(config.sinks.telemetry))
        if config.sinks.control_log.enabled:
            self.sinks.append(ControlLogSink(config.sinks.control_log))

        # ---- state ----
        self.fps = 0.0
        self._prev_ts: Optional[float] = None
        self._stop = False
        self._n_frames = 0

    # ------------------------------------------------------------------ #
    def stop(self):
        self._stop = True

    def run(self):
        print(f"[pipeline] source={self.cfg.source.type} "
              f"backend={self.cfg.detector.backend} "
              f"ocr={'on' if self.plate_ocr else 'off'} "
              f"controller={self.cfg.controller.backend}")
        try:
            for meta, frame in self.source:
                if self._stop:
                    break
                ctx = self.process_frame(meta, frame)
                for s in self.sinks:
                    s.write(ctx)
        except KeyboardInterrupt:
            print("\n[pipeline] interrupted by user")
        finally:
            self.close()

    # ------------------------------------------------------------------ #
    def process_frame(self, meta: FrameMeta, frame: np.ndarray) -> FrameContext:
        self._n_frames += 1

        # detect (primary, + optional plate detector)
        detections = self.detector.detect(frame)
        plates = self.detector.detect_plates(frame) if self.detector.plate_enabled else []

        # track
        tracks = self.tracker.update(frame, detections)
        self.tracker.record_for_interpolation(meta.idx, tracks)

        # ocr (throttled, per track)
        if self.plate_ocr is not None:
            self._run_ocr(frame, tracks, plates, meta.idx)

        # follow -> command -> controller
        follow_state, command = self.follower.step(tracks, meta.shape_hw, meta.ts)
        self.controller.send(command)

        # timing
        self._update_fps(meta.ts)

        return FrameContext(
            meta=meta, frame=frame,
            detections=detections, tracks=tracks,
            follow_state=follow_state, command=command,
            fps=self.fps, extra_stats=self._extra_stats(),
        )

    # ------------------------------------------------------------------ #
    def _run_ocr(self, frame, tracks, plates, idx):
        cfg = self.cfg.ocr
        if cfg.crop_mode == "plate_detection" and plates:
            self._ocr_plate_boxes(frame, tracks, plates, cfg.min_plate_area_px)
            return
        # default: lower-third of vehicle tracks
        vehicle = set(cfg.vehicle_classes_for_lower_third)
        for t in tracks:
            if t.name not in vehicle:
                continue
            if idx % max(1, cfg.every_n_frames) != 0:
                continue
            if t.area < cfg.min_plate_area_px:
                continue
            text = self.plate_ocr.recognize(lower_third_crop(frame, t.bbox))
            self.plate_voter.add(t.track_id, text)
            t.plate_text = self.plate_voter.majority(t.track_id)

    def _ocr_plate_boxes(self, frame, tracks, plates, min_area):
        for p in plates:
            if (p.x2 - p.x1) * (p.y2 - p.y1) < min_area:
                continue
            crop = frame[int(p.y1):int(p.y2), int(p.x1):int(p.x2)]
            text = self.plate_ocr.recognize(crop)
            if not text:
                continue
            # attach to the most-overlapping track
            best, best_iou = None, 0.1
            for t in tracks:
                v = _iou(t.bbox, p.as_xyxy())
                if v > best_iou:
                    best, best_iou = t, v
            if best is not None:
                self.plate_voter.add(best.track_id, text)
                best.plate_text = self.plate_voter.majority(best.track_id)

    def _extra_stats(self):
        stats = {}
        if self.tracker.cmc is not None:
            sev = float(self.tracker.last_motion.get("severity", 0.0))
            bar = ("#" * int(sev * 10)).ljust(10, "-")
            stats["motion"] = f"{bar} {sev:.2f}"
        return stats

    def _update_fps(self, ts: float):
        if self._prev_ts is not None and ts > self._prev_ts:
            instant = 1.0 / max(1e-6, ts - self._prev_ts)
            self.fps = instant if self.fps == 0.0 else 0.9 * self.fps + 0.1 * instant
        self._prev_ts = ts

    # ------------------------------------------------------------------ #
    def close(self):
        # post-processing interpolation (for offline MOT export; not drawn live)
        if self.cfg.tracker.interpolate_max_gap > 0 and self.tracker.frame_count > 0:
            interp = self.tracker.interpolate_tracks(self.cfg.tracker.interpolate_max_gap)
            if interp:
                print(f"[pipeline] interpolated {sum(len(v) for v in interp.values())} "
                      f"detections (max_gap={self.cfg.tracker.interpolate_max_gap})")

        self.source.release()
        for s in self.sinks:
            s.close()

        print("-" * 60)
        print(f"[pipeline] frames processed : {self._n_frames}")
        print(f"[pipeline] avg FPS          : {self.fps:.1f}")
        print(f"[pipeline] {self.tracker.get_stats()}")
        print("-" * 60)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
