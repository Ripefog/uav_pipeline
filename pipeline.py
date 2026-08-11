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
from .sinks import (ControlLogSink, DeferredSinkWriter, HUDAnnotatedSink, Sink,
                    SotResultSink, TelemetrySink)
from .sources import make_source
from .track import DroneByteTracker


class Pipeline:
    def __init__(self, config: Config):
        self.cfg = config

        # Fail-fast: scripts/run_pipeline.py đã validate trước khi tới đây, nhưng
        # eval_mot_visdrone.py và bất kỳ caller nào khác construct Pipeline(cfg)
        # trực tiếp thì không. Chặn ở đây trước khi load detector/MCITrack (tốn
        # thời gian) để lỗi cấu hình chết sớm, chết rõ — không phải AttributeError
        # 'NoneType' sau khi model đã load xong.
        errs = config.validate()
        if errs:
            raise SystemExit("[pipeline] cấu hình lỗi:\n  - " + "\n  - ".join(errs))

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

        # ---- track: MOT hoặc SOT, loại trừ nhau (Config.validate() đã chặn) ----
        self.tracker = DroneByteTracker(config.tracker) if config.tracker.enabled else None
        self.sot = None
        if config.sot.enabled:
            # batch>1 chỉ có nghĩa khi detector chạy MỌI frame. Với SOT thì detector
            # chỉ cần ở frame acquire/verify -> gom batch là detect thừa ~90%.
            if not config.sot.detect_every_frame and int(getattr(config.detector, "batch", 1)) > 1:
                print(f"[pipeline] SOT + detect_every_frame=false -> ép "
                      f"detector.batch {config.detector.batch} -> 1")
                config.detector.batch = 1
            from .sot import SotTracker
            from .sot.mcitrack_wrapper import build_mcitrack_model
            self.sot = SotTracker(config.sot, self.detector.names,
                                  build_mcitrack_model(config.sot))
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
        if config.sot.enabled and config.sinks.sot_result.enabled:
            self.sinks.append(SotResultSink(config.sinks.sot_result))

        # ---- state ----
        self.fps = 0.0            # full pipeline incl. video sink write
        self.fps_detect = 0.0     # detector inference only
        self.fps_pipeline = 0.0   # detect+track+follow+jsonl sinks, excl. video write
        self._prev_ts: Optional[float] = None
        self._t_start: Optional[float] = None   # wall-clock start for throughput
        self._t_detect_total = 0.0
        self._t_video_total = 0.0
        self._stop = False
        self._n_frames = 0

        # Guard có thể tuyên bố LOST rồi cắt LUI về frame đầu chuỗi. Stream thì
        # frame đã ghi không lấy lại được -> hoãn ghi motion.k-1 frame.
        self._deferred = None
        if self.sot is not None and self.sot.needs_deferral:
            # ⚠ max_hold PHẢI bằng guard.motion.k - 1 (không phải hằng số): guard
            # cắt lui tối đa k-1 frame về trước, writer phải giữ đủ để retract
            # không lọt box sai ra sink đã ghi. Lệch 2 số này -> writer log cảnh
            # báo "desync max_hold/motion.k" nhưng KHÔNG chặn được frame sai.
            hold = max(0, config.sot.guard.motion.k - 1)
            self._deferred = DeferredSinkWriter(self._write_sinks_now, max_hold=hold)
            print(f"[pipeline] SOT guard motion=on -> hoãn ghi sink {hold} frame "
                  f"(để cắt lui không lọt box sai)")

    # ------------------------------------------------------------------ #
    def stop(self):
        self._stop = True

    def run(self):
        print(f"[pipeline] source={self.cfg.source.type} "
              f"backend={self.cfg.detector.backend} "
              f"track={'SOT' if self.sot is not None else 'MOT'} "
              f"ocr={'on' if self.plate_ocr else 'off'} "
              f"controller={self.cfg.controller.backend}")
        batch = max(1, int(getattr(self.cfg.detector, "batch", 1)))
        try:
            if batch > 1:
                self._run_batched(batch)
            else:
                for meta, frame in self.source:
                    if self._stop:
                        break
                    ctx = self.process_frame(meta, frame)
                    self._write_sinks(ctx)
        except KeyboardInterrupt:
            print("\n[pipeline] interrupted by user")
        finally:
            self.close()

    # ------------------------------------------------------------------ #
    def _run_batched(self, batch: int):
        """Buffer `batch` frames, detect them in one inference, then run
        track -> ocr -> follow -> sinks sequentially (order preserved)."""
        buf = []
        for meta, frame in self.source:
            if self._stop:
                break
            buf.append((meta, frame))
            if len(buf) < batch:
                continue
            self._flush_batch(buf)
            buf = []
        if buf and not self._stop:
            self._flush_batch(buf)

    def _flush_batch(self, buf):
        frames = [f for _, f in buf]
        t0 = time.monotonic()
        det_lists = self.detector.detect_batch(frames)
        self._t_detect_total += time.monotonic() - t0
        for (meta, frame), detections in zip(buf, det_lists):
            ctx = self.process_frame(meta, frame, detections=detections)
            self._write_sinks(ctx)

    def _write_sinks(self, ctx: FrameContext):
        if self._deferred is not None:
            self._deferred.write(ctx, provisional=self.sot.provisional,
                                 retract_from=self.sot.retract_from)
        else:
            self._write_sinks_now(ctx)

    def _write_sinks_now(self, ctx: FrameContext):
        for s in self.sinks:
            if isinstance(s, HUDAnnotatedSink):
                t0 = time.monotonic()
                s.write(ctx)
                self._t_video_total += time.monotonic() - t0
            else:
                s.write(ctx)

    # ------------------------------------------------------------------ #
    def process_frame(self, meta: FrameMeta, frame: np.ndarray,
                      detections: Optional[List[Detection]] = None) -> FrameContext:
        self._n_frames += 1
        total = getattr(self.source, "total_frames", None)
        if self._n_frames == 1:
            h, w = frame.shape[:2]
            print(f"[pipeline] resolution={w}x{h}")
        print(f"[pipeline] frame {self._n_frames}/{total if total else '?'}")

        # detect (primary, + optional plate detector) + track
        def _detect_now():
            t0 = time.monotonic()
            d = self.detector.detect(frame)
            self._t_detect_total += time.monotonic() - t0
            return d

        if self.sot is not None:
            # SOT: detector chỉ chạy ở frame acquire/verify, trừ khi
            # detect_every_frame=true. SotTracker tự quyết (xem sot/tracker.py).
            if self.cfg.sot.detect_every_frame and detections is None:
                detections = _detect_now()
            tracks = self.sot.update(frame, meta.idx + 1, _detect_now,
                                     prefetched=detections)
            detections = self.sot.last_detections
            plates = []
        else:
            if detections is None:
                detections = _detect_now()
            plates = self.detector.detect_plates(frame) if self.detector.plate_enabled else []
            tracks = self.tracker.update(frame, detections)
            self.tracker.record_for_interpolation(meta.idx, tracks)

        # ocr (throttled, per track)
        if self.plate_ocr is not None:
            self._run_ocr(frame, tracks, plates, meta.idx)

        # follow -> command -> controller
        follow_state, command = self.follower.step(tracks, meta.shape_hw, meta.ts)
        self.controller.send(command)

        # timing
        self._update_fps()

        return FrameContext(
            meta=meta, frame=frame,
            detections=detections, tracks=tracks,
            follow_state=follow_state, command=command,
            fps=self.fps, fps_detect=self.fps_detect, fps_pipeline=self.fps_pipeline,
            extra_stats=self._extra_stats(),
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
        if self.tracker is not None and self.tracker.cmc is not None:
            sev = float(self.tracker.last_motion.get("severity", 0.0))
            bar = ("#" * int(sev * 10)).ljust(10, "-")
            stats["motion"] = f"{bar} {sev:.2f}"
        if self.sot is not None:
            stats["sot"] = self.sot.status
        return stats

    def _update_fps(self):
        # True end-to-end throughput = frames / wall-clock elapsed since start.
        # We do NOT use meta.ts (frame *read* time) nor a per-frame instant EMA:
        # in batch mode the detector runs once per N frames *outside* the
        # per-frame loop, so a per-frame delta would miss the detection cost and
        # report a bogus number (~1000 from read-stamps, ~152 from track-only
        # time). A cumulative average over wall-clock includes detection and is
        # correct in both single-frame and batch modes.
        now = time.monotonic()
        if self._t_start is None:
            self._t_start = now
            return
        elapsed = now - self._t_start
        if elapsed > 0:
            self.fps = self._n_frames / elapsed
        if self._t_detect_total > 0:
            self.fps_detect = self._n_frames / self._t_detect_total
        novideo = elapsed - self._t_video_total
        if novideo > 0:
            self.fps_pipeline = self._n_frames / novideo

    # ------------------------------------------------------------------ #
    def close(self):
        # flush frame đang bị hoãn trước khi đóng sink
        if self._deferred is not None:
            self._deferred.close()

        # post-processing interpolation (chỉ có ở đường MOT)
        if self.tracker is not None and self.cfg.tracker.interpolate_max_gap > 0 \
                and self.tracker.frame_count > 0:
            interp = self.tracker.interpolate_tracks(self.cfg.tracker.interpolate_max_gap)
            if interp:
                print(f"[pipeline] interpolated {sum(len(v) for v in interp.values())} "
                      f"detections (max_gap={self.cfg.tracker.interpolate_max_gap})")

        self.source.release()
        for s in self.sinks:
            s.close()

        print("-" * 60)
        print(f"[pipeline] frames processed : {self._n_frames}")
        print(f"[pipeline] avg FPS (full)   : {self.fps:.1f}")
        print(f"[pipeline] avg FPS (detect) : {self.fps_detect:.1f}")
        print(f"[pipeline] avg FPS (pipe)   : {self.fps_pipeline:.1f}  (detect+track+follow+jsonl, excl. video write)")
        if self.tracker is not None:
            print(f"[pipeline] {self.tracker.get_stats()}")
        if self.sot is not None:
            print(f"[pipeline] SOT {self.sot.status}"
                  + (f"  (LOST @{self.sot.lost_at}: {self.sot.lost_reason})"
                     if self.sot.lost_at is not None else ""))
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
