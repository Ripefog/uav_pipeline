"""validate_pipeline — smoke test for Track + Follow (no model required).

Exercises the visdrone_mot-style tracker on synthetic moving detections and the
PID follow controller, so you can confirm the pipeline core works without a
detector/weights/TensorRT. Pass ``--config`` to additionally attempt one real
detection frame (requires torch + a configured model).
"""
import argparse
import os
import sys

import numpy as np

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from uav_pipeline.config import FollowCfg, TrackerCfg  # noqa: E402
from uav_pipeline.contracts import Detection, FollowState, Track  # noqa: E402
from uav_pipeline.follow import FollowController  # noqa: E402
from uav_pipeline.track import DroneByteTracker  # noqa: E402


def _texture_frame(h=480, w=640, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_tracker_stability():
    print("[validate] tracker: 2 objects, 20 frames, static camera")
    cfg = TrackerCfg()
    tracker = DroneByteTracker(cfg)
    frame = _texture_frame()
    H, W = frame.shape[:2]

    # object A: moves right; object B: drifts left/down
    A = [100, 100, 140, 180]
    B = [400, 300, 460, 400]
    ids_early, ids_late = None, None
    for i in range(20):
        a = [A[0] + 3 * i, A[1], A[2] + 3 * i, A[3]]
        b = [B[0] - 2 * i, B[1] + i, B[2] - 2 * i, B[3] + i]
        dets = [
            Detection(*a, score=0.9, cls=0, name="pedestrian"),
            Detection(*b, score=0.9, cls=3, name="car"),
        ]
        active = tracker.update(frame, dets)
        tracker.record_for_interpolation(i, active)
        ids = {t.track_id for t in active}
        if i == 6:
            ids_early = ids
        if i == 18:
            ids_late = ids

    assert ids_early and ids_late, "tracker never confirmed tracks"
    assert ids_early == ids_late, f"ID switch! early={ids_early} late={ids_late}"
    assert len(ids_late) == 2, f"expected 2 stable tracks, got {ids_late}"
    print(f"   OK — stable IDs {sorted(ids_late)}; "
          f"CMC success_rate={tracker.cmc.success_rate:.2f}")

    # CMC on a static (repeated) frame should report near-zero motion.
    sev = tracker.last_motion.get("severity", 1.0)
    assert sev < 0.3, f"motion severity too high on static frames: {sev}"
    print(f"   OK — static-frame motion severity {sev:.3f} (EMAT near baseline)")

    # interpolation fills the single-frame gaps we never actually created here,
    # but the call must run without error.
    interp = tracker.interpolate_tracks(cfg.interpolate_max_gap)
    print(f"   OK — interpolate_tracks ran ({sum(len(v) for v in interp.values())} filled)")
    return tracker


def test_follow_signs():
    print("[validate] follow: PID sign conventions")
    fc = FollowCfg(enabled=True, deadzone_px=4.0,
                   target_area_norm=0.1, locked_id=None)
    follower = FollowController(fc)
    H, W = 480, 640

    # target to the right and below center -> yaw>0 (pan right), pitch>0 (tilt down)
    t = Track(track_id=1, bbox=[W * 0.70, H * 0.60, W * 0.80, H * 0.80],
              confidence=0.9, frame_id=0, cls=3, name="car")
    state, cmd = follower.step([t], (H, W), ts=0.0)
    assert state.mode == "tracking", state.mode
    assert cmd.yaw_rate > 0, f"yaw should be >0 for right-of-center target, got {cmd.yaw_rate}"
    assert cmd.pitch_rate > 0, f"pitch should be >0 for below-center target, got {cmd.pitch_rate}"
    print(f"   OK — target right/below -> yaw={cmd.yaw_rate:+.2f} pitch={cmd.pitch_rate:+.2f}")

    # no tracks -> coast for lost_recovery_frames, THEN brake (zeroed command)
    n_recover = fc.lost_recovery_frames
    for i in range(n_recover):
        s, c = follower.step([], (H, W), ts=1.0 + i)
        assert c.target_lost, "coast command must be marked target_lost"
        assert s.mode == "recover"
    # one more step past the recovery window -> brake
    state2, cmd2 = follower.step([], (H, W), ts=2.0 + n_recover)
    assert cmd2.target_lost and cmd2.yaw_rate == 0.0 and cmd2.forward_vel == 0.0
    print(f"   OK — lost target coasted {n_recover}f then braked "
          f"(mode={state2.mode}, yaw={cmd2.yaw_rate:.1f})")


def test_detect_optional(config_path):
    print(f"[validate] detect: optional one-frame test ({config_path})")
    try:
        from uav_pipeline.config import Config
        from uav_pipeline.detect.wrapper import UnifiedDetector
        import cv2
        cfg = Config.from_yaml(config_path)
        det = UnifiedDetector(cfg.detector)
        frame = _texture_frame(cfg.detector.imgsz, cfg.detector.imgsz, seed=7)
        dets = det.detect(frame)
        print(f"   OK — detector returned {len(dets)} detections on a noise frame")
    except Exception as e:  # pragma: no cover
        print(f"   SKIP — detect test unavailable ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser(description="Pipeline smoke test")
    ap.add_argument("--config", default="", help="Optional config to also test 1 detection frame")
    args = ap.parse_args()

    tracker = test_tracker_stability()
    test_follow_signs()
    if args.config:
        test_detect_optional(args.config)

    print("-" * 60)
    print("[validate] tracker stats:", tracker.get_stats())
    print("[validate] ALL CORE TESTS PASSED")
    print("-" * 60)


if __name__ == "__main__":
    main()
