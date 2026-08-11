"""validate_sot — test SOT không cần GPU / checkpoint / ONNX.

Repo không có pytest; đây là script chạy trực tiếp, cùng pattern với
scripts/validate_pipeline.py.

    cd /home/anlnm/UAV
    MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
    MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_config_mutual_exclusion
"""
import argparse
import os
import sys

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from uav_pipeline.config import Config  # noqa: E402
from uav_pipeline.config import SotGuardCfg  # noqa: E402  (thêm cùng import Config)
from uav_pipeline.contracts import Detection  # noqa: E402
from uav_pipeline.sot.class_groups import accepted_ids  # noqa: E402
from uav_pipeline.sot.guard import LostGuard, iou_xywh  # noqa: E402


def test_config_defaults_backward_compatible():
    """Config rỗng phải giữ hành vi CŨ: MOT bật, SOT tắt.

    Config.from_yaml load MỘT file, không merge default.yaml. Nếu đổi default của
    dataclass thì 4 config MOT đang có (local_onnx_batch16, jetson_trt,
    local_trt_*) tự chuyển sang SOT và eval_mot_visdrone.py hỏng theo.
    """
    cfg = Config.from_dict({})
    assert cfg.tracker.enabled is True, "tracker.enabled default phải là True"
    assert cfg.sot.enabled is False, "sot.enabled default phải là False"
    assert cfg.validate() == [], cfg.validate()
    print("[ok] test_config_defaults_backward_compatible")


def test_config_mutual_exclusion():
    both = Config.from_dict({"tracker": {"enabled": True}, "sot": {"enabled": True}})
    errs = both.validate()
    assert len(errs) == 1 and "loại trừ" in errs[0], errs

    neither = Config.from_dict({"tracker": {"enabled": False}, "sot": {"enabled": False}})
    errs = neither.validate()
    assert len(errs) == 1 and "không có tracker nào" in errs[0], errs

    ok = Config.from_dict({"tracker": {"enabled": False}, "sot": {"enabled": True}})
    assert ok.validate() == [], ok.validate()
    print("[ok] test_config_mutual_exclusion")


def test_config_sot_value_checks():
    base = {"tracker": {"enabled": False}, "sot": {"enabled": True}}

    bad_lost = Config.from_dict({**base, "sot": {"enabled": True, "on_lost": "xyz"}})
    assert any("on_lost" in e for e in bad_lost.validate()), bad_lost.validate()

    bad_gate = Config.from_dict({**base, "sot": {"enabled": True,
                                                 "guard": {"gate": "nope"}}})
    assert any("gate" in e for e in bad_gate.validate()), bad_gate.validate()

    bad_box = Config.from_dict({**base, "sot": {"enabled": True,
                                                "init_bbox": [10, 10, 0, 50]}})
    assert any("init_bbox" in e for e in bad_box.validate()), bad_box.validate()

    good_box = Config.from_dict({**base, "sot": {"enabled": True,
                                                 "init_bbox": [10, 10, 30, 50]}})
    assert good_box.validate() == [], good_box.validate()
    assert good_box.sot.init_bbox == [10, 10, 30, 50]
    print("[ok] test_config_sot_value_checks")


def test_config_nested_guard_defaults():
    """Ngưỡng guard là số đã calibrate — không được lệch."""
    cfg = Config.from_dict({"sot": {"enabled": True, "guard": {"enabled": True}}})
    g = cfg.sot.guard
    assert (g.gate, g.verify_every, g.K, g.iou_gate) == ("class", 10, 3, 0.3)
    assert (g.jump.enabled, g.jump.px, g.jump.area, g.jump.ref_width) == \
        (True, 90.0, 2.5, 1904.0)
    assert (g.motion.enabled, g.motion.iou, g.motion.k) == (True, 0.05, 2)
    print("[ok] test_config_nested_guard_defaults")


def test_config_warnings():
    cfg = Config.from_dict({
        "source": {"loop": True},
        "tracker": {"enabled": False},
        "sot": {"enabled": True, "on_lost": "reacquire", "guard": {"enabled": False}},
    })
    w = " | ".join(cfg.warnings())
    assert "loop" in w, w
    assert "reacquire" in w, w
    print("[ok] test_config_warnings")


VISDRONE = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}


def test_class_groups_gate_class():
    # person gộp pedestrian + people (VisDrone tách người thành 2 class)
    assert accepted_ids("class", 0, VISDRONE) == [0, 1]
    assert accepted_ids("class", 1, VISDRONE) == [0, 1]
    # car chỉ là car, KHÔNG kéo van/truck/bus vào (đó là việc của family)
    assert accepted_ids("class", 3, VISDRONE) == [3]
    assert accepted_ids("class", 5, VISDRONE) == [5]
    print("[ok] test_class_groups_gate_class")


def test_class_groups_gate_family():
    # detector lẫn bus/van/car nặng: tại vị trí GT bus nó gán van 27 frame,
    # bus 12, car 8, không có gì 17 frame (miss rate 78%)
    assert accepted_ids("family", 8, VISDRONE) == [3, 4, 5, 8]
    assert accepted_ids("family", 3, VISDRONE) == [3, 4, 5, 8]
    assert accepted_ids("family", 9, VISDRONE) == [2, 6, 7, 9]
    assert accepted_ids("family", 0, VISDRONE) == [0, 1]
    print("[ok] test_class_groups_gate_family")


def test_class_groups_presence_and_unknown():
    # presence: bỏ qua class hoàn toàn
    assert accepted_ids("presence", 3, VISDRONE) is None
    # tên không có trong bảng -> hạ xuống presence thay vì trả list rỗng
    # (list rỗng sẽ chặn mọi detection -> guard cắt oan mọi lúc)
    assert accepted_ids("class", 99, VISDRONE) is None
    assert accepted_ids("class", 0, {0: "airplane"}) is None
    print("[ok] test_class_groups_presence_and_unknown")


def test_class_groups_missing_name_in_names():
    """names.yaml thiếu 'people' -> chỉ trả id thực sự có, không KeyError."""
    partial = {0: "pedestrian", 3: "car"}
    assert accepted_ids("class", 0, partial) == [0]
    print("[ok] test_class_groups_missing_name_in_names")


def _guard(width=1904, init_cls=3, **over):
    """SotGuardCfg bật sẵn, cho phép override từng key lồng nhau."""
    cfg = SotGuardCfg(enabled=True)
    for k, v in over.items():
        if "." in k:
            a, b = k.split(".", 1)
            setattr(getattr(cfg, a), b, v)
        else:
            setattr(cfg, k, v)
    return LostGuard(cfg, width, init_cls, VISDRONE)


def _no_det():
    return []


def test_iou_xywh():
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xywh([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0
    assert abs(iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) - (50 / 150)) < 1e-9
    assert iou_xywh([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0   # box rỗng, không chia 0
    print("[ok] test_iou_xywh")


def test_guard_off_never_lost():
    """guard.enabled=false phải là hành vi MCITrack gốc: không bao giờ LOST."""
    cfg = SotGuardCfg(enabled=False)
    g = LostGuard(cfg, 1904, 3, VISDRONE)
    boxes = [[0, 0, 10, 10], [900, 900, 300, 300], [0, 0, 10, 10]]  # nhảy lung tung
    for i, b in enumerate(boxes, start=2):
        v = g.step(i, b, _no_det)
        assert v.alive and not v.provisional, (i, v)
    print("[ok] test_guard_off_never_lost")


def test_guard_first_frame_never_jumps():
    """Frame tracker ĐẦU TIÊN không được đem so với gì cả.

    Box detector và box MCITrack là 2 estimator khác nhau cho cùng vật -> chênh
    scale ở frame 2 là bình thường. Trước khi sửa, uav0000117_02622_v_car LOST oan
    ngay frame 2 (1/349 frame alive).
    """
    g = _guard()
    v = g.step(2, [1000, 500, 40, 40], _no_det)   # rất khác box init, vẫn phải sống
    assert v.alive and v.lost_at is None, v
    print("[ok] test_guard_first_frame_never_jumps")


def test_guard_jump_detector_catches_real_drift():
    """Cú nhảy thật của car uav0000339_00001_v f63->f64: 337px, 226x224 -> 74x64."""
    g = _guard(**{"motion.enabled": False})
    assert g.step(63, [500, 500, 226, 224], _no_det).alive
    v = g.step(64, [800, 640, 74, 64], _no_det)
    assert not v.alive and v.lost_at == 64, v
    assert "jump" in v.reason, v.reason
    print("[ok] test_guard_jump_detector_catches_real_drift")


def test_guard_jump_tolerates_real_gt_motion():
    """Chuyển động GT thật của 5 vật (n=708): max 44px/frame, max area x1.44."""
    g = _guard(**{"motion.enabled": False})
    box = [500.0, 500.0, 100.0, 100.0]
    for i in range(2, 30):
        box = [box[0] + 44.0, box[1], box[2] * 1.012, box[3] * 1.012]
        v = g.step(i, list(box), _no_det)
        assert v.alive, (i, v.reason)
    print("[ok] test_guard_jump_tolerates_real_gt_motion")


def test_guard_jump_px_scales_with_resolution():
    """VisDrone trải 1344x756 -> 3840x2160 (2.86x). Ngưỡng px TUYỆT ĐỐI không
    scale sẽ báo oan trên video 4K và bỏ sót drift trên video nhỏ."""
    small = _guard(width=1344, **{"motion.enabled": False})
    big = _guard(width=3840, **{"motion.enabled": False})
    assert abs(small.jump_px - 90.0 * 1344 / 1904) < 1e-6, small.jump_px
    assert abs(big.jump_px - 90.0 * 3840 / 1904) < 1e-6, big.jump_px
    # dịch 120px: quá ngưỡng trên video nhỏ (63.5), chưa quá trên video 4K (181.5)
    for g, expect_alive in ((small, False), (big, True)):
        g.step(2, [500, 500, 100, 100], _no_det)
        v = g.step(3, [620, 500, 100, 100], _no_det)
        assert v.alive is expect_alive, (g.jump_px, v)
    print("[ok] test_guard_jump_px_scales_with_resolution")


def test_guard_motion_gate_cuts_back_to_streak_start():
    """motion.k=2: frame đầu chuỗi là provisional (hoãn ghi), frame thứ 2 mới
    tuyên bố LOST và cắt LUI về frame đầu chuỗi — box sai không được lọt ra."""
    g = _guard(**{"jump.enabled": False})
    # 3 frame đi thẳng đều để có mốc dự đoán
    for i, x in ((2, 500), (3, 510), (4, 520)):
        assert g.step(i, [x, 500, 100, 100], _no_det).alive
    # f5, f6 lệch hẳn khỏi dự đoán (dự đoán ~530)
    v5 = g.step(5, [900, 900, 100, 100], _no_det)
    assert v5.alive and v5.provisional, v5
    v6 = g.step(6, [905, 905, 100, 100], _no_det)
    assert not v6.alive and v6.lost_at == 5, v6
    assert "motion" in v6.reason, v6.reason
    print("[ok] test_guard_motion_gate_cuts_back_to_streak_start")


def test_guard_motion_gate_recovers_without_lost():
    """Chuỗi bị ngắt -> frame đang giữ là hợp lệ, không LOST."""
    g = _guard(**{"jump.enabled": False})
    for i, x in ((2, 500), (3, 510), (4, 520)):
        g.step(i, [x, 500, 100, 100], _no_det)
    assert g.step(5, [900, 900, 100, 100], _no_det).provisional
    v = g.step(6, [540, 500, 100, 100], _no_det)   # về đúng quỹ đạo
    assert v.alive and not v.provisional and v.lost_at is None, v
    print("[ok] test_guard_motion_gate_recovers_without_lost")


def test_guard_motion_gate_does_not_poison_itself():
    """Mốc dự đoán CHỈ cập nhật từ frame đã được chấp nhận.

    Đo thật trên uav0000086 person rank 4: f148 giật 35px, f149-150 đã về đúng
    (IoU vs GT 0.66) nhưng nếu để f148 làm mốc thì IoU vs dự đoán vẫn 0.000 ->
    luôn sinh >=2 vi phạm liên tiếp -> LOST oan.
    """
    g = _guard(**{"jump.enabled": False})
    for i, x in ((2, 500), (3, 510), (4, 520)):
        g.step(i, [x, 500, 100, 100], _no_det)
    # DEVIATION (xem task-3-report.md): dự đoán tại f5 là [530,500,100,100]. Box
    # [560,535,100,100] của brief cho IoU=0.294 vs dự đoán -> KHÔNG dưới ngưỡng
    # motion.iou=0.05 nên không bao giờ provisional (test gốc tự mâu thuẫn với
    # threshold đã calibrate ở box 100x100). Giữ ý real "IoU vs dự đoán 0.000",
    # đổi sang lệch hẳn (0 overlap) để test đúng ý đồ mà không đụng threshold.
    assert g.step(5, [650, 600, 100, 100], _no_det).provisional      # 1 frame giật
    # 2 frame sau đã về đúng quỹ đạo cũ (530, 540) -> phải sống, không LOST
    assert g.step(6, [540, 500, 100, 100], _no_det).alive
    v = g.step(7, [550, 500, 100, 100], _no_det)
    assert v.alive and v.lost_at is None, v
    print("[ok] test_guard_motion_gate_does_not_poison_itself")


def test_guard_verify_needs_prior_confirmation():
    """Tầng verify chỉ được kết án track mà detector ĐÃ TỪNG xác nhận.

    Vật 26x41 px của uav0000339: detector 0 hit ở f10..f50 trong khi tracker vẫn
    đúng (IoU vs GT 0.67-0.75). Không có latch thì cắt oan ở f30.
    """
    g = _guard(**{"jump.enabled": False, "motion.enabled": False})
    for i in range(2, 61):
        v = g.step(i, [500, 500, 26, 41], _no_det)     # detector không thấy gì
        assert v.alive, (i, v.reason)
    print("[ok] test_guard_verify_needs_prior_confirmation")


def test_guard_verify_cuts_after_confirmation():
    """Đã xác nhận 1 lần rồi mất K=3 lần verify liên tiếp -> LOST."""
    g = _guard(**{"jump.enabled": False, "motion.enabled": False})
    hit = [Detection(x1=500, y1=500, x2=600, y2=600, score=0.9, cls=3, name="car")]

    def det_hit():
        return hit

    for i in range(2, 11):
        assert g.step(i, [500, 500, 100, 100], det_hit).alive
    assert g.step(10, [500, 500, 100, 100], det_hit).alive       # verify HIT -> latch
    lost = None
    for i in range(11, 60):
        v = g.step(i, [500, 500, 100, 100], _no_det)             # verify MISS
        if not v.alive:
            lost = v
            break
    # DEVIATION (xem task-3-report.md): K=3 nghia la 3 MISS lien tiep -> LOST (dung
    # voi "longest legitimate streak = 2" trong brief). MISS o f20,f30,f40 la du 3
    # lan -> LOST ngay f40, khong can f50 (brief goc ghi lost_at==50 la sai 1 nhip).
    assert lost is not None and lost.lost_at == 40, lost   # MISS ở f20,f30,f40
    assert "verify" in lost.reason.lower(), lost.reason
    print("[ok] test_guard_verify_cuts_after_confirmation")


def test_guard_verify_class_gate_rejects_wrong_class():
    """gate=class: detection trùng vị trí nhưng SAI class thì không tính là HIT."""
    g = _guard(init_cls=3, **{"jump.enabled": False, "motion.enabled": False})
    person = [Detection(x1=500, y1=500, x2=600, y2=600, score=0.9,
                        cls=0, name="pedestrian")]
    assert g._verify_hit(person, [500, 500, 100, 100]) is False
    car = [Detection(x1=500, y1=500, x2=600, y2=600, score=0.9, cls=3, name="car")]
    assert g._verify_hit(car, [500, 500, 100, 100]) is True
    # gate=presence: bỏ qua class -> person cũng tính là HIT
    gp = _guard(init_cls=3, gate="presence",
                **{"jump.enabled": False, "motion.enabled": False})
    assert gp._verify_hit(person, [500, 500, 100, 100]) is True
    print("[ok] test_guard_verify_class_gate_rejects_wrong_class")


from uav_pipeline.config import SotResultSinkCfg  # noqa: E402
from uav_pipeline.contracts import FrameContext, FrameMeta, Track  # noqa: E402
from uav_pipeline.sinks.sot_result import SotResultSink  # noqa: E402


def _ctx(idx, track=None):
    return FrameContext(meta=FrameMeta(idx=idx, ts=0.0, shape_hw=(1071, 1904)),
                        frame=None, tracks=[track] if track is not None else [])


def test_sot_result_sink_format():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "sot_result.txt")
    sink = SotResultSink(SotResultSinkCfg(enabled=True, path=path))
    sink.write(_ctx(0))                                  # acquire, chưa có box
    sink.write(_ctx(1))
    t = Track(track_id=1, bbox=[949.9, 576.5, 1043.4, 703.4],
              confidence=0.926, frame_id=2, cls=5, name="truck")
    sink.write(_ctx(2, t))                               # frame init
    sink.write(_ctx(3))                                  # LOST
    sink.close()

    lines = open(path).read().strip().split("\n")
    assert lines[0] == "1,-1,-1,-1,-1,-1,0", lines[0]
    assert lines[1] == "2,-1,-1,-1,-1,-1,0", lines[1]
    # frame 1-indexed (meta.idx+1); xyxy -> xywh; 2 chữ số box, 4 chữ số conf
    assert lines[2] == "3,949.90,576.50,93.50,126.90,0.9260,1", lines[2]
    assert lines[3] == "4,-1,-1,-1,-1,-1,0", lines[3]
    assert len(lines) == 4, "số dòng phải bằng số frame"
    print("[ok] test_sot_result_sink_format")


def test_sot_result_sink_disabled_writes_nothing():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "off.txt")
    sink = SotResultSink(SotResultSinkCfg(enabled=False, path=path))
    sink.write(_ctx(0))
    sink.close()
    assert not os.path.exists(path), "sink tắt thì không được tạo file"
    print("[ok] test_sot_result_sink_disabled_writes_nothing")


def test_sot_result_sink_flushes_every_frame():
    """Flush từng frame: crash giữa đường vẫn còn kết quả tới frame cuối."""
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "flush.txt")
    sink = SotResultSink(SotResultSinkCfg(enabled=True, path=path))
    sink.write(_ctx(0))
    assert open(path).read() == "1,-1,-1,-1,-1,-1,0\n", "chưa flush"
    sink.close()
    print("[ok] test_sot_result_sink_flushes_every_frame")


TESTS = [
    test_config_defaults_backward_compatible,
    test_config_mutual_exclusion,
    test_config_sot_value_checks,
    test_config_nested_guard_defaults,
    test_config_warnings,
    test_class_groups_gate_class,
    test_class_groups_gate_family,
    test_class_groups_presence_and_unknown,
    test_class_groups_missing_name_in_names,
    test_iou_xywh,
    test_guard_off_never_lost,
    test_guard_first_frame_never_jumps,
    test_guard_jump_detector_catches_real_drift,
    test_guard_jump_tolerates_real_gt_motion,
    test_guard_jump_px_scales_with_resolution,
    test_guard_motion_gate_cuts_back_to_streak_start,
    test_guard_motion_gate_recovers_without_lost,
    test_guard_motion_gate_does_not_poison_itself,
    test_guard_verify_needs_prior_confirmation,
    test_guard_verify_cuts_after_confirmation,
    test_guard_verify_class_gate_rejects_wrong_class,
    test_sot_result_sink_format,
    test_sot_result_sink_disabled_writes_nothing,
    test_sot_result_sink_flushes_every_frame,
]


def main():
    ap = argparse.ArgumentParser(description="SOT tests (no GPU)")
    ap.add_argument("--only", default="", help="chỉ chạy 1 test theo tên")
    args = ap.parse_args()
    tests = [t for t in TESTS if not args.only or t.__name__ == args.only]
    if not tests:
        sys.exit(f"[validate_sot] không có test tên '{args.only}'")
    for t in tests:
        t()
    print("-" * 60)
    print(f"[validate_sot] {len(tests)} TEST PASSED")
    print("-" * 60)


if __name__ == "__main__":
    main()
