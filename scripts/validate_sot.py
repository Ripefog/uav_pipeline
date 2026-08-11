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
from uav_pipeline.sot.class_groups import accepted_ids  # noqa: E402


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
