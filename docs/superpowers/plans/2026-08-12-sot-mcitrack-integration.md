# SOT (MCITrack) Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm module SOT (MCITrack) bật/tắt được vào `uav_pipeline`, nhận bbox từ config hoặc tự lấy box conf cao nhất của detector, loại trừ nhau với MOT hiện có.

**Architecture:** Package mới `uav_pipeline/sot/` gồm 4 file có biên rõ ràng: `mcitrack_wrapper.py` (file duy nhất biết MCITrack tồn tại), `class_groups.py` (dữ liệu), `guard.py` (hình học thuần, không import torch), `tracker.py` (state machine, bọc kết quả thành `Track` của `contracts.py`). `pipeline.py` chỉ rẽ **một** nhánh; nhờ SOT trả về đúng type `Track` nên `follow/`, HUD, telemetry chạy nguyên không sửa. Thêm `sinks/sot_result.py` (txt 7 cột) và `sinks/deferred.py` (hoãn ghi 1 frame để guard cắt lui được trong luồng stream).

**Tech Stack:** Python 3, numpy, opencv, PyYAML, torch 2.11+cu128 (chỉ trong `MCITrack/.venv`), onnxruntime (CPU), MCITrack (repo ngoài, import qua `sys.path`).

**Spec:** `docs/superpowers/specs/2026-08-12-sot-mcitrack-integration-design.md`

## Global Constraints

- **Env duy nhất chạy được**: `/home/anlnm/UAV/MCITrack/.venv`. Code root là `/home/anlnm/UAV` (vì `run_pipeline.py:14` lấy `../..`). Mọi lệnh chạy từ `/home/anlnm/UAV`.
- **Không sửa file nào trong `/home/anlnm/UAV/MCITrack/`** (kể cả `lib/`).
- **Không vendor code MCITrack** vào `_vendor/`.
- **Không đụng**: `track/*`, `detect/*`, `follow/*`, `ocr/*`, `sources/*`, `scripts/eval_mot_visdrone.py`, `_vendor/*`.
- **Repo không có pytest** (`import pytest` → ModuleNotFoundError) và không có `tests/`. Test theo đúng pattern sẵn có của repo: script chạy trực tiếp có `assert`, kiểu `scripts/validate_pipeline.py`. **Không** thêm pytest vào env.
- **`guard.py` và `class_groups.py` tuyệt đối không import torch / cv2 / MCITrack** — đó là lý do test chạy được không cần GPU.
- **Ngưỡng guard là số đã calibrate, không được đổi**: `jump.px=90` (GT thật max 44 px/frame, drift 337), `jump.area=2.5` (GT max 1.44, drift 10.73), `motion.iou=0.05` (drift 0.011 vs case đúng thấp nhất 0.094), `motion.k=2`, `K=3` (chuỗi verify-MISS oan dài nhất là 2), `ref_width=1904.0`.
- **Toạ độ**: `Detection`/`Track.bbox` là **xyxy** pixel ảnh gốc; MCITrack và file txt SOT dùng **xywh** góc trên-trái. Mọi chỗ chuyển đổi phải tường minh.
- **Frame id trong file kết quả là 1-indexed** (`meta.idx + 1`), khớp GT VisDrone và khớp `scripts/eval_mot_visdrone.py:39`.
- **Commit message ghi tên người dùng, KHÔNG có trailer `Co-Authored-By: Claude`.** Không commit gì ngoài các step `Commit` của plan này.
- Tiếng Việt không dấu trong code/comment là chấp nhận được (repo đang lẫn cả hai); giữ nguyên style file đang sửa.

## File Structure

**Tạo mới**

| File | Trách nhiệm |
|---|---|
| `sot/__init__.py` | export `SotTracker`, `LostGuard`, `GuardVerdict`, `iou_xywh`, `accepted_ids`. **KHÔNG** export `build_mcitrack_model`/`preflight`: import `mcitrack_wrapper` ở đây sẽ kéo cv2 + torch vào mọi lần `import uav_pipeline.sot`. `pipeline.py` import thẳng `from .sot.mcitrack_wrapper import build_mcitrack_model` bên trong nhánh `sot.enabled` (xem Task 7 Step 4) |
| `sot/mcitrack_wrapper.py` | file DUY NHẤT biết MCITrack: `sys.path`, 2 patch bắt buộc, `set_device`, BGR→RGB, reset `h_state`, preflight |
| `sot/class_groups.py` | bảng nhóm class theo TÊN + `accepted_ids()`. Dữ liệu, không logic |
| `sot/guard.py` | `iou_xywh`, `GuardVerdict`, `LostGuard` (3 tầng). Hình học thuần |
| `sot/tracker.py` | `SotTracker`: state machine acquire/tracking/lost, bọc thành `Track` |
| `sinks/sot_result.py` | `SotResultSink`: txt 7 cột |
| `sinks/deferred.py` | `DeferredSinkWriter`: hoãn ghi ≤ `motion.k-1` frame, cắt lui được |
| `configs/sot_mcitrack.yaml` | config chạy được ngay (onnx + yolox + CPU) |
| `scripts/validate_sot.py` | toàn bộ test không cần GPU |

**Sửa**

| File | Sửa gì |
|---|---|
| `config.py` | `+SotCfg/SotGuardCfg/SotJumpCfg/SotMotionCfg/SotResultSinkCfg`, `+TrackerCfg.enabled`, `Config.sot`, `SinksCfg.sot_result`, `Config.validate()`, `Config.warnings()` |
| `pipeline.py` | rẽ nhánh SOT/MOT, `tracker` có thể là None, `DeferredSinkWriter`, `extra_stats["sot"]`, ép `batch=1` |
| `sinks/__init__.py` | export `SotResultSink`, `DeferredSinkWriter` |
| `sinks/telemetry.py` | `+` field `"sot"` (jsonl) và cột `sot` (CSV) |
| `scripts/run_pipeline.py` | `--sot` / `--no-sot` / `--init-bbox`, gọi `validate()` + `warnings()` |
| `configs/default.yaml` | `+` block `sot:`, `tracker.enabled: false`, detector → onnx + yolox + `best_yoloxx.onnx` |
| `README.md` | mục SOT: cách bật, yêu cầu env, giới hạn guard |

---

### Task 1: Config schema + validation + test scaffold

**Files:**
- Modify: `config.py` (thêm dataclass sau `TrackerCfg` ở dòng 119-130; thêm `enabled` vào `TrackerCfg`; thêm `sot_result` vào `SinksCfg` dòng 203-207; thêm `sot` vào `Config` dòng 210-218; thêm 2 method vào `Config`)
- Create: `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `config.py` hiện có (`_build`, `TrackerCfg`, `SinksCfg`, `Config`)
- Produces:
  - `SotJumpCfg(enabled, px, area, ref_width)`
  - `SotMotionCfg(enabled, iou, k)`
  - `SotGuardCfg(enabled, gate, verify_every, K, iou_gate, jump, motion)`
  - `SotCfg(enabled, mcitrack_root, config, dataset_preset, device, init_bbox, init_classes, detect_every_frame, on_lost, guard)`
  - `SotResultSinkCfg(enabled, path)`
  - `TrackerCfg.enabled: bool = True`
  - `Config.sot: SotCfg`, `SinksCfg.sot_result: SotResultSinkCfg`
  - `Config.validate() -> List[str]` (rỗng = ok), `Config.warnings() -> List[str]`
  - `scripts/validate_sot.py` với `--only NAME`, mỗi test là `def test_xxx()`

- [ ] **Step 1: Viết test thất bại**

Tạo `scripts/validate_sot.py`:

```python
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


TESTS = [
    test_config_defaults_backward_compatible,
    test_config_mutual_exclusion,
    test_config_sot_value_checks,
    test_config_nested_guard_defaults,
    test_config_warnings,
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
```

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'sot'`

- [ ] **Step 3: Thêm dataclass vào `config.py`**

Chèn ngay sau `TrackerCfg` (kết thúc ở dòng 130):

```python
@dataclass
class SotJumpCfg:
    """Tầng 1 của guard: bắt cú nhảy bất khả thi về vật lý (chạy mỗi frame)."""
    enabled: bool = True
    px: float = 90.0          # ngưỡng dịch chuyển tâm/frame TẠI ref_width
    area: float = 2.5         # ngưỡng tỉ lệ diện tích/frame
    ref_width: float = 1904.0  # độ phân giải mà px được calibrate


@dataclass
class SotMotionCfg:
    """Tầng 2: bắt drift DẦN sang vật cùng class kề bên (dự đoán vận tốc không đổi)."""
    enabled: bool = True
    iou: float = 0.05
    k: int = 2                # số frame liên tiếp vi phạm -> LOST (cắt lui về đầu chuỗi)


@dataclass
class SotGuardCfg:
    enabled: bool = False     # OFF = hành vi MCITrack gốc, không bao giờ LOST
    gate: str = "class"       # class | family | presence
    verify_every: int = 10
    K: int = 3                # số lần verify MISS liên tiếp -> LOST
    iou_gate: float = 0.3
    jump: SotJumpCfg = field(default_factory=SotJumpCfg)
    motion: SotMotionCfg = field(default_factory=SotMotionCfg)


@dataclass
class SotCfg:
    enabled: bool = False
    mcitrack_root: str = "/home/anlnm/UAV/MCITrack"
    config: str = "mcitrack_l384"      # experiments/mcitrack/<config>.yaml
    dataset_preset: str = "uav"        # chọn preset UPT/UPH/INTER/MB
    device: str = "cuda:0"             # 'cuda' hoặc 'cuda:N'
    init_bbox: Optional[List[float]] = None   # [x,y,w,h]; None = detector tự lấy
    init_classes: List[int] = field(default_factory=list)  # [] = theo detector.classes_of_interest
    detect_every_frame: bool = False
    on_lost: str = "stop"              # stop | reacquire (chỉ có tác dụng khi guard bật)
    guard: SotGuardCfg = field(default_factory=SotGuardCfg)
```

Thêm `enabled` làm field đầu của `TrackerCfg`:

```python
@dataclass
class TrackerCfg:
    enabled: bool = True     # False = tắt MOT (dùng khi sot.enabled=true)
    high_conf: float = 0.4
```

Thêm sau `ControlLogSinkCfg` (dòng 198-200):

```python
@dataclass
class SotResultSinkCfg:
    enabled: bool = True
    path: str = "output/sot_result.txt"
```

`SinksCfg` thêm field:

```python
    sot_result: SotResultSinkCfg = field(default_factory=SotResultSinkCfg)
```

`Config` thêm field (sau `tracker`):

```python
    sot: SotCfg = field(default_factory=SotCfg)
```

- [ ] **Step 4: Thêm `validate()` và `warnings()` vào `Config`**

Chèn vào class `Config`, sau `from_yaml`:

```python
    def validate(self) -> List[str]:
        """Lỗi cấu hình chí tử. Rỗng = ok. Caller tự sys.exit.

        Chạy TRƯỚC khi load checkpoint MCITrack 1.44GB (~20s) để chết sớm, chết rõ.
        """
        errs: List[str] = []
        if self.sot.enabled and self.tracker.enabled:
            errs.append(
                "sot.enabled và tracker.enabled không được bật cùng lúc — SOT và MOT "
                "loại trừ nhau. Đặt tracker.enabled: false để chạy SOT, hoặc "
                "sot.enabled: false để chạy MOT.")
        if not self.sot.enabled and not self.tracker.enabled:
            errs.append("cả sot.enabled và tracker.enabled đều false — "
                        "không có tracker nào chạy.")
        if self.sot.enabled:
            if self.sot.on_lost not in ("stop", "reacquire"):
                errs.append(f"sot.on_lost phải là 'stop' hoặc 'reacquire', "
                            f"đang là '{self.sot.on_lost}'")
            g = self.sot.guard
            if g.gate not in ("class", "family", "presence"):
                errs.append(f"sot.guard.gate phải là class|family|presence, "
                            f"đang là '{g.gate}'")
            if g.enabled:
                if g.verify_every < 1:
                    errs.append("sot.guard.verify_every phải >= 1")
                if g.K < 1:
                    errs.append("sot.guard.K phải >= 1")
                if g.motion.k < 1:
                    errs.append("sot.guard.motion.k phải >= 1")
                if g.jump.ref_width <= 0:
                    errs.append("sot.guard.jump.ref_width phải > 0")
            if self.sot.init_bbox is not None:
                b = self.sot.init_bbox
                if len(b) != 4 or float(b[2]) <= 0 or float(b[3]) <= 0:
                    errs.append(f"sot.init_bbox phải là [x,y,w,h] với w>0 và h>0, "
                                f"đang là {b}")
        return errs

    def warnings(self) -> List[str]:
        """Cảnh báo không chết, chỉ in ra."""
        w: List[str] = []
        if self.sot.enabled:
            if self.source.loop:
                w.append("source.loop=true + SOT: video chạy vòng lại nên guard sẽ "
                         "tính chuyển động sai ở chỗ nối.")
            if self.sot.on_lost == "reacquire" and not self.sot.guard.enabled:
                w.append("sot.on_lost=reacquire nhưng sot.guard.enabled=false -> "
                         "MCITrack không bao giờ tuyên bố LOST nên reacquire vô hiệu.")
        return w
```

- [ ] **Step 5: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, in `[validate_sot] 5 TEST PASSED`

- [ ] **Step 6: Kiểm không hồi quy config cũ**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from uav_pipeline.config import Config
for f in ['local_onnx_batch16','jetson_trt','local_trt_fp16','local_trt_fp32','local_trt_int8','local_openvino_batch16']:
    c = Config.from_yaml(f'uav_pipeline/configs/{f}.yaml')
    print(f, 'tracker', c.tracker.enabled, 'sot', c.sot.enabled, 'errs', c.validate())
"
```
Expected: mọi file in `tracker True sot False errs []`

- [ ] **Step 7: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add config.py scripts/validate_sot.py
git commit -m "config: add SotCfg + tracker.enabled + validate()/warnings()"
```

---

### Task 2: `sot/class_groups.py` — nhóm class theo tên

**Files:**
- Create: `sot/__init__.py`, `sot/class_groups.py`
- Modify: `scripts/validate_sot.py`

**Interfaces:**
- Consumes: không (module độc lập, chỉ dùng `typing`)
- Produces: `accepted_ids(gate: str, init_cls: int, names: Dict[int, str]) -> Optional[List[int]]` — trả `None` nghĩa là bỏ qua điều kiện class. `GROUP_BY_NAME`, `FAMILY_BY_NAME` là `Dict[str, List[str]]`.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py` (import ở đầu file: `from uav_pipeline.sot.class_groups import accepted_ids`):

```python
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
```

Thêm 4 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_class_groups_gate_class
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sot'`

- [ ] **Step 3: Viết `sot/class_groups.py`**

```python
"""Nhóm class cho guard — map theo TÊN, không theo id.

Hardcode id class là SAI ÂM THẦM khi đổi weight sang taxonomy khác (id 3 của
VisDrone là 'car', của COCO là 'motorcycle'). Bảng này viết theo tên; id được
resolve lúc chạy qua detector.names (weights/names.yaml).
"""
from typing import Dict, List, Optional

# gate=class: nhóm hẹp. VisDrone tách người thành pedestrian (đang đi/đứng) và
# people (tư thế khác) — cùng một "person" nên gộp.
GROUP_BY_NAME: Dict[str, List[str]] = {
    "pedestrian": ["pedestrian", "people"],
    "people": ["pedestrian", "people"],
    "bicycle": ["bicycle"],
    "car": ["car"],
    "van": ["van"],
    "truck": ["truck"],
    "tricycle": ["tricycle"],
    "awning-tricycle": ["awning-tricycle"],
    "bus": ["bus"],
    "motor": ["motor"],
}

# gate=family: nới ra vì detector lẫn class nặng. Đo tại vị trí GT bus của
# uav0000305_00000_v: detector gán van 27 frame, bus 12, car 8, không có gì 17
# frame (miss rate 78%). Gate hẹp ở đây sẽ cắt oan.
FAMILY_BY_NAME: Dict[str, List[str]] = {
    "pedestrian": ["pedestrian", "people"],
    "people": ["pedestrian", "people"],
    "car": ["car", "van", "truck", "bus"],
    "van": ["car", "van", "truck", "bus"],
    "truck": ["car", "van", "truck", "bus"],
    "bus": ["car", "van", "truck", "bus"],
    "bicycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "tricycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "awning-tricycle": ["bicycle", "tricycle", "awning-tricycle", "motor"],
    "motor": ["bicycle", "tricycle", "awning-tricycle", "motor"],
}


def accepted_ids(gate: str, init_cls: int,
                 names: Dict[int, str]) -> Optional[List[int]]:
    """Id class được coi là 'cùng nhóm' với class lúc init.

    Trả None = BỎ QUA điều kiện class. Xảy ra khi gate='presence', hoặc khi tên
    class không có trong bảng — trả None (chứ không phải list rỗng) là cố ý: list
    rỗng sẽ chặn mọi detection và làm guard cắt oan mọi lúc.
    """
    if gate == "presence":
        return None
    table = FAMILY_BY_NAME if gate == "family" else GROUP_BY_NAME
    init_name = names.get(int(init_cls), "")
    group = table.get(init_name)
    if group is None:
        print(f"[sot/guard] class '{init_name}' (id {init_cls}) không có trong bảng "
              f"nhóm -> hạ gate '{gate}' xuống 'presence'")
        return None
    name_to_id = {v: k for k, v in names.items()}
    return sorted(name_to_id[n] for n in group if n in name_to_id)
```

Và `sot/__init__.py` — bản tạm, chỉ export những gì đã tồn tại ở task này (bản đầy đủ ở Task 7 Step 4, khi `guard.py` và `tracker.py` đã có):

```python
"""SOT (single object tracking) — MCITrack, bật/tắt bằng sot.enabled."""
from .class_groups import accepted_ids

__all__ = ["accepted_ids"]
```

- [ ] **Step 4: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 9 test

- [ ] **Step 5: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sot/__init__.py sot/class_groups.py scripts/validate_sot.py
git commit -m "sot: class groups for guard gate, resolved by name not id"
```

---

### Task 3: `sot/guard.py` — 3 tầng chặn drift

**Files:**
- Create: `sot/guard.py`
- Modify: `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `SotGuardCfg` (Task 1), `accepted_ids` (Task 2), `Detection` (`contracts.py`)
- Produces:
  - `iou_xywh(a, b) -> float`
  - `GuardVerdict(alive: bool, provisional: bool, lost_at: Optional[int], reason: str)`
  - `LostGuard(cfg: SotGuardCfg, frame_width: int, init_cls: int, names: Dict[int,str])` với `.step(frame_idx: int, box_xywh, detect_fn: Callable[[], List[Detection]]) -> GuardVerdict` và `.reset()`

`detect_fn` là **callable không tham số** trả `List[Detection]`; guard chỉ gọi nó ở frame verify. Nhờ vậy guard không import detector và test được không cần ONNX.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
from uav_pipeline.config import SotGuardCfg  # noqa: E402  (thêm cùng import Config)
from uav_pipeline.contracts import Detection  # noqa: E402
from uav_pipeline.sot.guard import LostGuard, iou_xywh  # noqa: E402


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
    # [650,600] chứ không phải [560,535]: dự đoán ở f5 là [530,500,100,100], mà
    # IoU([560,535],[530,500]) = 0.294 > motion.iou=0.05 -> không tính là vi phạm.
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
    assert lost is not None and lost.lost_at == 40, lost   # MISS f20,f30,f40 = K=3
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
```

Thêm 12 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_iou_xywh
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sot.guard'`

- [ ] **Step 3: Viết `sot/guard.py`**

```python
"""Ba tầng chặn drift cho SOT. Hình học thuần — KHÔNG import torch/cv2/MCITrack.

Vì sao phải có: MCITrack không có khái niệm "mất target". track() luôn trả 1 box
mỗi frame; clip_box(margin=10) (lib/utils/box_ops.py:101) còn ép box nằm trong ảnh
nên nó dính vào biên rồi bám sang vật khác. Confidence KHÔNG dùng làm ngưỡng được:
đo trên truck của uav0000339_00001_v, còn target conf min 0.466, mất target conf
max 0.713 — chồng lấn nặng.

Ba tầng bù nhau, đo trên 5 đối tượng: jump cắt car (f56) và motor (f191);
class-gate verify cắt truck (f170) và bus (f70); motion gate bắt drift dần mà cả
hai tầng kia đều mù.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..config import SotGuardCfg
from ..contracts import Detection
from .class_groups import accepted_ids


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU của 2 box dạng [x, y, w, h] (góc trên-trái)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / u) if u > 0 else 0.0


@dataclass
class GuardVerdict:
    alive: bool = True
    provisional: bool = False        # đang nghi ngờ, chưa đủ k -> hoãn ghi sink
    lost_at: Optional[int] = None    # frame ĐẦU chuỗi (cắt lui về đây)
    reason: str = ""


class LostGuard:
    def __init__(self, cfg: SotGuardCfg, frame_width: int, init_cls: int,
                 names: Dict[int, str]):
        self.cfg = cfg
        # jump.px calibrate ở 1904px. VisDrone trải 1344x756 -> 3840x2160 (2.86x):
        # cùng một chuyển động thật ra số px rất khác -> không scale là báo oan trên
        # video 4K và bỏ sót drift trên video nhỏ.
        self.jump_px = cfg.jump.px * (float(frame_width) / float(cfg.jump.ref_width))
        self.accept = accepted_ids(cfg.gate, init_cls, names)
        self.reset()

    def reset(self):
        # KHÔNG khởi tạo _prev bằng box của DETECTOR: box detector và box MCITrack là
        # 2 estimator khác nhau cho cùng một vật -> chênh scale ở frame đầu là bình
        # thường, không phải "nhảy". Trước khi sửa, uav0000117_02622_v_car LOST oan
        # ngay frame 2 (1/349 frame alive).
        self._prev: Optional[List[float]] = None
        self._motion_miss = 0
        self._streak_start: Optional[int] = None
        # Mốc dự đoán CHỈ cập nhật từ frame ĐÃ ĐƯỢC CHẤP NHẬN. Để frame giật làm mốc
        # thì metric tự đầu độc chính nó (uav0000086 person rank 4: f148 giật 35px,
        # f149-150 đã về đúng IoU_GT 0.66 nhưng IoU vs dự đoán vẫn 0.000).
        self._good_a: Optional[tuple] = None   # (box, frame_idx) cũ hơn
        self._good_b: Optional[tuple] = None   # (box, frame_idx) mới hơn
        self._verify_miss = 0
        # Chỉ kết án track mà detector ĐÃ TỪNG xác nhận. Vật 26x41 px của
        # uav0000339: detector 0 hit ở f10..f50 trong khi tracker vẫn đúng
        # (IoU 0.67-0.75) -> không latch thì cắt oan f30, và nâng K chỉ làm cắt muộn
        # hơn chứ không cứu được.
        self._verify_confirmed = False

    # ------------------------------------------------------------------ #
    def step(self, frame_idx: int, box_xywh: Sequence[float],
             detect_fn: Callable[[], List[Detection]]) -> GuardVerdict:
        """Chạy 3 tầng theo đúng thứ tự jump -> motion -> verify."""
        if not self.cfg.enabled:
            return GuardVerdict()
        box = [float(v) for v in box_xywh]

        v = self._jump(frame_idx, box)
        if v is not None:
            return v
        v = self._motion(frame_idx, box)
        if v is not None:
            return v

        # frame đã qua hết các gate -> mới được làm mốc
        self._prev = list(box)
        self._good_a, self._good_b = self._good_b, (list(box), frame_idx)
        self._motion_miss, self._streak_start = 0, None
        return self._verify(frame_idx, box, detect_fn)

    # ------------------------------------------------------------------ #
    def _jump(self, i: int, box: List[float]) -> Optional[GuardVerdict]:
        """Tầng 1, mỗi frame: cú nhảy bất khả thi về vật lý.

        Gate class MÙ HOÀN TOÀN khi drift sang vật CÙNG class — detection ở vị trí
        mới cũng là 'car' nên verify pass mọi lần. Tầng này không cần detector.
        """
        c = self.cfg.jump
        if not c.enabled or self._prev is None:
            return None
        p = self._prev
        dx = (box[0] + box[2] / 2.0) - (p[0] + p[2] / 2.0)
        dy = (box[1] + box[3] / 2.0) - (p[1] + p[3] / 2.0)
        disp = (dx * dx + dy * dy) ** 0.5
        pa, na = p[2] * p[3], box[2] * box[3]
        ratio = max(na / pa, pa / na) if pa > 0 and na > 0 else 1.0
        if disp > self.jump_px or ratio > c.area:
            # đang có frame bị hoãn (motion gate) thì cắt lui về đầu chuỗi luôn
            lost_at = self._streak_start if self._streak_start is not None else i
            return GuardVerdict(alive=False, lost_at=lost_at,
                                reason=f"jump {disp:.0f}px, area x{ratio:.1f}")
        return None

    def _motion(self, i: int, box: List[float]) -> Optional[GuardVerdict]:
        """Tầng 2, mỗi frame: drift DẦN sang vật cùng class kề bên.

        So box thật với dự đoán vận tốc không đổi từ CẶP FRAME TỐT cuối cùng
        (nhiều bước, không dùng frame đang bị nghi ngờ làm mốc).
        """
        c = self.cfg.motion
        if not c.enabled or self._good_a is None or self._good_b is None:
            return None
        (ba, fa), (bb, fb) = self._good_a, self._good_b
        dt = max(1, fb - fa)
        steps = i - fb
        pred = [bb[k] + (bb[k] - ba[k]) / dt * steps for k in range(4)]
        m_iou = iou_xywh(box, pred)
        if m_iou >= c.iou:
            return None
        self._motion_miss += 1
        if self._streak_start is None:
            self._streak_start = i
        # _prev VẪN cập nhật: tầng jump so với frame liền trước, không phải với mốc tốt
        self._prev = list(box)
        if self._motion_miss >= c.k:
            return GuardVerdict(
                alive=False, lost_at=self._streak_start,
                reason=(f"motion gate: {c.k} frame liên tiếp IoU với dự đoán < "
                        f"{c.iou} (cuối={m_iou:.3f}), cắt lui về frame "
                        f"{self._streak_start}"))
        return GuardVerdict(alive=True, provisional=True,
                            reason=(f"motion miss {self._motion_miss}/{c.k} "
                                    f"(IoU dự đoán {m_iou:.3f})"))

    def _verify(self, i: int, box: List[float],
                detect_fn: Callable[[], List[Detection]]) -> GuardVerdict:
        """Tầng 3, mỗi verify_every frame: chạy detector tại vị trí box tracker."""
        c = self.cfg
        if c.verify_every < 1 or i % c.verify_every != 0:
            return GuardVerdict()
        ok = self._verify_hit(detect_fn(), box)
        if ok:
            self._verify_confirmed = True
        self._verify_miss = 0 if ok else self._verify_miss + 1
        if self._verify_miss >= c.K and self._verify_confirmed:
            return GuardVerdict(alive=False, lost_at=i,
                                reason=(f"{c.K} verify MISS liên tiếp "
                                        f"(gate {c.gate})"))
        return GuardVerdict()

    def _verify_hit(self, dets: List[Detection], box: List[float]) -> bool:
        """HIT khi có detection thoả CẢ HAI: IoU > iou_gate và cùng nhóm class."""
        for d in dets:
            if self.accept is not None and int(d.cls) not in self.accept:
                continue
            if iou_xywh(box, [d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1]) > self.cfg.iou_gate:
                return True
        return False
```

- [ ] **Step 4: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 21 test

- [ ] **Step 5: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sot/guard.py scripts/validate_sot.py
git commit -m "sot: 3-tier lost guard (jump / motion / re-detect verify)"
```

---

### Task 4: `sinks/sot_result.py` — txt 7 cột

**Files:**
- Create: `sinks/sot_result.py`
- Modify: `sinks/__init__.py`, `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `SotResultSinkCfg` (Task 1), `Sink` (`sinks/base.py`), `FrameContext`/`Track` (`contracts.py`)
- Produces: `SotResultSink(cfg)` — `Sink` với `write(ctx)` / `close()`. Ghi `frame,x,y,w,h,conf,alive`; frame = `ctx.meta.idx + 1`; `ctx.tracks` rỗng → `-1,-1,-1,-1,-1,0`.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
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
```

Thêm 3 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_sot_result_sink_format
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sinks.sot_result'`

- [ ] **Step 3: Viết `sinks/sot_result.py`**

```python
"""SOT result sink — txt 7 cột, đúng format của repo MCITrack (tracking/*.py).

    frame,x,y,w,h,conf,alive     # x,y,w,h = xywh góc trên-trái, pixel ảnh gốc
    3,949.90,576.50,93.50,126.90,0.9260,1
    56,-1,-1,-1,-1,-1,0          # LOST hoặc chưa acquire

Hai chi tiết bắt buộc:
  * frame là 1-INDEXED (meta.idx + 1) — khớp GT VisDrone và khớp writer MOT
    (scripts/eval_mot_visdrone.py:39).
  * số dòng luôn bằng số frame (kể cả frame acquire/lost) nên align theo frame id
    không bao giờ lệch.

Ghi + flush TỪNG FRAME thay vì buffer tới close(): DeferredSinkWriter
(sinks/deferred.py) đã cắt lui TRƯỚC khi sink thấy ctx, nên sink không bao giờ
phải rút lại dòng đã ghi -> crash giữa đường vẫn còn kết quả.
"""
import os
from typing import Optional

from ..config import SotResultSinkCfg
from ..contracts import FrameContext
from .base import Sink


class SotResultSink(Sink):
    def __init__(self, cfg: SotResultSinkCfg):
        self.cfg = cfg
        self._f = None
        if cfg.enabled and cfg.path:
            os.makedirs(os.path.dirname(cfg.path) or ".", exist_ok=True)
            self._f = open(cfg.path, "w", encoding="utf-8")

    def write(self, ctx: FrameContext):
        if self._f is None:
            return
        fid = ctx.meta.idx + 1
        if ctx.tracks:
            t = ctx.tracks[0]
            x1, y1, x2, y2 = [float(v) for v in t.bbox]
            self._f.write(f"{fid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                          f"{t.confidence:.4f},1\n")
        else:
            self._f.write(f"{fid},-1,-1,-1,-1,-1,0\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
```

- [ ] **Step 4: Export trong `sinks/__init__.py`**

```python
"""Sinks — HUD video, telemetry JSONL/CSV, control command log, SOT result txt."""
from .base import Sink
from .control_log import ControlLogSink
from .sot_result import SotResultSink
from .telemetry import TelemetrySink
from .video import HUDAnnotatedSink

__all__ = ["Sink", "HUDAnnotatedSink", "TelemetrySink", "ControlLogSink",
           "SotResultSink"]
```

- [ ] **Step 5: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 24 test

- [ ] **Step 6: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sinks/sot_result.py sinks/__init__.py scripts/validate_sot.py
git commit -m "sinks: SOT result txt (frame,x,y,w,h,conf,alive)"
```

---

### Task 5: `sinks/deferred.py` — hoãn ghi để cắt lui được

**Files:**
- Create: `sinks/deferred.py`
- Modify: `sinks/__init__.py`, `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `FrameContext` (`contracts.py`)
- Produces: `DeferredSinkWriter(emit_fn: Callable[[FrameContext], None], max_hold: int)` với `write(ctx, provisional=False, retract_from=None)` và `close()`. `emit_fn` là hàm ghi thật (sẽ là `Pipeline._write_sinks_now`) — lớp này không biết sink là gì nên test được bằng list.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
from uav_pipeline.sinks.deferred import DeferredSinkWriter  # noqa: E402


def test_deferred_passthrough_when_not_provisional():
    out = []
    d = DeferredSinkWriter(out.append, max_hold=1)
    for i in range(3):
        d.write(_ctx(i, Track(1, [0, 0, 10, 10], 0.9, i)))
    d.close()
    assert [c.meta.idx for c in out] == [0, 1, 2], [c.meta.idx for c in out]
    assert all(c.tracks for c in out)
    print("[ok] test_deferred_passthrough_when_not_provisional")


def test_deferred_holds_provisional_then_flushes():
    """Chuỗi nghi ngờ bị ngắt -> frame đang giữ ghi ra NGUYÊN box."""
    out = []
    d = DeferredSinkWriter(out.append, max_hold=1)
    d.write(_ctx(0, Track(1, [0, 0, 10, 10], 0.9, 0)))
    d.write(_ctx(1, Track(1, [0, 0, 10, 10], 0.9, 1)), provisional=True)
    assert [c.meta.idx for c in out] == [0], "frame nghi ngờ phải được giữ lại"
    d.write(_ctx(2, Track(1, [0, 0, 10, 10], 0.9, 2)))
    assert [c.meta.idx for c in out] == [0, 1, 2]
    assert all(c.tracks for c in out), "flush phải giữ nguyên box"
    d.close()
    print("[ok] test_deferred_holds_provisional_then_flushes")


def test_deferred_retracts_held_frames_on_cut():
    """LOST cắt lui: frame đang giữ có frame_id >= lost_at bị xoá box.

    motion.k=2 -> max_hold=1 -> đúng 0 frame box sai lọt ra output.
    """
    out = []
    d = DeferredSinkWriter(out.append, max_hold=1)
    d.write(_ctx(0, Track(1, [0, 0, 10, 10], 0.9, 0)))          # f1 tốt
    d.write(_ctx(1, Track(1, [900, 900, 10, 10], 0.9, 1)), provisional=True)  # f2 nghi
    d.write(_ctx(2), retract_from=2)                             # f3: LOST, cắt về f2
    d.close()
    assert [c.meta.idx for c in out] == [0, 1, 2]
    assert out[0].tracks, "f1 tốt phải giữ box"
    assert out[1].tracks == [], "f2 phải bị xoá box (cắt lui)"
    assert out[2].tracks == []
    print("[ok] test_deferred_retracts_held_frames_on_cut")


def test_deferred_close_flushes_pending():
    """Hết video mà còn frame đang giữ (chuỗi chưa đủ k) -> coi là hợp lệ."""
    out = []
    d = DeferredSinkWriter(out.append, max_hold=1)
    d.write(_ctx(0, Track(1, [0, 0, 10, 10], 0.9, 0)), provisional=True)
    assert out == []
    d.close()
    assert [c.meta.idx for c in out] == [0] and out[0].tracks
    print("[ok] test_deferred_close_flushes_pending")


def test_deferred_max_hold_zero_is_passthrough():
    out = []
    d = DeferredSinkWriter(out.append, max_hold=0)
    d.write(_ctx(0, Track(1, [0, 0, 10, 10], 0.9, 0)), provisional=True)
    assert [c.meta.idx for c in out] == [0], "max_hold=0 -> ghi ngay"
    d.close()
    print("[ok] test_deferred_max_hold_zero_is_passthrough")
```

Thêm 5 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_deferred_passthrough_when_not_provisional
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sinks.deferred'`

- [ ] **Step 3: Viết `sinks/deferred.py`**

```python
"""DeferredSinkWriter — hoãn ghi vài frame để guard cắt lui được trong luồng stream.

Motion gate của SOT (sot/guard.py) có thể tuyên bố LOST rồi CẮT LUI về frame đầu
chuỗi nghi ngờ. Offline thì dễ: giữ frame trong buffer, chưa ghi video. Pipeline là
stream — frame đã ghi vào VideoWriter không lấy lại được.

Lớp này giữ tối đa (motion.k - 1) frame nghi ngờ. Khi guard cắt lui thì xoá tracks
của các frame đang giữ có frame_id >= lost_at rồi mới ghi -> 0 frame box sai lọt ra
output. Với motion.k=2 thì trễ đúng 1 frame (33ms @30fps).

Chỉ được dùng khi guard.enabled và guard.motion.enabled. Đường MOT và đường
guard-off không đi qua lớp này.
"""
from typing import Callable, List, Optional

from ..contracts import FrameContext


class DeferredSinkWriter:
    def __init__(self, emit_fn: Callable[[FrameContext], None], max_hold: int):
        self._emit = emit_fn
        self.max_hold = max(0, int(max_hold))
        self._held: List[FrameContext] = []

    def write(self, ctx: FrameContext, provisional: bool = False,
              retract_from: Optional[int] = None):
        """provisional=True -> giữ lại. retract_from=<frame 1-indexed> -> cắt lui."""
        if retract_from is not None:
            for h in self._held:
                if h.meta.idx + 1 >= retract_from:
                    h.tracks = []
            self._flush()
            self._emit(ctx)
            return
        if provisional and self.max_hold > 0:
            self._held.append(ctx)
            while len(self._held) > self.max_hold:
                self._emit(self._held.pop(0))
            return
        self._flush()
        self._emit(ctx)

    def _flush(self):
        while self._held:
            self._emit(self._held.pop(0))

    def close(self):
        # hết video mà còn frame đang giữ (chuỗi chưa đủ k) -> coi là hợp lệ
        self._flush()
```

- [ ] **Step 4: Export trong `sinks/__init__.py`**

```python
"""Sinks — HUD video, telemetry JSONL/CSV, control command log, SOT result txt."""
from .base import Sink
from .control_log import ControlLogSink
from .deferred import DeferredSinkWriter
from .sot_result import SotResultSink
from .telemetry import TelemetrySink
from .video import HUDAnnotatedSink

__all__ = ["Sink", "HUDAnnotatedSink", "TelemetrySink", "ControlLogSink",
           "SotResultSink", "DeferredSinkWriter"]
```

- [ ] **Step 5: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 29 test

- [ ] **Step 6: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sinks/deferred.py sinks/__init__.py scripts/validate_sot.py
git commit -m "sinks: DeferredSinkWriter so the SOT guard can cut back in a stream"
```

---

### Task 6: `sot/mcitrack_wrapper.py` — file duy nhất biết MCITrack

**Files:**
- Create: `sot/mcitrack_wrapper.py`
- Modify: `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `SotCfg` (Task 1)
- Produces:
  - `preflight(cfg: SotCfg) -> List[str]` — lỗi rẻ, chạy TRƯỚC khi load checkpoint 1.44 GB
  - `MCITrackModel(cfg)` với `initialize(frame_bgr, bbox_xywh) -> None` và `track(frame_bgr) -> Tuple[List[float], float]` (bbox xywh, score)
  - `build_mcitrack_model(cfg) -> MCITrackModel` (preflight rồi build)
  - `_device_index(device: str) -> int`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
from uav_pipeline.config import SotCfg  # noqa: E402
from uav_pipeline.sot.mcitrack_wrapper import _device_index, preflight  # noqa: E402


def test_device_index_parsing():
    assert _device_index("cuda") == 0
    assert _device_index("") == 0
    assert _device_index("cuda:1") == 1
    assert _device_index("CUDA:2") == 2
    try:
        _device_index("cpu")
    except ValueError as e:
        assert "cuda" in str(e), str(e)
    else:
        raise AssertionError("sot.device='cpu' phải báo lỗi: MCITrack hardcode .cuda()")
    print("[ok] test_device_index_parsing")


def test_preflight_bad_root():
    errs = preflight(SotCfg(mcitrack_root="/khong/ton/tai"))
    assert any("mcitrack_root" in e for e in errs), errs
    print("[ok] test_preflight_bad_root")


def test_preflight_good_root_reports_only_gpu_issues():
    """Root đúng -> không còn lỗi về path. Lỗi GPU (nếu có) là chuyện khác."""
    errs = preflight(SotCfg(mcitrack_root="/home/anlnm/UAV/MCITrack"))
    assert not any("mcitrack_root" in e for e in errs), errs
    print("[ok] test_preflight_good_root_reports_only_gpu_issues")


def test_preflight_bad_device_index():
    import torch
    n = torch.cuda.device_count()
    errs = preflight(SotCfg(mcitrack_root="/home/anlnm/UAV/MCITrack",
                            device=f"cuda:{n + 5}"))
    assert any("GPU" in e for e in errs), errs
    print("[ok] test_preflight_bad_device_index")
```

Thêm 4 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_device_index_parsing
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sot.mcitrack_wrapper'`

- [ ] **Step 3: Viết `sot/mcitrack_wrapper.py`**

```python
"""Bọc MCITrack (repo ngoài) thành API 3 hàm cho pipeline.

Đây là file DUY NHẤT trong uav_pipeline biết MCITrack tồn tại. Nó gánh toàn bộ
phần "bẩn" mà repo MCITrack bắt buộc phải có:

  1. sys.path.insert(mcitrack_root) — không copy code, không vendor.
  2. patch torch.load(weights_only=False): torch >= 2.6 mặc định True sẽ crash khi
     load checkpoint MCITrack.
  3. patch lib.models.mcitrack.encoder.is_main_process -> False: bỏ tải
     pretrained/fast_itpn_large_1600e_1k.pt (1.5GB) vì checkpoint bên dưới load
     strict=True nên ghi đè toàn bộ weight encoder anyway.
  4. torch.cuda.set_device(idx): lib/test/tracker/mcitrack.py hardcode .cuda()
     KHÔNG có index -> nó lấy CURRENT device.
  5. BGR -> RGB: pipeline giữ BGR (cv2), MCITrack cần RGB.
  6. reset h_state mỗi lần initialize: initialize() của repo KHÔNG reset (chỉ set
     trong __init__, lib/test/tracker/mcitrack.py:34) -> không tự reset thì hidden
     state Mamba của target cũ rò sang target mới khi reacquire.
"""
import os
import sys
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..config import SotCfg

_REQUIRED = os.path.join("lib", "test", "evaluation", "tracker.py")


def _device_index(device: str) -> int:
    d = (device or "").strip().lower()
    if d in ("", "cuda"):
        return 0
    if d.startswith("cuda:"):
        return int(d.split(":", 1)[1])
    raise ValueError(f"sot.device chỉ hỗ trợ 'cuda' hoặc 'cuda:N' (MCITrack hardcode "
                     f".cuda(), không chạy được CPU), đang là '{device}'")


def preflight(cfg: SotCfg) -> List[str]:
    """Mọi kiểm tra rẻ, chạy TRƯỚC khi load checkpoint 1.44GB (~20s)."""
    errs: List[str] = []
    root = os.path.abspath(cfg.mcitrack_root or "")
    if not os.path.isfile(os.path.join(root, _REQUIRED)):
        errs.append(f"không thấy {_REQUIRED} trong sot.mcitrack_root='{root}' — "
                    f"trỏ key sot.mcitrack_root vào clone của MCITrack")
    try:
        idx = _device_index(cfg.device)
    except ValueError as e:
        errs.append(str(e))
        return errs
    try:
        import torch
    except ImportError:
        errs.append("không import được torch (SOT cần torch cu128 — chạy bằng "
                    "/home/anlnm/UAV/MCITrack/.venv/bin/python)")
        return errs
    if not torch.cuda.is_available():
        errs.append("torch không thấy GPU nào (RTX 5080 là sm_120, cần "
                    "torch cu128; PyTorch 2.1.2+cu121 của install.sh không thấy GPU)")
    elif idx >= torch.cuda.device_count():
        errs.append(f"sot.device='{cfg.device}' nhưng máy chỉ thấy "
                    f"{torch.cuda.device_count()} GPU")
    return errs


class MCITrackModel:
    def __init__(self, cfg: SotCfg):
        import torch

        root = os.path.abspath(cfg.mcitrack_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        _orig_load = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)

        torch.load = _load   # patch toàn cục, giống script gốc của MCITrack

        import lib.models.mcitrack.encoder as enc_mod
        enc_mod.is_main_process = lambda: False

        torch.cuda.set_device(_device_index(cfg.device))

        from lib.test.evaluation.tracker import Tracker

        tr = Tracker("mcitrack", cfg.config, cfg.dataset_preset)
        params = tr.get_parameters()
        params.debug = 0
        if not os.path.isfile(params.checkpoint):
            raise FileNotFoundError(
                f"thiếu checkpoint MCITrack: {params.checkpoint}\n"
                f"(đường dẫn do {root}/lib/test/evaluation/local.py quyết định — "
                f"kiểm prj_dir/save_dir trong file đó)")
        self._t = tr.create_tracker(params)
        self._n_layers = self._t.cfg.MODEL.NECK.N_LAYERS
        print(f"[sot] MCITrack {cfg.config} preset={cfg.dataset_preset} "
              f"device={cfg.device} ckpt={os.path.basename(params.checkpoint)}")

    def initialize(self, frame_bgr: np.ndarray, bbox_xywh: Sequence[float]):
        self._t.h_state = [None] * self._n_layers
        self._t.initialize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                           {"init_bbox": [float(v) for v in bbox_xywh],
                            "seq_name": "pipeline"})

    def track(self, frame_bgr: np.ndarray) -> Tuple[List[float], float]:
        out = self._t.track(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # best_score là tensor -> phải float()
        return [float(v) for v in out["target_bbox"]], float(out["best_score"])


def build_mcitrack_model(cfg: SotCfg) -> MCITrackModel:
    errs = preflight(cfg)
    if errs:
        raise SystemExit("[sot] cấu hình SOT lỗi:\n  - " + "\n  - ".join(errs))
    return MCITrackModel(cfg)
```

- [ ] **Step 4: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 33 test

- [ ] **Step 5: Smoke test thật với GPU (load checkpoint + track 3 frame)**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -c "
import sys, glob, cv2; sys.path.insert(0,'.')
from uav_pipeline.config import SotCfg
from uav_pipeline.sot.mcitrack_wrapper import build_mcitrack_model
seq = 'uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v'
fs = sorted(glob.glob(seq + '/*.jpg'))[:3]
m = build_mcitrack_model(SotCfg(device='cuda:0'))
m.initialize(cv2.imread(fs[0]), [949.9, 576.5, 93.5, 126.9])
for f in fs[1:]:
    print(m.track(cv2.imread(f)))
"
```
Expected: in `[sot] MCITrack mcitrack_l384 preset=uav ...` rồi 2 dòng `([x, y, w, h], score)` với box gần `[949, 576, 93, 126]` và score > 0.4. Nếu GPU 0 bị chiếm thì đổi `device='cuda:1'`.

- [ ] **Step 6: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sot/mcitrack_wrapper.py scripts/validate_sot.py
git commit -m "sot: MCITrack wrapper (path bootstrap, required patches, h_state reset)"
```

---

### Task 7: `sot/tracker.py` — state machine

**Files:**
- Create: `sot/tracker.py`
- Modify: `sot/__init__.py`, `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `SotCfg` (Task 1), `LostGuard`/`GuardVerdict` (Task 3), `Detection`/`Track` (`contracts.py`)
- Produces: `SotTracker(cfg: SotCfg, names: Dict[int,str], model)` với
  - `update(frame, frame_idx: int, detect_fn: Callable[[], List[Detection]], prefetched: Optional[List[Detection]] = None) -> List[Track]`
  - thuộc tính đọc bởi `pipeline.py`: `mode` (`acquire|tracking|lost`), `status` (str cho HUD), `last_detections` (List[Detection]), `provisional` (bool), `retract_from` (Optional[int]), `needs_deferral` (bool), `lost_at`, `lost_reason`
  - `model` chỉ cần 2 method `initialize(frame_bgr, bbox_xywh)` và `track(frame_bgr) -> (bbox_xywh, score)` → test inject `FakeModel`, không cần GPU.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
import numpy as np  # noqa: E402  (thêm vào đầu file)
from uav_pipeline.sot.tracker import SotTracker  # noqa: E402


class FakeModel:
    """Model giả: trả trước một danh sách box, đếm số lần initialize."""

    def __init__(self, boxes, score=0.9):
        self.boxes = list(boxes)
        self.score = score
        self.n_init = 0
        self.init_boxes = []

    def initialize(self, frame, bbox_xywh):
        self.n_init += 1
        self.init_boxes.append(list(bbox_xywh))

    def track(self, frame):
        return list(self.boxes.pop(0)), self.score


def _frame(w=1904, h=1071):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(x1, y1, x2, y2, score, cls, name):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, cls=cls, name=name)


def test_sot_acquire_scans_until_first_box():
    """Frame đầu 0 box là bình thường (webcam/RTSP) -> quét tiếp, không crash."""
    cfg = SotCfg(enabled=True)
    empty_then = [[], [], [_det(100, 100, 140, 180, 0.8, 3, "car")]]
    calls = {"i": 0}

    def detect():
        d = empty_then[calls["i"]]
        calls["i"] += 1
        return d

    model = FakeModel([[100, 100, 40, 80]] * 5)
    sot = SotTracker(cfg, VISDRONE, model)
    assert sot.update(_frame(), 1, detect) == [] and sot.mode == "acquire"
    assert sot.update(_frame(), 2, detect) == [] and sot.mode == "acquire"
    tracks = sot.update(_frame(), 3, detect)
    assert len(tracks) == 1 and sot.mode == "tracking"
    assert model.n_init == 1
    # frame init xuất box + conf của DETECTOR
    t = tracks[0]
    assert [round(v, 2) for v in t.bbox] == [100.0, 100.0, 140.0, 180.0], t.bbox
    assert t.confidence == 0.8 and t.cls == 3 and t.name == "car"
    print("[ok] test_sot_acquire_scans_until_first_box")


def test_sot_picks_highest_conf_within_init_classes():
    cfg = SotCfg(enabled=True, init_classes=[0, 1])
    dets = [_det(0, 0, 100, 100, 0.95, 3, "car"),        # conf cao nhất nhưng sai class
            _det(10, 10, 30, 60, 0.60, 0, "pedestrian"),
            _det(50, 50, 70, 90, 0.80, 1, "people")]     # cao nhất TRONG nhóm
    model = FakeModel([[50, 50, 20, 40]])
    sot = SotTracker(cfg, VISDRONE, model)
    t = sot.update(_frame(), 1, lambda: dets)[0]
    assert t.cls == 1 and t.confidence == 0.80, (t.cls, t.confidence)
    assert model.init_boxes[0] == [50.0, 50.0, 20.0, 40.0], model.init_boxes
    print("[ok] test_sot_picks_highest_conf_within_init_classes")


def test_sot_init_bbox_skips_detector():
    """Có init_bbox thì KHÔNG gọi detector."""
    cfg = SotCfg(enabled=True, init_bbox=[10.0, 20.0, 30.0, 40.0])
    model = FakeModel([[10, 20, 30, 40]])

    def detect():
        raise AssertionError("không được gọi detector khi đã có init_bbox")

    sot = SotTracker(cfg, VISDRONE, model)
    t = sot.update(_frame(), 1, detect)[0]
    assert model.init_boxes[0] == [10.0, 20.0, 30.0, 40.0]
    assert [round(v, 2) for v in t.bbox] == [10.0, 20.0, 40.0, 60.0], t.bbox
    assert sot.last_detections == []
    print("[ok] test_sot_init_bbox_skips_detector")


def test_sot_tracking_updates_same_track_id_and_age():
    """Phải gọi Track.update() mỗi frame: follow/selector.py:14 lọc age == 0."""
    cfg = SotCfg(enabled=True)
    model = FakeModel([[110, 100, 40, 80], [120, 100, 40, 80]])
    sot = SotTracker(cfg, VISDRONE, model)
    sot.update(_frame(), 1, lambda: [_det(100, 100, 140, 180, 0.8, 3, "car")])
    t1 = sot.update(_frame(), 2, lambda: [])[0]
    t2 = sot.update(_frame(), 3, lambda: [])[0]
    assert t1.track_id == t2.track_id == 1
    assert t2.age == 0, "age phải là 0 nếu không follow sẽ bỏ qua target"
    assert len(t2.trajectory) == 3
    assert float(t2.velocity[0]) == 10.0, t2.velocity
    print("[ok] test_sot_tracking_updates_same_track_id_and_age")


def test_sot_lost_stop_emits_nothing_forever():
    cfg = SotCfg(enabled=True, on_lost="stop")
    cfg.guard.enabled = True
    cfg.guard.motion.enabled = False
    cfg.guard.verify_every = 0          # tắt tầng verify, chỉ dùng jump
    model = FakeModel([[110, 100, 40, 80], [900, 900, 40, 80],
                       [905, 905, 40, 80], [910, 910, 40, 80]])
    sot = SotTracker(cfg, VISDRONE, model)
    sot.update(_frame(), 1, lambda: [_det(100, 100, 140, 180, 0.8, 3, "car")])
    assert sot.update(_frame(), 2, lambda: [])
    assert sot.update(_frame(), 3, lambda: []) == [], "cú nhảy -> LOST"
    assert sot.mode == "lost" and sot.lost_at == 3, (sot.mode, sot.lost_at)
    assert sot.retract_from == 3
    assert sot.update(_frame(), 4, lambda: []) == []
    assert sot.retract_from is None, "retract_from chỉ set đúng 1 frame"
    assert "LOST" in sot.status
    print("[ok] test_sot_lost_stop_emits_nothing_forever")


def test_sot_reacquire_increments_id_and_reinitializes():
    cfg = SotCfg(enabled=True, on_lost="reacquire")
    cfg.guard.enabled = True
    cfg.guard.motion.enabled = False
    cfg.guard.verify_every = 0
    model = FakeModel([[110, 100, 40, 80], [900, 900, 40, 80], [210, 210, 40, 80]])
    car = [_det(100, 100, 140, 180, 0.8, 3, "car")]
    sot = SotTracker(cfg, VISDRONE, model)
    sot.update(_frame(), 1, lambda: car)
    sot.update(_frame(), 2, lambda: [])
    assert sot.update(_frame(), 3, lambda: []) == []
    assert sot.mode == "acquire", sot.mode
    t = sot.update(_frame(), 4, lambda: [_det(200, 200, 240, 280, 0.7, 3, "car")])[0]
    assert t.track_id == 2, "reacquire phải tăng track_id"
    assert model.n_init == 2, "phải initialize lại (reset h_state) chứ không track tiếp"
    print("[ok] test_sot_reacquire_increments_id_and_reinitializes")


def test_sot_needs_deferral_flag():
    on = SotCfg(enabled=True)
    on.guard.enabled = True
    assert SotTracker(on, VISDRONE, FakeModel([])).needs_deferral is True
    off = SotCfg(enabled=True)
    assert SotTracker(off, VISDRONE, FakeModel([])).needs_deferral is False
    no_motion = SotCfg(enabled=True)
    no_motion.guard.enabled = True
    no_motion.guard.motion.enabled = False
    assert SotTracker(no_motion, VISDRONE, FakeModel([])).needs_deferral is False
    print("[ok] test_sot_needs_deferral_flag")


def test_sot_prefetched_detections_not_detected_twice():
    """detect_every_frame=true: pipeline đã detect rồi -> không gọi lại."""
    cfg = SotCfg(enabled=True, detect_every_frame=True)
    dets = [_det(100, 100, 140, 180, 0.8, 3, "car")]

    def detect():
        raise AssertionError("đã có prefetched, không được detect lần 2")

    sot = SotTracker(cfg, VISDRONE, FakeModel([[100, 100, 40, 80]]))
    assert sot.update(_frame(), 1, detect, prefetched=dets)
    assert sot.last_detections == dets
    print("[ok] test_sot_prefetched_detections_not_detected_twice")
```

Thêm 8 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_sot_acquire_scans_until_first_box
```
Expected: FAIL — `ModuleNotFoundError: No module named 'uav_pipeline.sot.tracker'`

- [ ] **Step 3: Viết `sot/tracker.py`**

```python
"""SotTracker — state machine acquire -> tracking -> lost, bọc thành Track.

Cùng vai trò với DroneByteTracker.update() nên pipeline.py chỉ rẽ MỘT nhánh. Vì
kết quả là đúng type Track của contracts.py, follow/ + HUD + telemetry chạy nguyên
không phải sửa:
  * follow/selector.py:14 lọc age == 0 -> phải gọi Track.update() mỗi frame.
  * LOST -> trả [] -> follow._lost_step() tự coast rồi phanh.

Không biết MCITrack là gì: `model` chỉ cần 2 method initialize()/track() (xem
sot/mcitrack_wrapper.py), nhờ vậy test được bằng model giả, không cần GPU.
"""
from typing import Callable, Dict, List, Optional, Sequence

from ..config import SotCfg
from ..contracts import Detection, Track
from .guard import LostGuard


def _xywh_to_xyxy(b: Sequence[float]) -> List[float]:
    return [float(b[0]), float(b[1]), float(b[0]) + float(b[2]),
            float(b[1]) + float(b[3])]


class _DetectOnce:
    """Gọi detector nhiều nhất 1 lần cho 1 frame, giữ lại kết quả.

    Guard chỉ cần detection ở frame verify; acquire cần ở mọi frame. Lớp này để
    pipeline không detect 2 lần trên cùng frame, và để biết frame nào đã detect
    (ctx.detections).
    """

    def __init__(self, fn: Callable[[], List[Detection]],
                 prefetched: Optional[List[Detection]] = None):
        self._fn = fn
        self.result: Optional[List[Detection]] = prefetched

    def __call__(self) -> List[Detection]:
        if self.result is None:
            self.result = self._fn()
        return self.result


def _pick_init(dets: List[Detection],
               init_classes: Sequence[int]) -> Optional[Detection]:
    """Box conf cao nhất trong init_classes ([] = mọi class)."""
    cand = [d for d in dets if not init_classes or int(d.cls) in init_classes]
    if not cand:
        return None
    return max(cand, key=lambda d: d.score)


class SotTracker:
    def __init__(self, cfg: SotCfg, names: Dict[int, str], model):
        self.cfg = cfg
        self.names = names
        self.model = model
        self.mode = "acquire"          # acquire | tracking | lost
        self.status = "acquire"        # dòng cho HUD/telemetry
        self.last_detections: List[Detection] = []
        self.provisional = False       # frame đang bị motion gate nghi ngờ
        self.retract_from: Optional[int] = None   # set đúng 1 frame khi LOST
        self.lost_at: Optional[int] = None
        self.lost_reason = ""
        self.trk: Optional[Track] = None
        self.guard: Optional[LostGuard] = None
        self._next_id = 1
        self._acquire_frames = 0
        self._init_cls = -1

    @property
    def needs_deferral(self) -> bool:
        """True nếu guard có thể cắt lui -> pipeline phải hoãn ghi sink."""
        return bool(self.cfg.guard.enabled and self.cfg.guard.motion.enabled)

    # ------------------------------------------------------------------ #
    def update(self, frame, frame_idx: int,
               detect_fn: Callable[[], List[Detection]],
               prefetched: Optional[List[Detection]] = None) -> List[Track]:
        det = _DetectOnce(detect_fn, prefetched)
        self.provisional = False
        self.retract_from = None
        try:
            if self.mode == "acquire":
                return self._acquire(frame, frame_idx, det)
            if self.mode == "tracking":
                return self._tracking(frame, frame_idx, det)
            return []
        finally:
            self.last_detections = det.result or []

    # ------------------------------------------------------------------ #
    def _acquire(self, frame, frame_idx: int, det: _DetectOnce) -> List[Track]:
        # init_bbox chỉ dùng cho lần bám ĐẦU TIÊN: sau khi LOST nó đã lạc hậu.
        if self.cfg.init_bbox is not None and self._next_id == 1:
            box = [float(v) for v in self.cfg.init_bbox]
            score, cls, name = 1.0, -1, "init_bbox"
        else:
            self._acquire_frames += 1
            pick = _pick_init(det(), self.cfg.init_classes)
            if pick is None:
                self.status = f"acquire ({self._acquire_frames} frame, 0 box)"
                return []
            box = [pick.x1, pick.y1, pick.w, pick.h]
            score, cls, name = float(pick.score), int(pick.cls), pick.name

        h, w = frame.shape[:2]
        self.model.initialize(frame, box)
        self._init_cls = cls
        self.guard = LostGuard(self.cfg.guard, w, cls, self.names)
        self.trk = Track(track_id=self._next_id, bbox=_xywh_to_xyxy(box),
                         confidence=score, frame_id=frame_idx, cls=cls, name=name)
        self._next_id += 1
        self.mode = "tracking"
        self.status = f"tracking #{self.trk.track_id} init conf {score:.2f}"
        return [self.trk]

    def _tracking(self, frame, frame_idx: int, det: _DetectOnce) -> List[Track]:
        box, score = self.model.track(frame)
        v = self.guard.step(frame_idx, box, det)
        if not v.alive:
            self.lost_at, self.lost_reason = v.lost_at, v.reason
            self.retract_from = v.lost_at
            self.status = f"LOST @{v.lost_at} {v.reason}"
            self.trk = None
            if self.cfg.on_lost == "reacquire":
                self.mode = "acquire"
                self.guard = None
                self._acquire_frames = 0
            else:
                self.mode = "lost"
            return []
        self.trk.update(_xywh_to_xyxy(box), float(score), frame_idx,
                        cls=self._init_cls)
        self.provisional = v.provisional
        self.status = (f"tracking #{self.trk.track_id} conf {score:.2f}"
                       + (f"  [{v.reason}]" if v.provisional else ""))
        return [self.trk]
```

- [ ] **Step 4: Hoàn thiện `sot/__init__.py`**

```python
"""SOT (single object tracking) — MCITrack, bật/tắt bằng sot.enabled.

Loại trừ nhau với MOT (track/). Xem
docs/superpowers/specs/2026-08-12-sot-mcitrack-integration-design.md
"""
from .class_groups import accepted_ids
from .guard import GuardVerdict, LostGuard, iou_xywh
from .tracker import SotTracker

__all__ = ["SotTracker", "LostGuard", "GuardVerdict", "iou_xywh", "accepted_ids"]
```

⚠️ **Không** import `mcitrack_wrapper` ở đây: nó `import cv2` và sẽ kéo torch khi build. `pipeline.py` import trực tiếp `from .sot.mcitrack_wrapper import build_mcitrack_model` bên trong nhánh `sot.enabled`.

- [ ] **Step 5: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 41 test

- [ ] **Step 6: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add sot/tracker.py sot/__init__.py scripts/validate_sot.py
git commit -m "sot: SotTracker state machine (acquire/tracking/lost) wrapped as Track"
```

---

### Task 8: Nối vào `pipeline.py` + telemetry

**Files:**
- Modify: `pipeline.py` (imports; `__init__` dòng 27-80; `_write_sinks` dòng 131-138; `process_frame` dòng 141-178; `_extra_stats` dòng 217-223; `close` dòng 247-265)
- Modify: `sinks/telemetry.py` (dòng 38-40 header CSV, 46-64 record, 68-77 row)
- Modify: `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `SotTracker` (Task 7), `build_mcitrack_model` (Task 6), `SotResultSink` (Task 4), `DeferredSinkWriter` (Task 5), `Config.sot` (Task 1)
- Produces: `Pipeline.sot: Optional[SotTracker]`, `Pipeline.tracker: Optional[DroneByteTracker]`, `Pipeline._write_sinks_now(ctx)`, `ctx.extra_stats["sot"]`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
def test_pipeline_tracker_optional_when_sot_enabled():
    """Kiểm bằng đọc source: 3 chỗ dùng self.tracker phải chịu được None.

    Không khởi tạo Pipeline thật (cần model + video); test này chặn đúng lỗi
    AttributeError: 'NoneType' has no attribute 'cmc'/'frame_count'/'get_stats'.
    """
    src = open(os.path.join(_CODE_ROOT, "uav_pipeline", "pipeline.py")).read()
    assert "self.tracker = DroneByteTracker(config.tracker) if config.tracker.enabled" in src, \
        "tracker phải là None khi tracker.enabled=false"
    assert "if self.tracker is not None and self.tracker.cmc is not None" in src, \
        "_extra_stats phải guard self.tracker None"
    assert "if self.tracker is not None and self.cfg.tracker.interpolate_max_gap" in src, \
        "close() phải guard self.tracker None"
    assert '"sot"' in src, "extra_stats phải có key sot cho HUD"
    print("[ok] test_pipeline_tracker_optional_when_sot_enabled")


def test_telemetry_has_sot_field():
    import json
    import tempfile
    from uav_pipeline.config import TelemetrySinkCfg
    from uav_pipeline.sinks.telemetry import TelemetrySink
    d = tempfile.mkdtemp()
    sink = TelemetrySink(TelemetrySinkCfg(
        enabled=True, path=os.path.join(d, "t.jsonl"),
        csv_summary=os.path.join(d, "t.csv")))
    ctx = _ctx(0)
    ctx.extra_stats = {"sot": "tracking #1 conf 0.83"}
    sink.write(ctx)
    sink.close()
    rec = json.loads(open(os.path.join(d, "t.jsonl")).readline())
    assert rec["sot"] == "tracking #1 conf 0.83", rec
    head = open(os.path.join(d, "t.csv")).readline().strip().split(",")
    assert head[-1] == "sot", head
    print("[ok] test_telemetry_has_sot_field")
```

Thêm 2 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_telemetry_has_sot_field
```
Expected: FAIL — `KeyError: 'sot'`

- [ ] **Step 3: Sửa `sinks/telemetry.py`**

Header CSV (dòng 38-40) thêm `"sot"` vào cuối:

```python
                self._csv_writer.writerow(
                    ["frame", "fps", "n_det", "n_trk", "mode", "target_id",
                     "yaw", "pitch", "forward", "vertical", "target_lost", "sot"])
```

Record jsonl: thêm 1 dòng ngay sau `"motion"` (dòng 54):

```python
            "sot": ctx.extra_stats.get("sot", ""),
```

Row CSV: thêm phần tử cuối (sau `int(cmd.target_lost)`):

```python
                ctx.extra_stats.get("sot", ""),
```

- [ ] **Step 4: Sửa `pipeline.py` — imports và `__init__`**

Thêm import (cạnh dòng 21):

```python
from .sinks import (ControlLogSink, DeferredSinkWriter, HUDAnnotatedSink, Sink,
                    SotResultSink, TelemetrySink)
```

Trong `__init__`, thay dòng 57 (`self.tracker = DroneByteTracker(config.tracker)`) bằng:

```python
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
```

Trong khối sinks (dòng 62-69), thêm sau `ControlLogSink`:

```python
        if config.sot.enabled and config.sinks.sot_result.enabled:
            self.sinks.append(SotResultSink(config.sinks.sot_result))
```

Cuối `__init__` (sau khối `# ---- state ----`), thêm:

```python
        # Guard có thể tuyên bố LOST rồi cắt LUI về frame đầu chuỗi. Stream thì
        # frame đã ghi không lấy lại được -> hoãn ghi motion.k-1 frame.
        self._deferred = None
        if self.sot is not None and self.sot.needs_deferral:
            hold = max(0, config.sot.guard.motion.k - 1)
            self._deferred = DeferredSinkWriter(self._write_sinks_now, max_hold=hold)
            print(f"[pipeline] SOT guard motion=on -> hoãn ghi sink {hold} frame "
                  f"(để cắt lui không lọt box sai)")
```

- [ ] **Step 5: Sửa `pipeline.py` — `_write_sinks` tách 2 lớp**

Đổi `_write_sinks` (dòng 131-138) thành:

```python
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
```

- [ ] **Step 6: Sửa `pipeline.py` — `process_frame`**

Thay khối detect + track (dòng 150-159) bằng:

```python
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
```

⚠️ Dòng `plates = self.detector.detect_plates(...)` gốc (dòng 155) đã được chuyển vào nhánh `else` — xoá dòng gốc, không để lặp.

- [ ] **Step 7: Sửa `pipeline.py` — `_extra_stats` và `close`**

`_extra_stats`:

```python
    def _extra_stats(self):
        stats = {}
        if self.tracker is not None and self.tracker.cmc is not None:
            sev = float(self.tracker.last_motion.get("severity", 0.0))
            bar = ("#" * int(sev * 10)).ljust(10, "-")
            stats["motion"] = f"{bar} {sev:.2f}"
        if self.sot is not None:
            stats["sot"] = self.sot.status
        return stats
```

`close()` — 3 chỗ:

```python
    def close(self):
        # flush frame đang bị hoãn trước khi đóng sink
        if self._deferred is not None:
            self._deferred.close()

        # post-processing interpolation (chỉ có ở đường MOT)
        if (self.tracker is not None and self.cfg.tracker.interpolate_max_gap > 0
                and self.tracker.frame_count > 0):
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
                     if self.sot.lost_at else ""))
        print("-" * 60)
```

Và dòng in mở đầu `run()` (dòng 87-90) thêm mode:

```python
        print(f"[pipeline] source={self.cfg.source.type} "
              f"backend={self.cfg.detector.backend} "
              f"track={'SOT' if self.sot is not None else 'MOT'} "
              f"ocr={'on' if self.plate_ocr else 'off'} "
              f"controller={self.cfg.controller.backend}")
```

- [ ] **Step 8: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 43 test

- [ ] **Step 9: Kiểm đường MOT vẫn import và chạy được**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_pipeline.py
```
Expected: `[validate] ALL CORE TESTS PASSED`

- [ ] **Step 10: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add pipeline.py sinks/telemetry.py scripts/validate_sot.py
git commit -m "pipeline: branch SOT/MOT, deferred sink writes, sot in telemetry"
```

---

### Task 9: Config YAML + CLI + README

**Files:**
- Create: `configs/sot_mcitrack.yaml`
- Modify: `configs/default.yaml`, `scripts/run_pipeline.py`, `README.md`, `scripts/validate_sot.py`

**Interfaces:**
- Consumes: `Config.validate()` / `Config.warnings()` (Task 1)
- Produces: `configs/sot_mcitrack.yaml` chạy được ngay; `run_pipeline.py` có `--sot`, `--no-sot`, `--init-bbox`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `scripts/validate_sot.py`:

```python
def test_sot_config_yaml_loads_and_validates():
    p = os.path.join(_CODE_ROOT, "uav_pipeline", "configs", "sot_mcitrack.yaml")
    cfg = Config.from_yaml(p)
    assert cfg.sot.enabled is True and cfg.tracker.enabled is False
    assert cfg.validate() == [], cfg.validate()
    # ORT trong MCITrack/.venv chỉ có CPUExecutionProvider -> phải khai cpu tường
    # minh, nếu không numerics detector có thể lệch so với baseline đã đo.
    assert cfg.detector.device == "cpu", cfg.detector.device
    assert cfg.detector.backend == "onnx" and cfg.detector.preprocess == "yolox"
    assert cfg.detector.primary.onnx.endswith("best_yoloxx.onnx")
    assert cfg.sot.guard.enabled is False, "guard mặc định OFF"
    print("[ok] test_sot_config_yaml_loads_and_validates")


def test_default_yaml_is_sot():
    p = os.path.join(_CODE_ROOT, "uav_pipeline", "configs", "default.yaml")
    cfg = Config.from_yaml(p)
    assert cfg.sot.enabled is True and cfg.tracker.enabled is False
    assert cfg.validate() == [], cfg.validate()
    # openvino không có trong MCITrack/.venv -> default.yaml bật SOT thì phải onnx
    assert cfg.detector.backend == "onnx", cfg.detector.backend
    print("[ok] test_default_yaml_is_sot")


def test_mot_configs_untouched():
    for f in ["local_onnx_batch16", "jetson_trt", "local_trt_fp16",
              "local_trt_fp32", "local_trt_int8", "local_openvino_batch16"]:
        p = os.path.join(_CODE_ROOT, "uav_pipeline", "configs", f + ".yaml")
        cfg = Config.from_yaml(p)
        assert cfg.tracker.enabled is True, f
        assert cfg.sot.enabled is False, f
        assert cfg.validate() == [], (f, cfg.validate())
    print("[ok] test_mot_configs_untouched")
```

Thêm 3 test vào `TESTS`.

- [ ] **Step 2: Chạy test để chắc chắn nó fail**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py --only test_sot_config_yaml_loads_and_validates
```
Expected: FAIL — `FileNotFoundError: .../configs/sot_mcitrack.yaml`

- [ ] **Step 3: Tạo `configs/sot_mcitrack.yaml`**

```yaml
# SOT (MCITrack) — chạy được ngay trong /home/anlnm/UAV/MCITrack/.venv.
#
#   cd /home/anlnm/UAV
#   MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
#       --config uav_pipeline/configs/sot_mcitrack.yaml \
#       --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
#       --source-type image_dir
source:
  type: image_dir
  path: ""
  loop: false          # loop + SOT = guard tính chuyển động sai ở chỗ nối
  fps: 30

detector:
  backend: onnx
  # CPU tường minh: onnxruntime trong MCITrack/.venv chỉ có CPUExecutionProvider
  # (get_available_providers() -> Azure, CPU). Khai 'cuda' sẽ im lặng fallback về
  # CPU và làm khó truy vết khi so numerics với baseline.
  device: cpu
  preprocess: yolox    # YOLOX-X: BGR, KHÔNG /255, pad 114 góc trên-trái
  batch: 1             # SOT không dùng batch (detector chỉ chạy ở frame acquire/verify)
  imgsz: [736, 1280]   # export fixed 736x1280
  conf: 0.25
  iou: 0.45
  fp16: false
  classes_of_interest: []
  primary:
    onnx: weights/best_yoloxx.onnx
    names_yaml: weights/names.yaml

ocr:
  enabled: false

tracker:
  enabled: false       # MOT tắt: SOT và MOT loại trừ nhau

sot:
  enabled: true
  mcitrack_root: /home/anlnm/UAV/MCITrack
  config: mcitrack_l384
  dataset_preset: uav
  device: cuda:0       # đổi cuda:1 nếu GPU 0 đang bị chiếm
  init_bbox: null      # [x,y,w,h]; null = detector lấy box conf cao nhất
  init_classes: []     # vd [0,1] chỉ person; [] = mọi class
  detect_every_frame: false
  on_lost: stop        # stop | reacquire (chỉ có tác dụng khi guard bật)
  guard:
    enabled: false     # OFF = hành vi MCITrack gốc, không bao giờ LOST
    gate: class        # class | family | presence
    verify_every: 10
    K: 3
    iou_gate: 0.3
    jump:   {enabled: true, px: 90.0, area: 2.5, ref_width: 1904.0}
    motion: {enabled: true, iou: 0.05, k: 2}

follow:
  enabled: true
  default_policy: highest_score_area
  preferred_classes: []      # SOT chỉ có 1 track; lọc theo class ở đây là tự bắn chân
  target_area_norm: 0.12
  deadzone_px: 12.0
  lost_recovery_frames: 15
  pid:
    yaw:     {kp: 0.08, ki: 0.0, kd: 0.02, out_limit: 60.0}
    pitch:   {kp: 0.06, ki: 0.0, kd: 0.02, out_limit: 60.0}
    forward: {kp: 0.40, ki: 0.0, kd: 0.0, out_limit: 3.0}
    vertical: {kp: 0.0,  ki: 0.0, kd: 0.0, out_limit: 2.0}

controller:
  backend: mock

sinks:
  video:      {enabled: true, path: output/sot_pipeline.mp4, codec: mp4v, fps: 30, draw: true}
  telemetry:  {enabled: true, path: output/sot_telemetry.jsonl, csv_summary: output/sot_telemetry_summary.csv}
  control_log: {enabled: true, path: output/sot_commands.jsonl}
  sot_result: {enabled: true, path: output/sot_result.txt}
```

- [ ] **Step 4: Sửa `configs/default.yaml`**

Đổi khối `detector` (dòng 20-32) thành:

```yaml
detector:
  # SOT bật mặc định -> phải chạy trong MCITrack/.venv, mà env đó KHÔNG có
  # openvino. Vì vậy default là onnx + YOLOX-X. Các config MOT (local_onnx_batch16,
  # jetson_trt, local_trt_*) vẫn giữ backend riêng của chúng.
  backend: onnx          # torch | onnx | openvino | trt
  device: cpu            # ORT trong MCITrack/.venv chỉ có CPUExecutionProvider
  preprocess: yolox      # ultralytics (RGB/255/center-pad) | yolox (BGR/0-255/top-left-pad)
  imgsz: [736, 1280]
  batch: 1
  conf: 0.25
  iou: 0.45
  fp16: false            # torch cuda half; trt fp16
  classes_of_interest: []  # [] = all classes; e.g. [0,3,5]
  primary:
    pt:       weights/best_yolov26n_qat_int8_static.pt
    onnx:     weights/best_yoloxx.onnx
    openvino: weights/best_yolov26n_qat_int8_static.xml
    trt:      weights/best_yolov26n_qat_int8_static.engine
    names_yaml: weights/names.yaml
```

Thêm `enabled: false` làm dòng đầu của khối `tracker:` (dòng 49):

```yaml
tracker:                 # faithful port of pratap424/visdrone_mot thresholds
  enabled: false         # SOT và MOT loại trừ nhau; default.yaml chạy SOT
  high_conf: 0.4
```

Chèn khối `sot:` ngay sau khối `tracker:` (trước `follow:`):

```yaml
sot:                     # single object tracking (MCITrack) — loại trừ với tracker
  enabled: true
  mcitrack_root: /home/anlnm/UAV/MCITrack   # sys.path.insert, không copy code
  config: mcitrack_l384                     # experiments/mcitrack/<config>.yaml
  dataset_preset: uav                       # chọn preset UPT/UPH/INTER/MB
  device: cuda:0                            # 'cuda' | 'cuda:N' (MCITrack cần GPU)
  init_bbox: null         # [x,y,w,h] pixel ảnh gốc; null = detector lấy conf cao nhất
  init_classes: []        # [] = theo detector.classes_of_interest
  detect_every_frame: false  # false = detector chỉ chạy ở frame acquire/verify
  on_lost: stop           # stop | reacquire (chỉ có tác dụng khi guard bật)
  guard:                  # ngưỡng đã calibrate trên VisDrone — đừng đổi tuỳ tiện
    enabled: false        # OFF = hành vi MCITrack gốc, không bao giờ LOST
    gate: class           # class | family | presence
    verify_every: 10
    K: 3                  # chuỗi verify-MISS oan dài nhất đo được là 2
    iou_gate: 0.3
    jump:   {enabled: true, px: 90.0, area: 2.5, ref_width: 1904.0}
    motion: {enabled: true, iou: 0.05, k: 2}
```

Trong khối `sinks:`, thêm sau `control_log`:

```yaml
  sot_result:
    enabled: true
    path: output/sot_result.txt
```

- [ ] **Step 5: Sửa `scripts/run_pipeline.py`**

Thêm 3 arg sau `--no-follow` (dòng 29):

```python
    ap.add_argument("--sot", action="store_true",
                    help="bật SOT (MCITrack) và tắt MOT")
    ap.add_argument("--no-sot", action="store_true",
                    help="tắt SOT và bật MOT")
    ap.add_argument("--init-bbox", default="",
                    help="'x,y,w,h' pixel ảnh gốc cho SOT; rỗng = detector tự lấy "
                         "box conf cao nhất")
```

Thêm xử lý sau `if args.no_follow:` (dòng 41-42), trước kiểm `source.path`:

```python
    if args.sot and args.no_sot:
        sys.exit("[run_pipeline] --sot và --no-sot loại trừ nhau")
    if args.sot:
        cfg.sot.enabled, cfg.tracker.enabled = True, False
    if args.no_sot:
        cfg.sot.enabled, cfg.tracker.enabled = False, True
    if args.init_bbox:
        try:
            cfg.sot.init_bbox = [float(v) for v in args.init_bbox.split(",")]
        except ValueError:
            sys.exit(f"[run_pipeline] --init-bbox phải là 'x,y,w,h', "
                     f"đang là '{args.init_bbox}'")

    errs = cfg.validate()
    if errs:
        sys.exit("[run_pipeline] config lỗi:\n  - " + "\n  - ".join(errs))
    for w in cfg.warnings():
        print(f"[run_pipeline] CẢNH BÁO: {w}")
```

- [ ] **Step 6: Chạy test để chắc chắn nó pass**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
```
Expected: PASS, 46 test

- [ ] **Step 7: Kiểm CLI flip + validation không cần GPU**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml --no-sot --source /tmp/khong-co.mp4 2>&1 | head -3
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml --init-bbox "1,2,0,4" \
    --source /tmp/khong-co.mp4 2>&1 | head -3
```
Expected: lệnh 1 không báo lỗi config (đi tới bước mở source); lệnh 2 in `config lỗi: - sot.init_bbox phải là [x,y,w,h] với w>0 và h>0`

- [ ] **Step 8: Thêm mục SOT vào `README.md`**

Chèn một mục mới (sau phần nói về tracker/MOT):

```markdown
## SOT — single object tracking (MCITrack)

Bám **một** đối tượng bằng MCITrack (AAAI'25) thay cho MOT. Bật/tắt bằng
`sot.enabled`; **SOT và MOT loại trừ nhau** — bật cả hai thì pipeline báo lỗi và
dừng.

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
    --source-type image_dir
# chỉ định bbox đầu bằng tay
    ... --init-bbox "949.9,576.5,93.5,126.9"
# đổi qua MOT trong cùng một lệnh
    ... --no-sot
```

**Yêu cầu môi trường** (không thay được):
- Phải chạy bằng `/home/anlnm/UAV/MCITrack/.venv/bin/python`: cần torch cu128
  (RTX 5080 là sm_120; torch cu121 không thấy GPU). Env này **không có openvino**,
  nên `detector.backend` phải là `onnx` hoặc `torch`.
- Cần clone MCITrack ở `sot.mcitrack_root` kèm checkpoint
  `checkpoints/train/mcitrack/<sot.config>/MCITRACK_ep0300.pth.tar` (1.44 GB).
  `uav_pipeline` **không** copy code MCITrack.

**Chọn bbox đầu**: có `sot.init_bbox` thì dùng nó; không có thì chạy detector và
lấy box **conf cao nhất** trong `sot.init_classes` (`[]` = theo
`detector.classes_of_interest`). Frame đầu không có box là bình thường — pipeline
ở mode `acquire` và thử lại từng frame tới khi thấy box.

**Kết quả**: `sinks.sot_result` ghi `frame,x,y,w,h,conf,alive` (frame 1-indexed,
xywh pixel ảnh gốc, `-1,...,0` khi chưa acquire hoặc đã LOST). Video HUD và
telemetry dùng chung sink với MOT.

**Guard (`sot.guard.enabled`, mặc định OFF)**: MCITrack không có khái niệm "mất
target" — nó luôn trả 1 box mỗi frame, nên khi đối tượng ra khỏi khung nó sẽ bám
sang vật khác. Guard thêm 3 tầng cắt: nhảy bất khả thi (mỗi frame), lệch dự đoán
vận tốc (mỗi frame), và verify bằng detector mỗi `verify_every` frame.

⚠️ **Giới hạn đã đo của guard**: trên 15 video VisDrone test-dev nó cắt **đúng 4 /
oan 6**. Ba ngưỡng hiện tại được calibrate trên 5 đối tượng của **một** sequence
val. Bật guard là đánh đổi: bớt bám sai vật, thêm rủi ro cắt sớm. Tắt guard cho
kết quả y hệt MCITrack gốc.
```

- [ ] **Step 9: Commit**

```bash
cd /home/anlnm/UAV/uav_pipeline
git add configs/sot_mcitrack.yaml configs/default.yaml scripts/run_pipeline.py \
        README.md scripts/validate_sot.py
git commit -m "configs: SOT config + default.yaml runs SOT; CLI --sot/--no-sot/--init-bbox"
```

---

### Task 10: Kiểm chứng end-to-end với baseline thật

**Files:**
- Không sửa code. Nếu có kiểm nào fail thì sửa task tương ứng rồi chạy lại.

**Interfaces:**
- Consumes: toàn bộ Task 1-9
- Produces: kết luận đạt/không đạt cho 8 kiểm trong spec (kiểm #9 đã nằm trong `validate_sot.py`)

Baseline (đã kiểm tồn tại):
- MOT: `output/mot_eval/uav0000339_00001_v.txt` (+ `metrics.txt`, MOTA 0.4635)
- SOT guard-off: `/home/anlnm/UAV/MCITrack/output/uav0000161_00000_v_person_mcitrack_l384_raw.txt`, `..._uav0000077_00720_v_..._raw.txt`
- SOT guard-on: 17 file `/home/anlnm/UAV/MCITrack/output/*_person_*_guarded.txt`
- Baseline 5 object của `uav0000339` **đã bị xoá** → đối chiếu số trong `MCITrack/CLAUDE.md`.

- [ ] **Step 1: Kiểm 1 — MOT không hồi quy (quan trọng nhất)**

```bash
cd /home/anlnm/UAV
cp uav_pipeline/output/mot_eval/uav0000339_00001_v.txt /tmp/mot_baseline.txt
MCITrack/.venv/bin/python -m uav_pipeline.scripts.eval_mot_visdrone \
    --config uav_pipeline/configs/local_onnx_batch16.yaml \
    --seq uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v
diff -q /tmp/mot_baseline.txt uav_pipeline/output/mot_eval/uav0000339_00001_v.txt \
  && echo "KIEM 1 DAT: MOT khong hoi quy" || echo "KIEM 1 FAIL"
```
Expected: `KIEM 1 DAT`. Nếu `eval_mot_visdrone.py` có arg khác thì chạy `--help` để lấy đúng tên arg — **không sửa file đó**.

- [ ] **Step 2: Kiểm 2 — SOT guard=off trùng MCITrack tới 0.00 px**

```bash
cd /home/anlnm/UAV
for s in uav0000161_00000_v uav0000077_00720_v; do
  MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
      --config uav_pipeline/configs/sot_mcitrack.yaml \
      --source uav_pipeline/input/VisDrone2019-MOT-test-dev/sequences/$s \
      --source-type image_dir --no-video
  MCITrack/.venv/bin/python - "$s" <<'PY'
import sys
import numpy as np
s = sys.argv[1]
mine = np.loadtxt("uav_pipeline/output/sot_result.txt", delimiter=",")
base = np.loadtxt(f"/home/anlnm/UAV/MCITrack/output/{s}_person_mcitrack_l384_raw.txt",
                  delimiter=",")
n = min(len(mine), len(base))
d = np.abs(mine[:n, 1:5] - base[:n, 1:5]).max()
print(f"{s}: {n} frame, lech max {d:.4f} px ->", "DAT" if d < 0.005 else "FAIL")
PY
done
```
Expected: cả 2 in `lech max 0.0000 px -> DAT`.
⚠️ `init_classes` phải là `[0,1]` (person) để khớp baseline — sửa trong config hoặc thêm `sot: {init_classes: [0,1]}`. **Tuyệt đối không** truyền `--init-bbox` lấy từ dòng 1 của file baseline: dòng đó chỉ có 2 chữ số thập phân, init box làm tròn sẽ làm quỹ đạo lệch dần và ra kết luận "nondeterminism" sai (đã mắc 2 lần).

- [ ] **Step 3: Kiểm 3 + 4 — guard=on cắt đúng frame, không lọt box sai**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-test-dev/sequences/uav0000161_00000_v \
    --source-type image_dir --no-video
# ^ trước khi chạy: đặt sot.guard.enabled: true và sot.init_classes: [0,1] trong config
MCITrack/.venv/bin/python - <<'PY'
import numpy as np
mine = np.loadtxt("uav_pipeline/output/sot_result.txt", delimiter=",")
base = np.loadtxt("/home/anlnm/UAV/MCITrack/output/"
                  "uav0000161_00000_v_person_mcitrack_l384_guarded.txt", delimiter=",")
def first_lost(a):
    z = a[a[:, 6] == 0]
    return int(z[0, 0]) if len(z) else None
print("LOST cua toi:", first_lost(mine), " baseline:", first_lost(base))
print("so dong:", len(mine), len(base))
print("KIEM 3+4", "DAT" if first_lost(mine) == first_lost(base)
      and len(mine) == len(base) else "FAIL")
PY
```
Expected: 2 frame LOST bằng nhau và số dòng bằng nhau → `DAT`. Frame LOST **không được muộn hơn** baseline 1 frame — muộn 1 frame nghĩa là `DeferredSinkWriter` chưa cắt lui (Task 5/8 sai).

- [ ] **Step 4: Kiểm 5 — acquire khi cả sequence không có box hợp lệ**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
    --source-type image_dir --no-video 2>&1 | tail -5
# ^ đặt sot.init_classes: [8] (bus). uav0000339_00001_v: 0 box bus trong CẢ sequence
awk -F, '{if ($7 != 0) bad++} END {print "dong alive=1:", bad+0, "(phai la 0)"}' \
    uav_pipeline/output/sot_result.txt
wc -l < uav_pipeline/output/sot_result.txt   # phải là 275
```
Expected: không crash, `dong alive=1: 0`, 275 dòng, log in `acquire (275 frame, 0 box)`

- [ ] **Step 5: Kiểm 6 — init_bbox thủ công bỏ qua detector**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
    --source-type image_dir --no-video --init-bbox "949.9,576.5,93.5,126.9"
head -1 uav_pipeline/output/sot_result.txt
head -1 uav_pipeline/output/sot_telemetry.jsonl | \
    MCITrack/.venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print('n_det', r['n_det'], 'sot', r['sot'])"
```
Expected: dòng 1 của txt là `1,949.90,576.50,93.50,126.90,...,1`; `n_det 0` (guard off → không frame nào cần detector)

- [ ] **Step 6: Kiểm 7 — reacquire tăng track_id và reset h_state**

```bash
cd /home/anlnm/UAV
# đặt sot.guard.enabled: true, sot.on_lost: reacquire, sot.init_classes: [3] (car)
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
    --source-type image_dir --no-video
MCITrack/.venv/bin/python - <<'PY'
import json
ids = set()
for line in open("uav_pipeline/output/sot_telemetry.jsonl"):
    for t in json.loads(line)["tracks"]:
        ids.add(t["id"])
print("track_id thay duoc:", sorted(ids))
print("KIEM 7", "DAT" if len(ids) >= 2 else "FAIL (khong reacquire)")
PY
```
Expected: `track_id thay duoc: [1, 2, ...]` → `DAT`. Theo `CLAUDE.md` car bị cắt ở f56 nên phải có id 2.

- [ ] **Step 7: Kiểm 8 — follow chuyển mode và coast rồi phanh**

```bash
cd /home/anlnm/UAV
# dùng lại lần chạy Kiểm 6 (guard on, on_lost: stop, init_classes: [3])
MCITrack/.venv/bin/python - <<'PY'
import json
modes, cmds = [], []
for line in open("uav_pipeline/output/sot_telemetry.jsonl"):
    r = json.loads(line)
    modes.append(r["mode"])
    cmds.append(r["command"])
seq = [m for i, m in enumerate(modes) if i == 0 or m != modes[i - 1]]
print("chuoi mode:", seq)
lost = next(i for i, c in enumerate(cmds) if c and c["target_lost"])
after = cmds[lost:lost + 20]
n_coast = sum(1 for c in after if c and c["yaw_rate"] != 0.0)
print("coast frame:", n_coast, "(follow.lost_recovery_frames = 15)")
print("KIEM 8", "DAT" if "recover" in seq and n_coast <= 15 else "FAIL")
PY
```
Expected: chuỗi mode có `tracking` → `recover` → `acquire`, coast ≤ 15 frame → `DAT`

- [ ] **Step 8: Chạy lại toàn bộ test không GPU và test MOT core**

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_sot.py
MCITrack/.venv/bin/python uav_pipeline/scripts/validate_pipeline.py
```
Expected: `[validate_sot] 46 TEST PASSED` và `[validate] ALL CORE TESTS PASSED`

- [ ] **Step 9: Trả config về mặc định an toàn rồi commit**

Trước khi commit, đảm bảo `configs/sot_mcitrack.yaml` về đúng mặc định của Task 9
(`guard.enabled: false`, `on_lost: stop`, `init_classes: []`) — các kiểm ở trên có
sửa tạm.

```bash
cd /home/anlnm/UAV/uav_pipeline
git diff configs/sot_mcitrack.yaml     # phải rỗng
git status --short
git add -A docs/superpowers
git commit -m "docs: SOT integration spec + implementation plan"
```

---

## Self-Review

**Spec coverage** — đối chiếu từng mục của spec với task:

| Yêu cầu trong spec | Task |
|---|---|
| `sot.enabled` on/off, loại trừ MOT bằng validation | 1 (validate), 8 (rẽ nhánh), 9 (CLI) |
| Code MOT không xoá, dùng lại được | 8 (`tracker.enabled` → None, không sửa `track/`), test `test_mot_configs_untouched` (9), Kiểm 1 (10) |
| Input là bbox; không có thì detector lấy conf cao nhất | 7 (`_acquire`, `_pick_init`), 9 (`--init-bbox`) |
| Frame đầu 0 box → quét tiếp | 7 (`test_sot_acquire_scans_until_first_box`), 10 Kiểm 5 |
| `mcitrack_root` qua config, không copy code | 6 |
| 6 việc bắt buộc của wrapper (path, 2 patch, set_device, BGR→RGB, h_state) | 6 |
| Guard 3 tầng + 3 bài học bug | 3 |
| `class_groups` theo tên, fallback presence | 2 |
| `on_lost: stop \| reacquire` | 7, 10 Kiểm 6 |
| `DeferredSinkWriter` (cách B, 0 frame box sai) | 5, 8, 10 Kiểm 3+4 |
| `sinks.sot_result` txt 7 cột, 1-indexed, flush/frame | 4 |
| HUD `extra_stats["sot"]`, telemetry field `sot` | 8 |
| Ép `detector.batch = 1` | 8 |
| `sot.device` → `torch.cuda.set_device` | 6 |
| Bảng xử lý lỗi chết sớm | 1 (`validate`), 6 (`preflight`), 2 (fallback gate) |
| `default.yaml` bật SOT + đổi detector onnx/yolox | 9 |
| 9 kiểm chứng | 3/5/7 (kiểm 9 dạng unit), 10 (kiểm 1-8) |
| Ngoài phạm vi: không sửa `lib/`, không vendor, không recalibrate, không đổi UPH, không `eval_sot_visdrone.py` | không có task nào làm |

Không có mục nào của spec thiếu task.

**Placeholder scan**: không có "TBD"/"TODO"/"tương tự Task N". Mọi step có code thật hoặc lệnh thật kèm kết quả mong đợi. `sot/__init__.py` được viết 2 lần một cách có chủ ý: bản tạm ở Task 2 Step 3 (chỉ export `accepted_ids`) và bản đầy đủ ở Task 7 Step 4 — vì `guard.py`/`tracker.py` chưa tồn tại ở Task 2, import sớm sẽ làm mọi test fail.

**Type consistency** (đã soát chéo):
- `accepted_ids(gate, init_cls, names) -> Optional[List[int]]` — Task 2 định nghĩa, Task 3 dùng đúng thứ tự tham số.
- `LostGuard(cfg, frame_width, init_cls, names)` và `.step(frame_idx, box_xywh, detect_fn)` — Task 3 định nghĩa, Task 7 gọi đúng.
- `GuardVerdict.alive / .provisional / .lost_at / .reason` — dùng thống nhất ở Task 3, 7, 8.
- `MCITrackModel.initialize(frame_bgr, bbox_xywh)` / `.track(frame_bgr) -> (bbox_xywh, score)` — Task 6 định nghĩa; `FakeModel` ở Task 7 khớp đúng 2 method này.
- `SotTracker.update(frame, frame_idx, detect_fn, prefetched=None)` — Task 7 định nghĩa, Task 8 gọi đúng với `meta.idx + 1`.
- `DeferredSinkWriter(emit_fn, max_hold)` + `write(ctx, provisional, retract_from)` — Task 5 định nghĩa, Task 8 gọi đúng.
- `Config.validate()` / `Config.warnings()` — Task 1 định nghĩa, Task 9 gọi.
- Frame id: `sot_result` và `LostGuard` đều dùng 1-indexed (`meta.idx + 1`), khớp `retract_from` so với `h.meta.idx + 1` trong `DeferredSinkWriter`.
