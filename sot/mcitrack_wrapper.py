"""Bọc MCITrack (repo ngoài) thành API 3 hàm cho pipeline.

Đây là file DUY NHẤT trong uav_pipeline biết MCITrack tồn tại. Nó gánh toàn bộ
phần "bẩn" mà repo MCITrack bắt buộc phải có:

  1. sys.path.insert(mcitrack_root) — không copy code, không vendor.
  2. patch torch.load(weights_only=False): torch >= 2.6 mặc định True sẽ crash khi
     load checkpoint MCITrack. Patch này TOÀN CỤC, KHÔNG restore, và idempotent
     (xem docstring _patch_torch_load).
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


def _patch_torch_load(torch_mod) -> None:
    """Patch torch_mod.load để mặc định weights_only=False.

    torch >= 2.6 đổi default weights_only sang True; checkpoint MCITrack là
    dict thường (không chỉ tensor) lưu bằng torch.save đời cũ nên load thẳng sẽ
    crash. setdefault() giữ nguyên: nếu caller tự truyền weights_only thì giá
    trị đó vẫn thắng, patch chỉ đổi DEFAULT.

    Đây là patch TOÀN CỤC (ghi đè torch_mod.load, ảnh hưởng mọi torch.load gọi
    sau đó trong CÙNG process, không riêng gì MCITrack) và KHÔNG được restore
    lại — có thể tạo nhiều MCITrackModel trong một process (reacquire dài hạn,
    hoặc 2 phiên SOT nối tiếp) nên không có mốc "hết cần SOT" nào an toàn để
    unpatch; unpatch giữa lúc model khác còn sống sẽ làm nó crash lại.

    Đổi lại: hàm này idempotent — đánh dấu wrapper bằng thuộc tính
    ``_uav_sot_patched`` và bỏ qua nếu đã patch, để gọi lại (constructor thứ 2,
    thứ 3...) không lồng closure vào nhau vô hạn.
    """
    if getattr(torch_mod.load, "_uav_sot_patched", False):
        return
    _orig_load = torch_mod.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    _load._uav_sot_patched = True
    torch_mod.load = _load


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

        _patch_torch_load(torch)   # patch toàn cục, không restore — xem docstring

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
    """preflight(cfg) rồi build MCITrackModel(cfg).

    QUAN TRỌNG: nếu preflight() thấy lỗi, hàm này raise SystemExit — KHÔNG phải
    Exception thường. SystemExit kế thừa BaseException, nên
    ``except Exception`` ở caller sẽ KHÔNG bắt được nó và process sẽ thoát.
    Đây là chủ ý: cấu hình SOT sai là lỗi khởi động (fail-fast), không phải lỗi
    runtime có thể degrade-gracefully như OCR — operator cần thấy thông báo rõ
    và process dừng ngay, không có traceback rác. Caller KHÔNG được bắt
    SystemExit này để "catch-and-continue".
    """
    errs = preflight(cfg)
    if errs:
        raise SystemExit("[sot] cấu hình SOT lỗi:\n  - " + "\n  - ".join(errs))
    return MCITrackModel(cfg)
