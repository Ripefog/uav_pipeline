"""MOT + face identification overlay — mỗi person track hiển thị cosine với TỪNG người trong gallery.

Chạy MOT (YOLOX-X + DroneByteTracker) trên mọi frame; với mỗi track lớp người thì
head-crop -> SCRFD (kèm 5 landmark) -> align ArcFace 112x112 -> MobileFaceNet ->
cosine với mọi ảnh gallery. Vẽ conf của TẤT CẢ người trong gallery lên từng track.

Khác face/detector.py: bản này giải cả 3 tensor landmark (outputs[6:9]) để align.
face/detector.py bỏ landmark, mà ArcFace rất nhạy alignment -> không dùng lại được.

    MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_mot_face \
        --video uav_pipeline/input/beijing2015_100m_semi_300-315.mp4 \
        --gallery uav_pipeline/input/faces \
        --out uav_pipeline/output/mot_face.mp4
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from uav_pipeline.config import CMCCfg, DetectorCfg, PrimaryModelCfg, SotCfg, TrackerCfg
from uav_pipeline.detect.wrapper import UnifiedDetector
from uav_pipeline.face.recognizer import FaceRecognizer
from uav_pipeline.sot.mcitrack_wrapper import build_mcitrack_model
from uav_pipeline.track import DroneByteTracker

PERSON_CLASSES = {"pedestrian", "people"}

# Template 5 điểm chuẩn ArcFace ở 112x112 (mắt T, mắt P, mũi, khoé miệng T, khoé miệng P)
_ARCFACE_5PT = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class ScrfdLandmark:
    """SCRFD giải cả bbox và 5 landmark (face/detector.py chỉ giải bbox)."""

    _INPUT = (640, 640)
    _STRIDES = [8, 16, 32]
    _ANCHORS = 2

    def __init__(self, model_path, conf=0.35, nms=0.4, num_threads=8, device="cuda"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        self.conf, self.nms = conf, nms
        self._cache = {}

    def _centers(self, fh, fw, stride):
        key = (fh, fw, stride)
        if key not in self._cache:
            c = np.stack(np.mgrid[:fh, :fw][::-1], -1).astype(np.float32).reshape(-1, 2) * stride
            self._cache[key] = np.repeat(c, self._ANCHORS, axis=0)
        return self._cache[key]

    def detect(self, img):
        h0, w0 = img.shape[:2]
        iw, ih = self._INPUT
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, self._INPUT, (127.5,) * 3, swapRB=True)
        out = self.sess.run(None, {self.iname: blob})
        boxes, scores, kpss = [], [], []
        for i, st in enumerate(self._STRIDES):
            fh, fw = ih // st, iw // st
            sc = out[i].flatten()
            bb = out[i + 3].reshape(-1, 4) * st
            kp = out[i + 6].reshape(-1, 5, 2) * st
            ctr = self._centers(fh, fw, st)
            box = np.stack([ctr[:, 0] - bb[:, 0], ctr[:, 1] - bb[:, 1],
                            ctr[:, 0] + bb[:, 2], ctr[:, 1] + bb[:, 3]], axis=1)
            keep = sc >= self.conf
            boxes.append(box[keep]); scores.append(sc[keep]); kpss.append((kp + ctr[:, None, :])[keep])
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        kpss = np.concatenate(kpss)
        if not len(boxes):
            return []
        sx, sy = w0 / iw, h0 / ih
        boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
        kpss[:, :, 0] *= sx; kpss[:, :, 1] *= sy
        wh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0],
                       boxes[:, 3] - boxes[:, 1]], axis=1)
        idx = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), self.conf, self.nms)
        idx = np.array(idx).flatten() if len(idx) else []
        return [{"box": boxes[i], "score": float(scores[i]), "kps": kpss[i]} for i in idx]


def align_face(img, kps):
    """Similarity transform 5 điểm -> ArcFace 112x112. None nếu không giải được."""
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), _ARCFACE_5PT, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), borderValue=0)


def load_gallery(gallery_dir, scrfd, recognizer):
    """dataset/<person>/<img> -> (names, matrix). Ảnh gallery là chân dung cắt sát nên
    SCRFD thường KHÔNG bắt được (mặt chạm mép, không còn context) -> pad rồi thử lại;
    vẫn trượt thì embed thẳng ảnh gốc (đúng nhánh --no-detect của face/pipeline.py)."""
    names, vecs = [], []
    for pdir in sorted(glob.glob(os.path.join(gallery_dir, "*"))):
        if not os.path.isdir(pdir):
            continue
        person = os.path.basename(pdir)
        for f in sorted(glob.glob(os.path.join(pdir, "*"))):
            img = cv2.imread(f)
            if img is None:
                print(f"[gallery] BỎ QUA (không đọc được): {f}")
                continue
            pad = int(max(img.shape[:2]) * 0.6)
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            faces = scrfd.detect(padded)
            if faces:
                faces.sort(key=lambda x: -x["score"])
                crop = align_face(padded, faces[0]["kps"])
                how = f"aligned(det={faces[0]['score']:.2f})"
            else:
                crop, how = img, "raw(no-detect)"
            emb = recognizer.get_embedding(crop)
            if emb is None:
                continue
            names.append(person); vecs.append(emb)
            print(f"[gallery] {person:<20} {os.path.basename(f):<34} {how}")
    if not vecs:
        raise SystemExit(f"[gallery] không nạp được vector nào từ {gallery_dir}")
    return names, np.stack(vecs)


def put_text_bg(img, text, org, scale, color, thickness, bg=(0, 0, 0), pad=3):
    """putText với nền đen phía sau — không thì chữ chìm vào nền đường chạy nâu đỏ."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def head_region(frame, box, top_frac=0.45, pad_x=0.20, pad_y=0.08):
    """Vùng đầu = phần trên của box người, nới ra một chút. Cắt vùng này rồi mới
    chạy SCRFD (thay vì chạy trên cả frame) để mặt giữ được độ phân giải gốc."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a = int(max(0, x1 - w * pad_x)); b = int(max(0, y1 - h * pad_y))
    c = int(min(frame.shape[1], x2 + w * pad_x)); d = int(min(frame.shape[0], y1 + h * top_frac))
    return frame[b:d, a:c], a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--weights", default=os.path.join(_ROOT, "weights"))
    ap.add_argument("--sim-thresh", type=float, default=0.40,
                    help="cosine coi là KHỚP THẬT (ArcFace thường 0.4-0.5)")
    ap.add_argument("--display-thresh", type=float, default=None,
                    help="cosine để TÔ MÀU/in đậm trên video (chỉ hiển thị, không đổi "
                        "nghĩa 'khớp'). Mặc định = --sim-thresh; hạ xuống để soi mắt "
                        "thường lúc chưa calibrate xong ngưỡng thật.")
    ap.add_argument("--face-conf", type=float, default=0.35, help="ngưỡng SCRFD")
    ap.add_argument("--min-person-h", type=float, default=60.0)
    ap.add_argument("--person", default="",
                    help="chỉ dùng 1 người trong gallery (tên thư mục); \"\" = mọi người")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 | cuda:1 | cpu")
    ap.add_argument("--handoff", action="store_true",
                    help="MOT+face -> khi 1 track đạt sim>=sim-thresh đủ "
                        "--handoff-hits lần thì khoá bbox track đó, khởi tạo SOT "
                        "(MCITrack) và chuyển hẳn sang SOT (chỉ vẽ 1 box). Cần "
                        "--person (gallery đúng 1 người) và --device cuda:N.")
    ap.add_argument("--handoff-hits", type=int, default=2,
                    help="số lần sim>=sim-thresh trên CÙNG track_id để trigger handoff")
    args = ap.parse_args()
    if args.display_thresh is None:
        args.display_thresh = args.sim_thresh

    wd = args.weights
    dev_kind = "cpu" if args.device == "cpu" else "cuda"
    scrfd = ScrfdLandmark(os.path.join(wd, "scrfd_2.5g_bnkps.onnx"), conf=args.face_conf,
                          device=dev_kind)
    recog = FaceRecognizer(os.path.join(wd, "mobilefacenet.onnx"), device=dev_kind)
    names, G = load_gallery(args.gallery, scrfd, recog)
    if args.person:
        keep = [i for i, n in enumerate(names) if n == args.person]
        if not keep:
            raise SystemExit(f"[gallery] không có {args.person!r}; "
                             f"đang có: {sorted(set(names))}")
        names = [names[i] for i in keep]
        G = G[keep]
        print(f"[gallery] LỌC -> chỉ dùng {args.person!r} ({len(keep)} vector)")
    people = sorted(set(names))
    print(f"[gallery] {len(names)} vector / {len(people)} người: {people}")
    if len(names) > 1:
        off = [(names[i], names[j], float(G[i] @ G[j]))
               for i in range(len(names)) for j in range(i + 1, len(names)) if names[i] != names[j]]
        if off:
            print(f"[gallery] cosine khác-người: max={max(o[2] for o in off):.3f} "
                  f"(càng thấp càng tách tốt)")

    if args.handoff:
        if len(people) != 1:
            raise SystemExit(f"[handoff] cần gallery đúng 1 người (dùng --person), "
                             f"đang có {len(people)}: {people}")
        target_person = people[0]
        print(f"[handoff] BẬT: track nào đạt sim>={args.sim_thresh:.2f} với "
              f"{target_person!r} đủ {args.handoff_hits} lần -> khởi tạo SOT ngay "
              f"trên bbox track đó, chuyển hẳn sang SOT (bỏ MOT+face).")
        sot_model = build_mcitrack_model(SotCfg(device=args.device))
    else:
        target_person = None
        sot_model = None

    dcfg = DetectorCfg(backend="onnx", device=args.device, preprocess="yolox", imgsz=[736, 1280],
                       conf=0.25, iou=0.45, batch=1,
                       primary=PrimaryModelCfg(onnx=os.path.join(wd, "best_yoloxx.onnx"),
                                               names_yaml=os.path.join(wd, "names.yaml")))
    detector = UnifiedDetector(dcfg)
    tracker = DroneByteTracker(TrackerCfg(enabled=True, high_conf=0.4, low_conf=0.15, iou=0.3,
                                          max_age=50, min_hits=3, cmc=CMCCfg(enabled=True)))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"không mở được video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    jf = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None

    # bảng màu cố định theo người, để cùng một người luôn cùng màu qua các frame
    palette = [(80, 220, 80), (80, 180, 255), (255, 160, 60), (200, 120, 255)]
    pcolor = {p: palette[i % len(palette)] for i, p in enumerate(people)}

    best_per_track = defaultdict(lambda: defaultdict(float))
    hit_count = defaultdict(int)   # track_id -> số lần sim>=sim-thresh với target_person
    n_face = n_hit = 0
    mode = "mot"                   # mot -> sot (một chiều, không quay lại)
    handoff_info = None            # (frame, track_id, sim) lúc trigger
    sot_frames = 0
    sot_score_sum = 0.0
    idx = 0
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        idx += 1

        # ================= SOT: đã khoá người, chỉ bám + vẽ 1 box ==========
        if mode == "sot":
            bbox_xywh, score = sot_model.track(frame)
            x, y, w, h = bbox_xywh
            x1, y1, x2, y2 = x, y, x + w, y + h
            color = pcolor[target_person]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            put_text_bg(frame, f"{target_person}  SOT score={score:.2f}",
                       (int(x1), int(y1) - 10), 0.75, color, 2)
            sot_frames += 1
            sot_score_sum += score
            rec = [{"track_id": int(handoff_info[1]), "cls": "sot", "mode": "sot",
                    "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "target": target_person, "score": round(score, 4)}]

            cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
            cv2.putText(frame, f"frame {idx}/{total}  SOT  target={target_person} "
                               f" score={score:.2f}  (handoff @f{handoff_info[0]})",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(frame)
            if jf:
                jf.write(json.dumps({"frame": idx, "tracks": rec}) + "\n")
            if idx % 25 == 0:
                el = time.monotonic() - t0
                print(f"  frame {idx}/{total}  {el/idx:.2f}s/f  [SOT] score={score:.2f}")
            continue

        # ================= MOT + face id =====================================
        dets = detector.detect(frame)
        tracks = tracker.update(frame, dets)
        rec = []
        triggered_by = None
        for t in tracks:
            x1, y1, x2, y2 = [float(v) for v in t.bbox]
            is_person = t.name in PERSON_CLASSES
            sims, fbox = None, None
            if is_person and (y2 - y1) >= args.min_person_h:
                crop, ox, oy = head_region(frame, (x1, y1, x2, y2))
                if crop.size:
                    # phóng to vùng đầu nhỏ: SCRFD ép về 640x640 nên crop nhỏ mà
                    # không upscale thì mặt còn vài pixel sau resize.
                    s = max(1.0, 320.0 / max(1, crop.shape[0]))
                    big = cv2.resize(crop, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_CUBIC) if s > 1 else crop
                    faces = scrfd.detect(big)
                    if faces:
                        faces.sort(key=lambda f: -f["score"])
                        f0 = faces[0]
                        aligned = align_face(big, f0["kps"])
                        emb = recog.get_embedding(aligned) if aligned is not None else None
                        if emb is not None:
                            n_face += 1
                            raw = G @ emb
                            sims = {p: float(max(raw[i] for i in range(len(names))
                                                 if names[i] == p)) for p in people}
                            for p, v in sims.items():
                                best_per_track[t.track_id][p] = max(best_per_track[t.track_id][p], v)
                            bx = f0["box"] / s
                            fbox = (int(bx[0] + ox), int(bx[1] + oy),
                                    int(bx[2] + ox), int(bx[3] + oy))

            top = max(sims.items(), key=lambda kv: kv[1]) if sims else None
            hit = bool(top and top[1] >= args.sim_thresh)          # khớp THẬT (báo cáo/số liệu)
            shown = bool(top and top[1] >= args.display_thresh)    # chỉ để TÔ MÀU trên video
            n_hit += int(hit)

            if args.handoff and is_person and top and top[0] == target_person \
                    and top[1] >= args.sim_thresh:
                hit_count[t.track_id] += 1
                if hit_count[t.track_id] >= args.handoff_hits and triggered_by is None:
                    triggered_by = (t.track_id, top[1], (x1, y1, x2, y2))

            color = pcolor[top[0]] if shown else ((190, 190, 190) if is_person else (110, 110, 110))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3 if shown else 2)
            put_text_bg(frame, f"#{t.track_id} {t.name}", (int(x1), int(y1) - 8),
                       0.6, color, 2)
            if fbox:
                cv2.rectangle(frame, fbox[:2], fbox[2:], (0, 255, 255), 2)
            if sims:
                # conf cho TỪNG người trong gallery (identification conf = cosine
                # similarity, KHÁC face-detect conf của SCRFD), không chỉ top-1.
                for k, p in enumerate(people):
                    v = sims[p]
                    on = v >= args.display_thresh
                    put_text_bg(frame, f"{p}: {v:.2f}", (int(x1), int(y2) + 26 + 26 * k),
                               0.7, pcolor[p] if on else (210, 210, 210), 2)
            else:
                if is_person:
                    put_text_bg(frame, "no face", (int(x1), int(y2) + 26),
                               0.6, (170, 170, 170), 2)
            rec.append({"track_id": int(t.track_id), "cls": t.name, "mode": "mot",
                        "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                        "face_box": fbox, "sims": sims})

        cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {idx}/{total}  tracks={len(tracks)}  faces={n_face} "
                           f" thresh={args.sim_thresh:.2f}  MOT+face",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
        if jf:
            jf.write(json.dumps({"frame": idx, "tracks": rec}) + "\n")
        if idx % 25 == 0:
            el = time.monotonic() - t0
            print(f"  frame {idx}/{total}  {el/idx:.2f}s/f  faces={n_face}  hits={n_hit}")

        # ---- trigger handoff: khởi tạo SOT NGAY trên frame này, bbox MOT hiện tại ----
        if triggered_by is not None:
            tid, sim, (hx1, hy1, hx2, hy2) = triggered_by
            bbox_xywh = [hx1, hy1, hx2 - hx1, hy2 - hy1]
            print(f"[handoff] frame {idx}  track #{tid}  sim={sim:.3f} (lần thứ "
                  f"{hit_count[tid]}) >= {args.sim_thresh:.2f}  -> khởi tạo SOT bbox={bbox_xywh}")
            sot_model.initialize(frame, bbox_xywh)
            handoff_info = (idx, tid, sim)
            mode = "sot"

    cap.release(); vw.release()
    if jf:
        jf.close()

    print("-" * 68)
    print(f"frames={idx}  face_embeds={n_face}  hits(sim>={args.sim_thresh})={n_hit}")
    print(f"\nCosine CAO NHẤT từng track đạt được (n_track={len(best_per_track)}):")
    hdr = "  track  " + "".join(f"{p[:16]:>18}" for p in people)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for tid in sorted(best_per_track):
        row = "".join(f"{best_per_track[tid][p]:>18.3f}" for p in people)
        print(f"  #{tid:<6}{row}")
    if best_per_track:
        allv = [v for d in best_per_track.values() for v in d.values()]
        print(f"\n  max toàn video = {max(allv):.3f}   (cần >= {args.sim_thresh} mới coi là khớp)")
    if args.handoff:
        if handoff_info:
            f_idx, tid, sim = handoff_info
            avg = sot_score_sum / sot_frames if sot_frames else 0.0
            print(f"\n[handoff] TRIGGER tại frame {f_idx}, track #{tid} (sim={sim:.3f}) "
                  f"-> SOT chạy {sot_frames} frame, score trung bình={avg:.3f}")
        else:
            print(f"\n[handoff] KHÔNG bao giờ trigger trong {idx} frame "
                  f"(không track nào đạt sim>={args.sim_thresh:.2f} đủ {args.handoff_hits} lần)")
    print(f"\nvideo -> {os.path.abspath(args.out)}")
    if jf:
        print(f"jsonl -> {os.path.abspath(args.jsonl)}")


if __name__ == "__main__":
    main()
