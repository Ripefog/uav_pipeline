# Tích hợp SOT (MCITrack) vào uav_pipeline — design

Ngày: 2026-08-12

## Mục tiêu

`uav_pipeline` hiện chỉ có MOT (`track/drone_tracker.py`, constant-velocity + IoU
association). `MCITrack` (AAAI'25, single-object tracking) đã chạy tốt ở repo
riêng `/home/anlnm/UAV/MCITrack`. Đưa SOT vào pipeline như **một module bật/tắt
được**, giống các module khác (`ocr`, `follow`, `sinks.*`):

- Input là **bbox**. Có bbox thì dùng bbox đó; không có thì detector chạy và lấy
  box **conf cao nhất** làm init.
- Có biến on/off. **SOT và MOT loại trừ nhau — chỉ được bật một cái.**
- Code MOT **không xoá**, vẫn dùng lại được nguyên trạng.

## Ràng buộc đã kiểm (không phải giả định)

| Ràng buộc | Bằng chứng |
|---|---|
| `uav_pipeline` không có venv riêng | `ls -d /home/anlnm/UAV/*/.venv` → chỉ `MCITrack/.venv`, `eval_yolo/.venv` |
| Env duy nhất chạy được cả hai là `MCITrack/.venv` | có torch 2.11+cu128, torchvision, cv2, numpy, yaml, onnxruntime |
| Env đó **không có openvino** | `import openvino` → ModuleNotFoundError → không dùng được `configs/default.yaml` hiện tại (backend openvino) |
| `onnxruntime` ở env đó **chỉ có CPUExecutionProvider** | `get_available_providers()` → `['AzureExecutionProvider','CPUExecutionProvider']`. Detector chạy CPU, đúng như các lần đo baseline |
| Pipeline không cần scipy | `track/drone_tracker.py` chỉ import numpy |
| torch cu128 là bắt buộc | RTX 5080 = sm_120; PyTorch 2.1.2+cu121 của `install.sh` không thấy GPU |
| `Config.from_yaml()` load **một file**, không merge `default.yaml` | `config.py:225-230` |
| Detector của pipeline dùng **cùng NMS** với script MCITrack | `detect/postprocess.py:29` → `util.non_max_suppression_yolox` (bản vendored) |
| MCITrack hardcode `.cuda()` không index | `lib/test/tracker/mcitrack.py` → phải dùng `torch.cuda.set_device()`, không dựa vào `sot.device` truyền vào lib |
| `initialize()` của MCITrack **không** reset `h_state` | `lib/test/tracker/mcitrack.py:34` — chỉ set trong `__init__` |
| torch ≥2.6 mặc định `weights_only=True` | phải patch `torch.load` mới load được checkpoint 1.44 GB |

## Kiến trúc

Thêm package `uav_pipeline/sot/`. **Không sửa gì trong `track/`.** Điểm rẽ nhánh
duy nhất ở `pipeline.py:158`:

```
                          ┌─ sot.enabled=false ─→ DroneByteTracker.update() ─→ tracks[N]
source → detector.detect ─┤
                          └─ sot.enabled=true  ─→ SotTracker.update()       ─→ tracks[0..1]
                                                          │
                                    ctx.tracks ───────────┴──→ follow.step → controller
                                                           └──→ sinks (video / telemetry / sot_result)
```

### Điểm cốt lõi: SOT bọc kết quả thành đúng `Track` của `contracts.py`

`track_id=1`, `bbox` xyxy, `confidence=best_score`, `cls`/`name` lấy từ detection
lúc init. Mỗi frame gọi `Track.update()` để `velocity` / `trajectory` / `age` vẫn
đúng. Nhờ đó **không phải sửa module nào ở hạ nguồn**:

- `follow/selector.py:14` lọc `age == 0` → SOT track luôn thoả → luôn được chọn
  làm target.
- `HUDAnnotatedSink` vẽ nguyên; box SOT tự ra màu target (cyan) vì
  `follow_state.target_id` khớp `track_id`.
- LOST → `ctx.tracks = []` → `follow._lost_step()` coast `lost_recovery_frames`
  rồi phanh. Không viết thêm code điều khiển.

### State machine của `SotTracker`

| mode | mỗi frame làm gì | Track xuất ra |
|---|---|---|
| `acquire` | chỉ chạy detector, tìm box conf cao nhất trong `init_classes` | không có |
| `tracking` | `mcitrack.track()`; guard kiểm (nếu bật) | 1 Track |
| `lost` | không làm gì; hoặc về `acquire` nếu `on_lost: reacquire` | không có |

Frame đầu **không có box là bình thường** (webcam/RTSP): pipeline ở `acquire`,
vẫn xuất đủ frame video, thấy box hợp lệ đầu tiên thì `initialize` tại frame đó
rồi sang `tracking`.

### Hai quyết định phụ

1. **Detector không chạy mỗi frame khi SOT bật** (`sot.detect_every_frame: false`).
   MCITrack class-agnostic, không cần detection để track; detector chỉ chạy ở
   frame `acquire` và frame verify. Với YOLOX-X 736×1280 trên CPU đây là điều
   kiện để pipeline chạy nổi. Hệ quả: HUD không có box xám mờ ở frame không
   verify. Kèm theo: khi SOT bật và `detect_every_frame: false` thì **ép
   `detector.batch = 1`** + in thông báo (batch 16 chỉ có nghĩa khi detect mọi frame).
2. **GPU chọn qua `sot.device: cuda:1`** → `torch.cuda.set_device(1)`. Không cần
   `CUDA_VISIBLE_DEVICES` vì `.cuda()` không index lấy *current device*.

## Config

Default trong dataclass giữ hành vi cũ (`sot.enabled=False`, `tracker.enabled=True`)
để **4 config MOT đang có** (`local_onnx_batch16.yaml`, `jetson_trt.yaml`,
`local_trt_*.yaml`) không tự chuyển sang SOT — nếu đổi default trong `config.py`
thì `scripts/eval_mot_visdrone.py` hỏng theo. Còn `configs/default.yaml` ghi rõ
SOT bật, tức **config mặc định của repo là SOT** theo yêu cầu.

```yaml
tracker:
  enabled: false      # MỚI. Code MOT giữ nguyên, chỉ không chạy.
  high_conf: 0.4      # các key cũ không đổi

sot:
  enabled: true
  mcitrack_root: /home/anlnm/UAV/MCITrack   # sys.path.insert, KHÔNG copy code
  config: mcitrack_l384                     # experiments/mcitrack/<config>.yaml
  dataset_preset: uav                       # chọn preset UPT/UPH/INTER/MB
  device: cuda:1
  init_bbox: null          # [x,y,w,h] pixel ảnh gốc; null = detector lấy conf cao nhất
  init_classes: []         # [] = theo detector.classes_of_interest; [] cả hai = mọi class
  detect_every_frame: false
  on_lost: stop            # stop | reacquire
  guard:
    enabled: false         # OFF = hành vi MCITrack gốc, không bao giờ LOST
    gate: class            # class | family | presence
    verify_every: 10
    K: 3
    iou_gate: 0.3
    jump:   {enabled: true, px: 90.0, area: 2.5, ref_width: 1904.0}
    motion: {enabled: true, iou: 0.05, k: 2}

sinks:
  sot_result:
    enabled: true
    path: output/sot_result.txt
```

- **Loại trừ nhau bằng validation, không đoán ngầm**: bật cả hai →
  `sys.exit` kèm thông báo tắt cái nào (cùng kiểu `run_pipeline.py:44`).
- **`on_lost` chỉ có tác dụng khi `guard.enabled: true`.** Guard off thì MCITrack
  không bao giờ tuyên bố LOST (nó luôn trả 1 box mỗi frame) → không bao giờ vào
  mode `lost`. Đặt `on_lost: reacquire` mà quên bật guard là vô hiệu → in cảnh báo.
- CLI thêm `--sot` / `--no-sot` (bật cái này tắt cái kia trong một lệnh) và
  `--init-bbox "x,y,w,h"`.
- **`configs/default.yaml` phải đổi detector sang `backend: onnx` +
  `preprocess: yolox` + `primary.onnx: weights/best_yoloxx.onnx`** — hiện là
  openvino, mà env duy nhất có torch cu128 lại không có openvino → default config
  bật SOT sẽ crash lúc import. Các config MOT khác không đụng tới.
- Ngưỡng guard là **số đã calibrate**, không phải số bịa:

| Ngưỡng | Nguồn số |
|---|---|
| `jump.px = 90` | chuyển động GT thật của 5 vật (n=708): max 44 px/frame, p99 31.3. Cú nhảy drift thật: 337 px |
| `jump.area = 2.5` | GT max tỉ lệ diện tích/frame 1.44. Cú nhảy: 10.73 |
| `motion.iou = 0.05` | drift thật 0.011 vs case đúng thấp nhất 0.094 (đo trên 21 case) |
| `motion.k = 2` | 2 frame liên tiếp vi phạm → 17/21 case không kích hoạt |
| `K = 3` | chuỗi verify-MISS oan dài nhất là 2 (bus). K=2 báo oan bus tại f30 |
| `ref_width = 1904` | độ phân giải calibrate. VisDrone trải 1344×756 → 3840×2160; không scale là báo oan trên video 4K và bỏ sót drift trên video nhỏ |

## Bên trong `sot/`

Bốn file, chỉ file đầu biết MCITrack tồn tại:

```
sot/mcitrack_wrapper.py   MCITrack -> (bbox_xywh, score). Không biết pipeline là gì.
sot/class_groups.py       bảng nhóm class cho guard (dữ liệu, không logic)
sot/guard.py              3 tầng chặn drift. Hình học thuần, không import torch/MCITrack.
sot/tracker.py            SotTracker: state machine + bọc thành Track. Không biết MCITrack là gì.
```

### `mcitrack_wrapper.py`

```python
class MCITrackModel:
    def __init__(self, root, config, dataset_preset, device)
    def initialize(self, frame_bgr, bbox_xywh)     # tự reset h_state
    def track(self, frame_bgr) -> (bbox_xywh, score)
```

Gánh toàn bộ phần "bẩn" đã trả giá để biết:

1. `sys.path.insert(0, root)` rồi `from lib.test.evaluation.tracker import Tracker`.
2. Patch `torch.load(weights_only=False)` — torch ≥2.6 không load nổi checkpoint.
3. Patch `lib.models.mcitrack.encoder.is_main_process → False` — bỏ tải
   `pretrained/fast_itpn_large_1600e_1k.pt` (1.5 GB) vì checkpoint load
   `strict=True` ghi đè toàn bộ weight encoder.
4. `torch.cuda.set_device(idx)` trước khi build.
5. `cvtColor(BGR→RGB)`: pipeline giữ BGR, MCITrack cần RGB.
6. `h_state = [None] * cfg.MODEL.NECK.N_LAYERS` mỗi lần `initialize` — không tự
   reset thì hidden state Mamba của target cũ rò sang target mới khi reacquire.

Thiếu checkpoint → lỗi kèm đúng đường dẫn đang tìm.

### `guard.py`

Giữ nguyên 3 tầng và **toàn bộ bài học đã trả giá** (mỗi dòng dưới đây là một bug
đã thật sự xảy ra):

| tầng | tần suất | bắt được gì | chi tiết không được bỏ |
|---|---|---|---|
| jump | mỗi frame | nhảy sang vật cùng class | bỏ qua frame tracker **đầu tiên** (`prev_box=None`): box detector và box MCITrack là 2 estimator khác nhau cho cùng vật, so nhau ra LOST oan ngay frame 2 (`uav0000117_02622_v_car`, 1/349 alive) |
| motion | mỗi frame | drift **dần** sang vật cùng class kề bên | mốc dự đoán **chỉ** cập nhật từ frame đã được chấp nhận, dự đoán nhiều bước. Nếu để frame giật làm mốc thì metric tự đầu độc chính nó: 1 frame giật 35 px làm các frame sau dù đã về đúng (IoU_GT 0.66) vẫn ra IoU_pred 0.000 → luôn sinh ≥2 vi phạm liên tiếp → báo oan (`uav0000086` person rank 4, f148) |
| verify | mỗi `verify_every` | target đã ra khỏi khung | latch `verify_confirmed`: chỉ được kết án track mà detector **đã từng** xác nhận. Vật 26×41 px: detector 0 hit ở f10–f50 trong khi tracker vẫn đúng IoU 0.67–0.75 → không latch thì cắt oan f30 |

Interface:

```python
class LostGuard:
    def __init__(self, cfg, frame_width, init_cls, names)
    def step(self, frame_idx, box_xywh, detect_fn) -> GuardVerdict
    def reset(self)
```

`GuardVerdict`: `alive`, `lost_at` (frame đầu chuỗi khi cắt lui), `held` (số frame
đang giữ), `reason`. `detect_fn` là callback — guard **không** import detector,
nhờ đó test được không cần GPU/ONNX.

### `class_groups.py`

Guard gate `class`/`family` cần biết "cùng nhóm" nghĩa là gì. Bảng cũ hardcode id
VisDrone (`person:[0,1]`, `car:[3,4,5,8]`, `motor:[2,6,7,9]`); dùng thẳng id thì
đổi weight khác taxonomy là **sai âm thầm**. Nên map theo **tên** rồi resolve qua
`detector.names` lúc khởi tạo. Tên không có trong bảng → tự hạ xuống
`gate: presence` + cảnh báo, thay vì gate rỗng chặn mọi thứ.

Lý do có `family`: detector lẫn bus/van/car với nhau — tại vị trí GT bus nó gán
`van` 27 frame, `bus` 12, `car` 8, không có gì 17 frame (miss rate 78%).

### `tracker.py`

`SotTracker.update(frame, frame_idx, detect_fn) -> List[Track]` — cùng vai trò với
`DroneByteTracker.update()` nên `pipeline.py` chỉ rẽ một nhánh. `reacquire` tăng
`track_id` (1→2→3) để telemetry phân biệt lần bám mới, và reset cả guard.

**Ai gọi detector**: khác MOT ở chỗ này, nên nói rõ. Khi `sot.enabled` và
`detect_every_frame: false`, `pipeline.py` **không** gọi `detector.detect()` vô điều
kiện nữa; nó truyền `self.detector.detect` xuống làm callback `detect_fn`, và
`SotTracker` tự quyết định frame nào cần chạy (frame `acquire`, frame verify của
guard, frame reacquire). `ctx.detections` = kết quả của frame đó, **rỗng ở các frame
không detect** — đó là lý do HUD mất box xám mờ. Đặt `detect_every_frame: true` thì
pipeline gọi detector mỗi frame như MOT và truyền kết quả sẵn có xuống, `SotTracker`
không gọi thêm lần nào (không detect 2 lần trên cùng frame).

### Xung đột streaming: `DeferredSinkWriter`

Script bên MCITrack khi motion gate cắt thì **cắt lui về frame đầu chuỗi** — giữ
frame trong `pending`, chưa ghi video. Pipeline là stream: frame đã ghi vào
`VideoWriter` không lấy lại được.

**Chọn cách B**: pipeline giữ `motion.k - 1` frame cuối trước khi ghi sink; khi
guard cắt lui thì xoá `tracks` của các frame đang giữ.

- `motion.k = 2` → trễ đúng **1 frame (33 ms @30fps)**, drone không cảm nhận được.
- **0 frame box sai lọt ra output** (cách A báo muộn sẽ để lọt 1 frame).
- Kết quả trùng khít với các baseline đã đo; cách A làm số đo lệch nhẹ, mất khả
  năng so sánh.
- Lớp này **chỉ bật khi `guard.enabled and guard.motion.enabled`**. Đường MOT và
  đường guard-off không đi qua nó → giống hệt hôm nay.

## Output

### `sinks/sot_result.py` — sink mới, 7 cột

```
frame,x,y,w,h,conf,alive        # x,y,w,h = xywh góc trên-trái, pixel ảnh gốc
1,-1,-1,-1,-1,-1,0              # acquire: chưa có box
2,-1,-1,-1,-1,-1,0              # acquire
3,949.90,576.50,93.50,126.90,0.9260,1    # frame init: box + conf của DETECTOR
4,952.11,578.02,93.18,126.44,0.8734,1    # từ đây: box + best_score của MCITrack
56,-1,-1,-1,-1,-1,0             # LOST
```

- `frame` **1-indexed** (`meta.idx + 1`) — khớp GT VisDrone và khớp
  `eval_mot_visdrone.py:39`.
- Số dòng luôn bằng số frame (kể cả acquire/lost) → align theo frame id không lệch.
- Ghi + flush **từng frame**, không buffer tới `close()`. An toàn vì
  `DeferredSinkWriter` cắt lui **trước** khi sink thấy ctx → sink không bao giờ
  phải rút lại dòng đã ghi. Crash giữa đường vẫn còn kết quả.
- Dòng LOST dùng `-1` giống script cũ.

### HUD + telemetry

Không sửa `contracts.py`. Thêm một dòng vào `extra_stats` (dùng cơ chế `"motion"`
sẵn có ở `pipeline.py:220`):

```
SOT  tracking #1  conf 0.83
SOT  acquire (3 frame, 0 box)
SOT  LOST @56  jump 337px area x2.8
```

`TelemetrySink`: thêm `"sot"` vào record jsonl (cạnh `"motion"`, `telemetry.py:54`)
và một cột `sot` vào CSV. Thuần bổ sung — field cũ không đổi tên, không đổi thứ tự.

`scripts/eval_mot_visdrone.py` **không bị đụng tới**.

## Xử lý lỗi — chết sớm, chết rõ

Load checkpoint 1.44 GB mất ~20 s. Mọi kiểm tra dưới đây chạy **trước** khi load:

| tình huống | hành vi |
|---|---|
| `tracker.enabled` và `sot.enabled` cùng true | `sys.exit`, nói rõ tắt cái nào |
| `mcitrack_root` không có `lib/test/evaluation/tracker.py` | `sys.exit` + in path đang tìm + nhắc key `sot.mcitrack_root` |
| `init_bbox` có `w<=0`/`h<=0` hoặc nằm ngoài ảnh | `sys.exit` (MCITrack sẽ crash trong `sample_target` với lỗi không đọc nổi) |
| `init_classes` chứa id không có trong `names.yaml` | cảnh báo, bỏ id đó |
| `device: cuda:1` mà máy chỉ có 1 GPU | `sys.exit` + in số GPU thấy được |
| thiếu checkpoint | `sys.exit` + in đúng đường dẫn `checkpoints/train/mcitrack/<config>/` |
| guard `gate: class` mà tên class không có trong bảng nhóm | hạ xuống `presence` + cảnh báo |
| `source.loop: true` + SOT | cảnh báo: loop làm sequence chạy vòng lại, SOT/guard tính chuyển động sai qua chỗ nối |

## Kiểm chứng

Baseline có thật (đã kiểm tồn tại):

- MOT: `output/mot_eval/uav0000339_00001_v.txt` + `metrics.txt` (MOTA 0.4635)
- SOT guard-off: `MCITrack/output/uav0000161_00000_v_person_..._raw.txt`,
  `uav0000077_00720_v_person_..._raw.txt`
- SOT guard-on: 17 file `*_person_*_guarded.txt` (VisDrone test-dev)
- Baseline 5 object của `uav0000339` **đã bị xoá** → đối chiếu với số ghi trong
  `MCITrack/CLAUDE.md`, không có file để diff.

| # | Kiểm | Tiêu chí đạt |
|---|---|---|
| 1 | **MOT không hồi quy** — chạy lại `local_onnx_batch16.yaml` trên `uav0000339_00001_v` | txt trùng **từng dòng** với `output/mot_eval/`. Quan trọng nhất, vì có sửa `pipeline.py` |
| 2 | **SOT guard=off trùng MCITrack** — `uav0000161_00000_v`, `uav0000077_00720_v` | lệch **0.00 px** mọi frame. Init box để **detector tự sinh**; tuyệt đối không paste lại dòng 1 của txt cũ (chỉ 2 chữ số thập phân — lỗi này đã mắc 2 lần, ra kết luận "nondeterminism" sai) |
| 3 | **guard=on cắt đúng frame cũ** | frame LOST trùng 17 baseline test-dev; và trùng CLAUDE.md: car 56, truck 170, motor 191, bus 70, person không cắt |
| 4 | **DeferredSinkWriter** | frame LOST = frame **đầu chuỗi** (không muộn 1 frame); số dòng txt == số frame; số frame video == số frame input |
| 5 | **acquire** — `init_classes: [8]` (bus) trên `uav0000339` (cả sequence 0 box bus) | không crash, ở `acquire` tới hết, txt toàn `alive=0` |
| 6 | **init_bbox thủ công** | detector không chạy ở frame init (`n_det=0` trong telemetry); box init đúng số truyền vào |
| 7 | **reacquire** — car, `on_lost: reacquire` | sau f56 xuất hiện `track_id=2`; box mới không dính vị trí cũ (chứng minh `h_state` reset) |
| 8 | **follow tích hợp** | `mode` đi acquire→tracking→recover→acquire; `commands.jsonl` coast 15 frame rồi về 0 |
| 9 | **guard hình học — không cần GPU/checkpoint** | chuỗi GT thật (max 44 px/frame, area ×1.44) → không LOST; chuỗi tổng hợp 337 px / ×10.7 → LOST ngay |

Test 9 chạy vài giây, không GPU → dùng làm test hồi quy mỗi lần sửa `guard.py`.
Test 1–8 cần GPU, chạy tay.

```bash
cd /home/anlnm/UAV
MCITrack/.venv/bin/python -m uav_pipeline.scripts.run_pipeline \
    --config uav_pipeline/configs/sot_mcitrack.yaml \
    --source uav_pipeline/input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v \
    --source-type image_dir
```

## Ngoài phạm vi

- **Không** sửa file nào trong `MCITrack/lib/`.
- **Không** vendor code MCITrack vào `_vendor/`.
- **Không** train lại gì.
- **Không** recalibrate 3 ngưỡng guard. Đang có vấn đề đã biết: trên 15 video
  test-dev guard cắt **đúng 4 / oan 6**, và nguyên nhân **không phải** box nhỏ
  (đã bác bỏ bằng số: cắt oan ở 10823 px² trong khi 488 px² lại sạch 179/179).
  Ba ngưỡng hiện tại calibrate trên 5 object của **một** sequence val. Việc
  recalibrate là task riêng, không nằm trong lần tích hợp này.
- **Không** đổi `UPH.UAV=0.88` (ngưỡng xoá hidden state Mamba). Trên footage đêm
  VisDrone nó xoá 338/349 frame, gần như tắt đóng góp chính của paper — nhưng đổi
  sẽ làm mọi số đo cũ không so được nữa. Task riêng.
- **Không** thêm `scripts/eval_sot_visdrone.py` (đã cân nhắc, chốt bỏ ngoài phạm vi).

## Danh sách file

**Thêm mới**

```
sot/__init__.py
sot/mcitrack_wrapper.py
sot/class_groups.py
sot/guard.py
sot/tracker.py
sinks/sot_result.py
configs/sot_mcitrack.yaml
```

**Sửa**

```
config.py            + SotCfg / SotGuardCfg / SotResultSinkCfg; + tracker.enabled; validation loại trừ
pipeline.py           rẽ nhánh SOT/MOT; DeferredSinkWriter; extra_stats["sot"]
sinks/__init__.py    + SotResultSink
sinks/telemetry.py   + field "sot" (jsonl + CSV)
scripts/run_pipeline.py  + --sot / --no-sot / --init-bbox
configs/default.yaml  + block sot:; tracker.enabled: false; detector → onnx + yolox + best_yoloxx.onnx
README.md            mục SOT: cách bật, yêu cầu env, giới hạn của guard
```

**Không đụng**: `track/*`, `detect/*`, `follow/*`, `ocr/*`, `sources/*`,
`scripts/eval_mot_visdrone.py`, `_vendor/*`.
