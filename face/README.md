# face/ — face detection + recognition

Detection (SCRFD) + recognition (MobileFaceNet + turbovec gallery). Not yet
wired into `uav_pipeline.Pipeline` / `config.py` / `contracts.py` — this
folder is self-contained and runnable on its own for now.

## Layout

```
face/
├── __init__.py          # exports FaceDetector, FaceRecognizer, FaceDB, FaceVoter
├── detector.py           # FaceDetector (SCRFD)
├── recognizer.py          # FaceRecognizer, FaceDB, FaceVoter
├── build_index.py         # offline gallery builder (CLI)
├── pipeline.py             # standalone demo (CLI)
└── examples/face-data/    # shipped example gallery (face_index.tvim + .json)

weights/
├── scrfd_2.5g_bnkps.onnx  # detector.DEFAULT_MODEL
└── mobilefacenet.onnx     # recognizer.DEFAULT_MODEL
```

## Files

| File | What it is | How to run it |
|---|---|---|
| `detector.py` | `FaceDetector` — SCRFD ONNX face detector (own stride/anchor decode, not the `detect/backends` YOLO pipeline). Library only, no CLI. | `from uav_pipeline.face.detector import FaceDetector` |
| `recognizer.py` | `FaceRecognizer` (MobileFaceNet embedding), `FaceDB` (turbovec vector gallery: add/search/save/load), `FaceVoter` (per-track majority vote, mirrors `ocr.PlateVoter`). Library only, no CLI. | `from uav_pipeline.face.recognizer import FaceRecognizer, FaceDB, FaceVoter` |
| `build_index.py` | Offline gallery builder: scans `dataset/<person_name>/<image>` folders, embeds every image, writes `face_index.tvim` + `face_index.json`. | `python -m uav_pipeline.face.build_index --dataset /path/to/VN-celeb [--model weights/mobilefacenet.onnx] [--out face/examples/face-data/face_index]` |
| `pipeline.py` | Standalone demo/CLI: detect -> embed -> search gallery -> draw + print results on a single image. | `python -m uav_pipeline.face.pipeline <image_path> [k] [--no-detect]` |

## Typical order of operations

1. Build the gallery once (needs a labeled face dataset, not shipped in this repo):
   ```bash
   python -m uav_pipeline.face.build_index --dataset /path/to/VN-celeb
   ```
   Writes `face_index.tvim` + `face_index.json` (default base path:
   `recognizer.DEFAULT_INDEX` = `face/examples/face-data/face_index`, where
   the shipped example gallery already lives).

2. Try it on one image:
   ```bash
   python -m uav_pipeline.face.pipeline photo.jpg 3            # detect + search top-3
   python -m uav_pipeline.face.pipeline cropped_face.jpg 1 --no-detect  # skip detection, image is already a face crop
   ```
   Prints matches (`sim`, `person_name`, source `image_file`) and saves
   `photo_result.jpg` with boxes + top-1 label drawn.

## Models

`scrfd_2.5g_bnkps.onnx` (detector) and `mobilefacenet.onnx` (recognizer) live
in `weights/`, resolved via `DEFAULT_MODEL` in each module — pass
`model_path=` explicitly to override. `DEFAULT_INDEX` in `recognizer.py` is
the base path (no extension) for the gallery's `.tvim`/`.json` pair.

## Dependencies

`onnxruntime`, `opencv-python`, `numpy` (already in `requirements.txt`), plus
`turbovec` (pinned in `requirements.txt` under the Face section) — only
imported when `FaceDB` is actually constructed/loaded.
