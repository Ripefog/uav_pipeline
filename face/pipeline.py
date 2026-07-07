"""Face pipeline: detection (optional) -> recognition -> index search.

Standalone demo/CLI for the face/ module in isolation (not yet wired into the
main uav_pipeline.Pipeline). Usage:

    python -m uav_pipeline.face.pipeline <image_path> [k] [--no-detect]

Examples:
    python -m uav_pipeline.face.pipeline photo.jpg 3            # detect + search top-3
    python -m uav_pipeline.face.pipeline cropped_face.jpg 1 --no-detect  # skip detection
"""
import sys
import time

import cv2
import numpy as np

from .detector import FaceDetector
from .recognizer import DEFAULT_INDEX, FaceDB, FaceRecognizer


class FacePipeline:
    def __init__(self, use_detection: bool = True, index_path: str = DEFAULT_INDEX):
        self.use_detection = use_detection
        self.recognizer = FaceRecognizer()
        self.db = FaceDB.load(index_path)
        if use_detection:
            self.detector = FaceDetector()

    def run(self, img: np.ndarray, k: int = 1) -> list:
        """
        Input : BGR numpy array (camera frame or cv2.imread output)
        Output: list of dicts:
            {
              "box": [x1, y1, x2, y2] or None,
              "det_score": float,
              "matches": [(score, person_name, image_file), ...]
            }
        """
        if self.use_detection:
            faces = self.detector.detect(img)
        else:
            faces = [{"box": None, "score": 1.0, "crop": img}]

        results = []
        for face in faces:
            embedding = self.recognizer.get_embedding(face["crop"])
            matches = self.db.search(embedding, k=k)
            results.append({
                "box": face["box"],
                "det_score": face["score"],
                "matches": matches,
            })

        return results


def draw_results(img: np.ndarray, results: list) -> np.ndarray:
    """
    Draw bounding boxes and top-1 person_name label on a copy of img.
    Returns annotated BGR image.
    """
    out = img.copy()
    for face in results:
        box = face["box"]
        if not box or not face["matches"]:
            continue
        x1, y1, x2, y2 = box
        sim, person_name, _ = face["matches"][0]
        label = f"{person_name} ({sim:.2f})"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        bg_y1 = max(y1 - th - 8, 0)
        cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


if __name__ == "__main__":
    import os

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    image_path = args[0]
    k = int(args[1]) if len(args) > 1 and not args[1].startswith("--") else 1
    use_detection = "--no-detect" not in args

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read {image_path}")
        sys.exit(1)

    print(f"Image      : {image_path}")
    print(f"Detection  : {'on' if use_detection else 'off'}")
    print(f"Top-k      : {k}")
    print("-" * 50)

    pipeline = FacePipeline(use_detection=use_detection)

    t0 = time.perf_counter()
    results = pipeline.run(img, k=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Search time: {elapsed_ms:.2f} ms")
    print(f"Faces found: {len(results)}")
    print()

    for i, face in enumerate(results, start=1):
        box = face["box"]
        box_str = f"[{box[0]},{box[1]},{box[2]},{box[3]}]" if box else "N/A"
        print(f"Face #{i}  box={box_str}  det_score={face['det_score']:.3f}")
        for rank, (score, person_name, image_file) in enumerate(face["matches"], start=1):
            print(f"  #{rank}  sim={score:.4f}  person={person_name}  file={image_file}")

    # Save annotated image
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_result{ext}"
    annotated = draw_results(img, results)
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved: {out_path}")
