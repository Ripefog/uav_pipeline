"""Offline gallery builder — dataset/<person_name>/<image> -> face_index.tvim + .json.

Usage:
    python -m uav_pipeline.face.build_index --dataset /path/to/VN-celeb [--out face_index]
"""
import argparse
import os

import cv2

from .recognizer import DEFAULT_INDEX, DEFAULT_MODEL, FaceDB, FaceRecognizer


def build(dataset_path: str, model_path: str = DEFAULT_MODEL,
          out_path: str = DEFAULT_INDEX, dim: int = 512) -> int:
    recognizer = FaceRecognizer(model_path)
    db = FaceDB(dim=dim)
    total = 0
    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            embedding = recognizer.get_embedding(img)
            db.add(embedding, person_name, img_name)
            total += 1
    db.save(out_path)
    print(f"[build_index] saved {total} faces -> {out_path}.tvim / {out_path}.json")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="root dir with one subfolder per person")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_INDEX, help="base path (no extension)")
    args = ap.parse_args()
    build(args.dataset, args.model, args.out)
