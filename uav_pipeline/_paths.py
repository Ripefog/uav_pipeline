"""Path bootstrap so we can reuse ``eval_yolo`` / ``eval_ocr`` verbatim.

``eval_yolo`` has no ``__init__.py`` — its scripts import siblings directly
(``from utils import util``). To reuse those modules we put the ``eval_yolo``
directory on ``sys.path`` once, the same way the original infer scripts do.
Importing this module is idempotent and side-effect-free if the dirs are
missing (e.g. on a clean Jetson where only ``uav_pipeline`` was copied).
"""
import os
import sys

# repo layout: D:\UAV\code\uav_pipeline\_paths.py  ->  code root = ../..
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
EVAL_YOLO_DIR = os.path.join(CODE_ROOT, "eval_yolo")
EVAL_OCR_DIR = os.path.join(CODE_ROOT, "eval_ocr")
UAV_PIPELINE_DIR = _THIS_DIR


def ensure_paths():
    """Insert eval_yolo (and code root) on sys.path so `utils`/`nets` resolve."""
    for p in (CODE_ROOT, EVAL_YOLO_DIR):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# Run once at import so any `from utils... import ...` just works afterwards.
ensure_paths()
