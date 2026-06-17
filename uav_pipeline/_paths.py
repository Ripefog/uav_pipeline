"""Path bootstrap — makes ``uav_pipeline`` fully self-contained.

The detection layer vendors the YOLO helpers it needs under ``_vendor/``
(``utils/util.py`` for NMS + ``make_anchors``, ``nets/`` for the torch model
defs). Those vendored modules use the *original* import style
(``from utils import util``, ``from utils.util import make_anchors``,
``import nets.nn_v26``), so we put ``_vendor`` on ``sys.path`` once and they
resolve there — **no dependency on any sibling ``eval_yolo`` / ``eval_ocr``
folder**.

This module also exposes package-relative dirs (``PKG_DIR``, ``WEIGHTS_DIR``)
and ``resolve()``, which finds a config path relative to the package root when
it isn't found relative to the current working directory. That lets the bundled
``weights/`` (models + ``names.yaml`` + ``plate_config.yaml``) work no matter
where the pipeline is launched from.
"""
import os
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))          # .../uav_pipeline
VENDOR_DIR = os.path.join(PKG_DIR, "_vendor")
WEIGHTS_DIR = os.path.join(PKG_DIR, "weights")


def ensure_paths():
    """Prepend ``_vendor`` to sys.path so `utils`/`nets` resolve to our copies."""
    if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)


def resolve(path: str) -> str:
    """Resolve a config path to an absolute file.

    Order: absolute path / CWD-relative first, then package-relative (so the
    bundled ``weights/`` is found when run from anywhere). Empty -> "".
    """
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    cwd_rel = os.path.abspath(path)
    if os.path.exists(cwd_rel):
        return cwd_rel
    pkg_rel = os.path.join(PKG_DIR, path)
    if os.path.exists(pkg_rel):
        return pkg_rel
    # Fall back to the package-relative form even if missing (clearer errors).
    return pkg_rel


# Run once at import so any `from utils... import ...` / `import nets...` works.
ensure_paths()
