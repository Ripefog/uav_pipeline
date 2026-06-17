"""run_pipeline — launch the full Detect/Track/Follow pipeline from a YAML config.

Examples:
    # from D:\\UAV\\code
    python -m uav_pipeline.scripts.run_pipeline --config uav_pipeline/configs/windows_openvino.yaml
    python -m uav_pipeline.scripts.run_pipeline --config ... --max-frames 200 --no-video
"""
import argparse
import os
import sys

# Allow running both as `python -m uav_pipeline.scripts.run_pipeline` (from code
# root) and as a bare `python run_pipeline.py`. Bootstrap the code root on path.
_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from uav_pipeline.config import Config  # noqa: E402
from uav_pipeline.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="UAV Edge Pipeline runner")
    ap.add_argument("--config", "-c", required=True, help="Path to a pipeline YAML config")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    ap.add_argument("--source", default="", help="Override source.path (video file / image dir)")
    ap.add_argument("--source-type", default="", help="Override source.type")
    ap.add_argument("--no-video", action="store_true", help="Disable the annotated video sink")
    ap.add_argument("--no-follow", action="store_true", help="Disable the Follow layer")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.max_frames:
        cfg.source.max_frames = args.max_frames
    if args.source:
        cfg.source.path = args.source
    if args.source_type:
        cfg.source.type = args.source_type
    if args.no_video:
        cfg.sinks.video.enabled = False
    if args.no_follow:
        cfg.follow.enabled = False

    if cfg.source.type in ("video", "image_dir") and not cfg.source.path:
        sys.exit(f"[run_pipeline] source.path is empty for type '{cfg.source.type}'. "
                 f"Set it in {args.config} or pass --source <path>.")

    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
