"""Target selector — chooses which track the follower should chase."""
from typing import List, Optional, Tuple

from ..config import FollowCfg
from ..contracts import Track


class TargetSelector:
    def __init__(self, cfg: FollowCfg):
        self.cfg = cfg

    def select(self, tracks: List[Track], frame_hw: Tuple[int, int]) -> Optional[Track]:
        # Only consider tracks with a current match (age == 0).
        cands = [t for t in tracks if t.age == 0]
        pref = self.cfg.preferred_classes
        if pref:
            cands = [t for t in cands if t.name in pref]
        if not cands:
            return None

        # Sticky lock: keep following the configured id if it is still alive.
        if self.cfg.locked_id is not None:
            for t in cands:
                if t.track_id == self.cfg.locked_id:
                    return t

        policy = self.cfg.default_policy
        if policy == "largest_area":
            return max(cands, key=lambda t: t.area)
        if policy == "nearest_center":
            H, W = frame_hw
            cx, cy = W / 2.0, H / 2.0
            return min(cands, key=lambda t: (t.cx - cx) ** 2 + (t.cy - cy) ** 2)
        # default: highest_score_area
        return max(cands, key=lambda t: t.confidence * max(t.area, 1.0) ** 0.5)
