"""Camera Motion Compensation (CMC) for drone MOT.

Faithful port of ``pratap424/visdrone_mot`` ``src/camera_motion.py``: estimates
frame-to-frame camera motion with ORB feature matching + RANSAC, then warps
previous track predictions into the current frame's coordinate system. Also
exposes ``motion_severity`` — the feedback signal for EMAT
(Ego-Motion-Adaptive Tracking) in ``drone_tracker.py``.

Two modes:
  1. affine    (6 DOF) — small rotations/translations/zoom (default, stable)
  2. homography(8 DOF) — large perspective changes (optional)
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class CameraMotionCompensator:
    """ORB-based camera motion compensation for drone tracking."""

    def __init__(self, method: str = "affine", n_features: int = 1000,
                 match_ratio: float = 0.75, ransac_thresh: float = 5.0,
                 min_matches: int = 20, downscale: float = 0.5):
        self.method = method
        self.n_features = n_features
        self.match_ratio = match_ratio          # Lowe's ratio test
        self.ransac_thresh = ransac_thresh
        self.min_matches = min_matches
        self.downscale = downscale

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp = None
        self.prev_des = None

        self.n_matches_history: List[int] = []
        self.warp_history: List[Optional[np.ndarray]] = []

    # ------------------------------------------------------------------ #
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate camera motion prev->current. Returns a 2x3 affine / 3x3
        homography (None if estimation failed)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale != 1.0:
            h, w = gray.shape[:2]
            gray_small = cv2.resize(gray, (int(w * self.downscale), int(h * self.downscale)))
        else:
            gray_small = gray

        kp, des = self.orb.detectAndCompute(gray_small, None)
        warp_matrix = None
        if self.prev_gray is not None and self.prev_des is not None and des is not None:
            if len(kp) >= 10 and len(self.prev_kp) >= 10:
                warp_matrix = self._match_and_estimate(
                    self.prev_kp, self.prev_des, kp, des, gray_small.shape)

        self.prev_gray = gray_small
        self.prev_kp = kp
        self.prev_des = des
        self.warp_history.append(warp_matrix)
        return warp_matrix

    def _match_and_estimate(self, kp1, des1, kp2, des2, shape) -> Optional[np.ndarray]:
        try:
            matches = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return None

        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        self.n_matches_history.append(len(good_matches))
        if len(good_matches) < self.min_matches:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        if self.downscale != 1.0:
            scale = 1.0 / self.downscale
            src_pts *= scale
            dst_pts *= scale

        if self.method == "affine":
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh)
            if M is None:
                return None
            n_inliers = int(np.sum(inliers)) if inliers is not None else 0
            if n_inliers < self.min_matches * 0.4:
                return None
            return M  # 2x3 affine
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            if H is None:
                return None
            n_inliers = int(np.sum(mask)) if mask is not None else 0
            if n_inliers < self.min_matches * 0.4:
                return None
            return H  # 3x3 homography

    # ------------------------------------------------------------------ #
    def warp_bboxes(self, bboxes: np.ndarray, warp_matrix: np.ndarray,
                    frame_shape: Tuple[int, int]) -> np.ndarray:
        """Warp (N,4) [x1,y1,x2,y2] from prev-frame coords to current frame."""
        if bboxes is None or len(bboxes) == 0:
            return bboxes
        h, w = frame_shape[:2]
        n = len(bboxes)
        corners = np.zeros((n * 4, 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            corners[i * 4 + 0] = [x1, y1]
            corners[i * 4 + 1] = [x2, y1]
            corners[i * 4 + 2] = [x2, y2]
            corners[i * 4 + 3] = [x1, y2]
        corners = corners.reshape(-1, 1, 2)
        if warp_matrix.shape == (2, 3):
            warped = cv2.transform(corners, warp_matrix)
        else:
            warped = cv2.perspectiveTransform(corners, warp_matrix)
        warped = warped.reshape(-1, 4, 2)

        out = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            xs = warped[i, :, 0]
            ys = warped[i, :, 1]
            out[i] = [
                max(0, float(np.min(xs))),
                max(0, float(np.min(ys))),
                min(w, float(np.max(xs))),
                min(h, float(np.max(ys))),
            ]
        return out

    def warp_points(self, points: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        if warp_matrix.shape == (2, 3):
            warped = cv2.transform(pts, warp_matrix)
        else:
            warped = cv2.perspectiveTransform(pts, warp_matrix)
        return warped.reshape(-1, 2)

    # ------------------------------------------------------------------ #
    @property
    def avg_matches(self) -> float:
        if not self.n_matches_history:
            return 0.0
        recent = self.n_matches_history[-100:]
        return sum(recent) / len(recent)

    @property
    def success_rate(self) -> float:
        if not self.warp_history:
            return 0.0
        recent = self.warp_history[-100:]
        return sum(1 for w in recent if w is not None) / len(recent)

    def get_stats(self) -> Dict:
        return {
            "method": self.method,
            "avg_matches": round(self.avg_matches, 1),
            "success_rate": f"{self.success_rate * 100:.1f}%",
            "total_frames": len(self.warp_history),
        }

    def motion_severity(self, warp_matrix: Optional[np.ndarray] = None) -> Dict:
        """Decompose the warp into motion components — core of EMAT.

        Returns {translation_px, rotation_deg, scale_change, severity∈[0,1]}.
        """
        if warp_matrix is None:
            warp_matrix = self.warp_history[-1] if self.warp_history else None
        if warp_matrix is None:
            return {"translation_px": 0.0, "rotation_deg": 0.0,
                    "scale_change": 1.0, "severity": 0.0}

        M = warp_matrix[:2, :] if warp_matrix.shape == (3, 3) else warp_matrix
        a, b, tx = M[0]
        c, d, ty = M[1]

        translation_px = float(np.sqrt(tx ** 2 + ty ** 2))
        rotation_deg = float(np.degrees(np.arctan2(c, a)))
        scale_x = float(np.sqrt(a ** 2 + c ** 2))
        scale_y = float(np.sqrt(b ** 2 + d ** 2))
        scale_change = (scale_x + scale_y) / 2.0

        trans_sev = min(1.0, translation_px / 50.0)
        rot_sev = min(1.0, abs(rotation_deg) / 5.0)
        scale_sev = min(1.0, abs(scale_change - 1.0) / 0.1)
        severity = max(trans_sev, rot_sev, scale_sev)
        return {
            "translation_px": round(translation_px, 2),
            "rotation_deg": round(rotation_deg, 3),
            "scale_change": round(scale_change, 4),
            "severity": round(float(severity), 3),
        }

    def reset(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.n_matches_history.clear()
        self.warp_history.clear()
