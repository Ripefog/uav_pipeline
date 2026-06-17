"""OCR layer — license-plate recognition (fast-plate-ocr)."""
from .plate_ocr import PlateOCR, PlateVoter, lower_third_crop

__all__ = ["PlateOCR", "PlateVoter", "lower_third_crop"]
