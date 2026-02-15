"""Shared utilities for config resolution and geometry (e.g. IoU)."""

from .config import resolve_placeholders, lookup_path
from .boxes import calculate_iou

__all__ = ["resolve_placeholders", "lookup_path", "calculate_iou"]
