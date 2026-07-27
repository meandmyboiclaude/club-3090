"""Bounded hardware scenarios for profile/compose compatibility checks."""

CANONICAL_SCENARIOS = [
    {"name": "2x-3090-pcie", "hardware": ["rtx-3090", "rtx-3090"], "nvlink_active": False},
    {"name": "2x-3090-nvlink", "hardware": ["rtx-3090", "rtx-3090"], "nvlink_active": True},
    {"name": "4x-3090-pcie", "hardware": ["rtx-3090", "rtx-3090", "rtx-3090", "rtx-3090"], "nvlink_active": False},
    {"name": "1x-3090", "hardware": ["rtx-3090"], "nvlink_active": False},
    {"name": "2x-4090", "hardware": ["rtx-4090", "rtx-4090"], "nvlink_active": False},
    {"name": "1x-5090", "hardware": ["rtx-5090"], "nvlink_active": False},
    {"name": "2x-5090", "hardware": ["rtx-5090", "rtx-5090"], "nvlink_active": False},
    {"name": "2x-a100-40gb", "hardware": ["a100-40gb", "a100-40gb"], "nvlink_active": True},
    {"name": "heterogeneous-3090-5090", "hardware": ["rtx-3090", "rtx-5090"], "nvlink_active": False},
]
