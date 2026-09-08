"""Public module entry point for DAU-Net.

The implementation remains in :mod:`models.SCUNet` so that existing imports
and checkpoints remain compatible with earlier releases.
"""

from .SCUNet import Model

__all__ = ['Model']
