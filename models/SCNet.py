"""Legacy public module entry point for SC-Net.

The implementation is kept in :mod:`models.SCUNet` for checkpoint and
experiment compatibility. New code should import ``Model`` from
:mod:`models.DAUNet` instead.
"""

from .SCUNet import Model

__all__ = ['Model']
