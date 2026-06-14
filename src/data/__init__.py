"""Training-data utilities."""

from .cfd_supervision import CFDSupervisionPool, build_cavity_cfd_supervision

__all__ = ["CFDSupervisionPool", "build_cavity_cfd_supervision"]
