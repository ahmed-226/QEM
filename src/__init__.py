"""
High-Quality Mesh Decimation using Quadric Error Metrics (QEM)
===============================================================

A Python implementation of the classic mesh simplification algorithm
from "Surface Simplification Using Quadric Error Metrics" 
by Michael Garland and Paul S. Heckbert (SIGGRAPH 1997).
"""

from .qem import QuadricErrorMetrics
from .mesh_decimator import MeshDecimator
from .visualization import MeshVisualizer
from .evaluation import MeshEvaluator

__version__ = "1.0.0"
__author__ = "Mesh Decimation Project"
__all__ = ["QuadricErrorMetrics", "MeshDecimator", "MeshVisualizer", "MeshEvaluator"]
