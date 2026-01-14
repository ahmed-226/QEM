"""
Mesh Evaluation Module
======================

Provides quantitative evaluation metrics for mesh simplification:
- Hausdorff distance
- Chamfer distance
- Vertex/Face count statistics
- Boundary preservation metrics
- Comparison with baseline methods
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import trimesh
from scipy.spatial import cKDTree
import time


class MeshEvaluator:
    """
    Evaluation tools for assessing mesh simplification quality.
    
    Provides both geometric distance metrics and topological statistics.
    """
    
    def __init__(self, sample_points: int = 10000):
        """
        Initialize evaluator.
        
        Args:
            sample_points: Number of points to sample for distance metrics
        """
        self.sample_points = sample_points
    
    def compute_all_metrics(self, original: trimesh.Trimesh,
                            simplified: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            original: Original high-resolution mesh
            simplified: Simplified mesh
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Count statistics
        metrics['original_faces'] = len(original.faces)
        metrics['simplified_faces'] = len(simplified.faces)
        metrics['original_vertices'] = len(original.vertices)
        metrics['simplified_vertices'] = len(simplified.vertices)
        metrics['face_reduction_ratio'] = len(simplified.faces) / len(original.faces)
        metrics['vertex_reduction_ratio'] = len(simplified.vertices) / len(original.vertices)
        
        # Geometric metrics
        hausdorff, hausdorff_forward, hausdorff_backward = self.hausdorff_distance(
            original, simplified
        )
        metrics['hausdorff_distance'] = hausdorff
        metrics['hausdorff_forward'] = hausdorff_forward
        metrics['hausdorff_backward'] = hausdorff_backward
        
        chamfer = self.chamfer_distance(original, simplified)
        metrics['chamfer_distance'] = chamfer
        
        # Mesh quality metrics
        metrics['simplified_volume'] = float(simplified.volume) if simplified.is_watertight else np.nan
        metrics['original_volume'] = float(original.volume) if original.is_watertight else np.nan
        
        if not np.isnan(metrics['original_volume']) and not np.isnan(metrics['simplified_volume']):
            metrics['volume_error'] = abs(metrics['simplified_volume'] - metrics['original_volume']) / \
                                      max(abs(metrics['original_volume']), 1e-10)
        else:
            metrics['volume_error'] = np.nan
        
        # Surface area
        metrics['simplified_area'] = float(simplified.area)
        metrics['original_area'] = float(original.area)
        metrics['area_error'] = abs(metrics['simplified_area'] - metrics['original_area']) / \
                                max(metrics['original_area'], 1e-10)
        
        # Boundary metrics
        boundary_metrics = self.boundary_preservation_metrics(original, simplified)
        metrics.update(boundary_metrics)
        
        return metrics
    
    def hausdorff_distance(self, mesh1: trimesh.Trimesh, 
                           mesh2: trimesh.Trimesh) -> Tuple[float, float, float]:
        """
        Compute symmetric Hausdorff distance between two meshes.
        
        Uses point sampling on the mesh surfaces.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            
        Returns:
            Tuple of (symmetric_hausdorff, forward, backward) distances
        """
        # Sample points on both meshes
        try:
            points1 = mesh1.sample(self.sample_points)
            points2 = mesh2.sample(self.sample_points)
        except Exception:
            # Fallback to vertices if sampling fails
            points1 = mesh1.vertices
            points2 = mesh2.vertices
        
        # Build KD-trees
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Forward distance (mesh1 -> mesh2)
        distances_forward, _ = tree2.query(points1)
        hausdorff_forward = float(np.max(distances_forward))
        
        # Backward distance (mesh2 -> mesh1)
        distances_backward, _ = tree1.query(points2)
        hausdorff_backward = float(np.max(distances_backward))
        
        # Symmetric Hausdorff
        hausdorff_symmetric = max(hausdorff_forward, hausdorff_backward)
        
        return hausdorff_symmetric, hausdorff_forward, hausdorff_backward
    
    def chamfer_distance(self, mesh1: trimesh.Trimesh,
                         mesh2: trimesh.Trimesh) -> float:
        """
        Compute symmetric Chamfer distance between two meshes.
        
        Chamfer distance is the average of nearest-neighbor distances.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            
        Returns:
            Chamfer distance value
        """
        # Sample points on both meshes
        try:
            points1 = mesh1.sample(self.sample_points)
            points2 = mesh2.sample(self.sample_points)
        except Exception:
            points1 = mesh1.vertices
            points2 = mesh2.vertices
        
        # Build KD-trees
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Forward distance (mesh1 -> mesh2)
        distances_forward, _ = tree2.query(points1)
        chamfer_forward = float(np.mean(distances_forward ** 2))
        
        # Backward distance (mesh2 -> mesh1)
        distances_backward, _ = tree1.query(points2)
        chamfer_backward = float(np.mean(distances_backward ** 2))
        
        # Symmetric Chamfer (sum of both directions)
        return chamfer_forward + chamfer_backward
    
    def mean_surface_distance(self, mesh1: trimesh.Trimesh,
                              mesh2: trimesh.Trimesh) -> float:
        """
        Compute mean surface distance from mesh1 to mesh2.
        
        Args:
            mesh1: Source mesh
            mesh2: Target mesh
            
        Returns:
            Mean distance value
        """
        try:
            points1 = mesh1.sample(self.sample_points)
            points2 = mesh2.sample(self.sample_points)
        except Exception:
            points1 = mesh1.vertices
            points2 = mesh2.vertices
        
        tree2 = cKDTree(points2)
        distances, _ = tree2.query(points1)
        
        return float(np.mean(distances))
    
    def boundary_preservation_metrics(self, original: trimesh.Trimesh,
                                       simplified: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute metrics for boundary preservation.
        
        Args:
            original: Original mesh
            simplified: Simplified mesh
            
        Returns:
            Dictionary of boundary metrics
        """
        metrics = {}
        
        # Get boundary edges
        orig_boundaries = self._get_boundary_edges(original)
        simp_boundaries = self._get_boundary_edges(simplified)
        
        metrics['original_boundary_edges'] = len(orig_boundaries)
        metrics['simplified_boundary_edges'] = len(simp_boundaries)
        
        # Compute total boundary length
        orig_length = self._compute_boundary_length(original, orig_boundaries)
        simp_length = self._compute_boundary_length(simplified, simp_boundaries)
        
        metrics['original_boundary_length'] = orig_length
        metrics['simplified_boundary_length'] = simp_length
        
        if orig_length > 0:
            metrics['boundary_length_change'] = abs(simp_length - orig_length) / orig_length
        else:
            metrics['boundary_length_change'] = 0.0
        
        # Check if meshes are closed
        metrics['original_is_watertight'] = int(original.is_watertight)
        metrics['simplified_is_watertight'] = int(simplified.is_watertight)
        
        return metrics
    
    def _get_boundary_edges(self, mesh: trimesh.Trimesh) -> set:
        """Find boundary edges of a mesh."""
        edge_count = {}
        for face in mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        return {edge for edge, count in edge_count.items() if count == 1}
    
    def _compute_boundary_length(self, mesh: trimesh.Trimesh, 
                                  boundary_edges: set) -> float:
        """Compute total length of boundary edges."""
        total_length = 0.0
        for edge in boundary_edges:
            v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
            total_length += np.linalg.norm(v2 - v1)
        return total_length
    
    def compare_methods(self, original: trimesh.Trimesh,
                        target_ratio: float,
                        methods: Dict[str, callable]) -> Dict[str, Dict]:
        """
        Compare different simplification methods.
        
        Args:
            original: Original mesh
            target_ratio: Target face ratio
            methods: Dictionary of method names to simplification functions
                    Each function takes (mesh, target_ratio) and returns simplified mesh
            
        Returns:
            Dictionary of method names to their metrics and runtime
        """
        results = {}
        
        for name, method_fn in methods.items():
            print(f"Running {name}...")
            
            start_time = time.time()
            try:
                simplified = method_fn(original, target_ratio)
                runtime = time.time() - start_time
                
                metrics = self.compute_all_metrics(original, simplified)
                metrics['runtime'] = runtime
                metrics['success'] = True
                
                results[name] = {
                    'mesh': simplified,
                    'metrics': metrics
                }
                
            except Exception as e:
                results[name] = {
                    'mesh': None,
                    'metrics': {'success': False, 'error': str(e)}
                }
        
        return results
    
    def generate_report(self, metrics: Dict[str, float], 
                        method_name: str = "QEM") -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            metrics: Dictionary of metric values
            method_name: Name of the simplification method
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            f"Mesh Simplification Report - {method_name}",
            "=" * 60,
            "",
            "MESH STATISTICS",
            "-" * 40,
            f"  Original:    {metrics.get('original_faces', 'N/A'):>8} faces, "
            f"{metrics.get('original_vertices', 'N/A'):>8} vertices",
            f"  Simplified:  {metrics.get('simplified_faces', 'N/A'):>8} faces, "
            f"{metrics.get('simplified_vertices', 'N/A'):>8} vertices",
            f"  Reduction:   {metrics.get('face_reduction_ratio', 0)*100:>7.2f}% of original faces",
            "",
            "GEOMETRIC ACCURACY",
            "-" * 40,
            f"  Hausdorff Distance:    {metrics.get('hausdorff_distance', np.nan):>12.6f}",
            f"    Forward:             {metrics.get('hausdorff_forward', np.nan):>12.6f}",
            f"    Backward:            {metrics.get('hausdorff_backward', np.nan):>12.6f}",
            f"  Chamfer Distance:      {metrics.get('chamfer_distance', np.nan):>12.6f}",
            "",
            "MESH PROPERTIES",
            "-" * 40,
        ]
        
        vol_error = metrics.get('volume_error', np.nan)
        if not np.isnan(vol_error):
            lines.append(f"  Volume Error:          {vol_error*100:>11.4f}%")
        else:
            lines.append(f"  Volume Error:          N/A (non-watertight)")
        
        lines.extend([
            f"  Area Error:            {metrics.get('area_error', 0)*100:>11.4f}%",
            "",
            "BOUNDARY PRESERVATION",
            "-" * 40,
            f"  Original Boundaries:   {metrics.get('original_boundary_edges', 0):>8} edges",
            f"  Simplified Boundaries: {metrics.get('simplified_boundary_edges', 0):>8} edges",
            f"  Length Change:         {metrics.get('boundary_length_change', 0)*100:>11.4f}%",
            "",
            "TOPOLOGY",
            "-" * 40,
            f"  Original Watertight:   {'Yes' if metrics.get('original_is_watertight') else 'No'}",
            f"  Simplified Watertight: {'Yes' if metrics.get('simplified_is_watertight') else 'No'}",
            "",
            "=" * 60,
        ])
        
        if 'runtime' in metrics:
            lines.insert(-1, f"  Runtime:               {metrics['runtime']:>11.4f} seconds")
        
        return "\n".join(lines)
    
    def print_report(self, metrics: Dict[str, float], method_name: str = "QEM"):
        """Print the evaluation report to console."""
        print(self.generate_report(metrics, method_name))


def sample_mesh_surface(mesh: trimesh.Trimesh, n_samples: int) -> np.ndarray:
    """
    Sample points uniformly on mesh surface.
    
    Args:
        mesh: Mesh to sample from
        n_samples: Number of samples
        
    Returns:
        (N, 3) array of sample points
    """
    try:
        return mesh.sample(n_samples)
    except:
        # Fallback: return vertices
        if len(mesh.vertices) >= n_samples:
            indices = np.random.choice(len(mesh.vertices), n_samples, replace=False)
            return mesh.vertices[indices]
        return mesh.vertices
