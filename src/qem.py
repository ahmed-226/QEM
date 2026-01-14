"""
Quadric Error Metrics (QEM) Implementation
==========================================

Core implementation of the Quadric Error Metrics algorithm for 
computing vertex error quadrics and optimal collapse positions.

Based on: "Surface Simplification Using Quadric Error Metrics"
by Michael Garland and Paul S. Heckbert (SIGGRAPH 1997)
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class QuadricErrorMetrics:
    """
    Implements Quadric Error Metrics for mesh simplification.
    
    The fundamental quadric Q for a plane ax + by + cz + d = 0 is the 4x4 matrix:
    Q = p * p^T where p = [a, b, c, d]^T
    
    The error of a vertex v = [x, y, z, 1]^T with respect to Q is:
    error(v) = v^T * Q * v
    
    When collapsing an edge (v1, v2) -> v_new, the combined quadric is:
    Q_new = Q1 + Q2
    """
    
    def __init__(self, boundary_weight: float = 10.0):
        """
        Initialize QEM calculator.
        
        Args:
            boundary_weight: Weight multiplier for boundary edge quadrics.
                            Higher values preserve boundaries better.
        """
        self.boundary_weight = boundary_weight
        
    def compute_face_plane(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute the plane equation coefficients for a triangle face.
        
        The plane equation is: ax + by + cz + d = 0
        where [a, b, c] is the unit normal and d = -dot(normal, point_on_plane)
        
        Args:
            v0, v1, v2: Triangle vertices as 3D points
            
        Returns:
            Plane coefficients [a, b, c, d] where [a,b,c] is unit normal
        """
        # Compute edge vectors
        e1 = v1 - v0
        e2 = v2 - v0
        
        # Compute normal via cross product
        normal = np.cross(e1, e2)
        norm_length = np.linalg.norm(normal)
        
        if norm_length < 1e-12:
            # Degenerate triangle
            return np.zeros(4)
        
        # Normalize
        normal = normal / norm_length
        
        # Compute d = -dot(n, v0)
        d = -np.dot(normal, v0)
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def compute_fundamental_quadric(self, plane: np.ndarray) -> np.ndarray:
        """
        Compute the fundamental error quadric for a plane.
        
        Q = p * p^T where p = [a, b, c, d]
        
        Args:
            plane: Plane coefficients [a, b, c, d]
            
        Returns:
            4x4 symmetric matrix Q
        """
        return np.outer(plane, plane)
    
    def compute_vertex_quadrics(self, vertices: np.ndarray, faces: np.ndarray,
                                 boundary_edges: Optional[set] = None) -> np.ndarray:
        """
        Compute initial error quadrics for all vertices.
        
        For each vertex, the quadric is the sum of fundamental quadrics
        of all faces incident to that vertex.
        
        Args:
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of face indices
            boundary_edges: Optional set of boundary edge tuples for weighting
            
        Returns:
            (N, 4, 4) array of vertex quadrics
        """
        n_vertices = len(vertices)
        quadrics = np.zeros((n_vertices, 4, 4))
        
        # Add quadrics from each face
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            plane = self.compute_face_plane(v0, v1, v2)
            Q = self.compute_fundamental_quadric(plane)
            
            # Add to each vertex of the face
            for vi in face:
                quadrics[vi] += Q
        
        # Add boundary constraint quadrics if provided
        if boundary_edges is not None:
            self._add_boundary_quadrics(vertices, faces, boundary_edges, quadrics)
        
        return quadrics
    
    def _add_boundary_quadrics(self, vertices: np.ndarray, faces: np.ndarray,
                                boundary_edges: set, quadrics: np.ndarray):
        """
        Add weighted quadrics for boundary edges to preserve mesh boundaries.
        
        For each boundary edge, we create a perpendicular plane constraint
        that helps preserve the boundary during simplification.
        """
        # Get face normals for boundary plane computation
        face_normals = {}
        for fi, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                face_normals[fi] = normal / norm_len
            else:
                face_normals[fi] = np.array([0, 0, 1])
        
        # Build edge to face map
        edge_to_face = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge not in edge_to_face:
                    edge_to_face[edge] = []
                edge_to_face[edge].append(fi)
        
        for edge in boundary_edges:
            v0_idx, v1_idx = edge
            v0, v1 = vertices[v0_idx], vertices[v1_idx]
            
            # Get adjacent face normal
            edge_key = tuple(sorted(edge))
            if edge_key in edge_to_face and len(edge_to_face[edge_key]) > 0:
                face_normal = face_normals[edge_to_face[edge_key][0]]
            else:
                face_normal = np.array([0, 0, 1])
            
            # Create perpendicular plane through the boundary edge
            edge_vec = v1 - v0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-12:
                continue
            edge_vec = edge_vec / edge_len
            
            # Plane normal is perpendicular to both edge and face normal
            plane_normal = np.cross(edge_vec, face_normal)
            norm_len = np.linalg.norm(plane_normal)
            if norm_len < 1e-12:
                continue
            plane_normal = plane_normal / norm_len
            
            # Plane through edge midpoint
            midpoint = (v0 + v1) / 2
            d = -np.dot(plane_normal, midpoint)
            
            plane = np.array([plane_normal[0], plane_normal[1], plane_normal[2], d])
            Q = self.compute_fundamental_quadric(plane) * self.boundary_weight
            
            quadrics[v0_idx] += Q
            quadrics[v1_idx] += Q
    
    def compute_optimal_position(self, Q: np.ndarray, v1: np.ndarray, 
                                  v2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the optimal position for an edge collapse.
        
        Tries to find position that minimizes v^T * Q * v by solving:
        [Q[0:3, 0:3]  Q[0:3, 3] ] [x]   [0]
        [    0    0    0      1 ] [1] = [1]
        
        If the matrix is singular, falls back to testing endpoints and midpoint.
        
        Args:
            Q: Combined 4x4 quadric matrix
            v1, v2: Edge endpoint positions
            
        Returns:
            Tuple of (optimal_position, error)
        """
        # Try to solve for optimal position
        A = Q.copy()
        A[3, :] = [0, 0, 0, 1]
        b = np.array([0, 0, 0, 1])
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Check if matrix is well-conditioned
                cond = np.linalg.cond(A)
                if cond < 1e10:
                    v_opt = np.linalg.solve(A, b)
                    error = self.compute_error(Q, v_opt[:3])
                    return v_opt[:3], error
        except np.linalg.LinAlgError:
            pass
        
        # Fallback: test endpoints and midpoint
        candidates = [v1, v2, (v1 + v2) / 2]
        best_pos = v1
        best_error = float('inf')
        
        for pos in candidates:
            error = self.compute_error(Q, pos)
            if error < best_error:
                best_error = error
                best_pos = pos
        
        return best_pos.copy(), best_error
    
    def compute_error(self, Q: np.ndarray, v: np.ndarray) -> float:
        """
        Compute the quadric error for a vertex position.
        
        error = v^T * Q * v where v is [x, y, z, 1]
        
        Args:
            Q: 4x4 quadric matrix
            v: 3D vertex position
            
        Returns:
            Quadric error value
        """
        v_homo = np.array([v[0], v[1], v[2], 1.0])
        error = v_homo @ Q @ v_homo
        return max(0.0, error)  # Clamp to non-negative
    
    def compute_edge_collapse_error(self, Q1: np.ndarray, Q2: np.ndarray,
                                     v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the error and optimal position for collapsing edge (v1, v2).
        
        Args:
            Q1, Q2: Quadrics of the two edge vertices
            v1, v2: Positions of the two edge vertices
            
        Returns:
            Tuple of (optimal_position, error)
        """
        Q_combined = Q1 + Q2
        return self.compute_optimal_position(Q_combined, v1, v2)
