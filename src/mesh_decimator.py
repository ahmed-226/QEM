"""
Mesh Decimator
==============

Main mesh simplification class that performs iterative edge collapse
using Quadric Error Metrics with a priority queue.
"""

import numpy as np
import heapq
from typing import Dict, Set, Tuple, List, Optional, Callable
from dataclasses import dataclass, field
import trimesh

from .qem import QuadricErrorMetrics


@dataclass(order=True)
class EdgeCollapseCandidate:
    """Priority queue entry for edge collapse candidates."""
    error: float
    edge: Tuple[int, int] = field(compare=False)
    optimal_pos: np.ndarray = field(compare=False)
    version: int = field(compare=False)  # For lazy deletion


class MeshDecimator:
    """
    Mesh simplification using Quadric Error Metrics (QEM).
    
    Implements iterative edge collapse with:
    - Priority queue based on collapse error
    - Lazy updates for efficiency
    - Boundary preservation
    - Manifold edge collapse validation
    """
    
    def __init__(self, boundary_weight: float = 10.0, 
                 preserve_boundaries: bool = True,
                 max_error: Optional[float] = None):
        """
        Initialize the mesh decimator.
        
        Args:
            boundary_weight: Weight for boundary preservation (higher = stronger)
            preserve_boundaries: Whether to add boundary constraints
            max_error: Maximum allowed error per collapse (None = no limit)
        """
        self.boundary_weight = boundary_weight
        self.preserve_boundaries = preserve_boundaries
        self.max_error = max_error
        self.qem = QuadricErrorMetrics(boundary_weight=boundary_weight)
        
        # State variables (initialized per decimation)
        self._vertices: Optional[np.ndarray] = None
        self._faces: Optional[List[List[int]]] = None
        self._quadrics: Optional[np.ndarray] = None
        self._vertex_versions: Optional[Dict[int, int]] = None
        self._vertex_faces: Optional[Dict[int, Set[int]]] = None
        self._edge_candidates: Optional[Dict[Tuple[int, int], EdgeCollapseCandidate]] = None
        self._priority_queue: Optional[List[EdgeCollapseCandidate]] = None
        self._deleted_vertices: Optional[Set[int]] = None
        self._deleted_faces: Optional[Set[int]] = None
        self._boundary_edges: Optional[Set[Tuple[int, int]]] = None
        self._collapse_history: Optional[List[dict]] = None
        
    def decimate(self, mesh: trimesh.Trimesh, 
                 target_faces: Optional[int] = None,
                 target_ratio: Optional[float] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> trimesh.Trimesh:
        """
        Decimate the mesh to target face count or ratio.
        
        Args:
            mesh: Input trimesh object
            target_faces: Target number of faces (mutually exclusive with target_ratio)
            target_ratio: Target ratio of faces to keep (0.0 to 1.0)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Simplified trimesh object
        """
        if target_faces is None and target_ratio is None:
            raise ValueError("Must specify either target_faces or target_ratio")
        
        if target_ratio is not None:
            target_faces = max(4, int(len(mesh.faces) * target_ratio))
        
        # Initialize data structures
        self._initialize(mesh)
        
        initial_faces = len(self._faces) - len(self._deleted_faces)
        faces_to_remove = initial_faces - target_faces
        
        print(f"Starting decimation: {initial_faces} -> {target_faces} faces")
        print(f"  Boundary edges: {len(self._boundary_edges)}")
        
        collapses_done = 0
        last_progress = 0
        
        while self._get_active_face_count() > target_faces:
            # Get best edge to collapse
            candidate = self._pop_best_candidate()
            
            if candidate is None:
                print("No more valid edges to collapse")
                break
            
            # Check max error constraint
            if self.max_error is not None and candidate.error > self.max_error:
                print(f"Reached max error threshold: {candidate.error:.6f} > {self.max_error}")
                break
            
            # Perform the collapse
            success = self._collapse_edge(candidate)
            
            if success:
                collapses_done += 1
                
                # Progress callback
                if progress_callback is not None:
                    progress = collapses_done / max(1, faces_to_remove)
                    if progress - last_progress >= 0.05:  # Update every 5%
                        progress_callback(min(1.0, progress))
                        last_progress = progress
        
        final_faces = self._get_active_face_count()
        print(f"Decimation complete: {final_faces} faces, {collapses_done} collapses")
        
        return self._build_output_mesh()
    
    def _initialize(self, mesh: trimesh.Trimesh):
        """Initialize all data structures for decimation."""
        # Copy vertex and face data
        self._vertices = mesh.vertices.copy()
        self._faces = [list(f) for f in mesh.faces]
        
        # Track deletions
        self._deleted_vertices = set()
        self._deleted_faces = set()
        
        # Vertex versioning for lazy updates
        self._vertex_versions = {i: 0 for i in range(len(self._vertices))}
        
        # Build vertex-face adjacency
        self._vertex_faces = {i: set() for i in range(len(self._vertices))}
        for fi, face in enumerate(self._faces):
            for vi in face:
                self._vertex_faces[vi].add(fi)
        
        # Find boundary edges
        self._boundary_edges = self._find_boundary_edges()
        
        # Compute initial quadrics
        vertices_array = self._vertices
        faces_array = np.array(self._faces)
        
        if self.preserve_boundaries and self._boundary_edges:
            self._quadrics = self.qem.compute_vertex_quadrics(
                vertices_array, faces_array, self._boundary_edges
            )
        else:
            self._quadrics = self.qem.compute_vertex_quadrics(
                vertices_array, faces_array
            )
        
        # Initialize priority queue with all edges
        self._edge_candidates = {}
        self._priority_queue = []
        self._collapse_history = []
        
        self._initialize_edge_queue()
    
    def _find_boundary_edges(self) -> Set[Tuple[int, int]]:
        """Find all boundary edges (edges with only one adjacent face)."""
        edge_count = {}
        
        for face in self._faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        return {edge for edge, count in edge_count.items() if count == 1}
    
    def _initialize_edge_queue(self):
        """Initialize the priority queue with all edges."""
        edges_seen = set()
        
        for face in self._faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edges_seen:
                    edges_seen.add(edge)
                    self._add_edge_candidate(edge)
    
    def _add_edge_candidate(self, edge: Tuple[int, int]):
        """Add or update an edge collapse candidate."""
        v1_idx, v2_idx = edge
        
        if v1_idx in self._deleted_vertices or v2_idx in self._deleted_vertices:
            return
        
        # Compute optimal collapse
        Q1 = self._quadrics[v1_idx]
        Q2 = self._quadrics[v2_idx]
        v1 = self._vertices[v1_idx]
        v2 = self._vertices[v2_idx]
        
        optimal_pos, error = self.qem.compute_edge_collapse_error(Q1, Q2, v1, v2)
        
        # Create candidate with current version
        version = self._vertex_versions[v1_idx] + self._vertex_versions[v2_idx]
        candidate = EdgeCollapseCandidate(
            error=error,
            edge=edge,
            optimal_pos=optimal_pos,
            version=version
        )
        
        self._edge_candidates[edge] = candidate
        heapq.heappush(self._priority_queue, candidate)
    
    def _pop_best_candidate(self) -> Optional[EdgeCollapseCandidate]:
        """Get the best valid edge collapse candidate."""
        while self._priority_queue:
            candidate = heapq.heappop(self._priority_queue)
            edge = candidate.edge
            v1_idx, v2_idx = edge
            
            # Skip if vertices are deleted
            if v1_idx in self._deleted_vertices or v2_idx in self._deleted_vertices:
                continue
            
            # Check if candidate is still valid (version matches)
            current_version = self._vertex_versions[v1_idx] + self._vertex_versions[v2_idx]
            if candidate.version != current_version:
                continue
            
            # Validate the collapse
            if not self._is_collapse_valid(edge):
                continue
            
            return candidate
        
        return None
    
    def _is_collapse_valid(self, edge: Tuple[int, int]) -> bool:
        """
        Check if an edge collapse is valid (maintains manifold property).
        
        Uses the link condition: the collapse is valid if the intersection
        of the 1-rings of the two vertices equals exactly the two common
        vertices of the edge's adjacent faces.
        """
        v1_idx, v2_idx = edge
        
        # Get 1-ring neighbors (vertices connected by an edge)
        neighbors1 = self._get_vertex_neighbors(v1_idx)
        neighbors2 = self._get_vertex_neighbors(v2_idx)
        
        # Common neighbors
        common = neighbors1 & neighbors2
        
        # For a manifold edge, there should be exactly 2 common neighbors
        # (the two vertices opposite to the edge in the adjacent triangles)
        # For boundary edges, there may be only 1
        
        is_boundary = edge in self._boundary_edges or tuple(reversed(edge)) in self._boundary_edges
        
        if is_boundary:
            # Boundary edge: at most 1 common neighbor
            return len(common) <= 1
        else:
            # Interior edge: exactly 2 common neighbors
            return len(common) == 2
    
    def _get_vertex_neighbors(self, v_idx: int) -> Set[int]:
        """Get all vertices connected to v_idx by an edge."""
        neighbors = set()
        
        for fi in self._vertex_faces.get(v_idx, []):
            if fi in self._deleted_faces:
                continue
            face = self._faces[fi]
            for vi in face:
                if vi != v_idx and vi not in self._deleted_vertices:
                    neighbors.add(vi)
        
        return neighbors
    
    def _collapse_edge(self, candidate: EdgeCollapseCandidate) -> bool:
        """
        Perform an edge collapse.
        
        v1 is kept and moved to optimal position, v2 is deleted.
        All faces/edges referencing v2 are updated to reference v1.
        """
        v1_idx, v2_idx = candidate.edge
        optimal_pos = candidate.optimal_pos
        
        # Move v1 to optimal position
        self._vertices[v1_idx] = optimal_pos
        
        # Update quadric for v1
        self._quadrics[v1_idx] = self._quadrics[v1_idx] + self._quadrics[v2_idx]
        
        # Increment version for v1
        self._vertex_versions[v1_idx] += 1
        
        # Mark v2 as deleted
        self._deleted_vertices.add(v2_idx)
        
        # Find faces to delete (faces containing both v1 and v2)
        faces_to_delete = self._vertex_faces[v1_idx] & self._vertex_faces[v2_idx]
        
        # Update faces: replace v2 with v1 where v2 appears alone
        faces_to_update = self._vertex_faces[v2_idx] - faces_to_delete
        
        for fi in faces_to_delete:
            self._deleted_faces.add(fi)
        
        for fi in faces_to_update:
            face = self._faces[fi]
            for i in range(3):
                if face[i] == v2_idx:
                    face[i] = v1_idx
            # Update vertex-face adjacency
            self._vertex_faces[v1_idx].add(fi)
        
        # Clear v2's face list
        self._vertex_faces[v2_idx] = set()
        
        # Update boundary edges
        self._update_boundary_edges(candidate.edge)
        
        # Re-add affected edges to priority queue
        affected_edges = set()
        for fi in self._vertex_faces[v1_idx]:
            if fi in self._deleted_faces:
                continue
            face = self._faces[fi]
            for i in range(3):
                if face[i] in self._deleted_vertices:
                    continue
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge[0] not in self._deleted_vertices and edge[1] not in self._deleted_vertices:
                    affected_edges.add(edge)
        
        for edge in affected_edges:
            self._add_edge_candidate(edge)
        
        # Record collapse history
        self._collapse_history.append({
            'edge': candidate.edge,
            'error': candidate.error,
            'optimal_pos': optimal_pos.copy()
        })
        
        return True
    
    def _update_boundary_edges(self, collapsed_edge: Tuple[int, int]):
        """Update boundary edge set after a collapse."""
        # Remove the collapsed edge
        self._boundary_edges.discard(collapsed_edge)
        self._boundary_edges.discard((collapsed_edge[1], collapsed_edge[0]))
        
        # Recompute boundary for affected vertices
        v1_idx = collapsed_edge[0]
        
        # Count edge occurrences in active faces
        edge_count = {}
        for fi in self._vertex_faces[v1_idx]:
            if fi in self._deleted_faces:
                continue
            face = self._faces[fi]
            for i in range(3):
                if face[i] in self._deleted_vertices or face[(i+1)%3] in self._deleted_vertices:
                    continue
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Update boundary set
        for edge, count in edge_count.items():
            if count == 1:
                self._boundary_edges.add(edge)
            else:
                self._boundary_edges.discard(edge)
    
    def _get_active_face_count(self) -> int:
        """Get the number of non-deleted faces."""
        return len(self._faces) - len(self._deleted_faces)
    
    def _build_output_mesh(self) -> trimesh.Trimesh:
        """Build the output mesh from current state."""
        # Collect active vertices and create remapping
        active_vertices = []
        vertex_remap = {}
        
        for i, v in enumerate(self._vertices):
            if i not in self._deleted_vertices:
                vertex_remap[i] = len(active_vertices)
                active_vertices.append(v)
        
        # Collect active faces with remapped indices
        active_faces = []
        for fi, face in enumerate(self._faces):
            if fi not in self._deleted_faces:
                # Check for degenerate faces
                new_face = [vertex_remap[vi] for vi in face]
                if len(set(new_face)) == 3:  # Not degenerate
                    active_faces.append(new_face)
        
        vertices_array = np.array(active_vertices)
        faces_array = np.array(active_faces) if active_faces else np.zeros((0, 3), dtype=int)
        
        return trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
    
    def get_collapse_history(self) -> List[dict]:
        """Get the history of edge collapses performed."""
        return self._collapse_history.copy() if self._collapse_history else []
    
    def get_vertex_errors(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute the quadric error at each vertex of a mesh.
        
        Useful for error visualization after decimation.
        """
        quadrics = self.qem.compute_vertex_quadrics(
            mesh.vertices, mesh.faces
        )
        
        errors = np.zeros(len(mesh.vertices))
        for i, (v, Q) in enumerate(zip(mesh.vertices, quadrics)):
            errors[i] = self.qem.compute_error(Q, v)
        
        return errors
