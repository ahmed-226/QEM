"""
Baseline Methods for Comparison
===============================

Simple baseline methods for comparing against QEM:
- Midpoint collapse (edge collapse without QEM optimization)
- Vertex clustering (grid-based simplification)
"""

import numpy as np
import trimesh
from typing import Optional
from collections import defaultdict


def midpoint_collapse_simplify(mesh: trimesh.Trimesh, 
                                target_ratio: float) -> trimesh.Trimesh:
    """
    Simple midpoint edge collapse without QEM optimization.
    
    This is a baseline method that shows the importance of optimal
    vertex placement in QEM.
    
    Args:
        mesh: Input mesh
        target_ratio: Target ratio of faces to keep
        
    Returns:
        Simplified mesh
    """
    vertices = mesh.vertices.copy()
    faces = [list(f) for f in mesh.faces]
    
    target_faces = max(4, int(len(faces) * target_ratio))
    
    # Build edge list
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1) % 3]]))
            edges.add(edge)
    
    edges = list(edges)
    deleted_vertices = set()
    deleted_faces = set()
    
    # Vertex remapping
    vertex_remap = {i: i for i in range(len(vertices))}
    
    def get_root(v):
        while vertex_remap[v] != v:
            v = vertex_remap[v]
        return v
    
    edge_idx = 0
    np.random.shuffle(edges)  # Random order for baseline
    
    while len(faces) - len(deleted_faces) > target_faces and edge_idx < len(edges):
        edge = edges[edge_idx]
        edge_idx += 1
        
        v1, v2 = get_root(edge[0]), get_root(edge[1])
        
        if v1 == v2 or v1 in deleted_vertices or v2 in deleted_vertices:
            continue
        
        # Collapse to midpoint
        midpoint = (vertices[v1] + vertices[v2]) / 2
        vertices[v1] = midpoint
        
        # Remap v2 to v1
        vertex_remap[v2] = v1
        deleted_vertices.add(v2)
        
        # Update faces
        for fi, face in enumerate(faces):
            if fi in deleted_faces:
                continue
            
            # Update references
            for i in range(3):
                face[i] = get_root(face[i])
            
            # Check for degenerate face
            if len(set(face)) < 3:
                deleted_faces.add(fi)
    
    # Build output mesh
    active_vertices = []
    new_remap = {}
    
    for i, v in enumerate(vertices):
        if i not in deleted_vertices:
            new_remap[i] = len(active_vertices)
            active_vertices.append(v)
    
    active_faces = []
    for fi, face in enumerate(faces):
        if fi not in deleted_faces:
            new_face = [new_remap[get_root(vi)] for vi in face]
            if len(set(new_face)) == 3:
                active_faces.append(new_face)
    
    return trimesh.Trimesh(
        vertices=np.array(active_vertices),
        faces=np.array(active_faces) if active_faces else np.zeros((0, 3), dtype=int)
    )


def vertex_clustering_simplify(mesh: trimesh.Trimesh,
                                target_ratio: float,
                                grid_size: Optional[float] = None) -> trimesh.Trimesh:
    """
    Vertex clustering (grid-based) simplification.
    
    Merges all vertices within the same grid cell to their centroid.
    This is similar to what Open3D's simplify_vertex_clustering does.
    
    Args:
        mesh: Input mesh
        target_ratio: Approximate target ratio (used to determine grid size)
        grid_size: Optional explicit grid cell size
        
    Returns:
        Simplified mesh
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # Compute bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Estimate grid size from target ratio
    if grid_size is None:
        # Rough estimate: cube root of inverse target ratio
        scale_factor = (1.0 / target_ratio) ** (1.0 / 3.0)
        avg_edge_length = np.mean([
            np.linalg.norm(vertices[f[1]] - vertices[f[0]])
            for f in faces[:min(1000, len(faces))]
        ])
        grid_size = avg_edge_length * scale_factor
    
    # Assign vertices to grid cells
    cell_indices = np.floor((vertices - bbox_min) / grid_size).astype(int)
    
    # Group vertices by cell
    cell_to_vertices = defaultdict(list)
    for vi, cell in enumerate(cell_indices):
        cell_key = tuple(cell)
        cell_to_vertices[cell_key].append(vi)
    
    # Create new vertices (centroid of each cell)
    new_vertices = []
    vertex_map = {}
    
    for cell_key, vertex_indices in cell_to_vertices.items():
        cell_centroid = vertices[vertex_indices].mean(axis=0)
        new_idx = len(new_vertices)
        new_vertices.append(cell_centroid)
        
        for vi in vertex_indices:
            vertex_map[vi] = new_idx
    
    # Remap faces
    new_faces = []
    for face in faces:
        new_face = [vertex_map[vi] for vi in face]
        # Keep only non-degenerate faces
        if len(set(new_face)) == 3:
            new_faces.append(new_face)
    
    return trimesh.Trimesh(
        vertices=np.array(new_vertices),
        faces=np.array(new_faces) if new_faces else np.zeros((0, 3), dtype=int)
    )


def open3d_vertex_clustering(mesh: trimesh.Trimesh,
                              target_ratio: float) -> trimesh.Trimesh:
    """
    Use Open3D's vertex clustering for comparison.
    
    Args:
        mesh: Input mesh
        target_ratio: Target ratio (used to estimate voxel size)
        
    Returns:
        Simplified mesh
    """
    try:
        import open3d as o3d
        
        # Convert to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
        # Compute voxel size from target ratio
        bbox = mesh.bounding_box.extents
        avg_extent = np.mean(bbox)
        scale_factor = (1.0 / target_ratio) ** (1.0 / 3.0)
        
        # Estimate average edge length
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        avg_edge = np.mean(edge_lengths)
        
        voxel_size = avg_edge * scale_factor
        
        # Simplify
        simplified = o3d_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        
        # Convert back to trimesh
        return trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles)
        )
        
    except ImportError:
        print("Open3D not available, falling back to custom vertex clustering")
        return vertex_clustering_simplify(mesh, target_ratio)


def open3d_quadric_decimation(mesh: trimesh.Trimesh,
                               target_ratio: float) -> trimesh.Trimesh:
    """
    Use Open3D's quadric decimation for comparison.
    
    Args:
        mesh: Input mesh
        target_ratio: Target ratio of faces
        
    Returns:
        Simplified mesh
    """
    try:
        import open3d as o3d
        
        # Convert to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
        target_faces = max(4, int(len(mesh.faces) * target_ratio))
        
        # Simplify using quadric decimation
        simplified = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        
        # Convert back to trimesh
        return trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles)
        )
        
    except ImportError:
        print("Open3D not available for quadric decimation comparison")
        return midpoint_collapse_simplify(mesh, target_ratio)
