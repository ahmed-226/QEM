"""
Utility Functions
=================

Mesh loading, downloading, and sample mesh creation utilities.
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file.
    
    Supports: OBJ, PLY, STL, OFF, and other formats supported by trimesh.
    
    Args:
        path: Path to mesh file
        
    Returns:
        Loaded trimesh object
    """
    mesh = trimesh.load(path, force='mesh')
    
    if isinstance(mesh, trimesh.Scene):
        # Convert scene to single mesh
        meshes = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in file")
    
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str):
    """
    Save a mesh to file.
    
    Args:
        mesh: Mesh to save
        path: Output path
    """
    mesh.export(path)
    print(f"Saved mesh to: {path}")


def download_sample_meshes(output_dir: Path) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Download standard test meshes from web sources.
    
    Args:
        output_dir: Directory to save downloaded meshes
        
    Returns:
        List of (name, mesh) tuples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for standard test meshes
    mesh_urls = {
        "bunny": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj",
        "spot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj",
    }
    
    # Alternative simple URLs
    backup_urls = {
        "bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bunny.obj",
    }
    
    meshes = []
    
    for name, url in mesh_urls.items():
        output_path = output_dir / f"{name}.obj"
        
        if output_path.exists():
            print(f"Loading cached: {name}")
            try:
                mesh = load_mesh(str(output_path))
                meshes.append((name, mesh))
                continue
            except Exception as e:
                print(f"Failed to load cached {name}: {e}")
        
        print(f"Downloading: {name}...")
        try:
            urllib.request.urlretrieve(url, str(output_path))
            mesh = load_mesh(str(output_path))
            meshes.append((name, mesh))
            print(f"  Downloaded: {len(mesh.faces)} faces")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")
            
            # Try backup URL
            if name in backup_urls:
                try:
                    print(f"  Trying backup URL...")
                    urllib.request.urlretrieve(backup_urls[name], str(output_path))
                    mesh = load_mesh(str(output_path))
                    meshes.append((name, mesh))
                    print(f"  Downloaded from backup: {len(mesh.faces)} faces")
                except Exception as e2:
                    print(f"  Backup also failed: {e2}")
    
    if not meshes:
        print("No meshes downloaded, creating sample meshes...")
        for name in ["bunny", "sphere", "torus"]:
            mesh = create_sample_mesh(name)
            meshes.append((name, mesh))
    
    return meshes


def create_sample_mesh(mesh_type: str = "bunny") -> trimesh.Trimesh:
    """
    Create a sample mesh for testing.
    
    Args:
        mesh_type: Type of mesh to create:
            - "bunny": Approximation of Stanford Bunny
            - "sphere": UV sphere
            - "torus": Torus
            - "cube": Subdivided cube
            - "cylinder": Cylinder
            
    Returns:
        Generated trimesh object
    """
    if mesh_type == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    elif mesh_type == "torus":
        mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, 
                                       major_sections=64, minor_sections=32)
    elif mesh_type == "cube":
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        # Subdivide for more faces
        for _ in range(3):
            mesh = mesh.subdivide()
    elif mesh_type == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=64)
    elif mesh_type == "bunny":
        # Create a bunny-like shape using sphere combinations
        mesh = _create_bunny_approximation()
    else:
        # Default to sphere
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    
    print(f"Created {mesh_type} mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def _create_bunny_approximation() -> trimesh.Trimesh:
    """
    Create a bunny-like mesh using primitive combinations.
    
    This is a placeholder for when the Stanford Bunny is not available.
    """
    meshes = []
    
    # Body (ellipsoid)
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.vertices[:, 0] *= 0.8  # Flatten X
    body.vertices[:, 1] *= 1.0  # Y stays
    body.vertices[:, 2] *= 0.7  # Flatten Z
    meshes.append(body)
    
    # Head (smaller sphere, offset)
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    head.vertices += np.array([0, 0.9, 0.3])
    meshes.append(head)
    
    # Left ear
    left_ear = trimesh.creation.icosphere(subdivisions=2, radius=0.15)
    left_ear.vertices[:, 1] *= 3.0  # Elongate in Y
    left_ear.vertices += np.array([-0.2, 1.5, 0.4])
    meshes.append(left_ear)
    
    # Right ear
    right_ear = trimesh.creation.icosphere(subdivisions=2, radius=0.15)
    right_ear.vertices[:, 1] *= 3.0  # Elongate in Y
    right_ear.vertices += np.array([0.2, 1.5, 0.4])
    meshes.append(right_ear)
    
    # Tail (small sphere)
    tail = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    tail.vertices += np.array([0, -0.3, -0.7])
    meshes.append(tail)
    
    # Combine all parts
    combined = trimesh.util.concatenate(meshes)
    
    # Subdivide once for more detail
    combined = combined.subdivide()
    
    return combined


def create_mesh_with_boundary(rows: int = 20, cols: int = 20) -> trimesh.Trimesh:
    """
    Create a mesh with boundaries (open surface) for testing boundary preservation.
    
    Creates a wavy surface grid.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        Open surface mesh
    """
    # Create grid vertices
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    
    # Create wavy surface
    Z = 0.2 * np.sin(3 * X) * np.cos(3 * Y)
    
    # Flatten to vertex array
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Create faces (two triangles per grid cell)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            # First triangle
            faces.append([idx, idx + 1, idx + cols])
            # Second triangle
            faces.append([idx + 1, idx + cols + 1, idx + cols])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    
    # Add some random noise for more interesting decimation
    mesh.vertices += np.random.randn(*mesh.vertices.shape) * 0.01
    
    return mesh


def create_mesh_with_features(size: float = 1.0, 
                               feature_sharpness: float = 0.1) -> trimesh.Trimesh:
    """
    Create a mesh with sharp features for testing feature preservation.
    
    Creates a box with beveled edges.
    
    Args:
        size: Size of the box
        feature_sharpness: Controls edge sharpness
        
    Returns:
        Mesh with sharp features
    """
    # Start with a cube
    mesh = trimesh.creation.box(extents=[size, size, size])
    
    # Subdivide edges for more detail
    for _ in range(4):
        mesh = mesh.subdivide()
    
    # Push vertices slightly inward near edges to create features
    for i, v in enumerate(mesh.vertices):
        # Distance to each face of the cube
        dist_to_faces = np.abs(np.abs(v) - size/2)
        min_dist = np.min(dist_to_faces)
        
        if min_dist < feature_sharpness * 2:
            # Near an edge or corner, push inward slightly
            direction = -v / (np.linalg.norm(v) + 1e-10)
            mesh.vertices[i] += direction * feature_sharpness * 0.5
    
    return mesh


def get_mesh_info(mesh: trimesh.Trimesh) -> dict:
    """
    Get comprehensive information about a mesh.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Dictionary of mesh properties
    """
    info = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'edges': len(mesh.edges) if hasattr(mesh, 'edges') else 'N/A',
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'euler_number': mesh.euler_number,
        'bounds': mesh.bounds.tolist(),
        'centroid': mesh.centroid.tolist(),
        'scale': float(mesh.scale),
    }
    
    # Surface area
    try:
        info['area'] = float(mesh.area)
    except:
        info['area'] = 'N/A'
    
    # Volume (only for watertight meshes)
    if mesh.is_watertight:
        try:
            info['volume'] = float(mesh.volume)
        except:
            info['volume'] = 'N/A'
    else:
        info['volume'] = 'N/A (not watertight)'
    
    # Count boundary edges
    edge_count = {}
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    info['boundary_edges'] = boundary_edges
    
    return info


def print_mesh_info(mesh: trimesh.Trimesh, name: str = "Mesh"):
    """
    Print mesh information to console.
    
    Args:
        mesh: Input mesh
        name: Name to display
    """
    info = get_mesh_info(mesh)
    
    print(f"\n{name} Information:")
    print("-" * 40)
    print(f"  Vertices:        {info['vertices']}")
    print(f"  Faces:           {info['faces']}")
    print(f"  Edges:           {info['edges']}")
    print(f"  Boundary Edges:  {info['boundary_edges']}")
    print(f"  Watertight:      {info['is_watertight']}")
    print(f"  Euler Number:    {info['euler_number']}")
    print(f"  Surface Area:    {info['area']}")
    print(f"  Volume:          {info['volume']}")
    print(f"  Scale:           {info['scale']:.4f}")
