"""
Mesh Visualization Module
=========================

Provides visualization utilities for mesh comparison,
error heatmaps, and animated before/after views.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List, Tuple
import trimesh


class MeshVisualizer:
    """
    Visualization tools for mesh simplification results.
    
    Provides:
    - Side-by-side mesh comparison
    - Wireframe and shaded views
    - Error heatmap visualization
    - Multi-resolution comparison
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 7)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colormap = cm.viridis
        self.error_colormap = cm.RdYlGn_r  
        
    def plot_mesh_comparison(self, original: trimesh.Trimesh, 
                              simplified: trimesh.Trimesh,
                              title: str = "Mesh Comparison",
                              show_wireframe: bool = True,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create side-by-side comparison of original and simplified meshes.
        
        Args:
            original: Original high-resolution mesh
            simplified: Simplified mesh
            title: Plot title
            show_wireframe: Whether to show wireframe overlay
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, 
                                  subplot_kw={'projection': '3d'})
        
        # Plot original
        self._plot_single_mesh(axes[0], original, 
                               f"Original\n({len(original.faces)} faces, {len(original.vertices)} vertices)",
                               show_wireframe)
        
        # Plot simplified
        self._plot_single_mesh(axes[1], simplified,
                               f"Simplified\n({len(simplified.faces)} faces, {len(simplified.vertices)} vertices)",
                               show_wireframe)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        return fig
    
    def _plot_single_mesh(self, ax: Axes3D, mesh: trimesh.Trimesh, 
                          title: str, show_wireframe: bool,
                          vertex_colors: Optional[np.ndarray] = None):
        """Plot a single mesh on a 3D axis."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Normalize to unit cube centered at origin
        center = vertices.mean(axis=0)
        scale = np.max(np.abs(vertices - center))
        vertices_normalized = (vertices - center) / scale
        
        # Create polygon collection
        triangles = vertices_normalized[faces]
        
        if vertex_colors is not None:
            # Use vertex colors (average for each face)
            face_colors = vertex_colors[faces].mean(axis=1)
            face_colors = self.error_colormap(face_colors)
        else:
            # Default gray coloring with lighting
            face_colors = self._compute_face_colors(mesh, vertices_normalized)
        
        triangles_rotated = triangles.copy()
        
        
        # Rotate around Y axis (vertical) and convert to Z-up
        triangles_rotated[..., 0] = triangles[..., 2]   # X gets Depth
        triangles_rotated[..., 1] = triangles[..., 0]   # Y gets Width
        triangles_rotated[..., 2] = triangles[..., 1]   # Z gets Height (Standard Up)

        poly = Poly3DCollection(triangles_rotated, facecolors=face_colors,
                                edgecolors='black' if show_wireframe else 'none',
                                linewidths=0.1 if show_wireframe else 0,
                                alpha=0.9)
        ax.add_collection3d(poly)
        
        # Set axis limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    def _compute_face_colors(self, mesh: trimesh.Trimesh, 
                             vertices: np.ndarray) -> np.ndarray:
        """Compute face colors based on normals for shading."""
        # Light direction
        light_dir = np.array([1, 1, 2])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Compute face normals
        if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
            normals = mesh.face_normals
        else:
            normals = np.zeros((len(mesh.faces), 3))
            for i, face in enumerate(mesh.faces):
                v0, v1, v2 = vertices[face]
                e1 = v1 - v0
                e2 = v2 - v0
                n = np.cross(e1, e2)
                norm = np.linalg.norm(n)
                if norm > 1e-10:
                    normals[i] = n / norm
        
        # Compute diffuse lighting
        intensity = np.clip(np.dot(normals, light_dir), 0.2, 1.0)
        
        # Create grayscale colors with blue tint
        colors = np.zeros((len(mesh.faces), 4))
        colors[:, 0] = 0.3 + 0.4 * intensity  # R
        colors[:, 1] = 0.4 + 0.4 * intensity  # G
        colors[:, 2] = 0.6 + 0.3 * intensity  # B
        colors[:, 3] = 1.0                     # A
        
        return colors
    
    def plot_error_heatmap(self, mesh: trimesh.Trimesh, 
                           vertex_errors: np.ndarray,
                           title: str = "Vertex Error Heatmap",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize vertex errors as a heatmap on the mesh.
        
        Args:
            mesh: The mesh to visualize
            vertex_errors: Per-vertex error values
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize,
                                  gridspec_kw={'width_ratios': [3, 1]})
        
        # 3D plot
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Normalize errors to [0, 1]
        errors_normalized = vertex_errors.copy()
        if errors_normalized.max() > errors_normalized.min():
            errors_normalized = (errors_normalized - errors_normalized.min()) / \
                               (errors_normalized.max() - errors_normalized.min())
        
        self._plot_single_mesh(ax3d, mesh, title, show_wireframe=False,
                               vertex_colors=errors_normalized)
        
        # Remove the empty subplot and add colorbar
        axes[1].remove()
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.error_colormap,
                                    norm=plt.Normalize(vertex_errors.min(), 
                                                       vertex_errors.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3d, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Quadric Error', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_multi_resolution(self, meshes: List[trimesh.Trimesh],
                               labels: Optional[List[str]] = None,
                               title: str = "Multi-Resolution Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple meshes at different resolutions.
        
        Args:
            meshes: List of meshes at different resolutions
            labels: Optional labels for each mesh
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n = len(meshes)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        
        for i, mesh in enumerate(meshes):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            if labels and i < len(labels):
                label = labels[i]
            else:
                ratio = len(mesh.faces) / len(meshes[0].faces) if meshes else 1.0
                label = f"{len(mesh.faces)} faces ({ratio*100:.1f}%)"
            
            self._plot_single_mesh(ax, mesh, label, show_wireframe=True)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-resolution plot to {save_path}")
        
        return fig
    
    def plot_statistics(self, original: trimesh.Trimesh,
                        simplified_meshes: List[trimesh.Trimesh],
                        ratios: List[float],
                        hausdorff_distances: Optional[List[float]] = None,
                        runtimes: Optional[List[float]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot decimation statistics.
        
        Args:
            original: Original mesh
            simplified_meshes: List of simplified meshes
            ratios: Target reduction ratios
            hausdorff_distances: Optional Hausdorff distances
            runtimes: Optional runtimes in seconds
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_plots = 2 + (1 if hausdorff_distances else 0) + (1 if runtimes else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Face count
        face_counts = [len(m.faces) for m in simplified_meshes]
        axes[plot_idx].bar(range(len(ratios)), face_counts, color='steelblue')
        axes[plot_idx].axhline(y=len(original.faces), color='red', linestyle='--', 
                               label=f'Original ({len(original.faces)})')
        axes[plot_idx].set_xticks(range(len(ratios)))
        axes[plot_idx].set_xticklabels([f'{r*100:.0f}%' for r in ratios])
        axes[plot_idx].set_xlabel('Target Ratio')
        axes[plot_idx].set_ylabel('Face Count')
        axes[plot_idx].set_title('Face Count vs Target')
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Vertex count
        vertex_counts = [len(m.vertices) for m in simplified_meshes]
        axes[plot_idx].bar(range(len(ratios)), vertex_counts, color='forestgreen')
        axes[plot_idx].axhline(y=len(original.vertices), color='red', linestyle='--',
                               label=f'Original ({len(original.vertices)})')
        axes[plot_idx].set_xticks(range(len(ratios)))
        axes[plot_idx].set_xticklabels([f'{r*100:.0f}%' for r in ratios])
        axes[plot_idx].set_xlabel('Target Ratio')
        axes[plot_idx].set_ylabel('Vertex Count')
        axes[plot_idx].set_title('Vertex Count vs Target')
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Hausdorff distance
        if hausdorff_distances:
            axes[plot_idx].plot(ratios, hausdorff_distances, 'o-', color='crimson')
            axes[plot_idx].set_xlabel('Target Ratio')
            axes[plot_idx].set_ylabel('Hausdorff Distance')
            axes[plot_idx].set_title('Geometric Error vs Reduction')
            plot_idx += 1
        
        # Runtime
        if runtimes:
            axes[plot_idx].plot(ratios, runtimes, 's-', color='purple')
            axes[plot_idx].set_xlabel('Target Ratio')
            axes[plot_idx].set_ylabel('Runtime (seconds)')
            axes[plot_idx].set_title('Runtime vs Target Ratio')
        
        plt.suptitle('Decimation Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved statistics to {save_path}")
        
        return fig
    
    def interactive_view(self, mesh: trimesh.Trimesh):
        """
        Open an interactive 3D viewer for the mesh.
        
        Uses trimesh's built-in viewer.
        """
        try:
            mesh.show()
        except Exception as e:
            print(f"Could not open interactive viewer: {e}")
            print("Falling back to matplotlib view...")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_single_mesh(ax, mesh, f"Mesh ({len(mesh.faces)} faces)", True)
            plt.show()
    
    def create_comparison_animation(self, original: trimesh.Trimesh,
                                     simplified: trimesh.Trimesh,
                                     output_path: str,
                                     frames: int = 60,
                                     fps: int = 15):
        """
        Create a rotating animation comparing original and simplified meshes.
        
        Args:
            original: Original mesh
            simplified: Simplified mesh
            output_path: Path to save the animation (GIF or MP4)
            frames: Number of frames in the animation
            fps: Frames per second
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("Animation requires matplotlib animation support")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize,
                                  subplot_kw={'projection': '3d'})
        
        def init():
            self._plot_single_mesh(axes[0], original, 
                                   f"Original ({len(original.faces)} faces)", True)
            self._plot_single_mesh(axes[1], simplified,
                                   f"Simplified ({len(simplified.faces)} faces)", True)
            return []
        
        def animate(frame):
            angle = frame * (360 / frames)
            for ax in axes:
                ax.view_init(elev=20, azim=angle)
            return []
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                            frames=frames, interval=1000/fps, blit=True)
        
        if output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        else:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps)
        
        anim.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")
        plt.close(fig)
