"""
High-Quality Mesh Decimation - Main Demo
=========================================

Demonstrates mesh simplification using Quadric Error Metrics (QEM).

This script:
1. Downloads/loads test meshes (Stanford Bunny, etc.)
2. Performs decimation at various reduction levels
3. Visualizes before/after comparisons
4. Computes quantitative metrics
5. Compares with baseline methods
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import trimesh
import matplotlib.pyplot as plt

from src.mesh_decimator import MeshDecimator
from src.visualization import MeshVisualizer
from src.evaluation import MeshEvaluator
from src.baselines import (
    midpoint_collapse_simplify,
    vertex_clustering_simplify,
    open3d_quadric_decimation
)
from src.utils import download_sample_meshes, create_sample_mesh, load_mesh


def run_single_decimation(mesh: trimesh.Trimesh, 
                          target_ratio: float,
                          boundary_weight: float = 10.0,
                          preserve_boundaries: bool = True) -> tuple:
    """
    Run a single decimation and return simplified mesh with metrics.
    """
    decimator = MeshDecimator(
        boundary_weight=boundary_weight,
        preserve_boundaries=preserve_boundaries
    )
    
    start_time = time.time()
    simplified = decimator.decimate(mesh, target_ratio=target_ratio)
    runtime = time.time() - start_time
    
    return simplified, runtime


def demo_basic_decimation(mesh: trimesh.Trimesh, 
                          output_dir: Path,
                          mesh_name: str = "mesh"):
    """
    Demonstrate basic mesh decimation with visualization.
    """
    print("\n" + "=" * 60)
    print("BASIC DECIMATION DEMO")
    print("=" * 60)
    
    visualizer = MeshVisualizer()
    evaluator = MeshEvaluator()
    
    # Test different reduction levels
    ratios = [0.5, 0.25, 0.1, 0.05]
    simplified_meshes = []
    metrics_list = []
    runtimes = []
    
    for ratio in ratios:
        print(f"\n--- Decimating to {ratio*100:.0f}% ---")
        
        simplified, runtime = run_single_decimation(mesh, target_ratio=ratio)
        simplified_meshes.append(simplified)
        runtimes.append(runtime)
        
        # Compute metrics
        metrics = evaluator.compute_all_metrics(mesh, simplified)
        metrics['runtime'] = runtime
        metrics_list.append(metrics)
        
        # Print summary
        print(f"  Faces: {len(mesh.faces)} -> {len(simplified.faces)} "
              f"({len(simplified.faces)/len(mesh.faces)*100:.1f}%)")
        print(f"  Vertices: {len(mesh.vertices)} -> {len(simplified.vertices)}")
        print(f"  Hausdorff: {metrics['hausdorff_distance']:.6f}")
        print(f"  Runtime: {runtime:.3f}s")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Side-by-side comparison at 25%
    fig = visualizer.plot_mesh_comparison(
        mesh, simplified_meshes[1],
        title=f"{mesh_name} - Original vs 25% Simplified",
        save_path=str(output_dir / f"{mesh_name}_comparison_25pct.png")
    )
    plt.close(fig)
    
    # Multi-resolution view
    all_meshes = [mesh] + simplified_meshes
    labels = ["Original (100%)"] + [f"{r*100:.0f}%" for r in ratios]
    fig = visualizer.plot_multi_resolution(
        all_meshes, labels,
        title=f"{mesh_name} - Multi-Resolution",
        save_path=str(output_dir / f"{mesh_name}_multi_resolution.png")
    )
    plt.close(fig)
    
    # Statistics plot
    hausdorff_dists = [m['hausdorff_distance'] for m in metrics_list]
    fig = visualizer.plot_statistics(
        mesh, simplified_meshes, ratios,
        hausdorff_distances=hausdorff_dists,
        runtimes=runtimes,
        save_path=str(output_dir / f"{mesh_name}_statistics.png")
    )
    plt.close(fig)
    
    # Generate report for 25% reduction
    print("\n" + evaluator.generate_report(metrics_list[1], "QEM (25% target)"))
    
    # Save simplified meshes
    for ratio, simplified in zip(ratios, simplified_meshes):
        output_path = output_dir / f"{mesh_name}_simplified_{int(ratio*100)}pct.ply"
        simplified.export(str(output_path))
        print(f"Saved: {output_path}")
    
    return simplified_meshes, metrics_list


def demo_boundary_weight_comparison(mesh: trimesh.Trimesh,
                                     output_dir: Path,
                                     mesh_name: str = "mesh"):
    """
    Demonstrate effect of boundary weight on decimation.
    """
    print("\n" + "=" * 60)
    print("BOUNDARY WEIGHT COMPARISON")
    print("=" * 60)
    
    visualizer = MeshVisualizer()
    evaluator = MeshEvaluator()
    
    weights = [0.0, 1.0, 10.0, 100.0]
    target_ratio = 0.25
    
    results = []
    
    for weight in weights:
        print(f"\n--- Boundary weight: {weight} ---")
        
        decimator = MeshDecimator(
            boundary_weight=weight,
            preserve_boundaries=(weight > 0)
        )
        
        start_time = time.time()
        simplified = decimator.decimate(mesh, target_ratio=target_ratio)
        runtime = time.time() - start_time
        
        metrics = evaluator.compute_all_metrics(mesh, simplified)
        metrics['runtime'] = runtime
        metrics['boundary_weight'] = weight
        
        results.append({
            'mesh': simplified,
            'metrics': metrics
        })
        
        print(f"  Faces: {len(simplified.faces)}")
        print(f"  Boundary change: {metrics['boundary_length_change']*100:.2f}%")
        print(f"  Hausdorff: {metrics['hausdorff_distance']:.6f}")
    
    # Visualize comparison
    meshes = [r['mesh'] for r in results]
    labels = [f"Weight={w}" for w in weights]
    
    fig = visualizer.plot_multi_resolution(
        meshes, labels,
        title=f"{mesh_name} - Boundary Weight Comparison (25% target)",
        save_path=str(output_dir / f"{mesh_name}_boundary_weights.png")
    )
    plt.close(fig)
    
    return results


def demo_method_comparison(mesh: trimesh.Trimesh,
                           output_dir: Path,
                           mesh_name: str = "mesh"):
    """
    Compare QEM with baseline methods.
    """
    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)
    
    visualizer = MeshVisualizer()
    evaluator = MeshEvaluator()
    
    target_ratio = 0.25
    
    # Define methods to compare
    methods = {
        "QEM ": lambda m, r: MeshDecimator(boundary_weight=10.0).decimate(m, target_ratio=r),
        "Midpoint Collapse": midpoint_collapse_simplify,
        "Vertex Clustering": vertex_clustering_simplify,
    }
    
    # Try to add Open3D methods
    try:
        import open3d
        methods["Open3D Quadric"] = open3d_quadric_decimation
    except ImportError:
        print("Open3D not available, skipping Open3D comparison")
    
    results = evaluator.compare_methods(mesh, target_ratio, methods)
    
    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'Method':<20} {'Faces':>10} {'Hausdorff':>12} {'Chamfer':>12} {'Runtime':>10}")
    print("-" * 80)
    
    for name, result in results.items():
        if result['metrics'].get('success', True):
            m = result['metrics']
            print(f"{name:<20} {m.get('simplified_faces', 'N/A'):>10} "
                  f"{m.get('hausdorff_distance', float('nan')):>12.6f} "
                  f"{m.get('chamfer_distance', float('nan')):>12.6f} "
                  f"{m.get('runtime', 0):>10.3f}s")
        else:
            print(f"{name:<20} {'FAILED':>10} {result['metrics'].get('error', '')}")
    
    print("-" * 80)
    
    # Visualize comparison
    valid_meshes = []
    valid_labels = []
    for name, result in results.items():
        if result['mesh'] is not None:
            valid_meshes.append(result['mesh'])
            valid_labels.append(name)
    
    if valid_meshes:
        fig = visualizer.plot_multi_resolution(
            valid_meshes, valid_labels,
            title=f"{mesh_name} - Method Comparison (25% target)",
            save_path=str(output_dir / f"{mesh_name}_method_comparison.png")
        )
        plt.close(fig)
    
    return results


def demo_error_visualization(mesh: trimesh.Trimesh,
                              output_dir: Path,
                              mesh_name: str = "mesh"):
    """
    Demonstrate error heatmap visualization.
    """
    print("\n" + "=" * 60)
    print("ERROR VISUALIZATION")
    print("=" * 60)
    
    visualizer = MeshVisualizer()
    decimator = MeshDecimator(boundary_weight=10.0)
    
    # Decimate
    simplified = decimator.decimate(mesh, target_ratio=0.25)
    
    # Compute vertex errors on simplified mesh
    vertex_errors = decimator.get_vertex_errors(simplified)
    
    print(f"Vertex error range: [{vertex_errors.min():.6f}, {vertex_errors.max():.6f}]")
    print(f"Mean vertex error: {vertex_errors.mean():.6f}")
    
    # Visualize
    fig = visualizer.plot_error_heatmap(
        simplified, vertex_errors,
        title=f"{mesh_name} - Quadric Error Heatmap (25% simplified)",
        save_path=str(output_dir / f"{mesh_name}_error_heatmap.png")
    )
    plt.close(fig)
    
    return simplified, vertex_errors


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="High-Quality Mesh Decimation Demo using QEM"
    )
    parser.add_argument(
        "--mesh", "-m", type=str, default=None,
        help="Path to input mesh file. If not provided, uses sample meshes."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--download-samples", action="store_true",
        help="Download sample meshes (Stanford Bunny, etc.)"
    )
    parser.add_argument(
        "--ratio", "-r", type=float, default=0.25,
        help="Target reduction ratio (default: 0.25)"
    )
    parser.add_argument(
        "--boundary-weight", "-b", type=float, default=10.0,
        help="Boundary preservation weight (default: 10.0)"
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Quick mode - skip some demos for faster testing"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Open interactive 3D viewer after decimation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HIGH-QUALITY MESH DECIMATION")
    print("Using Quadric Error Metrics (QEM)")
    print("=" * 60)
    
    # Load or create mesh
    if args.mesh:
        print(f"\nLoading mesh from: {args.mesh}")
        mesh = load_mesh(args.mesh)
        mesh_name = Path(args.mesh).stem
    elif args.download_samples:
        print("\nDownloading sample meshes...")
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        meshes = download_sample_meshes(sample_dir)
        if meshes:
            mesh = meshes[0][1]
            mesh_name = meshes[0][0]
        else:
            print("Failed to download samples, using generated mesh")
            mesh = create_sample_mesh("sphere")
            mesh_name = "sphere"
    else:
        print("\nNo mesh specified, creating sample mesh...")
        mesh = create_sample_mesh("bunny")
        mesh_name = "sample_bunny"
    
    print(f"\nMesh loaded: {mesh_name}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    
    # Run demos
    demo_basic_decimation(mesh, output_dir, mesh_name)
    
    if not args.quick:
        demo_boundary_weight_comparison(mesh, output_dir, mesh_name)
        demo_method_comparison(mesh, output_dir, mesh_name)
        demo_error_visualization(mesh, output_dir, mesh_name)
    
    # Quick single decimation
    if args.ratio != 0.25:
        print(f"\n--- Custom decimation at {args.ratio*100:.0f}% ---")
        simplified, runtime = run_single_decimation(
            mesh, 
            target_ratio=args.ratio,
            boundary_weight=args.boundary_weight
        )
        output_path = output_dir / f"{mesh_name}_simplified_{int(args.ratio*100)}pct.ply"
        simplified.export(str(output_path))
        print(f"Saved: {output_path}")
        
        if args.interactive:
            print("\nOpening interactive viewer...")
            MeshVisualizer().interactive_view(simplified)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print(f"Results saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
