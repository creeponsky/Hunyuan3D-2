import glob
import os
import subprocess

import numpy as np
import trimesh


def apply_blender_transform(mesh):
    """Apply Blender-style coordinate transformation (Z-up to Y-up)"""
    # Fix: Rotate +90 degrees around X-axis to convert Z-up to Y-up (opposite of before)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Apply rotation to vertices
    mesh.vertices = mesh.vertices @ rotation_matrix.T

    # If mesh has vertex normals, rotate them too
    if (
        hasattr(mesh.visual, "vertex_normals")
        and mesh.visual.vertex_normals is not None
    ):
        mesh.visual.vertex_normals = mesh.visual.vertex_normals @ rotation_matrix.T

    return mesh


def process_obj_file(input_path: str, output_path: str) -> bool:
    """
    Process OBJ file to calculate normals and apply coordinate transformation

    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file with normals

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Loading mesh from: {input_path}")

    # Load mesh
    try:
        mesh = trimesh.load(input_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return False

    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Apply coordinate transformation
    print("Applying coordinate transformation...")
    mesh = apply_blender_transform(mesh)

    # Recalculate normals
    print("Recalculating normals...")
    mesh.vertex_normals  # This triggers calculation of smooth vertex normals
    mesh.face_normals  # This triggers calculation of face normals

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Export mesh
    print(f"Exporting mesh to: {output_path}")
    try:
        mesh.export(output_path)
        print("Export successful!")
        return True
    except Exception as e:
        print(f"Error exporting mesh: {e}")
        return False


def generate_gif(obj_path: str, gif_path: str) -> bool:
    """
    Generate GIF from OBJ file using external gif generator

    Args:
        obj_path: Path to OBJ file
        gif_path: Path to output GIF file

    Returns:
        bool: True if successful, False otherwise
    """
    gif_generator_path = "/home/hprt-gpu-2/CompanyProject/gifGenerator2/gifGenerator"

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(gif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cmd = [
        gif_generator_path,
        "-w",
        "512",
        "-h",
        "512",
        "-n",
        "26",
        "-delay",
        "15",
        "-name",
        "model",
        "-f",
        obj_path,
        "-o",
        gif_path,
    ]

    try:
        print(f"Generating GIF: {gif_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("GIF generation successful!")
            return True
        else:
            print(f"GIF generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error generating GIF: {e}")
        return False


def process_model(
    input_path: str, output_path: str = None, gif_path: str = None
) -> bool:
    """
    Process a single model: calculate normals and optionally generate GIF

    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file with normals (optional, auto-generated if None)
        gif_path: Path to output GIF file (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    # Generate output path if not provided
    if output_path is None:
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}-normal.obj"

    # Process normals
    if not process_obj_file(input_path, output_path):
        return False

    # Generate GIF if requested
    if gif_path:
        if not generate_gif(output_path, gif_path):
            return False

    return True


def process_folder(folder_path: str) -> bool:
    """
    Process all OBJ files in a folder

    Args:
        folder_path: Path to folder containing OBJ files

    Returns:
        bool: True if all files processed successfully, False otherwise
    """
    obj_files = glob.glob(os.path.join(folder_path, "*.obj"))

    if not obj_files:
        print(f"No OBJ files found in {folder_path}")
        return False

    success_count = 0
    total_count = len(obj_files)

    for obj_file in obj_files:
        # Skip if it's already a normal version
        if obj_file.endswith("-normal.obj"):
            continue

        base_path = os.path.splitext(obj_file)[0]
        normal_path = f"{base_path}-normal.obj"
        gif_path = f"{base_path}-normal.gif"

        # Check if normal version already exists
        if os.path.exists(normal_path):
            print(f"Normal version already exists: {normal_path}")
            # Still try to generate GIF if it doesn't exist
            if not os.path.exists(gif_path):
                if generate_gif(normal_path, gif_path):
                    success_count += 1
            else:
                success_count += 1
        else:
            # Process the model
            if process_model(obj_file, normal_path, gif_path):
                success_count += 1

    print(f"Processed {success_count}/{total_count} files successfully")
    return success_count == total_count


def main():
    # Test paths
    # input_path = "output/20250618/9c7d9731-6fc4-43a1-b48e-81f4d7110973.obj"
    input_path = ""
    input_folder_path = "output/20250618"

    # Test single file processing
    if os.path.exists(input_path):
        print("=== Processing single file ===")
        output_path = "output/20250618/9c7d9731-6fc4-43a1-b48e-81f4d7110973-normal.obj"
        gif_path = "output/20250618/9c7d9731-6fc4-43a1-b48e-81f4d7110973-normal.gif"

        success = process_model(input_path, output_path, gif_path)
        if success:
            print("Single file processing completed successfully!")
        else:
            print("Single file processing failed!")

    # Test folder processing
    if os.path.exists(input_folder_path):
        print("\n=== Processing folder ===")
        success = process_folder(input_folder_path)
        if success:
            print("Folder processing completed successfully!")
        else:
            print("Folder processing completed with some failures!")


if __name__ == "__main__":
    main()
