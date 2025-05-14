import os
import time
from pathlib import Path

import vedo


# 20250514081827_az45_el30_z1.2_s2 is better
def render_model_cover(
    model_path, output_path, azimuth=45, elevation=30, zoom=1.2, scale=2
):
    """
    Render a 3D model from specified angle and save it as a cover image.

    Args:
        model_path (str): Path to the 3D model file (.obj, .stl, etc.)
        output_path (str): Path where the rendered image will be saved
        azimuth (float): Horizontal rotation angle in degrees (around the y-axis)
        elevation (float): Vertical elevation angle in degrees (above the x-z plane)
        zoom (float): Zoom factor to ensure the model fits in the frame
        scale (int): Resolution scale factor for the screenshot

    Returns:
        str: Path to the saved image
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {output_dir}: {e}")
    # Load the 3D model
    mesh = vedo.load(model_path)
    mesh.color("white")

    # Initialize the plotter with offscreen rendering
    plotter = vedo.Plotter(offscreen=True)

    # Add the mesh to the plotter
    plotter.add(mesh)
    plotter.background("black")

    # First reset camera to ensure object is centered
    plotter.reset_camera()

    # Apply specified rotations
    plotter.camera.Azimuth(azimuth)  # Rotate horizontally (around y-axis)
    plotter.camera.Elevation(elevation)  # Rotate vertically (above x-z plane)
    plotter.camera.Zoom(zoom)  # Adjust zoom to fit the model

    # Render and capture a screenshot
    try:
        plotter.screenshot(output_path, scale=scale).close()
        return output_path
    except Exception as e:
        print(f"Error generating screenshot: {e}")
        return None


def find_all_obj_files(directory):
    """
    Recursively find all .obj files in the given directory.

    Args:
        directory (str): Root directory to search in

    Returns:
        list: List of paths to .obj files
    """
    obj_files = []
    for path in Path(directory).rglob("*.obj"):
        obj_files.append(str(path))
    return obj_files


# Main test function
if __name__ == "__main__":
    # Find all .obj files in the output directory
    output_dir = "output"
    obj_files = find_all_obj_files(output_dir)

    if not obj_files:
        print("No .obj files found in the output directory")
        exit(1)

    print(f"Found {len(obj_files)} .obj files")

    # Create renders directory
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)

    # Process each .obj file
    timestamp = time.strftime("%Y%m%d%H%M%S")
    results = []

    for i, obj_path in enumerate(obj_files, 1):
        # Create a filename based on the original obj filename
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
        output_path = os.path.join(renders_dir, f"{timestamp}_{obj_name}_cover.png")

        # Render image with default parameters (which are the best settings)
        result = render_model_cover(obj_path, output_path)

        if result:
            results.append((output_path, obj_path))
            print(f"[{i}/{len(obj_files)}] Generated: {output_path}")
        else:
            print(f"[{i}/{len(obj_files)}] Failed to generate: {output_path}")

    print(f"\nGenerated {len(results)} cover images:")
    for output_path, obj_path in results:
        print(f"- {output_path} (from {obj_path})")
