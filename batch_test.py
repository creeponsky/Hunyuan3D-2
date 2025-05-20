import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from model_config import ModelConfig, ModelRegistry
from renderer_utils import render_model_cover


@dataclass
class ProcessingResult:
    image_name: str
    model_name: str
    shape_time: float
    texture_time: float = 0.0
    shape_path: str = ""
    texture_path: str = ""
    preview_path: Optional[str] = None


class Hunyuan3DPerformanceTester:
    def __init__(self, output_base_dir: str = "output"):
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%y%m%d")
        self.results: List[ProcessingResult] = []
        self.model_registry = ModelRegistry()

        # Initialize pipelines as None, will be loaded on demand
        self.shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
        self.paint_pipeline: Optional[Hunyuan3DPaintPipeline] = None
        self.background_remover = BackgroundRemover()

    def _setup_output_dir(self, image_name: str) -> str:
        """Create output directory for a specific image"""
        output_dir = os.path.join(self.output_base_dir, self.timestamp, image_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _process_image(self, image_path: str) -> Image.Image:
        """Process input image (remove background if needed)"""
        image = Image.open(image_path)
        if image.mode == "RGB":
            image = self.background_remover(image)
        return image

    def _load_shape_pipeline(
        self, model_config: ModelConfig
    ) -> Hunyuan3DDiTFlowMatchingPipeline:
        """Load shape generation pipeline for specific model"""
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_config.model_id, subfolder=model_config.subfolder
        )

    def _load_paint_pipeline(self, model_config: ModelConfig) -> Hunyuan3DPaintPipeline:
        """Load paint pipeline for specific model"""
        return Hunyuan3DPaintPipeline.from_pretrained(
            model_config.model_id, subfolder=model_config.subfolder
        )

    def process_single_image(
        self, image_path: str, model_config: ModelConfig, generate_preview: bool = True
    ) -> ProcessingResult:
        """Process a single image with a specific model

        Args:
            image_path: Path to the input image
            model_config: Model configuration to use
            generate_preview: Whether to generate preview image (default: True)
        """
        image_name = Path(image_path).stem
        output_dir = self._setup_output_dir(image_name)

        # Process image
        image = self._process_image(image_path)

        # Initialize result
        result = ProcessingResult(
            image_name=image_name,
            model_name=model_config.name,
            shape_time=0.0,
            texture_time=0.0,
        )

        # Shape generation
        start_time = time.time()
        shape_pipeline = self._load_shape_pipeline(model_config)
        mesh = shape_pipeline(image=image)[0]
        shape_time = time.time() - start_time
        result.shape_time = shape_time

        # Save shape
        shape_path = os.path.join(output_dir, f"{model_config.name}_shape.obj")
        mesh.export(shape_path)
        result.shape_path = shape_path

        # Generate preview only if requested
        if generate_preview:
            try:
                preview_path = os.path.join(
                    output_dir, f"{model_config.name}_preview.png"
                )
                render_model_cover(shape_path, preview_path)
                result.preview_path = preview_path
            except Exception as e:
                print(
                    f"Warning: Failed to generate preview for {image_name} with {model_config.name}: {e}"
                )
                result.preview_path = None

        # Texture generation if needed
        if model_config.is_paint:
            start_time = time.time()
            paint_pipeline = self._load_paint_pipeline(model_config)
            textured_mesh = paint_pipeline(mesh, image=image)
            texture_time = time.time() - start_time
            result.texture_time = texture_time

            # Save textured model
            texture_path = os.path.join(output_dir, f"{model_config.name}_textured.glb")
            textured_mesh.export(texture_path)
            result.texture_path = texture_path

        return result

    def run_performance_test(
        self,
        image_paths: List[str],
        model_names: List[str],
        generate_preview: bool = True,
    ):
        """Run performance test for all images and models

        Args:
            image_paths: List of paths to input images
            model_names: List of model names to test
            generate_preview: Whether to generate preview images (default: True)
        """
        for image_path in image_paths:
            for model_name in model_names:
                try:
                    model_config = self.model_registry.get_model_config(model_name)
                    result = self.process_single_image(
                        image_path, model_config, generate_preview
                    )
                    self.results.append(result)
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue

    def generate_report(self):
        """Generate performance report and visualizations"""
        if not self.results:
            print("No results to generate report from")
            return None, None

        # Create DataFrame for results
        df = pd.DataFrame([vars(r) for r in self.results])

        # Generate performance plot
        plt.figure(figsize=(15, 8))
        for model_name in df["model_name"].unique():
            model_data = df[df["model_name"] == model_name]
            plt.bar(
                model_data["image_name"], model_data["shape_time"], label=model_name
            )

        plt.title("Shape Generation Performance by Model")
        plt.xlabel("Image")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            self.output_base_dir, self.timestamp, "performance_plot.png"
        )
        plt.savefig(plot_path)

        # Generate HTML report
        html_content = self._generate_html_report(df)
        report_path = os.path.join(self.output_base_dir, self.timestamp, "report.html")
        with open(report_path, "w") as f:
            f.write(html_content)

        return plot_path, report_path

    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML report with results and previews"""
        html = """
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid black; padding: 8px; text-align: left; }
                img { max-width: 200px; }
            </style>
        </head>
        <body>
            <h1>Hunyuan3D Performance Test Report</h1>
            <h2>Results Summary</h2>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Model</th>
                    <th>Shape Time (s)</th>
                    <th>Texture Time (s)</th>
        """

        # Only add preview column if any previews exist
        if df["preview_path"].notna().any():
            html += "<th>Preview</th>"

        html += "</tr>"

        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td>{row["image_name"]}</td>
                    <td>{row["model_name"]}</td>
                    <td>{row["shape_time"]:.2f}</td>
                    <td>{row["texture_time"]:.2f}</td>
            """

            # Only add preview cell if preview exists
            if pd.notna(row["preview_path"]):
                html += f'<td><img src="{row["preview_path"]}" alt="Preview"></td>'

            html += "</tr>"

        html += """
            </table>
        </body>
        </html>
        """
        return html


# Usage example
if __name__ == "__main__":
    # Example usage
    tester = Hunyuan3DPerformanceTester()

    # Define test parameters
    image_paths = [
        "assets/1.png",
        "assets/1.jpg",
        "assets/3.png",
        # Add more image paths here
    ]

    # Get all available models
    model_registry = ModelRegistry()
    shape_models = model_registry.get_shape_models()
    paint_models = model_registry.get_paint_models()

    # You can choose which models to test
    model_names = [
        "DiT-v2-0",  # Base model
        "DiT-v2-0-Turbo",  # Turbo version
        "DiT-v2-0-Fast",
        # "Paint-v2-0",
        # "Paint-v2-0-Turbo",
        "DiT-v2-mini",
        "DiT-v2-mini-Turbo",
        "DiT-v2-mini-Fast",
    ]

    # Run tests
    tester.run_performance_test(image_paths, model_names, generate_preview=False)

    # Generate report
    plot_path, report_path = tester.generate_report()
    if plot_path and report_path:
        print(f"Performance plot saved to: {plot_path}")
        print(f"HTML report saved to: {report_path}")
