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
    num_inference_steps: int
    octree_resolution: int
    shape_time: float
    texture_time: float = 0.0
    shape_path: str = ""
    texture_path: str = ""
    preview_path: Optional[str] = None


class Hunyuan3DPerformanceTester:
    def __init__(self, output_base_dir: str = "output"):
        """
        初始化性能测试器，包括输出目录、模型注册表、结果列表等。
        """
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%y%m%d")
        self.results: List[ProcessingResult] = []
        self.model_registry = ModelRegistry()
        # Initialize pipelines as None, will be loaded on demand
        self.shape_pipeline: Optional[Hunyuan3DDiTFlowMatchingPipeline] = None
        self.paint_pipeline: Optional[Hunyuan3DPaintPipeline] = None
        self.background_remover = BackgroundRemover()

    def _setup_output_dir(self, image_name: str) -> str:
        """
        为每张图片创建对应的输出目录，便于分类保存结果。
        """
        output_dir = os.path.join(self.output_base_dir, self.timestamp, image_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _process_image(self, image_path: str) -> Image.Image:
        """
        读取图片并根据需要去除背景，返回处理后的图片对象。
        """
        image = Image.open(image_path)
        if image.mode == "RGB":
            image = self.background_remover(image)
        return image

    def _load_shape_pipeline(
        self, model_config: ModelConfig
    ) -> Hunyuan3DDiTFlowMatchingPipeline:
        """
        根据模型配置加载3D形状生成的pipeline。
        """
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_config.model_id, subfolder=model_config.subfolder, use_safetensors=False
        )

    def _load_paint_pipeline(self, model_config: ModelConfig) -> Hunyuan3DPaintPipeline:
        """
        根据模型配置加载纹理生成的pipeline。
        """
        return Hunyuan3DPaintPipeline.from_pretrained(
            model_config.model_id, subfolder=model_config.subfolder
        )

    def process_single_image(
        self,
        image_path: str,
        model_config: ModelConfig,
        num_inference_steps: int = 50,
        octree_resolution: int = 256,
        generate_preview: bool = True
    ) -> ProcessingResult:
        """
        对单张图片、单个模型、指定推理步数和octree分辨率进行完整的3D生成和（可选）纹理生成，并保存结果。
        """
        image_name = Path(image_path).stem
        output_dir = self._setup_output_dir(image_name)
        # Process image
        image = self._process_image(image_path)
        # Initialize result
        result = ProcessingResult(
            image_name=image_name,
            model_name=model_config.name,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            shape_time=0.0,
            texture_time=0.0,
        )
        # Shape generation
        start_time = time.time()
        shape_pipeline = self._load_shape_pipeline(model_config)
        mesh = shape_pipeline(
            image=image,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution
        )[0]
        shape_time = time.time() - start_time
        result.shape_time = shape_time
        # Save shape
        shape_path = os.path.join(output_dir, f"{model_config.name}_steps{num_inference_steps}_res{octree_resolution}_shape.obj")
        mesh.export(shape_path)
        result.shape_path = shape_path
        # Generate preview only if requested
        if generate_preview:
            try:
                preview_path = os.path.join(
                    output_dir, f"{model_config.name}_steps{num_inference_steps}_res{octree_resolution}_preview.png"
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
            texture_path = os.path.join(output_dir, f"{model_config.name}_steps{num_inference_steps}_res{octree_resolution}_textured.glb")
            textured_mesh.export(texture_path)
            result.texture_path = texture_path
        return result

    def run_performance_test(
        self,
        image_paths: List[str],
        model_names: List[str],
        num_inference_steps_list: List[int] = [50],
        octree_resolution_list: List[int] = [256],
        generate_preview: bool = True,
    ):
        """
        批量遍历所有图片、模型、推理步数和octree分辨率，执行性能测试，收集所有结果。
        """
        for image_path in image_paths:
            for model_name in model_names:
                for num_inference_steps in num_inference_steps_list:
                    for octree_resolution in octree_resolution_list:
                        try:
                            model_config = self.model_registry.get_model_config(model_name)
                            result = self.process_single_image(
                                image_path, model_config, num_inference_steps, octree_resolution, generate_preview
                            )
                            self.results.append(result)
                        except ValueError as e:
                            print(f"Warning: {e}")
                            continue

    def generate_report(self):
        """
        根据所有测试结果生成性能可视化图表和HTML报告。
        """
        if not self.results:
            print("No results to generate report from")
            return None, None
        # Create DataFrame for results
        df = pd.DataFrame([vars(r) for r in self.results])
        # Generate performance plot
        plt.figure(figsize=(15, 8))
        for model_name in df["model_name"].unique():
            for num_inference_steps in df["num_inference_steps"].unique():
                for octree_resolution in df["octree_resolution"].unique():
                    model_data = df[
                        (df["model_name"] == model_name) &
                        (df["num_inference_steps"] == num_inference_steps) &
                        (df["octree_resolution"] == octree_resolution)
                    ]
                    if not model_data.empty:
                        plt.bar(
                            model_data["image_name"] + f"\n{model_name}_steps{num_inference_steps}_res{octree_resolution}",
                            model_data["shape_time"],
                            label=f"{model_name}_steps{num_inference_steps}_res{octree_resolution}"
                        )
        plt.title("Shape Generation Performance by Model, Steps, and Octree Resolution")
        plt.xlabel("Image & Model & Steps & Resolution")
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
        """
        根据结果DataFrame生成HTML格式的详细报告，包含表格和预览图。
        """
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
                    <th>Steps</th>
                    <th>Octree Resolution</th>
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
                    <td>{row["num_inference_steps"]}</td>
                    <td>{row["octree_resolution"]}</td>
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
    tester = Hunyuan3DPerformanceTester()

    image_paths = [
        "./test_input/4.png",
        "./test_input/8.png",
        "./test_input/10.png",
        "./test_input/11.png",
        # Add more image paths here
    ]

    model_registry = ModelRegistry()
    shape_models = model_registry.get_shape_models()
    paint_models = model_registry.get_paint_models()

    model_names = [
        "DiT-v2-0",  # Base model
        # "DiT-v2-mini",
        # "DiT-v2-mini-Turbo",
        # "DiT-v2-1",
    ]

    num_inference_steps_list = [10, 50]  # low, high
    octree_resolution_list = [128, 256] # low, medium, high resolution
    tester.run_performance_test(
        image_paths,
        model_names,
        num_inference_steps_list,
        octree_resolution_list,
        generate_preview=False
    )

    plot_path, report_path = tester.generate_report()
    if plot_path and report_path:
        print(f"Performance plot saved to: {plot_path}")
        print(f"HTML report saved to: {report_path}")
