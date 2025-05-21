import os
import time
import traceback
from enum import Enum
from typing import Any, Dict

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from models import model_manager
from renderer_utils import render_model_cover


class QualityLevel(str, Enum):
    low = "low"  # 粗略: steps=10, resolution=64
    medium_low = "medium_low"  # 中低: steps=20, resolution=128
    medium = "medium"  # 中等: steps=30, resolution=192
    medium_high = "medium_high"  # 中高: steps=40, resolution=256
    high = "high"  # 精细: steps=50, resolution=384


# Quality level parameters
QUALITY_PARAMS = {
    QualityLevel.low: {"num_inference_steps": 10, "octree_resolution": 64},
    QualityLevel.medium_low: {"num_inference_steps": 20, "octree_resolution": 128},
    QualityLevel.medium: {"num_inference_steps": 30, "octree_resolution": 192},
    QualityLevel.medium_high: {"num_inference_steps": 40, "octree_resolution": 256},
    QualityLevel.high: {"num_inference_steps": 50, "octree_resolution": 384},
}


def init_worker(gpu_id: int) -> None:
    """Initialize worker process with shape model on specified GPU"""
    model_manager.initialize(gpu_id)


def process_model_generation(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process shape generation task in a separate process"""
    try:
        task_id = task_data["task_id"]
        image_path = task_data["image_path"]
        quality = QualityLevel(task_data["quality"])
        obj_path = task_data["obj_path"]
        obj_cover_path = task_data.get("obj_cover_path")

        worker_id = os.getpid()
        print(f"Worker process {worker_id}: Starting shape generation task {task_id}")
        print(f"Task parameters: quality={quality}")
        model_manager.log_gpu_memory()

        start_time = time.time()

        # Load and process image
        print(f"Task {task_id}: Loading image from {image_path}")
        image = Image.open(image_path)
        if image.mode == "RGB":
            print(f"Task {task_id}: Removing background")
            rembg = BackgroundRemover()
            image = rembg(image)
        model_manager.log_gpu_memory()

        # Generate 3D mesh
        print(f"Task {task_id}: Generating shape with quality {quality}")
        try:
            params = QUALITY_PARAMS[quality]
            mesh = model_manager.get_model("shape_pipeline")(
                image=image,
                num_inference_steps=params["num_inference_steps"],
                octree_resolution=params["octree_resolution"],
            )[0]
            model_manager.log_gpu_memory()
        except Exception as e:
            print(f"Task {task_id}: Error during shape generation: {str(e)}")
            print(f"Task {task_id}: GPU memory state before error:")
            model_manager.log_gpu_memory()
            raise

        # Save OBJ file
        mesh.export(obj_path)
        result = {"obj_path": obj_path}

        # Render OBJ preview
        if obj_cover_path:
            print(f"Task {task_id}: Rendering OBJ preview")
            render_result = render_model_cover(obj_path, obj_cover_path)
            if render_result:
                result["obj_cover_path"] = obj_cover_path

        end_time = time.time()
        result["status"] = "completed"
        result["execution_time"] = end_time - start_time

        print(f"Task {task_id}: Completed in {end_time - start_time:.2f} seconds")
        return result

    except Exception as e:
        error_msg = f"Error in shape generation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        try:
            import psutil

            process = psutil.Process()
            print(f"Process memory info: {process.memory_info().rss / 1024**2:.2f}MB")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"CUDA device name: {torch.cuda.get_device_name()}")
        except Exception as sys_info_error:
            print(f"Failed to get system info: {str(sys_info_error)}")
        return {"status": "failed", "error": str(e)}
