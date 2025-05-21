import os
from typing import Dict, Optional

import torch


class ModelManager:
    def __init__(self):
        self.models: Dict[str, Optional[torch.nn.Module]] = {
            "shape_pipeline": None,
            "paint_pipeline": None,
        }
        self.gpu_id: Optional[int] = None

    def initialize(self, gpu_id: int) -> None:
        """Initialize models on specified GPU"""
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA devices available")

            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise RuntimeError("No CUDA devices found")

            # Check if CUDA_VISIBLE_DEVICES is already set
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                # Only set environment variables if not already set
                os.environ.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
                    }
                )
            else:
                # Use the already set GPU
                gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
                print(f"Using already set GPU {gpu_id}")

            # Verify GPU setting
            if (
                torch.cuda.current_device() != 0
            ):  # Because of CUDA_VISIBLE_DEVICES, current device should always be 0
                raise RuntimeError(
                    f"Failed to set GPU device. Current device: {torch.cuda.current_device()}"
                )

            self.gpu_id = gpu_id
            print(
                f"ModelManager initialized on GPU {gpu_id} ({torch.cuda.get_device_name(0)})"
            )

            # Load models
            self._load_models()

        except Exception as e:
            print(f"ModelManager initialization failed: {str(e)}")
            raise

    def _load_models(self) -> None:
        """Load all required models"""
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            print("Loading shape generation model...")
            self.models["shape_pipeline"] = (
                Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini-turbo"
                )
            )

            print("Loading paint pipeline model...")
            self.models["paint_pipeline"] = Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
            )
            print("All models loaded successfully")

        except Exception as e:
            print(f"Failed to load models: {str(e)}")
            raise

    def get_model(self, model_name: str) -> torch.nn.Module:
        """Get a specific model by name"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        if self.models[model_name] is None:
            raise RuntimeError(f"Model {model_name} not loaded")
        return self.models[model_name]

    def log_gpu_memory(self) -> None:
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
            print(
                f"GPU {gpu_id} Memory - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB"
            )


# Global model manager instance
model_manager = ModelManager()
