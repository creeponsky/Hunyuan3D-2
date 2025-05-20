from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelConfig:
    name: str
    model_id: str
    subfolder: str
    is_turbo: bool = False
    is_paint: bool = False
    size: str = ""
    date: str = ""
    description: str = ""


class ModelRegistry:
    def __init__(self):
        # Shape generation models (excluding paint models)
        self.shape_models: Dict[str, ModelConfig] = {
            # Hunyuan3D-2mini Series
            "DiT-v2-mini": ModelConfig(
                name="DiT-v2-mini",
                model_id="tencent/Hunyuan3D-2mini",
                subfolder="hunyuan3d-dit-v2-mini",
                size="0.6B",
                date="2025-03-18",
                description="Mini Image to Shape Model",
            ),
            "DiT-v2-mini-Turbo": ModelConfig(
                name="DiT-v2-mini-Turbo",
                model_id="tencent/Hunyuan3D-2mini",
                subfolder="hunyuan3d-dit-v2-mini-turbo",
                is_turbo=True,
                size="0.6B",
                date="2025-03-19",
                description="Step Distillation Version",
            ),
            "DiT-v2-mini-Fast": ModelConfig(
                name="DiT-v2-mini-Fast",
                model_id="tencent/Hunyuan3D-2mini",
                subfolder="hunyuan3d-dit-v2-mini-fast",
                size="0.6B",
                date="2025-03-18",
                description="Guidance Distillation Version",
            ),
            # Hunyuan3D-2mv Series
            "DiT-v2-mv": ModelConfig(
                name="DiT-v2-mv",
                model_id="tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv",
                size="1.1B",
                date="2025-03-18",
                description="Multiview Image to Shape Model",
            ),
            "DiT-v2-mv-Turbo": ModelConfig(
                name="DiT-v2-mv-Turbo",
                model_id="tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv-turbo",
                is_turbo=True,
                size="1.1B",
                date="2025-03-19",
                description="Step Distillation Version",
            ),
            "DiT-v2-mv-Fast": ModelConfig(
                name="DiT-v2-mv-Fast",
                model_id="tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv-fast",
                size="1.1B",
                date="2025-03-18",
                description="Guidance Distillation Version",
            ),
            # Hunyuan3D-2 Series
            "DiT-v2-0": ModelConfig(
                name="DiT-v2-0",
                model_id="tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-dit-v2-0",
                size="1.1B",
                date="2025-01-21",
                description="Image to Shape Model",
            ),
            "DiT-v2-0-Turbo": ModelConfig(
                name="DiT-v2-0-Turbo",
                model_id="tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-dit-v2-0-turbo",
                is_turbo=True,
                size="1.1B",
                date="2025-03-19",
                description="Step Distillation Model",
            ),
            "DiT-v2-0-Fast": ModelConfig(
                name="DiT-v2-0-Fast",
                model_id="tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-dit-v2-0-fast",
                size="1.1B",
                date="2025-02-03",
                description="Guidance Distillation Model",
            ),
        }

        # Paint models
        self.paint_models: Dict[str, ModelConfig] = {
            "Paint-v2-0": ModelConfig(
                name="Paint-v2-0",
                model_id="tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-paint-v2-0",
                is_paint=True,
                size="1.3B",
                date="2025-01-21",
                description="Texture Generation Model",
            ),
            "Paint-v2-0-Turbo": ModelConfig(
                name="Paint-v2-0-Turbo",
                model_id="tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-paint-v2-0-turbo",
                is_paint=True,
                is_turbo=True,
                size="1.3B",
                date="2025-04-01",
                description="Distillation Texture Model",
            ),
        }

    def get_shape_models(self) -> List[str]:
        """Get list of all shape model names"""
        return list(self.shape_models.keys())

    def get_paint_models(self) -> List[str]:
        """Get list of all paint model names"""
        return list(self.paint_models.keys())

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration by name"""
        if model_name in self.shape_models:
            return self.shape_models[model_name]
        if model_name in self.paint_models:
            return self.paint_models[model_name]
        raise ValueError(f"Model {model_name} not found in registry")
