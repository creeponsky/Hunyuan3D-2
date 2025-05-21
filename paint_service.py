import argparse
import os
import time
import traceback
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import model_manager
from renderer_utils import render_model_cover


class PaintRequest(BaseModel):
    task_id: str
    mesh_path: str
    image_path: str
    glb_path: str
    glb_cover_path: Optional[str] = None


class PaintResponse(BaseModel):
    status: str  # "completed" or "failed"
    glb_path: Optional[str] = None
    glb_cover_path: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


app = FastAPI()

# 全局变量
model_initialized = False


def process_paint_generation(request: PaintRequest) -> Dict:
    """Process paint generation task"""
    try:
        if not model_initialized:
            raise RuntimeError("Paint model not initialized")

        print(f"Task {request.task_id}: Starting paint generation")
        model_manager.log_gpu_memory()

        start_time = time.time()

        # Load mesh and image
        print(f"Task {request.task_id}: Loading mesh and image")
        from PIL import Image
        from trimesh import load

        mesh = load(request.mesh_path)
        image = Image.open(request.image_path)

        # Apply texture
        print(f"Task {request.task_id}: Applying texture")
        paint_pipeline = model_manager.get_model("paint_pipeline")
        textured_mesh = paint_pipeline(mesh, image)
        model_manager.log_gpu_memory()

        # Export GLB
        textured_mesh.export(request.glb_path)
        result = {"glb_path": request.glb_path}

        # Render GLB preview
        if request.glb_cover_path and False:
            print(f"Task {request.task_id}: Rendering GLB preview")
            render_result = render_model_cover(request.glb_path, request.glb_cover_path)
            if render_result:
                result["glb_cover_path"] = request.glb_cover_path

        end_time = time.time()
        result["status"] = "completed"
        result["execution_time"] = end_time - start_time

        print(
            f"Task {request.task_id}: Completed in {end_time - start_time:.2f} seconds"
        )
        return result

    except Exception as e:
        error_msg = f"Error in paint generation: {str(e)}\n{traceback.format_exc()}"
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


@app.post("/paint", response_model=PaintResponse)
async def generate_texture(request: PaintRequest):
    """Generate texture for a 3D model"""
    if not model_initialized:
        raise HTTPException(status_code=503, detail="Paint service not initialized")

    result = process_paint_generation(request)
    return result


@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "status": "ready" if model_initialized else "initializing",
        "gpu_id": model_manager.gpu_id if model_initialized else None,
        "gpu_name": torch.cuda.get_device_name(0)
        if model_initialized and torch.cuda.is_available()
        else None,
    }


def main():
    """Service entry point"""
    parser = argparse.ArgumentParser(description="Hunyuan3D Paint Service")
    parser.add_argument("--port", type=int, default=42122, help="Service port")
    parser.add_argument("--gpu", type=int, default=3, help="GPU device ID")
    args = parser.parse_args()

    global model_initialized

    try:
        # Set GPU environment variables before any other operations
        os.environ.update(
            {
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
            }
        )

        # Verify GPU setting
        if torch.cuda.is_available():
            if (
                torch.cuda.current_device() != 0
            ):  # Should be 0 because of CUDA_VISIBLE_DEVICES
                raise RuntimeError(
                    f"Failed to set GPU device. Current device: {torch.cuda.current_device()}"
                )
            print(
                f"Paint service will use GPU {args.gpu} ({torch.cuda.get_device_name(0)})"
            )

        # Initialize model
        print(f"Initializing paint service on GPU {args.gpu}")
        model_manager.initialize(args.gpu)  # This will now use the already set GPU
        model_initialized = True
        print("Paint service initialized successfully")

        # Start server
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except Exception as e:
        print(f"Failed to initialize paint service: {str(e)}")
        raise


if __name__ == "__main__":
    main()
