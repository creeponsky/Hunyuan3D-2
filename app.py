import argparse
import asyncio
import concurrent.futures
import multiprocessing
import os
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from renderer_utils import render_model_cover

# 配置参数
parser = argparse.ArgumentParser(description="Hunyuan3D API服务")
parser.add_argument("--port", type=int, default=42121, help="服务器端口号")
parser.add_argument(
    "--gpus",
    type=str,
    default="3",
    help="使用的GPU设备ID，多个设备用逗号分隔，如'0,1,2'",
)
parser.add_argument("--max_workers", type=int, default=3, help="最大并行处理任务数")
parser.add_argument("--preload_models", default=True, help="启动时预加载模型")
args = parser.parse_args()

# 解析GPU设备
gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
MAX_WORKERS = min(args.max_workers, len(gpu_ids))  # 确保工作进程数不超过GPU数量

app = FastAPI()

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 全局变量
process_pool = None
task_semaphore = None
models_preloaded = False
loader_pid = None

# 工作进程全局变量
worker_models = {"shape_pipeline": None, "paint_pipeline": None}


class GenerateType(str, Enum):
    only_model = "only_model"
    both = "both"


class QualityLevel(str, Enum):
    low = "low"  # 粗略: steps=10, resolution=64
    medium = "medium"  # 中等: steps=30, resolution=192
    high = "high"  # 精细: steps=50, resolution=384


class GenerateRequest(BaseModel):
    image_url: HttpUrl
    type: GenerateType = GenerateType.both
    quality: QualityLevel = QualityLevel.high  # 默认使用中等质量


class TaskResponse(BaseModel):
    task_id: str


class TaskInfoResponse(BaseModel):
    status: str  # "pending", "processing", "completed", "failed"
    obj_path: Optional[str] = None
    obj_cover_path: Optional[str] = None
    glb_cover_path: Optional[str] = None
    glb_path: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


# 初始化工作进程
def init_worker():
    """设置工作进程的环境变量和GPU设备，并预加载模型"""
    global worker_models
    worker_id = multiprocessing.current_process()._identity[0] - 1
    gpu_id = gpu_ids[worker_id % len(gpu_ids)]

    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        }
    )
    print(f"Worker process {worker_id} initialized on GPU {gpu_id}")

    # 预加载模型
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        print(f"Worker {worker_id}: Loading shape generation model...")
        worker_models["shape_pipeline"] = (
            Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini-turbo"
            )
        )

        print(f"Worker {worker_id}: Loading paint pipeline model...")
        worker_models["paint_pipeline"] = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
        )
        print(f"Worker {worker_id}: Models loaded successfully")
    except Exception as e:
        print(f"Worker {worker_id}: Failed to load models: {str(e)}")
        raise


# 在单独进程中执行的GPU处理函数
def process_model_generation(task_data):
    """
    在单独进程中运行的模型生成函数
    """
    try:
        # 导入GPU相关模块
        import torch
        import trimesh
        from PIL import Image

        from hy3dgen.rembg import BackgroundRemover

        # 记录GPU内存使用情况
        def log_gpu_memory():
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                print(
                    f"GPU {gpu_id} Memory - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB"
                )

        # 解析任务数据
        task_id = task_data["task_id"]
        image_path = task_data["image_path"]
        output_type = task_data["output_type"]
        quality = QualityLevel(task_data["quality"])
        obj_path = task_data["obj_path"]
        glb_path = task_data["glb_path"]
        obj_cover_path = task_data.get("obj_cover_path")
        glb_cover_path = task_data.get("glb_cover_path")

        # 根据质量级别设置参数
        quality_params = {
            QualityLevel.low: {"num_inference_steps": 10, "octree_resolution": 64},
            QualityLevel.medium: {"num_inference_steps": 30, "octree_resolution": 192},
            QualityLevel.high: {"num_inference_steps": 50, "octree_resolution": 384},
        }
        params = quality_params[quality]

        worker_id = multiprocessing.current_process()._identity[0] - 1
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        current_pid = os.getpid()

        print(
            f"Worker process {worker_id} (PID {current_pid}) on GPU {gpu_id}: Starting task {task_id}"
        )
        print(f"Task parameters: quality={quality}, output_type={output_type}")
        log_gpu_memory()

        start_time = time.time()

        # 加载图像
        print(f"Task {task_id}: Loading image from {image_path}")
        image = Image.open(image_path)
        if image.mode == "RGB":
            print(f"Task {task_id}: Removing background")
            rembg = BackgroundRemover()
            image = rembg(image)
        log_gpu_memory()

        # 使用预加载的模型生成3D网格
        print(f"Task {task_id}: Generating shape with quality {quality}")
        try:
            mesh = worker_models["shape_pipeline"](
                image=image,
                num_inference_steps=params["num_inference_steps"],
                octree_resolution=params["octree_resolution"],
            )[0]
            log_gpu_memory()
        except Exception as e:
            print(f"Task {task_id}: Error during shape generation: {str(e)}")
            print(f"Task {task_id}: GPU memory state before error:")
            log_gpu_memory()
            raise

        # 保存OBJ文件
        mesh.export(obj_path)
        result = {"obj_path": obj_path}

        # 渲染OBJ模型预览图
        if obj_cover_path:
            print(f"Task {task_id}: Rendering OBJ preview")
            render_result = render_model_cover(obj_path, obj_cover_path)
            if render_result:
                result["obj_cover_path"] = obj_cover_path

        # 如果需要生成带纹理的GLB
        if output_type == "both":
            print(f"Task {task_id}: Starting texturing")
            try:
                # 加载生成的网格
                trimesh_mesh = trimesh.load(obj_path)
                log_gpu_memory()

                # 使用预加载的模型应用纹理
                textured_mesh = worker_models["paint_pipeline"](
                    trimesh_mesh, image=[image]
                )
                log_gpu_memory()

                # 导出为GLB
                textured_mesh.export(glb_path)
                result["glb_path"] = glb_path

                # 渲染GLB模型预览图
                if glb_cover_path:
                    print(f"Task {task_id}: Rendering GLB preview")
                    render_result = render_model_cover(glb_path, glb_cover_path)
                    if render_result:
                        result["glb_cover_path"] = glb_cover_path
            except Exception as e:
                print(f"Task {task_id}: Error during texturing: {str(e)}")
                print(f"Task {task_id}: GPU memory state before error:")
                log_gpu_memory()
                raise

        end_time = time.time()
        result["status"] = "completed"
        result["execution_time"] = end_time - start_time

        print(
            f"Task {task_id}: Completed in {end_time - start_time:.2f} seconds on GPU {gpu_id}"
        )
        return result

    except Exception as e:
        error_msg = f"Error in worker process: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # 尝试记录更多系统信息
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


async def handle_model_task(
    task_id: str, image_url: str, output_type: str, quality: str
):
    """
    异步处理模型生成任务，在单独进程中执行GPU操作
    """
    # 使用信号量限制并发任务数
    async with task_semaphore:
        try:
            # 更新任务状态
            tasks[task_id]["status"] = "downloading"
            tasks[task_id]["quality"] = quality  # 记录质量级别

            # 为当前任务创建目录
            date_str = datetime.now().strftime("%Y%m%d")
            task_dir = OUTPUT_DIR / date_str
            task_dir.mkdir(exist_ok=True)

            # 生成唯一文件名
            obj_filename = f"{task_id}.obj"
            glb_filename = f"{task_id}.glb"
            image_filename = f"{task_id}.png"
            obj_cover_filename = f"{task_id}_obj_cover.png"
            glb_cover_filename = f"{task_id}_glb_cover.png"

            obj_path = str(task_dir / obj_filename)
            glb_path = str(task_dir / glb_filename)
            image_path = str(task_dir / image_filename)
            obj_cover_path = str(task_dir / obj_cover_filename)
            glb_cover_path = str(task_dir / glb_cover_filename)

            # 下载图像并保存到本地 - 这在主进程中进行是安全的
            print(f"Downloading image for task {task_id}")
            response = requests.get(image_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {response.status_code}")

            # 保存图像到本地文件系统
            with open(image_path, "wb") as f:
                f.write(response.content)

            # 准备发送到工作进程的任务数据
            task_data = {
                "task_id": task_id,
                "image_path": image_path,
                "output_type": output_type,
                "quality": quality,  # 添加质量参数
                "obj_path": obj_path,
                "glb_path": glb_path,
                "obj_cover_path": obj_cover_path,
                "glb_cover_path": glb_cover_path if output_type == "both" else None,
            }

            # 更新任务状态
            tasks[task_id]["status"] = "processing"

            # 在进程池中运行GPU密集型操作
            loop = asyncio.get_event_loop()
            print(f"Submitting task {task_id} to process pool")
            result = await loop.run_in_executor(
                process_pool, process_model_generation, task_data
            )

            # 更新任务状态
            if result.get("status") == "failed":
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = result.get("error", "Unknown error")
            else:
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["obj_path"] = result.get("obj_path")
                tasks[task_id]["obj_cover_path"] = result.get("obj_cover_path")
                if "glb_path" in result:
                    tasks[task_id]["glb_path"] = result.get("glb_path")
                if "glb_cover_path" in result:
                    tasks[task_id]["glb_cover_path"] = result.get("glb_cover_path")
                tasks[task_id]["execution_time"] = result.get("execution_time")

        except Exception as e:
            print(f"Error in handle_model_task: {str(e)}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)


@app.post("/generate", response_model=TaskResponse, status_code=202)
async def generate_3d_model(request: GenerateRequest):
    """启动3D模型生成任务"""
    # 创建任务ID
    task_id = str(uuid.uuid4())

    # 初始化任务信息
    tasks[task_id] = {
        "status": "pending",
        "type": request.type,
        "quality": request.quality,  # 记录质量级别
        "image_url": str(request.image_url),
        "created_at": datetime.now().isoformat(),
        "obj_path": None,
        "obj_cover_path": None,
        "glb_path": None,
        "glb_cover_path": None,
        "execution_time": None,
    }

    # 创建后台任务，但不等待其完成
    asyncio.create_task(
        handle_model_task(
            task_id, str(request.image_url), request.type, request.quality
        )
    )

    return {"task_id": task_id}


@app.get("/task/{task_id}", response_model=TaskInfoResponse)
async def get_task_info(task_id: str):
    """获取任务状态，这个函数必须快速返回，不能阻塞"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # 创建任务状态的副本，避免任何阻塞
    task_info = {
        "status": tasks[task_id]["status"],
        "obj_path": tasks[task_id]["obj_path"],
        "obj_cover_path": tasks[task_id]["obj_cover_path"],
        "glb_path": tasks[task_id]["glb_path"],
        "glb_cover_path": tasks[task_id]["glb_cover_path"],
        "execution_time": tasks[task_id]["execution_time"],
        "error": tasks[task_id].get("error"),
    }

    return task_info


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """下载生成的文件"""
    # 安全检查 - 确保路径在output目录内
    full_path = Path(file_path)

    # 确保路径以"output/"开头
    if not str(full_path).startswith("output/"):
        raise HTTPException(status_code=403, detail="Access denied")

    # 检查文件是否存在
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@app.get("/status")
async def get_status():
    """获取服务状态信息"""
    return {
        "gpu_ids": gpu_ids,
        "max_workers": MAX_WORKERS,
        "active_tasks": MAX_WORKERS - task_semaphore._value,
        "queue_length": len(task_semaphore._waiters) if task_semaphore._waiters else 0,
        "models_preloaded": models_preloaded,
        "loader_pid": loader_pid,
        "total_tasks": len(tasks),
        "active_tasks_ids": [
            task_id
            for task_id, task in tasks.items()
            if task["status"] in ["pending", "downloading", "processing"]
        ],
    }


@app.on_event("shutdown")
async def shutdown_event():
    """关闭应用时清理资源"""
    global process_pool
    if process_pool:
        print("Shutting down process pool...")
        process_pool.shutdown(wait=False)
    print("Application shutdown complete")


def main():
    """应用程序主入口点"""
    global process_pool, task_semaphore, models_preloaded, loader_pid

    print("Starting Hunyuan3D API service with:")
    print(f"- Using GPUs: {gpu_ids}")
    print(f"- Max concurrent tasks: {MAX_WORKERS}")
    print(f"- Preload models: {args.preload_models}")

    # 创建进程池和信号量
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        # 确保在启动新进程前设置spawn方法
        multiprocessing.set_start_method("spawn", force=True)

    # 创建进程池
    process_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
    )

    # 信号量控制最大并行处理任务数
    task_semaphore = asyncio.Semaphore(MAX_WORKERS)

    # 启动服务器
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    finally:
        # 确保进程池关闭
        if process_pool:
            process_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
