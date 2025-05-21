import argparse
import asyncio
import concurrent.futures
import multiprocessing
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from task_executor import QualityLevel, init_worker, process_model_generation

# 配置参数
parser = argparse.ArgumentParser(description="Hunyuan3D API服务")
parser.add_argument("--port", type=int, default=42121, help="服务器端口号")
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help="使用的GPU设备ID，多个设备用逗号分隔，如'0,1,2'",
)
parser.add_argument("--max_workers", type=int, default=3, help="最大并行处理任务数")
parser.add_argument("--preload_models", default=True, help="启动时预加载模型")
parser.add_argument(
    "--paint_service_url",
    type=str,
    default="http://localhost:42122",
    help="Paint服务URL",
)
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


class GenerateType(str, Enum):
    only_model = "only_model"
    both = "both"


class GenerateRequest(BaseModel):
    image_url: HttpUrl
    type: GenerateType = GenerateType.both
    quality: QualityLevel = QualityLevel.high  # 默认使用高质量


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


async def handle_model_task(
    task_id: str, image_url: str, output_type: str, quality: str
):
    """异步处理模型生成任务，在单独进程中执行GPU操作"""
    async with task_semaphore:
        try:
            # 更新任务状态
            tasks[task_id]["status"] = "downloading"
            tasks[task_id]["quality"] = quality

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

            # 下载图像
            print(f"Downloading image for task {task_id}")
            response = requests.get(image_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download image: {response.status_code}")

            with open(image_path, "wb") as f:
                f.write(response.content)

            # 准备任务数据
            task_data = {
                "task_id": task_id,
                "image_path": image_path,
                "quality": quality,
                "obj_path": obj_path,
                "obj_cover_path": obj_cover_path,
            }

            # 更新任务状态
            tasks[task_id]["status"] = "processing"

            # 在进程池中运行shape生成
            loop = asyncio.get_event_loop()
            print(f"Submitting shape generation task {task_id}")
            shape_result = await loop.run_in_executor(
                process_pool, process_model_generation, task_data
            )

            if shape_result.get("status") == "failed":
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = shape_result.get(
                    "error", "Shape generation failed"
                )
                return

            # 如果需要生成带纹理的GLB，尝试调用paint服务
            if output_type == "both":
                print(f"Attempting paint generation task {task_id}")
                try:
                    # 调用paint服务
                    async with aiohttp.ClientSession() as session:
                        paint_request = {
                            "task_id": task_id,
                            "mesh_path": obj_path,
                            "image_path": image_path,
                            "glb_path": glb_path,
                            "glb_cover_path": glb_cover_path,
                        }

                        try:
                            async with session.post(
                                f"{args.paint_service_url}/paint",
                                json=paint_request,
                                timeout=300,  # 5分钟超时
                            ) as response:
                                if response.status == 200:
                                    paint_result = await response.json()
                                    if paint_result.get("status") == "completed":
                                        # 更新GLB相关的结果
                                        tasks[task_id].update(
                                            {
                                                "glb_path": paint_result.get(
                                                    "glb_path"
                                                ),
                                                "glb_cover_path": paint_result.get(
                                                    "glb_cover_path"
                                                ),
                                                "execution_time": (
                                                    shape_result.get(
                                                        "execution_time", 0
                                                    )
                                                    or 0
                                                )
                                                + (
                                                    paint_result.get(
                                                        "execution_time", 0
                                                    )
                                                    or 0
                                                ),
                                            }
                                        )
                                        tasks[task_id]["status"] = "completed"
                                        print(
                                            f"Task {task_id}: Paint generation completed successfully"
                                        )
                                    else:
                                        tasks[task_id]["status"] = "failed"
                                        tasks[task_id]["error"] = paint_result.get(
                                            "error", "Paint generation failed"
                                        )
                                        print(
                                            f"Task {task_id}: Paint generation failed: {paint_result.get('error')}"
                                        )
                                else:
                                    error_text = await response.text()
                                    tasks[task_id]["status"] = "failed"
                                    tasks[task_id]["error"] = (
                                        f"Paint service error (HTTP {response.status}): {error_text}"
                                    )
                                    print(
                                        f"Task {task_id}: Paint service error (HTTP {response.status}): {error_text}"
                                    )
                        except asyncio.TimeoutError:
                            tasks[task_id]["status"] = "failed"
                            tasks[task_id]["error"] = "Paint service request timed out"
                            print(f"Task {task_id}: Paint service request timed out")
                        except aiohttp.ClientError as e:
                            tasks[task_id]["status"] = "failed"
                            tasks[task_id]["error"] = (
                                f"Paint service connection error: {str(e)}"
                            )
                            print(
                                f"Task {task_id}: Paint service connection error: {str(e)}"
                            )
                        except Exception as e:
                            tasks[task_id]["status"] = "failed"
                            tasks[task_id]["error"] = (
                                f"Unexpected error during paint generation: {str(e)}"
                            )
                            print(
                                f"Task {task_id}: Unexpected error during paint generation: {str(e)}"
                            )
                except Exception as e:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["error"] = (
                        f"Failed to start paint generation: {str(e)}"
                    )
                    print(f"Task {task_id}: Failed to start paint generation: {str(e)}")
            else:
                # 如果不需要paint，直接标记任务完成
                tasks[task_id]["status"] = "completed"

            # 更新shape生成的结果
            tasks[task_id].update(
                {
                    "status": "completed",
                    "obj_path": shape_result.get("obj_path"),
                    "obj_cover_path": shape_result.get("obj_cover_path"),
                    "execution_time": shape_result.get("execution_time"),
                }
            )
        except Exception as e:
            print(f"Error in handle_model_task: {str(e)}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)


@app.post("/generate", response_model=TaskResponse, status_code=202)
async def generate_3d_model(request: GenerateRequest):
    """启动3D模型生成任务"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "type": request.type,
        "quality": request.quality,
        "image_url": str(request.image_url),
        "created_at": datetime.now().isoformat(),
        "obj_path": None,
        "obj_cover_path": None,
        "glb_path": None,
        "glb_cover_path": None,
        "execution_time": None,
    }

    asyncio.create_task(
        handle_model_task(
            task_id, str(request.image_url), request.type, request.quality
        )
    )

    return {"task_id": task_id}


@app.get("/task/{task_id}", response_model=TaskInfoResponse)
async def get_task_info(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

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
    full_path = Path(file_path)
    if not str(full_path).startswith("output/"):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@app.get("/status")
async def get_status():
    """获取服务状态信息"""
    paint_service_status = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.paint_service_url}/status") as response:
                if response.status == 200:
                    paint_service_status = await response.json()
    except Exception as e:
        print(f"Failed to get paint service status: {str(e)}")

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
        "paint_service": paint_service_status,
    }


@app.on_event("shutdown")
async def shutdown_event():
    """关闭应用时清理资源"""
    global process_pool
    if process_pool:
        print("Shutting down process pool...")
        process_pool.shutdown(wait=False)
    print("Application shutdown complete")


def check_gpu_availability():
    """检查GPU可用性并返回可用的GPU ID列表"""
    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: No CUDA devices available. Running in CPU mode.")
            return []

        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("Warning: No CUDA devices found.")
            return []

        available_gpus = []
        for i in range(device_count):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(1, device=f"cuda:{i}")
                del test_tensor
                torch.cuda.empty_cache()
                available_gpus.append(i)
                print(f"GPU {i} ({torch.cuda.get_device_name(i)}) is available")
            except Exception as e:
                print(f"GPU {i} is not available: {str(e)}")

        return available_gpus
    except Exception as e:
        print(f"Error checking GPU availability: {str(e)}")
        return []


def main():
    """应用程序主入口点"""
    global \
        process_pool, \
        task_semaphore, \
        models_preloaded, \
        loader_pid, \
        gpu_ids, \
        MAX_WORKERS

    # 检查GPU可用性
    available_gpus = check_gpu_availability()
    if not available_gpus:
        print("No available GPUs found. Please check your CUDA installation.")
        return

    # 验证并调整用户指定的GPU
    requested_gpus = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
    max_gpu_id = max(available_gpus)
    invalid_gpus = [gpu for gpu in requested_gpus if gpu > max_gpu_id]

    if invalid_gpus:
        print(
            f"Warning: GPU IDs {invalid_gpus} are invalid (max available GPU ID is {max_gpu_id})"
        )
        print(f"Available GPUs: {available_gpus}")
        print("Using GPU 0 instead")
        gpu_ids = [0]
    else:
        valid_gpus = [gpu for gpu in requested_gpus if gpu in available_gpus]
        if not valid_gpus:
            print(
                f"Warning: None of the requested GPUs {requested_gpus} are available."
            )
            print("Using GPU 0 instead")
            gpu_ids = [0]
        else:
            if len(valid_gpus) < len(requested_gpus):
                print("Warning: Some requested GPUs are not available.")
                print(f"Requested GPUs: {requested_gpus}")
                print(f"Available GPUs: {available_gpus}")
                print(f"Using available GPUs: {valid_gpus}")
            gpu_ids = valid_gpus

    # 更新MAX_WORKERS
    MAX_WORKERS = min(args.max_workers, len(gpu_ids))
    if MAX_WORKERS < args.max_workers:
        print(
            f"Warning: Reducing max_workers from {args.max_workers} to {MAX_WORKERS} to match available GPUs"
        )

    print("Starting Hunyuan3D API service with:")
    print(f"- Using GPUs: {gpu_ids}")
    print(f"- Max concurrent tasks: {MAX_WORKERS}")
    print(f"- Preload models: {args.preload_models}")

    # 创建进程池和信号量
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocessing.set_start_method("spawn", force=True)

    # 创建进程池
    process_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(gpu_ids[0],),  # 使用第一个GPU初始化所有worker
    )

    # 信号量控制最大并行处理任务数
    task_semaphore = asyncio.Semaphore(MAX_WORKERS)

    # 启动服务器
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    finally:
        if process_pool:
            process_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
