import os
import uuid
import shutil
import multiprocessing
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
from pathlib import Path
import time
import traceback
import concurrent.futures
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
from PIL import Image
import requests
from io import BytesIO
import trimesh

# 确保在导入GPU相关库之前设置多进程启动方式
multiprocessing.set_start_method("spawn", force=True)

app = FastAPI()

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 初始化工作进程
def init_worker():
    """设置工作进程的环境变量"""
    os.environ.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
    })
    # 避免在工作进程中导入这些模块，确保在需要时才导入
    print("Worker process initialized")

# 创建进程池
process_pool = concurrent.futures.ProcessPoolExecutor(
    max_workers=1,  # 限制为1个进程处理GPU任务，避免GPU内存争用
    initializer=init_worker
)

class GenerateType(str, Enum):
    only_model = "only_model"
    both = "both"

class GenerateRequest(BaseModel):
    image_url: HttpUrl
    type: GenerateType = GenerateType.both

class TaskResponse(BaseModel):
    task_id: str

class TaskInfoResponse(BaseModel):
    status: str  # "pending", "processing", "completed", "failed"
    obj_path: Optional[str] = None
    glb_path: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

# 在单独进程中执行的GPU处理函数
def process_model_generation(task_data):
    """
    在单独进程中运行的模型生成函数
    """
    try:
        # 这里才导入GPU相关模块，避免在主进程中加载
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        # 解析任务数据
        task_id = task_data["task_id"]
        image_path = task_data["image_path"]
        output_type = task_data["output_type"]
        obj_path = task_data["obj_path"]
        glb_path = task_data["glb_path"]
        
        print(f"Worker process: Starting task {task_id}")
        start_time = time.time()
        
        # 加载图像
        image = Image.open(image_path)
        if image.mode == "RGB":
            rembg = BackgroundRemover()
            image = rembg(image)
        
        # 加载模型
        print(f"Worker process: Loading shape generation model for task {task_id}")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
        
        # 生成3D网格
        print(f"Worker process: Generating shape for task {task_id}")
        mesh = shape_pipeline(image=image)[0]
        
        # 保存OBJ文件
        mesh.export(obj_path)
        result = {"obj_path": obj_path}
        
        # 如果需要生成带纹理的GLB
        if output_type == "both":
            print(f"Worker process: Starting texturing for task {task_id}")
            paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2", 
                subfolder="hunyuan3d-paint-v2-0-turbo"
            )
            
            # 加载生成的网格
            trimesh_mesh = trimesh.load(obj_path)
            
            # 应用纹理
            textured_mesh = paint_pipeline(trimesh_mesh, image=[image])
            
            # 导出为GLB
            textured_mesh.export(glb_path)
            result["glb_path"] = glb_path
        
        end_time = time.time()
        result["status"] = "completed"
        result["execution_time"] = end_time - start_time
        
        print(f"Worker process: Task {task_id} completed in {end_time - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = f"Error in worker process: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "status": "failed",
            "error": str(e)
        }

async def handle_model_task(task_id: str, image_url: str, output_type: str):
    """
    异步处理模型生成任务，在单独进程中执行GPU操作
    """
    try:
        # 更新任务状态
        tasks[task_id]["status"] = "downloading"
        
        # 为当前任务创建目录
        date_str = datetime.now().strftime("%Y%m%d")
        task_dir = OUTPUT_DIR / date_str
        task_dir.mkdir(exist_ok=True)
        
        # 生成唯一文件名
        obj_filename = f"{task_id}.obj"
        glb_filename = f"{task_id}.glb"
        image_filename = f"{task_id}.png"
        
        obj_path = str(task_dir / obj_filename)
        glb_path = str(task_dir / glb_filename)
        image_path = str(task_dir / image_filename)
        
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
            "obj_path": obj_path,
            "glb_path": glb_path
        }
        
        # 更新任务状态
        tasks[task_id]["status"] = "processing"
        
        # 在进程池中运行GPU密集型操作
        loop = asyncio.get_event_loop()
        print(f"Submitting task {task_id} to process pool")
        result = await loop.run_in_executor(
            process_pool, 
            process_model_generation, 
            task_data
        )
        
        # 更新任务状态
        if result.get("status") == "failed":
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = result.get("error", "Unknown error")
        else:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["obj_path"] = result.get("obj_path")
            if "glb_path" in result:
                tasks[task_id]["glb_path"] = result.get("glb_path")
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
        "image_url": str(request.image_url),
        "created_at": datetime.now().isoformat(),
        "obj_path": None,
        "glb_path": None,
        "execution_time": None
    }
    
    # 创建后台任务，但不等待其完成
    asyncio.create_task(
        handle_model_task(task_id, str(request.image_url), request.type)
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
        "glb_path": tasks[task_id]["glb_path"],
        "execution_time": tasks[task_id]["execution_time"],
        "error": tasks[task_id].get("error")
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

@app.on_event("shutdown")
async def shutdown_event():
    """关闭应用时清理资源"""
    if process_pool:
        print("Shutting down process pool...")
        process_pool.shutdown(wait=False)
    print("Application shutdown complete")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=42121)
    finally:
        # 确保进程池关闭
        if process_pool:
            process_pool.shutdown(wait=False) 