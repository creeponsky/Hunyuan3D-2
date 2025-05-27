import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
import torch
from task_executor import QualityLevel, init_worker, process_model_generation

# 定义更细粒度的质量参数组合
QUALITY_COMBINATIONS = [
    # steps=10, different resolutions
    {"steps": 10, "resolution": 64},
    {"steps": 10, "resolution": 96},
    {"steps": 10, "resolution": 128},
    {"steps": 10, "resolution": 192},
    {"steps": 10, "resolution": 256},
    {"steps": 10, "resolution": 384},
    
    # steps=20, different resolutions
    {"steps": 20, "resolution": 64},
    {"steps": 20, "resolution": 96},
    {"steps": 20, "resolution": 128},
    {"steps": 20, "resolution": 192},
    {"steps": 20, "resolution": 256},
    {"steps": 20, "resolution": 384},
    
    # steps=30, different resolutions
    {"steps": 30, "resolution": 64},
    {"steps": 30, "resolution": 96},
    {"steps": 30, "resolution": 128},
    {"steps": 30, "resolution": 192},
    {"steps": 30, "resolution": 256},
    {"steps": 30, "resolution": 384},
    
    # steps=40, different resolutions
    {"steps": 40, "resolution": 64},
    {"steps": 40, "resolution": 96},
    {"steps": 40, "resolution": 128},
    {"steps": 40, "resolution": 192},
    {"steps": 40, "resolution": 256},
    {"steps": 40, "resolution": 384},
    
    # steps=50, different resolutions
    {"steps": 50, "resolution": 64},
    {"steps": 50, "resolution": 96},
    {"steps": 50, "resolution": 128},
    {"steps": 50, "resolution": 192},
    {"steps": 50, "resolution": 256},
    {"steps": 50, "resolution": 384},
]

def process_single_model(params: Dict, image_path: str, output_dir: Path) -> Dict:
    """处理单个模型生成任务"""
    try:
        # 生成任务ID和文件名
        task_id = f"steps{params['steps']}_res{params['resolution']}"
        date_str = datetime.now().strftime("%Y%m%d")
        task_dir = output_dir / date_str
        task_dir.mkdir(exist_ok=True)
        
        # 生成文件路径
        obj_filename = f"{task_id}.obj"
        image_filename = f"{task_id}.png"
        obj_cover_filename = f"{task_id}_obj_cover.png"
        
        obj_path = str(task_dir / obj_filename)
        obj_cover_path = str(task_dir / obj_cover_filename)
        
        # 准备任务数据
        task_data = {
            "task_id": task_id,
            "image_path": image_path,
            "quality": QualityLevel.high,  # 使用high作为基础，但会覆盖具体参数
            "obj_path": obj_path,
            "obj_cover_path": obj_cover_path,
            "num_inference_steps": params["steps"],
            "octree_resolution": params["resolution"]
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行模型生成
        result = process_model_generation(task_data)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        if result.get("status") == "completed":
            return {
                "params": f"steps={params['steps']}, resolution={params['resolution']}",
                "execution_time": round(execution_time, 2),
                "obj_path": obj_path,
                "obj_cover_path": obj_cover_path,
                "status": "success"
            }
        else:
            return {
                "params": f"steps={params['steps']}, resolution={params['resolution']}",
                "execution_time": round(execution_time, 2),
                "obj_path": None,
                "obj_cover_path": None,
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        return {
            "params": f"steps={params['steps']}, resolution={params['resolution']}",
            "execution_time": 0,
            "obj_path": None,
            "obj_cover_path": None,
            "status": "failed",
            "error": str(e)
        }

def main():
    # 设置GPU
    torch.cuda.set_device(0)  # 使用GPU 2
    
    # 创建输出目录
    output_dir = Path("batchoutput")
    output_dir.mkdir(exist_ok=True)
    
    # 输入图片路径
    image_path = "assets/1.png"
    if not os.path.exists(image_path):
        print(f"Error: Input image {image_path} not found!")
        return
    
    # 初始化worker
    init_worker(0)  # 使用GPU 2初始化
    
    # 存储所有结果
    results = []
    
    # 顺序处理每个参数组合
    for params in QUALITY_COMBINATIONS:
        print(f"Processing model with params: steps={params['steps']}, resolution={params['resolution']}")
        result = process_single_model(params, image_path, output_dir)
        results.append(result)
        print(f"Status: {result['status']}")
        if result['status'] == 'failed':
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("-" * 50)
    
    # 生成Excel报告
    df = pd.DataFrame(results)
    excel_path = output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\nBatch processing completed. Report saved to: {excel_path}")

if __name__ == "__main__":
    main() 