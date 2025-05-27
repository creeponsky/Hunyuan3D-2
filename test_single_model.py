import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import aiohttp
import asyncio
from task_executor import QualityLevel, init_worker, process_model_generation

async def generate_painted_model(obj_path: str, image_path: str, output_dir: Path, task_id: str, paint_service_url: str) -> dict:
    """Generate GLB version with texture using paint service"""
    try:
        glb_path = str(output_dir / f"{task_id}.glb")
        glb_cover_path = str(output_dir / f"{task_id}_glb_cover.png")
        
        async with aiohttp.ClientSession() as session:
            paint_request = {
                "task_id": task_id,
                "mesh_path": obj_path,
                "image_path": image_path,
                "glb_path": glb_path,
                "glb_cover_path": glb_cover_path,
            }
            
            async with session.post(
                f"{paint_service_url}/paint",
                json=paint_request,
                timeout=300  # 5 minutes timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("status") == "completed":
                        return {
                            "status": "success",
                            "glb_path": result.get("glb_path"),
                            "glb_cover_path": result.get("glb_cover_path"),
                            "execution_time": result.get("execution_time", 0)
                        }
                    else:
                        return {
                            "status": "failed",
                            "error": result.get("error", "Paint generation failed")
                        }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "error": f"Paint service error (HTTP {response.status}): {error_text}"
                    }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Paint service error: {str(e)}"
        }

def process_model(params: dict, image_path: str, output_dir: Path, paint_service_url: str = "http://localhost:42122") -> dict:
    """Process single model with custom parameters"""
    try:
        # Generate task ID and filenames
        task_id = f"test_steps{params['steps']}_res{params['resolution']}"
        date_str = datetime.now().strftime("%Y%m%d")
        task_dir = output_dir / date_str
        task_dir.mkdir(exist_ok=True)
        
        # Generate file paths
        obj_filename = f"{task_id}.obj"
        obj_cover_filename = f"{task_id}_obj_cover.png"
        
        obj_path = str(task_dir / obj_filename)
        obj_cover_path = str(task_dir / obj_cover_filename)
        
        # Prepare task data
        task_data = {
            "task_id": task_id,
            "image_path": image_path,
            "quality": QualityLevel.high,  # Use high as base, but will override with specific params
            "obj_path": obj_path,
            "obj_cover_path": obj_cover_path,
            "num_inference_steps": params["steps"],
            "octree_resolution": params["resolution"]
        }
        
        # Record start time
        start_time = time.time()
        
        # Execute model generation
        result = process_model_generation(task_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        if result.get("status") == "completed":
            # Generate GLB version
            paint_result = asyncio.run(generate_painted_model(
                obj_path, image_path, task_dir, task_id, paint_service_url
            ))
            
            return {
                "params": f"steps={params['steps']}, resolution={params['resolution']}",
                "execution_time": {
                    "shape": round(execution_time, 2),
                    "paint": round(paint_result.get("execution_time", 0), 2) if paint_result.get("status") == "success" else 0,
                    "total": round(execution_time + (paint_result.get("execution_time", 0) if paint_result.get("status") == "success" else 0), 2)
                },
                "obj_path": obj_path,
                "obj_cover_path": obj_cover_path,
                "glb_path": paint_result.get("glb_path") if paint_result.get("status") == "success" else None,
                "glb_cover_path": paint_result.get("glb_cover_path") if paint_result.get("status") == "success" else None,
                "status": "success" if paint_result.get("status") == "success" else "partial",
                "paint_error": paint_result.get("error") if paint_result.get("status") == "failed" else None
            }
        else:
            return {
                "params": f"steps={params['steps']}, resolution={params['resolution']}",
                "execution_time": {
                    "shape": round(execution_time, 2),
                    "paint": 0,
                    "total": round(execution_time, 2)
                },
                "obj_path": None,
                "obj_cover_path": None,
                "glb_path": None,
                "glb_cover_path": None,
                "status": "failed",
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        return {
            "params": f"steps={params['steps']}, resolution={params['resolution']}",
            "execution_time": {
                "shape": 0,
                "paint": 0,
                "total": 0
            },
            "obj_path": None,
            "obj_cover_path": None,
            "glb_path": None,
            "glb_cover_path": None,
            "status": "failed",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test single model generation with custom parameters")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps (10-50)")
    parser.add_argument("--resolution", type=int, default=128, help="Octree resolution (64-384)")
    parser.add_argument("--image", type=str, default="assets/1.png", help="Input image path")
    parser.add_argument("--output", type=str, default="batchoutput/testoutput", help="Output directory")
    parser.add_argument("--paint-service", type=str, default="http://localhost:42122", help="Paint service URL")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()
    
    # Validate parameters
    if not 10 <= args.steps <= 50:
        print("Error: steps must be between 10 and 50")
        return
    if args.resolution not in [64, 96, 128, 192, 256, 384]:
        print("Error: resolution must be one of: 64, 96, 128, 192, 256, 384")
        return
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Check input image
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} not found!")
        return
    
    # Initialize worker
    init_worker(args.gpu)
    
    # Process model
    params = {
        "steps": args.steps,
        "resolution": args.resolution
    }
    
    print(f"Processing model with params: steps={params['steps']}, resolution={params['resolution']}")
    result = process_model(params, args.image, output_dir, args.paint_service)
    
    # Print results
    print("\nResults:")
    print(f"Status: {result['status']}")
    print(f"Execution time:")
    print(f"  - Shape generation: {result['execution_time']['shape']}s")
    print(f"  - Paint generation: {result['execution_time']['paint']}s")
    print(f"  - Total: {result['execution_time']['total']}s")
    
    if result['status'] == 'success':
        print("\nGenerated files:")
        print(f"OBJ: {result['obj_path']}")
        print(f"OBJ Preview: {result['obj_cover_path']}")
        print(f"GLB: {result['glb_path']}")
        print(f"GLB Preview: {result['glb_cover_path']}")
    elif result['status'] == 'partial':
        print("\nPartially generated files:")
        print(f"OBJ: {result['obj_path']}")
        print(f"OBJ Preview: {result['obj_cover_path']}")
        print(f"\nPaint generation failed: {result['paint_error']}")
    else:
        print(f"\nGeneration failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 