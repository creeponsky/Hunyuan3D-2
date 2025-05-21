import time

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

start_time = time.time()
time1 = time.time()
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini-turbo"
)

input_image = "input/3.png"
input_image2 = "input/1.png"
mesh = pipeline(image=input_image, num_inference_steps=10, octree_resolution=64)[0]
time2 = time.time()
mesh2 = pipeline(image=input_image2, num_inference_steps=10, octree_resolution=64)[0]
time3 = time.time()

print(f"第一次生成时间: {time2 - time1} 秒")
print(f"第二次生成时间: {time3 - time2} 秒")

# pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
#     "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv"
# )
# mesh = pipeline(
#     image={"front": "assets/5.png"},
#     num_inference_steps=30,
#     octree_resolution=380,
#     num_chunks=20000,
#     generator=torch.manual_seed(12345),
#     output_type="trimesh",
# )[0]
end_time1 = time.time()
print(f"Shape generation time: {end_time1 - start_time} seconds")
print(mesh)

# 导出mesh为OBJ文件
output_path = f"output/output-{time.time()}.obj"
mesh.export(output_path)
print(f"Model exported to {output_path}")

# 其他常见格式导出方法（取消注释使用）
# mesh.export("output.stl")  # STL格式
# mesh.export("output.glb")  # GLB格式
# mesh.export("output.ply")  # PLY格式


# images_path = [
#     "assets/5.png",
# ]

# images = []
# for image_path in images_path:
#     image = Image.open(image_path)
#     if image.mode == "RGB":
#         rembg = BackgroundRemover()
#         image = rembg(image)
#     images.append(image)
end_time2 = time.time()
print(f"Background removal time: {end_time2 - end_time1} seconds")

texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    "tencent/Hunyuan3D-2",
    subfolder="hunyuan3d-paint-v2-0-turbo",
)

# mesh = trimesh.load("output.obj")
end_time3 = time.time()
print(f"Mesh loading time: {end_time3 - end_time2} seconds")

mesh = texture_pipeline(mesh, image=input_image)
mesh.export("demo_textured.obj")
