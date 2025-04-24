from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import time

start_time = time.time()
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
mesh = pipeline(image="assets/5.png")[0]
end_time1 = time.time()
print(f"Shape generation time: {end_time1 - start_time} seconds")
print(mesh)

# 导出mesh为OBJ文件
output_path = "output.obj"
mesh.export(output_path)
print(f"Model exported to {output_path}")

# 其他常见格式导出方法（取消注释使用）
# mesh.export("output.stl")  # STL格式
# mesh.export("output.glb")  # GLB格式
# mesh.export("output.ply")  # PLY格式


images_path = [
    "assets/5.png",
]

images = []
for image_path in images_path:
    image = Image.open(image_path)
    if image.mode == "RGB":
        rembg = BackgroundRemover()
        image = rembg(image)
    images.append(image)
end_time2 = time.time()
print(f"Background removal time: {end_time2 - end_time1} seconds")

texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
)

mesh = trimesh.load("output.obj")
end_time3 = time.time()
print(f"Mesh loading time: {end_time3 - end_time2} seconds")

mesh = texture_pipeline(mesh, image=images)
mesh.export("demo_textured.glb")
