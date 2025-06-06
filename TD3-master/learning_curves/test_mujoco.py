import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
print(f"MuJoCo path: {mj_path}")  # 添加这行检查路径
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
print(f"Model path: {xml_path}")  # 添加这行检查模型文件
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
sim.step()
print(sim.data.qpos)

