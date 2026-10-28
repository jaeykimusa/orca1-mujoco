import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

script_dir = Path(__file__).resolve().parent
xml_path = script_dir / "orca1_description" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Find the part's mesh geometry.
geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "base_visual"
)
if geom_id < 0:
    raise ValueError(f"base_visual was not found in {xml_path}")

mesh_id = model.geom_dataid[geom_id]
start = model.mesh_vertadr[mesh_id]
count = model.mesh_vertnum[mesh_id]
vertices = model.mesh_vert[start:start + count]

# Convert compiled mesh vertices into world coordinates.
rotation = data.geom_xmat[geom_id].reshape(3, 3)
world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]

lower = world_vertices.min(axis=0)
upper = world_vertices.max(axis=0)
center = (lower + upper) / 2
size = np.linalg.norm(upper - lower)

print("Loaded XML:", xml_path)
print("Part dimensions (m):", upper - lower)
print("Part center (m):", center)
print("Part minimum Z (m):", lower[2])

print("MuJoCo version:", mujoco.__version__)
print("Mesh vertices / faces:", model.mesh_vertnum[mesh_id],
      model.mesh_facenum[mesh_id])
print("Geometry world position:", data.geom_xpos[geom_id])
print("Geometry RGBA:", model.geom_rgba[geom_id])


with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(1.5 * size, 0.05)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        # Ensure the part's geometry group is visible.
        viewer.opt.geomgroup[model.geom_group[geom_id]] = 1

    # Static inspection: keep the viewer open.
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)