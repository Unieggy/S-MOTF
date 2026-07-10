"""Render a qpos rollout to mp4 + gif for the README (offscreen, no mjpython).

Run on your Mac in an env with mujoco:
  python save_media.py [INPUT.npz] [OUTPUT_NAME]
  e.g. python save_media.py rollout_walk.npz walk   -> walk.gif, walk.mp4
Needs the Go1 model:  git clone https://github.com/google-deepmind/mujoco_menagerie.git
"""

import sys
import imageio
import mujoco
import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "rollout_qpos.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "rollout"

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go1/scene.xml")
data = mujoco.MjData(model)
qpos = np.load(IN)["qpos"]                            # [T, 19]

renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 2.5, 120.0, -20.0   # nice 3/4 view

frames = []
for q in qpos:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)                   # replay recorded pose (no physics)
    cam.lookat[:] = data.qpos[:3]                    # follow the base so it stays centered
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

imageio.mimsave(f"{OUT}.mp4", frames, fps=50)
imageio.mimsave(f"{OUT}.gif", frames[::2], fps=25)  # every other frame -> lighter gif
print(f"saved {OUT}.mp4 and {OUT}.gif")
