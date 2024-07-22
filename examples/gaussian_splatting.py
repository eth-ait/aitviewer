import json
import os

import imgui
import numpy as np

from aitviewer.renderables.gaussian_splats import GaussianSplats
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node
from aitviewer.viewer import Viewer

# Update this variable to point to the Gaussian Splatting dataset that can be downloaded
# from here https://github.com/graphdeco-inria/gaussian-splatting link is "Pre-trained Models (14 GB)"".
#
# This variable should point to the top level directory containing a directory for each scene.
PATH = ""

if not PATH:
    print(
        "Update this variable to point to the Gaussian Splatting dataset that can be downloaded"
        ' from here https://github.com/graphdeco-inria/gaussian-splatting clicking on "Pre-trained Models"'
    )
    exit(1)

dataset = {f: os.path.join(PATH, f) for f in sorted(os.listdir(PATH))}


gs = None
cameras = Node("Cameras")


def set_scene(viewer, name, iteration):
    global gs, cameras
    if gs is not None:
        viewer.scene.remove(gs)
    for c in cameras.nodes:
        cameras.remove(c)

    path = os.path.join(dataset[name], "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    to_y_up = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    gs = GaussianSplats.from_ply(path, rotation=to_y_up[:3, :3])
    gs._debug_gui = True
    print(f"Loaded {gs.num_splats} splats")

    cams = json.load(open(os.path.join(dataset[name], "cameras.json")))
    for c in cams:
        t = np.array(c["position"])
        R = np.array(c["rotation"]).reshape(3, 3)
        fx = c["fx"]
        fy = c["fy"]
        w = c["width"]
        h = c["height"]
        name = str(c["id"])

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt = (np.linalg.inv(Rt) @ to_y_up)[:3, :4]

        K = np.array(
            [
                [fx, 0, w / 2],
                [0, fy, h / 2],
                [0, 0, 1],
            ]
        )

        c = OpenCVCamera(K, Rt, w, h, name=name, viewer=viewer)
        cameras.add(c)

    viewer.scene.add(gs)


v = Viewer(size=(1600, 900))
v.auto_set_floor = False
v.auto_set_camera_target = False
v.scene.floor.enabled = False
v.scene.background_color = (0, 0, 0, 1)
v.scene.add(cameras, enabled=False)


def gui_dataset():
    imgui.set_next_window_position(v.window_size[0] - 200, 50, imgui.FIRST_USE_EVER)
    imgui.set_next_window_size(v.window_size[0] * 0.2, v.window_size[1] * 0.5, imgui.FIRST_USE_EVER)
    expanded, _ = imgui.begin("Dataset", None)
    if expanded:
        for i, k in enumerate(dataset.keys()):
            space = imgui.get_content_region_available()[0]
            imgui.text(k)
            imgui.same_line()
            imgui.set_cursor_pos_x(space - 150)
            if imgui.button(f"7000##{i}", width=70):
                set_scene(v, k, 7000)
            imgui.same_line()
            if imgui.button(f"30000##{i}", width=70):
                set_scene(v, k, 30000)
    imgui.end()


v.gui_controls["gs_dataset"] = gui_dataset

set_scene(v, "bicycle", 30000)

v.run()
