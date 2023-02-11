---
title: Billboards
layout: default
parent: Renderables
---

# Billboards

![Billboards](../assets/images/billboard.png)

Billboards can be used to display images and videos inside a 3D scene. They are especially useful to visualize pictures taken from a real-world camera and 3D objects on top of them (see [Cameras How-To]({% link cameras.md %}) for more information about adding cameras to the scene).

## Example Usage

Billboards can be created from vertex positions and a set of images (either as paths to image files or as numpy arrays). Alternatively, instead of manually specifying the vertex positions a billboard can be placed in the scene using the `from_camera_and_distance()` class method, which returns a billboard object positioned in the view frustum of the camera at the specified distance.

Here is an excerpt from the [`load_ROMP.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/load_ROMP.py) example in which a billboard is created from camera data:

```python
# Create an OpenCV camera.
cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], cols, rows, viewer=viewer)

# Load the reference image and create a Billboard.
pc = Billboard.from_camera_and_distance(cameras, 4.0, cols, rows, [img_path])

# Add all the objects to the scene.
viewer.scene.add(pc, romp_smpl, cameras)
```