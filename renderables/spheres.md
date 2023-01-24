---
title: Spheres
layout: default
parent: Renderables
---

# Spheres

![Spheres](../assets/images/spheres.png)


We render spheres using a mesh shader. Sphere `rings` and `sectors` can be customized for lower or higher resolution spheres.


## Example Usage
An excerpt from the [`render_primitives.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/render_primitives.py) example shows how to use spheres:

```python
# Draw 10k lines.
grid_xz = np.mgrid[-5:5:0.1, -5:5:0.1]
n_lines = grid_xz.shape[1] * grid_xz.shape[2]
print("Number of lines", n_lines)

xz_coords = np.reshape(grid_xz, (2, -1)).T
line_starts = np.concatenate([xz_coords[:, 0:1], np.zeros((n_lines, 1)), xz_coords[:, 1:2]], axis=-1)
line_ends = line_starts.copy()
line_ends[:, 1] = 1.0
line_strip = np.zeros((2 * n_lines, 3))
line_strip[::2] = line_starts
line_strip[1::2] = line_ends
line_renderable = Lines(line_strip, mode="lines")

# Draw some spheres on top of each line.
line_dirs = line_ends - line_starts
sphere_positions = line_ends + 0.1 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
spheres = Spheres(sphere_positions, color=(1.0, 0.0, 1.0, 1.0))

# Draw rigid bodies on top of each sphere (a rigid body is just a sphere with three axes representing its
# orientation).
rb_positions = line_ends + 0.4 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
angles = np.arange(0.0, 2 * np.pi, step=2 * np.pi / n_lines)[:, None]
axes = np.zeros((n_lines, 3))
axes[:, 2] = 1.0
rb_orientations = aa2rot(angles * axes)
rbs = RigidBodies(rb_positions, rb_orientations)

# Display in viewer.
v = Viewer()
v.scene.add(line_renderable, spheres, rbs)
v.scene.camera.position = np.array([0.0, 1.3, 5.0])
v.run()
``` 