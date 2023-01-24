---
title: Arrows
layout: default
parent: Renderables
---

# Arrows
![Arrows](../assets/images/arrows.png) 

Arrows can be useful for displaying normals on a mesh, or vector quantities.  


## Example Usage
Arrows are constucted when showing normals on a mesh. The following function, which can be found in [`meshes.py`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/meshes.py#L574), displays the normal of each vertex. Arrows expect a set of positions for `origins` and `tips` at a minimum. In the example below, the length of the arrows is computed proportional to the bounding box of the mesh, but this is optional.


```python
def _show_normals(self):
    """Create and add normals at runtime"""
    vn = self.vertex_normals

    bounds = self.bounds
    diag = np.linalg.norm(bounds[:, 0] - bounds[:, 1])

    length = 0.0025 * max(diag, 1) / self.scale
    vn = vn / np.linalg.norm(vn, axis=-1, keepdims=True) * length

    # Must import here because if we do it at the top we create a circular dependency.
    from aitviewer.renderables.arrows import Arrows

    positions = self.vertices
    self.normals_r = Arrows(
        origins=positions,
        tips=positions + vn,
        r_base=length / 20,
        r_head=2 * length / 20,
        p=0.25,
        name="Normals",
    )
    self.normals_r.current_frame_id = self.current_frame_id
    self.add(self.normals_r)
```

