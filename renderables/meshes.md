---
title: Meshes
layout: default
parent: Renderables
---

# Meshes
![Meshes](../assets/images/meshes.png) 

Standard Mesh component that also supports the batch dimension allowing for animation, hence, Meshes. 

## Example Usage:
As an example, meshes are used to display the dense surface of SMPL sequences: [`smpl.py`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/smpl.py#L169). 


```python
# Create Meshes
self.mesh_seq = Meshes(
            self.vertices,
            self.faces,
            is_selectable=False,
            gui_affine=False,
            color=kwargs.get("color", (160 / 255, 160 / 255, 160 / 255, 1.0)),
            name="Mesh",
        )
self._add_node(self.mesh_seq)
```


## Variable Topology Meshes
An extended version of Meshes that allows for a unique toplogy (faces/triangles) for each frame of animation. 
