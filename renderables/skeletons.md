---
title: Skeletons
layout: default
parent: Renderables
---

# Skeletons

![Skeletons](../assets/images/skeletons.png) 


Human bodies are articulated and thus require visualization of underlying skeletal structure. 

## Example Usage:
Skeletons are used in the SMPL Sequence renderable [`smpl.py`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/smpl.py#L151) whenever the sequence is specified to be rigged. They are initialized and added as follows:


```python
# Create Skeleton Sequence
# self.joints = (F, J, 3) Skeleton joint positions
# self.skeleton = (J, 2) Joint connections - e.g. for SMPL Human Body: [[-1  0], [ 0  1], [ 0  2], [ 0  3], [ 1  4], [ 2  5], [ 3  6], [ 4  7], [ 5  8], [ 6  9], [ 7 10], [ 8 11], [ 9 12], [ 9 13], [ 9 14], [12 15], [13 16], [14 17], [16 18], [17 19], [18 20], [19 21]]

self.skeleton_seq = Skeletons(
                self.joints,
                self.skeleton,
                gui_affine=False,
                color=(1.0, 177 / 255, 1 / 255, 1.0),
                name="Skeleton",
            )
self._add_node(self.skeleton_seq)
```
