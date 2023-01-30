---
title: Working with the SMPL Family
layout: default
nav_order: 1
parent: Parametric Human Models
---

# Working with the SMPL Family
This page covers some more details on the parametric human body models from the SMPL family. It focuses on the original SMPL model, but the other models (SMPL+H, SMPL-X, etc.) work analogously.

## Overview
A `SMPLSequence` is a composite renderable, i.e. it keeps track of three separate renderables: the SMPL mesh (`self.mesh_seq` which is of type [`Meshes`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/meshes.py)), the SMPL joints (`self.skeleton_seq` which is of type [`Skeletons`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/skeletons.py)), and the joint angles (`self.rbs` which is of type [`RigidBodies`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/rigid_bodies.py)). The joints are only displayed if `is_rigged` is set to true in the initializer (true by default) and the joint angles are only rendered if `show_joint_angles` is set accordingly (false by default). 

## The `fk` Function
A `SMPLSequence` has a a [`fk`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/smpl.py#L331) function (short for "forward kinematics" (which should probably be renamed)) that evaluates the underlying body model. I.e., given the parameters expected by the body model, it executes a forward pass through the model (on the GPU, potentially batched) and stores the results that are then used for rendering.

The `fk` function is potentially expensive. It is run when the renderable is initialized and when it is updated. To force a call to the `fk` function, it's best to use the [`redraw`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/smpl.py#L427) function and not the `fk` function directly.

It is possible to update the a renderable from the outside after creation, e.g., by setting new pose parameters. However, the renderable currently does not automatically detect changes from outside. Hence, if you change data of the renderable after creation, make sure to call `redraw` on the renderable for the changes to have an effect.

## The `post_fk_func` Parameter
The `SMPLSequence` initializer accepts a parameter `post_fk_func`, which is a function that can optionally be run to do something with the outputs of the `fk` function. It has the following signature:
```python
def post_fk_func(
        self: SMPLSequence,
        vertices: torch.Tensor,
        joints: torch.Tensor,
        current_frame_only: bool,
    ):
```

This is for example used when running the [GLAMR](https://github.com/eth-ait/aitviewer/blob/main/examples/load_GLAMR.py) model.

## Editing SMPL Joints in the Viewer
