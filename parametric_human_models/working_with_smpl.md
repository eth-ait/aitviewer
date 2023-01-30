---
title: Working with SMPL & Co
layout: default
nav_order: 1
parent: Parametric Human Models
---

# Working with SMPL & Co
This page covers some more details on how we interface with parametric human body models from the SMPL family. It focuses on the original SMPL model, but the other models (SMPL+H, SMPL-X, etc.) work analogously.

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

## Load from Existing Data
A `SMPLSequence` needs a minimum of two things: the body model (`smpl_layer`) and the body pose parameters as a numpy array. It additionally accepts the root orientation and translation, and the shape parameters, but default values are used for those if they are not supplied.

The parameters should always be supplied as a numpy array of shape whose first dimension is the time dimension and the second dimension is the flattened data per frame. E.g., for the body poses, it expects a numpy array of shape `(F, N_JOINTS*3)`, where `F` is the number of frames. Shape parameters `betas` can either be supplied as a numpy array of shape `(N_BETAS, )`, `(1, N_BETAS)`, or `(F, N_BETAS)`.

We provide convenience functions to load sequences from [AMASS](https://amass.is.tue.mpg.de/) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/). The respective functions are class methods [`SMPLSequence.from_amass`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/smpl.py#L178) and [`SMPLSequence.from_3dpw`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/smpl.py#L228). These functions expect a path to a sample sequence of the respective dataset instead of the pose parameters directly.

## Editing SMPL Joints in the Viewer
The default viewer implements an edit mode for `SMPLSequence`s. With a right-click on a SMPL mesh, the "Edit" mode can be selected. Afterwards, the mesh appears transparent and the joint angles become visible. By clicking on one of the joints, the current rotation of that joint is displayed in the Scene menu. The invdividual joint angles can then be adjusted in the menu by clicking and dragging or entering a number.

Without any further action, the change in joint angle is only temporary, i.e. when the same frame is loaded again, the original joint angles are loaded. To apply the change permanently for the current frame, click on the `Apply` button. To apply the same change to not just the current frame, but all frames in the sequence, click on `Apply to all`. To reset the joint angles to their original values of this frame, click on `Reset`.

An example interaction is shown in the following.
![AITV SMPL Editing](https://user-images.githubusercontent.com/5639197/188625764-351100e9-992e-430c-b170-69d4f142f5dd.gif)
