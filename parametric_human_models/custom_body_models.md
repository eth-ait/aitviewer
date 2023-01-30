---
title: Adding Custom Body Models
layout: default
nav_order: 2
parent: Parametric Human Models
---

# Adding Custom Body Models
To add a new body model you need two things:
 - A class that implements your body model (e.g. the `SMPLLayer` or `STARLayer` currently in [`aitviewer.models`](https://github.com/eth-ait/aitviewer/tree/main/aitviewer/models)).
 - A renderable that calls your body model, gets the results and displays them (e.g. the `SMPLSequence` or `STARSequence` in [`aitviewer.renderables.smpl`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/smpl.py) or [`aitviewer.renderables.star`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/star.py).

## Body Model Class
Currently all supported body models are Pytorch modules, i.e., they inherit from `torch.nn.Module` and need to implement the `forward` function. The `forward` function implements the forward pass and is the function that is called from the renderable class in order to obtain the outputs of the body model that we want to visualize. For the `SMPLSequence` the forward function just forwards the call to the `fk` function.

A new body model does not necessarily have to be a Pytorch module. It can be any class and the function that evaluates the body model can have any name - as long as the corresponding renderable knows how to call that function.

## Renderable Class
The renderable class is responsible to feed your custom body model with the data it requires and to display the output. If your body model is somewhat related with SMPL, it might be easiest to subclass `SMPLSequence` (as for example the [`STARSequence`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/star.py) does it). If this is not suitable, follow the documentation of how to create a custom renderable, with the following additional tips:
 - Refer to the [`SMPLSequence`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/smpl.py) renderable for an example.
 - The `SMPLSequence` mostly just keeps three renderables as children: a mesh, a skeleton, and rigid bodies for the joint angles.
 - The `fk` function is the function that calls the forward function of the body model. It is slightly more complicated than that because we support an edit mode for SMPL sequences. Also, because it might be expensive to evaluate the model on the entire sequence, it accepts a parameter `current_frame_only` if the body model should only be evaluated for the current frame.
 - We currently push the entire sequence through the body model in the initializer of the `SMPLSequence`. This is because we benefit from GPU acceleration and we gain higher frame rates at the cost of slightly higher initialization times. You don't have to follow this - you could evaluate the body model on-the-fly, i.e. whenever a frame is loaded (see below for more information).
 - The `redraw` updates the state of the renderable. It evaluates the body model (via a call to `fk`) and updates the child renderables with the new data. This update automatically triggers redraws on the child renderables.
 - The `SMPLSequence` does not have a `render` function. That's because the `SMPLSequence` renderable is a composite renderable, i.e. it does not have data that it directly wants to render, but just defers it to the child renderables. Because we add the child renderables via `self._add_node` in the initializer, they will be automatically rendered (the scene class just loops over all renderables and its children and renders them). If you would like to change this, you can implement a custom `render` function. This is for example where you could evaluate your body model on-the-fly instead of pre-computing everything.
