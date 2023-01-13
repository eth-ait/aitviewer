---
title: Rendering Pipeline
layout: default
nav_order: 1
parent: Technical Details
---

# Rendering Pipeline
The main loop of the viewer is composed of the following steps:
- Process input events.
- Advance timestep and update current frame of all nodes.
- Render all objects to the screen.
- Render the GUI.

The rendering of all objects can be further broken down to the following steps:
- Clear framebuffer and setup initial renderer state.
- Render objects to the picking map, this is a framebuffer which contains world space position, object id, triangle id and instance id of each pixel.
- Render objects to shadow maps, there is one shadow map per light with shadows enabled.
- Render objects to the framebuffer.
- Render objects to the outline buffer and then use it to render outlines to the framebuffer.

