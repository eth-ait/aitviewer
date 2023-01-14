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
- Clear framebuffer and setup initial renderer state.
- Render the scene.
- Render the GUI.

The rendering of the scene can be further broken down into these render passes:
1. Render objects to the picking map, this is a framebuffer which contains world space position, object id, triangle id and instance id of each pixel.
2. Render objects to shadow maps, there is one shadow map per light with shadows enabled.
3. Render objects to the framebuffer (potentially rendering a depth prepass, see [Rendering of Transparent Objects]({{ site.baseurl }}{% link technical_details/rendering_of_transparent_objects.md %})).
4. Render objects to the outline buffer and then use it to render outlines to the framebuffer.

## Render passes

The actual rendering pass (first part of number 3 in the list above) is implemented in the `render()` method by subclasses of the `Node` class.

The other rendering passes take advantage of the `render_positions()` method, which is also implemented by subclasses of the `Node` class. This method receives as an argument the corresponding shader program and it is responsible for drawing the object with only the vertex buffer containing positions bound (this is called `in_position` in shaders to follow ModernGL naming conventions). Therefore renderables can just implement this one method to enable all other rendering passes. These other passes are implemented on the `Node` class in the methods `render_framgap()`, `render_shadowmap()`, `render_depth_prepass()` and `render_outline()`, these methods check that the relevant shaders and flags are set, setup the rendering state and call `render_positions()` to issue the draw call.

If a renderable wants to enable a render pass it has to do the following three things:
- Implement `render_positions()` as described above.
- Set the relevant flag to `True`: `self.cast_shadow`, `self.depth_prepass`, `self.fragmap` and `self.outline`.
- Set the relevant shader program to a valid shader program for this pass: `self.depth_only_program` (used by both the shadow map pass and the depth prepass), `self.fragmap_program` and `self.outline_program`.

## Shader programs

All shaders are stored in the `aitviewer/shaders` directory. The script `aitviewer/shaders.py` provides utility functions for loading and caching shaders used by the renderer and by renderable classes. All renderables require custom vertex shaders to be drawn, but many fragment shaders can be reused across different renderables. To do this conveniently the `aitviewer/shaders.py` script exposes functions that take a vertex shader and use it to create a shader program for a specific pass.

Tipically renderables use a vertex shader for the normal rendering pass (e.g. `spheres_instanced.vs.glsl` for rendering spheres) and a simpler shader that only processes vertex positions (e.g. `spheres_instanced_positions.vs.glsl`) for the other render passes. In `make_renderable()` renderables use the utility functions to load a shader for each render pass, passing as argument the positions-only vertex shader.

Here is an example of what this looks like for the `Spheres` class.

```py
@Node.once
def make_renderable(self, ctx: moderngl.Context):
    self.prog = get_sphere_instanced_program()

    vs_path = "sphere_instanced_positions.vs.glsl"
    self.outline_program = get_outline_program(vs_path)
    self.depth_only_program = get_depth_only_program(vs_path)
    self.fragmap_program = get_fragmap_program(vs_path)

    ...
```

Here `self.prog` is the program used internally in the `Spheres` class for rendering. The other programs are loaded with utilities functions from `aiviwer/shaders.py` and will be passed by the `Node` class to the `render_positions()` method for drawing the corresponding render passes.