---
title: Rendering of Transparent Objects
layout: default
nav_order: 2
parent: Technical Details
usemathjax: true
---

# Rendering of Transparent Objects
Rendering transparent objects in a 3D rasterization-based graphics pipeline, such as the one exposed by OpenGL, poses some challenges related to the way that alpha blending works. In this page we discuss where problems arise and how we deal with them in our rendering pipeline.


## Alpha blending
Transparency is tipically implemented using the alpha blending functionality, which lets you specify a function to "blend" the color of the current pixel that is being drawn with the color that was already present in the framebuffer that we are rendering to. Standard alpha blending uses the following formula:

$$
\begin{equation*}

C_{\text{new}} = (1 - \alpha_\text{pixel})C_{\text{old}} + \alpha_\text{pixel} C_\text{pixel}

\end{equation*}
$$

Here $$C_{\text{new}}$$ is the new color that we will write to the framebuffer, $$C_{\text{old}}$$ is the previous color in the framebuffer and $$C_\text{pixel}$$ and $$\alpha_\text{pixel}$$ are the color and alpha channel of the pixel we are currently drawing.
The $$\alpha$$ value of the pixel we are rendering controls the opacity, $$\alpha = 0$$ means that the object is fully transparent and $$\alpha=1$$ makes the object completely opaque.

To ensure that the correct final color is computed we need to make sure that before rendering a transparent object, the object behind it has already been drawn to the framebuffer, this is usually handled by drawing all fully opaque objects first and then rendering transparent objects afterwards. Problems arise when we have more than one transparent object covering the same pixel, in this case to compute the correct color of the pixel we need to ensure that transparent objects are drawn in the correct order from back to front. To make things worse, if the object is not convex, it's also possible that the object itself covers the same pixels more than once.

## Overlapping objects
To handle overlap between different transparent objects we need to ensure the object that is further away is drawn before the object that is closer to the camera. However, this is not always possible, the objects could be intersecting and one could be in front of the other for some pixels and behind for some others. Correctly dealing with this case is complex, one option would be to compute which parts are intersecting, split the object in those parts and correctly order them before drawing. For simplicity and efficiency our renderer only attempts to avoid this issue by ordering objects back to front based on their distance to the camera. This heuristic works fine for distant objects but does not handle intersecting objects. Furthermore distance to the camera is computed with respect to the node position, which may or may not closely match the position of the object depending on the vertex positions. A better heuristic would be to compute the center position of the object instead, but this requires going through all vertices of the object which is more expensive (but can be cached).

## Self overlap
For transparent objects that are non-convex, different parts of the object could end up covering the same pixel and we need to ensure that they are rendered in the right order. Triangles that are drawn in the same OpenGL draw call are processed in submission order, meaning that triangles are blended into the framebuffer in the same order as they are specified in the index buffer. Therefore the blending order might or might not be correct based on the order of triangles and the view position. Since meshes usually don't follow strict ordering conventions for triangles this could result in unpleasant artifacts when some nearby triangles aren't ordered consistently.
Correctly handling self-overlap would require either splitting the object into convex parts and sorting them or sorting the triangle themselves depending on the view position (assuming there aren't any intersecting triangles).
Both of these options are fairly involved, therefore our rendering pipeline takes a simpler approach of that doesn't correctly handle this case but avoids the artifacts of self overlapping objects by rendering a depth-prepass.

| ![alt-text-1](/aitviewer/img/depth_prepass_without.png) | ![alt-text-2](/aitviewer/img/depth_prepass_with.png) |
|:--:| :--:|
| _**Without** depth-prepass_ | _**With** depth-prepass_ |


The way this works is that when we draw a transparent object, we first draw it only to the depth buffer, we do this by using an empty fragment shader and setting the color mask of the framebuffer to all zeros. After this depth-only prepass we draw the object again normally, but using an `<=` depth comparison function, meaning that instead of keeping only pixels that are closer to the camera than what has already been drawn to the depth buffer, we also keep pixels that are exactly at the same depth. This ensures that only the first layer of triangles closer to the camera survives the depth test and is blended with the background. The result is that we avoid order-related blending artifacts, but we only blend the side of the object that is closer to the camera with the background and not with itself.

## Implementation details
The transparency rendering logic is implemented in the `Scene.render()` method in `aitviewer/scene/scene.py`. The function implements the following steps:
- Draw all opaque objects.
- Sort all transparent objects (those for which `Node.is_transparent()` returns `True`) from back to front based on distance to camera.
- For each transparent object in order, draw it to the depth buffer only and then draw it again with the `<=` depth function.

Drawing to the depth buffer is implemented in the `Node.render_depth_prepasss()` method, which sets up uniform buffers and only draws the triangles with an empty fragment shader bound using the `Node.render_positions()` method discussed in the [Rendering Pipeline]({% link technical_details/rendering_pipeline.md %}) section.

## Order independent transparency
The main limitation of the current implementation is that the intersecting objects and self overlap issues are not completely solved. There are other approaches to handle transparency that don't rely on sorting. These are often referred to as order independent transparency methods, one simpler approach is [depth peeling](https://developer.download.nvidia.com/assets/gamedev/docs/OrderIndependentTransparency.pdf) but [more advanced approaches](https://interplayoflight.wordpress.com/2022/06/25/order-independent-transparency-part-1/) also exist. In the future one of these methods could be implemented to improve the quality of scenes with more complex transparent objects.


