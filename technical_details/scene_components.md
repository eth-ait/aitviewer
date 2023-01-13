---
title: Scene Components
layout: default
parent: Technical Details
nav_order: 0
---

# Scene Components

## Nodes

All objects that can be rendered in the viewer are subtypes of the `Node` class.
The `Node` class contains some information common to all renderable objects:
- references to parent (`parent`) and children (`nodes`) nodes to construct the scene hierarchy.
- object to world transform information (`position`, `rotation` and `scale`).
- GUI name (`name`) and a unique identifier (`uid`).
- number of frames (`n_frames`) and index of the current frame in the sequence (`current_frame_id`).
- internal flags and properties related to the state and rendering options of the node.


## Renderables

Renderables are created by subtyping the Node class and overriding the relevant methods.
To expose parameters of the Node constructors to users, the `__init__()` method of subclasses usually take
a `**kwargs` argument that is forwarded to the `Node.__init__()` method.

The node object does not render anything to the screen when drawn, other renderables
override its methods to implement their own rendering logic. Here is a list of
the important rendering methods that a renderable needs to override:

- `make_renderable()`: this method is always implemented with the `Node.once` decorator to ensure that it is only called one time
when the object is not yet renderable. The purpose of this method is to initialize
all OpenGL resources required by the node to be rendered, such as shader programs,
buffers and textures.

- `render()`: this method is called once per frame when the object should be drawn to the screen.
The usual structure of this method is to bind a shader program, set the required uniforms, buffers and textures and invoke a draw method.

- `render_positions()`: this is another rendering method that renderables need to implement.
This method is called when the object should be drawn
with only vertex positions bound, this is used for drawing to shadow maps, drawing to the buffer used for picking objects and drawing the depth prepass for transparent objects (see [Rendering Pipeline]({% link technical_details/rendering_pipeline.md %}) for more information about render passes).

- `release()` this method is invoked to release all rendering resources allocated by the node. It should always be implemented if the node allocates any OpenGL resource to avoid memory leaks.

- `redraw()` this method can be invoked on a node when some change to its internal
state has happened that modifies the way it should be rendered. Generally
the user does not have to call this method directly, as setters of properties that change
the appearance of the node should call it instead.

Other useful methods that a renderable can override are the following:
- `on_frame_update()`: this method is called whenever the currently selected frame of the object changed. This is called at the start of a new frame and can be used for uploading per frame rendering data or changing any frame related state, `self.current_frame_id` contains the index of the current frame after the update.
- `on_before_frame_update()`: this method is called just before the currently selected frame of the object is about to change. This can be used to free resources related to the current frame before it's updated, `self.current_frame_id` contains the index of the current frame before the update.
- `bounds()` and `current_bounds()`: these methods return an axis-aligned bounding box (AABB) of the renderable in world coordinates, for the whole sequence of frames and just for the current frame respectively.
This bounding box is used for centering the camera to the object and to automatically position the floor below all objects.
- `on_selection()`: this method is called whenever the object is selected, this can happen when the object is clicked or when the `scene.select()` method is called on the object. When the objecte is clicked additional information about the click is passed in, otherwise all parameters are `None`. The additional information can be used to handle selection of specific parts of the object, when it's not provided the whole object should be considered selected.
- `gui()` and `gui_*()`: those methods are responsible for drawing custom GUI that is specific for this node. There are many `gui_*()` methods, each method is responsible for drawing in a different section of the UI or in special windows such as the context menu that appears when right clicking on an object.
- `key_event()`: this method is called on a node that is selected whenever a key is pressed. Renderables can override this method to implement custom actions.
- `is_transparent()`: this method returns `True` if the object should be considered transparent when rendering. This changes the order in which the object is drawn and performs a depth-prepass when rendering this object to avoid rendering artifacts due to self overlap (see [Rendering of transparent objects]({% link technical_details/rendering_of_transparent_objects.md %}) for more information about how transparency is handled by the renderer).


## Scene

The Scene node is a special renderable, it is created by the `Viewer` class and it's always the root of the node hierarchy, it can be accessed from the `Viewer.scene` property. The scene class is responsible for managing the rendering of the whole scene and contains properties such as references to the camera (`Scene.camera`), the lights (`Scene.lights`) and to other default objects in the scene such as the origin (`Scene.origin`) and floor (`Scene.floor`) objects.
