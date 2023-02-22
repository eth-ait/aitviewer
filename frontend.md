---
title: Viewer Frontend
layout: default
nav_order: 1
has_children: false
---

# Viewer Frontend
The viewer frontend is visualized below. Useful keyboard shortcuts are shown.

![Frontend Help](/aitviewer/assets/images/frontend_help.png)

## Using the Frontend
The basic functionality is illustrated - *all* sequence data can be played via the playback controls natively. The main drawing canvas is interactive - meaning that objects can be clicked on and selected (and in the case of SMPL - edited). We support a comprehensive video export dialog (File > Export) with custom animation framerates and optional 360 degree rotation.   




## Configure the Viewer
The viewer loads default configuration parameters from [`aitvconfig.yaml`](aitviewer/aitvconfig.yaml). There are three ways how to override these parameters:
  - Create a file named `aitvconfig.yaml` and have the environment variable `AITVRC` point to it. Alternatively, you can point `AITVRC` to the directory containing `aitvconfig.yaml`.
  - Create a file named `aitvconfig.yaml` in your current working directory, i.e. from where you launch your python program.
  - Calling `C.update_conf()` directly from a python script with a dictionary of option value pairs. This function should likely be called before creating any object to ensure that the new values are used.
    ```python
    from aitviewer.configuration import CONFIG as C
    C.update_conf({"run_animations": True})
    ```

Note that the configuration files are loaded in this order, i.e. the config file in your working directory overrides all previous parameters.

The configuration management is using [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/index.html). You will probably want to override the following parameters at your convenience:
- `datasets.amass`: where [AMASS](https://amass.is.tue.mpg.de/) is stored if you want to load AMASS sequences.
- `smplx_models`: where SMPLX models are stored, preprocessed as required by the [`smplx` package](https://github.com/vchoutas/smplx).
- `star_models`: where the [STAR model](https://github.com/ahmedosman/STAR) is stored if you want to use it.
- `export_dir`: where videos and other outputs are stored by default.


## Full Set of Keyboard Shortcuts
The viewer supports the following keyboard shortcuts, all of this functionality is also accessible from the menus and windows in the GUI.
This list can be shown directly in the viewer by clicking on the `Help -> Keyboard shortcuts` menu.

- `SPACE` Start/stop playing animation.
- `.` Go to next frame.
- `,` Go to previous frame.
- `G` Open a window to change frame by typing the frame number.
- `X` Center view on the selected object.
- `O` Enable/disable orthographic camera.
- `T` Show the camera target in the scene.
- `C` Save the camera position and orientation to disk.
- `L` Load the camera position and orientation from disk.
- `K` Lock the selection to the currently selected object.
- `S` Show/hide shadows.
- `D` Enabled/disable dark mode.
- `P` Save a screenshot to the the `export/screenshots` directory.
- `I` Change the viewer mode to `inspect`.
- `V` Change the viewer mode to `view`.
- `E` If a mesh is selected, show the edges of the mesh.
- `F` If a mesh is selected, switch between flat and smooth shading.
- `Z` Show a debug visualization of the object IDs.
- `ESC` Exit the viewer.


## Custom Viewer
You can create your own custom viewer by overriding the main `Viewer()` class. An example is included in the repo that shows how to interactively handle clicking on vertices of a mesh (For example, if you want to place markers, or identify nearby vertices manually).
 * [`vertex_clicking.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/vertex_clicking.py): An example how to subclass the basic Viewer class for custom interaction.

