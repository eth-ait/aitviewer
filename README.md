# [![AITV](aitv_logo.svg)](https://github.com/eth-ait/aitviewer) AITViewer

A set of tools to visualize and interact with sequences of 3D data with cross-platform support on Windows, Linux, and Mac OS X.

![AITV Sample](aitv_sample.png)

## Features
* Easy to use Python interface.
* Load [SMPL[-H | -X]](https://smpl.is.tue.mpg.de/) / [MANO](https://mano.is.tue.mpg.de/) / [FLAME](https://flame.is.tue.mpg.de/) sequences and display them in an interactive viewer.
* Built-in extensible GUI (based on Dear ImGui).
* Prebuilt renderable primitives (cylinders, spheres, point clouds, etc).
* Render videos of the currently loaded sequences.
* Headless/Offscreen rendering.
* Support live data feeds and rendering (e.g., webcam).
* Modern OpenGL shader-based rendering pipeline for high performance (via ModernGL / ModernGL Window).

![AITV Interface](aitv_screenshot.png)

## Installation
Install directly from git:
```commandline
pip install git+https://github.com/eth-ait/aitviewer.git
```

Or install locally (if you need to extend or modify code)
```commandline
git clone git@github.com:eth-ait/aitviewer.git
cd aitviewer
pip install -e .
```

Note that this does not install the GPU-version of PyTorch automatically. If your environment already contains it, you should be good to go, otherwise install it manually.

## Configuration
The viewer loads default configuration parameters from [`aitvconfig.yaml`](aitviewer/aitvconfig.yaml). There are two ways how to override these parameters:
  - Create a file named `aitvconfig.yaml` and have the environment variable `AITVRC` point to it. Alternatively, you can point `AITVRC` to the directory containing `aitvconfig.yaml`.
  - Create a file named `aitvconfig.yaml` in your current working directory, i.e. from where you launch your python program.

Note that the configuration files are loaded in this order, i.e. the config file in your working directory overrides all previous parameters.

The configuration management is using [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/index.html). You will probably want to override the following parameters at your convenience:
- `datasets.amass`: where [AMASS](https://amass.is.tue.mpg.de/) is stored if you want to load AMASS sequences.
- `smplx_models`: where SMPLX models are stored, preprocessed as required by the [`smplx` package](https://github.com/vchoutas/smplx).
- `export_dir`: where videos and other outputs are stored by default.


## Quickstart
View an SMPL template

```py
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

smpl_template = SMPLSequence.t_pose()

# Display in viewer.
v = Viewer()
v.scene.add(smpl_template)
v.run()
```

## Examples

Check out the [examples](examples/) for a few examples how to use the viewer. Some examples are:

**`load_3dpw.py`**: Loads an SMPL sequence from the 3DPW dataset and displays it in the viewer.

**`load_amass.py`**: Loads an SMPL sequence from the AMASS dataset and displays it in the viewer.

**`load_obj.py`**: Loads meshes from OBJ files.

**`load_template.py`**: Loads the template meshes of SMPL-H, MANO, and FLAME.

**`render_primitives.py`**: Renders a bunch of spheres and lines.

**`stream.py`**: Streams your webcam into the viewer.

**`vertex_clicking.py`**: An example how to subclass the basic Viewer class for custom interaction.

## Projects using the AITViewer
The following projects have used the AITViewer:
- Dong et al., [Shape-aware Multi-Person Pose Estimation from Multi-view Images](https://ait.ethz.ch/projects/2021/multi-human-pose/), ICCV 2021
- Kaufmann et al., [EM-POSE: 3D Human Pose Estimation from Sparse Electromagnetic Trackers](https://ait.ethz.ch/projects/2021/em-pose/), ICCV 2021
- Vechev et al., [Computational Design of Kinesthetic Garments](https://ait.ethz.ch/projects/2022/cdkg/), Eurographics 2021
- Guo et al., [Human Performance Capture from Monocular Video in the Wild](https://ait.ethz.ch/projects/2021/human-performance-capture/index.php), 3DV 2021
- Dong and Guo et al., [PINA: Learning a Personalized Implicit Neural Avatar from a Single RGB-D Video Sequence](https://zj-dong.github.io/pina/), CVPR 2022

## Citation
If you use this software, please cite it as below.
```commandline
@software{Kaufmann_Vechev_AITViewer_2022,
  author = {Kaufmann, Manuel and Vechev, Velko},
  doi = {10.5281/zenodo.1234},
  month = {7},
  title = {{AITViewer}},
  url = {https://github.com/eth-ait/aitviewer},
  version = {1.0},
  year = {2022}
}
```

## Contact & Contributions
This software was developed by [Manuel Kaufmann](mailto:manuel.kaufmann@inf.ethz.ch) and [Velko Vechev](mailto:velko.vechev@inf.ethz.ch). 
For questions please create an issue.
We welcome and encourage module and feature contributions from the community.
