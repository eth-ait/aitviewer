---
title: Introduction
layout: home
nav_order: 0
---

# [![AITV](https://raw.githubusercontent.com/eth-ait/aitviewer/main/aitv_logo.svg)](https://github.com/eth-ait/aitviewer) AITViewer

A set of tools to visualize and interact with sequences of 3D data with cross-platform support on Windows, Linux, and Mac OS X.

## Features
* Easy to use Python interface.
* Load [SMPL[-H/-X]](https://smpl.is.tue.mpg.de/) / [MANO](https://mano.is.tue.mpg.de/) / [FLAME](https://flame.is.tue.mpg.de/) sequences and display them in an interactive viewer.
* Support for the [STAR model](https://github.com/ahmedosman/STAR).
* Manually editable SMPL sequences.
* Render 3D data on top of images via weak-perspective or OpenCV camera models.
* Built-in extensible GUI (based on Dear ImGui).
* Export the scene to a video (mp4/gif) via the GUI or render videos/images in headless mode.
* Animatable camera paths.
* Prebuilt renderable primitives (cylinders, spheres, point clouds, etc).
* Support live data feeds and rendering (e.g., webcam).
* Modern OpenGL shader-based rendering pipeline for high performance (via ModernGL / ModernGL Window).

![AITV SMPL Editing](https://user-images.githubusercontent.com/5639197/188625764-351100e9-992e-430c-b170-69d4f142f5dd.gif)

## Installation
Basic Installation:
```commandline
pip install aitviewer
```

Or install locally (if you need to extend or modify code)
```commandline
git clone git@github.com:eth-ait/aitviewer.git
cd aitviewer
pip install -e .
```

Note that this does not install the GPU-version of PyTorch automatically. If your environment already contains it, you should be good to go, otherwise install it manually.

If you would like to visualize STAR, please install the package manually via
```commandline
pip install git+https://github.com/ahmedosman/STAR.git
```
and download the respective body models from the official website.



## Quickstart
Display the SMPL T-pose:
```py
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == '__main__':
    v = Viewer()
    v.scene.add(SMPLSequence.t_pose())
    v.run()
```


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
  author = {Kaufmann, Manuel and Vechev, Velko and Mylonopoulos, Dario},
  doi = {10.5281/zenodo.1234},
  month = {7},
  title = {{AITViewer}},
  url = {https://github.com/eth-ait/aitviewer},
  year = {2022}
}
```

## Contact & Contributions
This software was developed by [Manuel Kaufmann](mailto:manuel.kaufmann@inf.ethz.ch), [Velko Vechev](mailto:velko.vechev@inf.ethz.ch) and Dario Mylonopoulos.
For questions please create an issue.
We welcome and encourage module and feature contributions from the community.

![AITV Sample](https://raw.githubusercontent.com/eth-ait/aitviewer/main/aitv_sample.png)
