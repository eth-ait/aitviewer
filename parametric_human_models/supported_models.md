---
title: Supported Body Models
layout: default
nav_order: 0
parent: Parametric Human Models
---
# Supported Body Models

## Overview
We support the following parametric human models:
 - SMPL: https://smpl.is.tue.mpg.de/
 - SMPL+H / MANO: https://mano.is.tue.mpg.de/
 - SMPL-X: https://smpl-x.is.tue.mpg.de/
 - FLAME: https://flame.is.tue.mpg.de/
 - STAR: https://star.is.tue.mpg.de/
 - SUPR: https://supr.is.tue.mpg.de/

To make interfacing with our renderables easier, we provide wrappers around these body models:
 - [`aitviewer.models.smpl`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/models/smpl.py). This contains the `SMPLLayer`, which essentially just forwards calls to the various models implemented in the [`smplx`](https://github.com/vchoutas/smplx) package. I.e., it supports SMPL, SMPL+H, SMPL-X, MANO, and FLAME.
 - [`aitviewer.models.star`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/models/star.py). The `STARLayer` wraps the STAR model provided by https://github.com/ahmedosman/STAR.
 - [`aitviewer.models.supr`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/models/supr.py). The `SUPRLayer` wraps the SUPR model provided by https://github.com/ahmedosman/SUPR.

## Renderables
For each of the body models mentioned above, there is a corresponding renderable.
 - [`SMPLSequence`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/smpl.py#L43) is the corresponding renderable for the `SMPLLayer`.
 - [`STARSequence`](https://github.com/eth-ait/aitviewer/blob/c3e0de4a44e2ccae06c67714765bb1db9db68951/aitviewer/renderables/star.py#L26) is the corresponding renderable for the `STARLayer`.
 - [`SUPRSequence`](https://github.com/eth-ait/aitviewer/blob/main/aitviewer/renderables/supr.py#L25) is the corresponding renderable for the `SUPRLayer`.

A body model renderable requires an instance of the respective body model layer, as well as the data that should be displayed. E.g. to display a T-Pose with the SMPL+H model:
```python
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")
poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
smpl_seq = SMPLSequence(poses, smpl_layer)
```

For more information on how to work with body models, please refer to [Working with the SMPL Family](https://eth-ait.github.io/aitviewer/parametric_human_models/working_with_smpl.html).

## Installation
Please note that you only have to install the SMPL/STAR/SUPR models if you plan to use them. If you do not need these models, feel free to skip this section.

### SMPL models
The `smplx` package is automatically installed as one of the dependencies. However, you have to download the various body models. Please follow the instructions provided on the [SMPL-X Github](https://github.com/vchoutas/smplx#downloading-the-model) page to do so. We expect the same directory structure as the `smplx` package does.

After the download of the body models, configure aitviewer to point the root directory of where you stored the body models by updating the `smplx_models` parameter in the `aitvconfig.yaml`. Please refer to the [configuration section here](https://eth-ait.github.io/aitviewer/frontend.html#configure-the-viewer) to find out about various ways how to create your custom configuration file.

### STAR
If you would like to use STAR, you have to manually install the package via
```
pip install git+https://github.com/ahmedosman/STAR.git
```
and download the body model from the [STAR project page](https://github.com/ahmedosman/STAR). After the download, update the `star_models` parameter in the `aitvconfig`, the same way as you did for the SMPL models.

### SUPR
If you would like to use SUPR, you have to manually install the package via
```
pip install git+https://github.com/ahmedosman/SUPR.git
```
and download the body model from the [SUPR project page](https://github.com/ahmedosman/SUPR). After the download, update the `supr_models` parameter in the `aitvconfig`, the same way as you did for the SMPL models.