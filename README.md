# aitviewer - SKEL

This fork of AitViewer enables the vizualization of Marker sequences, OpenSim models sequences, the BSM model and the SKEL model.

This repo contain a visualization tool. If you are interested in the SKEL model code, please refer to the [SKEL repository](https://download.is.tue.mpg.de/skel/main_paper.pdf). 

For more info on SKEL, BSM and BioaAmass, check our [project page](https://skel.is.tue.mpg.de) and our [paper](https://download.is.tue.mpg.de/skel/main_paper.pdf).

aitviewer is a set of tools to visualize and interact with sequences of 3D data with cross-platform support on Windows, Linux, and macOS. See the official page at [https://eth-ait.github.io/aitviewer](https://eth-ait.github.io/aitviewer/) for all the details.

![aitviewer skel gif](assets/skel_sequence.gif)
⇧ *aitviewer-Skel enables visualization of motion sequences of the SKEL model.*


## Installation

Clone this repository and install it using:
```
git clone https://github.com/MarilynKeller/aitviewer-skel.git
cd aitviewer-skel
pip install -e .
```

To set up the paths to SMPLX and AMASS, please refer to the [aitviewer instructions](https://eth-ait.github.io/aitviewer/frontend.html#configure-the-viewer)

## BSM model

You can download the BSM model `bsm.osim` from the dowload page at [https://skel.is.tue.mpg.de](https://skel.is.tue.mpg.de). To visualize it, run:

```python load_osim.py --osim /path/to/bsm.osim```

You can find motion sequences in the BioAmass dataset at [https://skel.is.tue.mpg.de](https://skel.is.tue.mpg.de).

To visualize an OpenSim motion sequence:

```
python load_osim.py --osim /path/to/bsm.osim --mot /path/to/trial.mot
```

![aitviewer osim vizu](assets/osim_apose.png)

## SKEL model

You can download the SKEL model from the dowload page at [https://skel.is.tue.mpg.de](https://skel.is.tue.mpg.de). 
Edit then the file aitviewer/aitvconfig.yaml` to point to the SKEL folder:
```skel_models: "/path/to/skel_models_v1.0"```

Install the SKEL loader by following the guidelines here: https://github.com/MarilynKeller/SKEL 

Vizualize the SKEL model's shape space:

```
python examples/load_SKEL.py
```

Vizualize a SKEL sequence. You can find a sample SKEL motion in `skel_models_v1.0/sample_motion/ ` and the corresponding SMPL motion.

```
python examples/load_SKEL.py -s 'skel_models_v1.1/sample_motion/01_01_poses_skel.pkl' --z_up
```


## BioAmass Dataset

First download the models and dataset from [https://skel.is.tue.mpg.de](https://skel.is.tue.mpg.de) and in `aitconfig.yaml` set the following paths:

```
osim_geometry : /path/to/skel_models_v1.0/Geometry
bioamass : /path/to/bioamass_v1.0
```

To visualize a sequence from the BioAmass dataset, run:

```
python examples/load_bioamass.py
```

## Mocap data
    
We enable loading .c3d and .trc motion capture data. Sample CMU mocap data can be downloaded at http://mocap.cs.cmu.edu/subjects.php.  Set the path to the mocap data folder in `aitvconfig.yaml` in `datasets.mocap`.

To visualize an example mocap sequence, run:

```python load_markers.py```

 

## Citation
If you use this software, please cite the following work and software:

```
@inproceedings{keller2023skel,
  title = {From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans},
  author = {Keller, Marilyn and Werling, Keenon and Shin, Soyong and Delp, Scott and 
            Pujades, Sergi and Liu, C. Karen and Black, Michael J.},
  booktitle = {ACM ToG, Proc.~SIGGRAPH Asia},
  volume = {42},
  number = {6},
  month = dec,
  year = {2023},
}
```

```
@software{Kaufmann_Vechev_aitviewer_2022,
  author = {Kaufmann, Manuel and Vechev, Velko and Mylonopoulos, Dario},
  doi = {10.5281/zenodo.10013305},
  month = {7},
  title = {{aitviewer}},
  url = {https://github.com/eth-ait/aitviewer},
  year = {2022}
}
```

## Licencing

For use of SKEL and BSM, please refer to our project page https://skel.is.tue.mpg.de/license.html.

## Contact 

For any question on the OpenSim model or SKEL loading, please contact skel@tuebingen.mpg.de.

For commercial licensing, please contact ps-licensing@tue.mpg.de
