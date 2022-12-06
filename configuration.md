---
title: Configuration
layout: default
nav_order: 2
---

The viewer loads default configuration parameters from [`aitvconfig.yaml`](aitviewer/aitvconfig.yaml). There are three ways how to override these parameters:
  - Create a file named `aitvconfig.yaml` and have the environment variable `AITVRC` point to it. Alternatively, you can point `AITVRC` to the directory containing `aitvconfig.yaml`.
  - Create a file named `aitvconfig.yaml` in your current working directory, i.e. from where you launch your python program.
  - Pass a `config` parameter to the `Viewer` constructor.

Note that the configuration files are loaded in this order, i.e. the config file in your working directory overrides all previous parameters.

The configuration management is using [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/index.html). You will probably want to override the following parameters at your convenience:
- `datasets.amass`: where [AMASS](https://amass.is.tue.mpg.de/) is stored if you want to load AMASS sequences.
- `smplx_models`: where SMPLX models are stored, preprocessed as required by the [`smplx` package](https://github.com/vchoutas/smplx).
- `star_models`: where the [STAR model](https://github.com/ahmedosman/STAR) is stored if you want to use it.
- `export_dir`: where videos and other outputs are stored by default.
