# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import os

from setuptools import find_packages, setup

from aitviewer import __version__

INSTALL_PYQT6 = os.getenv("AITVIEWER_INSTALL_PYQT6", 0)

requirements = [
    "torch>=1.6.0",
    "numpy>=1.18,<2",
    "opencv-contrib-python-headless>=4.5.1.48",
    "smplx",
    "moderngl-window>=2.4.3",
    "moderngl>=5.8.2,<6",
    "imgui==2.0.0",
    "tqdm>=4.60.0",
    "trimesh>=3.9.15,<4",
    "scipy>=1.5.2",
    "omegaconf>=2.1.1",
    "roma>=1.2.3",
    "joblib",
    "scikit-image>=0.9.0",
    "scikit-video",
    "Pillow",
    "websockets",
    "usd-core>=23.5",
]

# Choose PyQt version depending on environment variable.
if INSTALL_PYQT6:
    requirements.append("PyQt6>=6.5.2")
else:
    requirements.append("PyQt5>=5.15.4")

setup(
    name="aitviewer",
    description="Viewing and rendering of sequences of 3D data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eth-ait/aitviewer",
    version=__version__,
    author="Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos",
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "viewer",
        "moderngl",
        "machine learning",
        "sequences",
        "smpl",
        "computer graphics",
        "computer vision",
        "3D",
        "meshes",
        "visualization",
    ],
    platforms=["any"],
    python_requires=">=3.7,<3.11",
    install_requires=requirements,
    project_urls={
        "Documentation": "https://eth-ait.github.io/aitviewer/",
        "Source": "https://github.com/eth-ait/aitviewer",
        "Bug Tracker": "https://github.com/eth-ait/aitviewer/issues",
    },
)
