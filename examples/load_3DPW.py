"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import glob
import os

import cv2

from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Load a 3DPW sequence and overlay it on the provided images. The data is expected to be in the same folder
    # structure that the download provides, i.e.:
    #  3DPW_root
    #  |--- sequenceFiles
    #       |--- test
    #       |--- train
    #       |--- validation
    #            |--- courtyard_jumpBench_01.pkl
    # |--- imageFiles
    #      |--- courtyard_jumpBench_01

    root_3dpw = r"E:\data\3dpw_with_imgs"
    sequence_name = "outdoors_slalom_01"

    pkl_file = None
    for split in ("test", "train", "validation"):
        pkl_file = os.path.join(root_3dpw, "sequenceFiles", split, sequence_name + ".pkl")
        if os.path.exists(pkl_file):
            break

    if not os.path.exists(pkl_file):
        raise ValueError("Could not find sequence {} in any of the splits.".format(sequence_name))

    # Load 3DPW sequence. This uses the SMPL model. This might return more than one sequence because some 3DPW
    # sequences contain multiple people.
    seqs_3dpw, camera_info = SMPLSequence.from_3dpw(
        pkl_data_path=pkl_file, name=sequence_name, color=(24 / 255, 106 / 255, 153 / 255, 1.0)
    )

    # Get image paths.
    image_folder = os.path.join(root_3dpw, "imageFiles", sequence_name)
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    image0 = cv2.imread(images[0])
    cols, rows = image0.shape[1], image0.shape[0]

    # Display in the viewer.
    v = Viewer(size=(cols // 2, rows // 2))
    v.playback_fps = 30.0
    v.scene.floor.enabled = False
    v.scene.origin.enabled = False

    cam = OpenCVCamera(camera_info["intrinsics"], camera_info["extrinsics"][:, :3], cols=cols, rows=rows, viewer=v)
    billboard = Billboard.from_camera_and_distance(cam, 15.0, cols=cols, rows=rows, textures=images)
    v.scene.add(*seqs_3dpw, cam, billboard)

    v.set_temp_camera(cam)
    v.run()
