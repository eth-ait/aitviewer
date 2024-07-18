# # Code Developed by Marilyn Keller, marilyn.keller@tuebingen.mpg.de
# # Do not share or distribute without permission of the author
import c3d
import numpy as np
import tqdm

try:
    import nimblephysics as nimble
except ImportError:
    raise ("nimblephysics not found. Please install nimblephysics with 'pip install nimblephysics' to use this module.")


def clean_CMU_mocap_labels(c3dFile: nimble.biomechanics.C3D):
    "Rename all the labels with the pattern AAAA-XX and replace them by AAAA"

    c3dFile.markers = [name for name in c3dFile.markers if "-" not in name]

    markerTimesteps = c3dFile.markerTimesteps.copy()

    for markers_dict in markerTimesteps:
        markers_dict_clean = markers_dict.copy()
        for key in markers_dict:
            if "-" in key:
                key_clean = key.split("-")[0]
                markers_dict_clean[key_clean] = markers_dict_clean.pop(key)
        markers_dict.clear()
        markers_dict.update(markers_dict_clean)

    c3dFile.markerTimesteps = markerTimesteps

    return c3dFile


def load_markers(c3d_path, nb_markers_expected=None):
    # Load the marker trajectories
    import os

    assert os.path.exists(c3d_path), f"File {c3d_path} not found."
    try:
        import nimblephysics as nimble
    except:
        raise ImportError("Please install nimblephysics to load c3d files")

    try:
        c3dFile: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(os.path.abspath(c3d_path))
    except Exception as e:
        print(f"Error loading c3d file {c3d_path}: {e}")
        raise e

    c3dFile = clean_CMU_mocap_labels(c3dFile)

    # This c3dFile.markerTimesteps is cryptonite, it keeps doing weird stuff (aka changing values, or you can not edit it),
    # it behaves normaly if you make a copy
    markers_data_list = c3dFile.markerTimesteps.copy()

    markers_labels = c3dFile.markers
    markers_labels.sort()
    nb_markers = len(markers_labels)

    if nb_markers_expected is not None:
        assert len(markers_labels) == nb_markers_expected, "Expected {} markers, found {}".format(
            nb_markers_expected, len(markers_labels)
        )
    print(f"Found {nb_markers} markers: {markers_labels}")

    # List of per frame pc array
    markers_array = np.zeros((len(markers_data_list), nb_markers, 3))  # FxMx3
    for frame_id, marker_data in (pbar := tqdm.tqdm(enumerate(markers_data_list))):
        pbar.set_description("Generating markers point clouds ")
        for marker_id, marker_name in enumerate(markers_labels):
            if marker_name in marker_data:
                marker_pos = marker_data[marker_name]
                if np.any(np.abs(marker_pos) > 10e2):
                    print(
                        "Warning: marker {} is too far away on frame {}, will be displayed in (0,0,0)".format(
                            marker_name, frame_id
                        )
                    )
                    marker_pos = np.nan * np.zeros((3))
            else:
                marker_pos = np.nan * np.zeros((3))
            markers_array[frame_id, marker_id, :] = marker_pos

    fps = c3dFile.framesPerSecond

    return markers_array, markers_labels, fps
