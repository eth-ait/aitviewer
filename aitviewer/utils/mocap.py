# Code Developed by Marilyn Keller, marilyn.keller@tuebingen.mpg.de
# Do not share or distribute without permission of the author

import nimblephysics as nimble


def clean_CMU_mocap_labels(c3dFile: nimble.biomechanics.C3D):
    "Rename all the labels with the pattern AAAA-XX and replace them by AAAA"

    c3dFile.markers = [name for name in c3dFile.markers if "-" not in name]

    markerTimesteps = c3dFile.markerTimesteps.copy()

    for markers_dict in markerTimesteps:
        markers_dict_clean = markers_dict.copy()
        for key in markers_dict:
            if '-' in key:
                key_clean = key.split('-')[0]
                markers_dict_clean[key_clean] = markers_dict_clean.pop(key)
        markers_dict.clear()
        markers_dict.update(markers_dict_clean)

    c3dFile.markerTimesteps = markerTimesteps


    return c3dFile