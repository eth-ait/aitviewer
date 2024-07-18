# Copyright (C) 2024 Max Planck Institute for Intelligent Systems, Marilyn Keller, marilyn.keller@tuebingen.mpg.de
import argparse
import os

import pyvista  # required as a .vtp reader


def convert_meshes(src_folder, dst_folder):
    src = src_folder
    if src[-1] != "/":
        src += "/"
    if dst_folder is None:
        target = src + "../Geometry_ply/"
    else:
        target = dst_folder
        if target[-1] != "/":
            target += "/"

    os.makedirs(target, exist_ok=True)

    # for each file in src
    for filename in os.listdir(src):
        ext = os.path.splitext(filename)[-1]
        if ext not in [".vtp", ".obj"]:
            print("Skipping " + filename)
            continue
        try:
            reader = pyvista.get_reader(src + filename)
        except:
            print("Could not read " + filename + ". Skipping.")
            continue

        mesh = reader.read()
        mesh = mesh.triangulate()
        # mesh.plot()

        mesh.save(target + filename + ".ply")
        print("Converted mesh: " + target + filename + ".ply")


if __name__ == "__main__":
    # Parse a vtp file and convert it to a ply file
    parser = argparse.ArgumentParser(description="Convert a folder of vtp files to a folder of ply files")
    parser.add_argument(
        "src_folder",
        help="folder containing the vtp files to convert",
        default="/home/kellerm/Dropbox/MPI/TML/Fullbody_TLModels_v2.0_OS4x/Geometry/",
        type=str,
    )
    parser.add_argument("dst_folder", help="folder to save the ply files", default=None, type=str)

    args = parser.parse_args()

    src_folder = args.src_folder
    dst_folder = args.dst_folder
