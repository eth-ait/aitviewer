import os
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from utils import (
    RESOURCE_DIR,
    noreference,
    reference,
    requires_ffmpeg,
    requires_smpl,
    viewer,
)

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.camera import OpenCVCamera, WeakPerspectiveCamera
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer


@noreference
@requires_ffmpeg
def test_headless(viewer: HeadlessRenderer):
    sphere_positions = np.array(
        [
            [[0.0, 0, 0]],
            [[0.1, 0, 0]],
            [[0.2, 0, 0]],
            [[0.3, 0, 0]],
            [[0.4, 0, 0]],
        ],
        np.float32,
    )
    spheres = Spheres(sphere_positions, radius=0.05, color=(1.0, 0.0, 1.0, 1.0))

    viewer.playback_fps = 30.0
    viewer.scene.add(spheres)
    with TemporaryDirectory() as temp:
        mp4_path = os.path.join(temp, "video.mp4")
        gif_path = os.path.join(temp, "video.gif")
        frame_path = os.path.join(temp, "frames")

        viewer.save_video(frame_dir=frame_path, video_dir=mp4_path, output_fps=30)
        assert os.path.exists(os.path.join(temp, "video_0.mp4")), "MP4 file not found"
        frames = os.path.join(frame_path, "0000")
        assert os.path.exists(frames) and len(os.listdir(frames)) == sphere_positions.shape[0]

        viewer.save_video(frame_dir=frame_path, video_dir=gif_path, output_fps=30)
        assert os.path.exists(os.path.join(temp, "video_0.gif")), "GIF file not found"
        frames = os.path.join(frame_path, "0001")
        assert os.path.exists(frames) and len(os.listdir(frames)) == sphere_positions.shape[0]

        viewer.save_video(video_dir=mp4_path)
        assert os.path.exists(os.path.join(temp, "video_1.mp4")), "MP4 file not found"
        frames = os.path.join(frame_path, "0002")
        assert not os.path.exists(frames)


@reference()
@requires_smpl
def test_normals(viewer: Viewer):
    smpl_transparent = SMPLSequence.t_pose(
        SMPLLayer(model_type="smpl", gender="male", device=C.device),
        name="SMPL",
        position=np.array((-1, 0.0, 0.0)),
    )
    smpl_opaque = SMPLSequence.t_pose(
        SMPLLayer(model_type="smpl", gender="male", device=C.device),
        name="SMPL",
        position=np.array((1, 0.0, 0.0)),
    )

    smpl_transparent.mesh_seq.norm_coloring = True
    smpl_transparent.mesh_seq.color = smpl_transparent.mesh_seq.color[:3] + (0.5,)

    smpl_opaque.mesh_seq.norm_coloring = True
    viewer.scene.add(smpl_transparent, smpl_opaque)

    viewer.scene.camera.position = np.array([0.0, 0.5, 3.5])


@reference()
def test_empty(viewer: Viewer):
    viewer.scene.add(viewer.scene.floor)
    viewer.scene.add(viewer.scene.origin)
    viewer.scene.camera.position = np.array([0.2, 0.2, 0.2])
    viewer.scene.camera.target = np.array([0.0, 0.0, 0.0])


def add_cube(viewer: Viewer, pos):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, "cube.obj"), process=False)
    cube_mesh = Meshes(cube.vertices, cube.faces, name="Cube", position=pos, flat_shading=True)
    viewer.scene.add(cube_mesh)


@reference()
def test_weak_perspective_camera(viewer: Viewer):
    add_cube(viewer, np.array((0, 0, -5)))
    camera = WeakPerspectiveCamera(
        np.array([0.36266993, 0.62252431]),
        np.array([0.0100468, 0.36682156]),
        1920,
        1080,
        viewer=viewer,
    )
    viewer.scene.origin.enabled = False
    viewer.set_temp_camera(camera)


@reference()
def test_opencv_camera(viewer: Viewer):
    add_cube(viewer, np.array((0, 0, -5)))
    extrinsics = np.array(
        [
            [1.0, 0.0, 0.0, -0.09347329],
            [0.0, -1.0, 0.0, 0.08958537],
            [0.0, 0.0, -1.0, 3.0267725],
        ],
        np.float32,
    )
    intrinsics = np.array(
        [
            [1.10851252e03, 0.00000000e00, 6.40000000e02],
            [0.00000000e00, 1.10851252e03, 3.60000000e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        np.float32,
    )
    camera = OpenCVCamera(intrinsics, extrinsics, 1280, 720, viewer=viewer)
    viewer.scene.origin.enabled = False
    viewer.set_temp_camera(camera)


@reference()
def test_vertex_face_colors(viewer: Viewer):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, "cube.obj"), process=False)

    face_colors = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    per_face0 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        face_colors=face_colors,
        position=[-3.0, 0.0, -3.0],
        flat_shading=True,
    )
    per_face1 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[-3.0, 0.0, 3.0],
        flat_shading=True,
    )
    per_face1.face_colors = face_colors
    per_face2 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        face_colors=face_colors,
        position=[-3.0, 0.0, 0.0],
        flat_shading=True,
    )
    per_face2.face_colors = face_colors

    vertex_colors = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    per_vertex0 = Meshes(
        cube.vertices,
        cube.faces,
        vertex_colors=vertex_colors,
        name="Cube",
        position=[3.0, 0.0, -3.0],
        flat_shading=True,
    )
    per_vertex1 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[3.0, 0.0, 0.0],
        flat_shading=True,
    )
    per_vertex1.vertex_colors = vertex_colors
    per_vertex2 = Meshes(
        cube.vertices,
        cube.faces,
        vertex_colors=vertex_colors,
        name="Cube",
        position=[3.0, 0.0, 3.0],
        flat_shading=True,
    )
    per_vertex2.vertex_colors = vertex_colors

    color = color = (0.8, 0.5, 0.1, 1)
    uniform1 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[0.0, 0.0, -3.0],
        color=color,
        flat_shading=True,
    )
    uniform2 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[0.0, 0.0, 0.0],
        color=color,
        flat_shading=True,
    )
    uniform2.color = color
    uniform3 = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[0.0, 0.0, 3.0],
        flat_shading=True,
    )
    uniform3.color = color

    viewer.scene.camera.dolly_zoom(-100.0)
    viewer.scene.camera.rotate_azimuth_elevation(250, 250)
    viewer.scene.floor.enabled = False
    viewer.shadows_enabled = False
    viewer.scene.add(
        per_vertex0,
        per_vertex1,
        per_vertex2,
        per_face0,
        per_face1,
        per_face2,
        uniform1,
        uniform2,
        uniform3,
    )


@reference()
def test_instancing(viewer: Viewer):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, "cube.obj"), process=False)
    p = np.linspace(np.array([-8, 0, 0]), np.array([8, 0, 0]), num=10)
    r = aa2rot_numpy(np.linspace([0, 0, 0], np.array([0, 2 * np.pi, 0]), num=10))
    s = np.linspace(0.4, 0.8, num=10)

    viewer.scene.camera.position = (10, 5, 7)
    viewer.scene.camera.target = (1.5, -1, -2)

    cube_mesh = Meshes.instanced(cube.vertices, cube.faces, positions=p, rotations=r, scales=s, flat_shading=True)
    viewer.scene.add(cube_mesh)


@reference()
def test_sphere_and_line_colors(viewer: Viewer):
    s_pos = np.linspace(np.array([-1, 0, 0]), np.array([1, 0, 0]), 10)
    ls_pos = np.linspace(np.array([-1, 0, 1]), np.array([1, 0, 1]), 10)
    ll_pos = np.linspace(np.array([-1, 0, -1]), np.array([1, 0, -1]), 10)

    s_cols = np.linspace(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]), 10)
    ls_cols = np.linspace(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]), 9)
    ll_cols = np.linspace(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]), 5)

    viewer.scene.camera.position = (2, 2, 2)
    viewer.scene.camera.target = (0, 0, 0)

    viewer.scene.add(Spheres(s_pos, color=s_cols, radius=0.05))
    viewer.scene.add(Lines(ls_pos, color=ls_cols, r_base=0.05, mode="line_strip"))
    viewer.scene.add(Lines(ll_pos, color=ll_cols, r_base=0.05, mode="lines"))
