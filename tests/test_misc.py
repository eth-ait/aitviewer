from utils import reference, viewer, noreference, requires_smpl, RESOURCE_DIR

from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.smpl import SMPLSequence, SMPLLayer
from aitviewer.scene.camera import OpenCVCamera, WeakPerspectiveCamera
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C

import trimesh
import numpy as np
import os
from tempfile import TemporaryDirectory


@noreference
def test_headless(viewer: HeadlessRenderer):
    sphere_positions = np.array([
        [[0.0, 0, 0]],
        [[0.1, 0, 0]],
        [[0.2, 0, 0]],
        [[0.3, 0, 0]],
        [[0.4, 0, 0]],
    ], np.float32)
    spheres = Spheres(sphere_positions, radius=0.05, color=(1.0, 0.0, 1.0, 1.0))

    viewer.playback_fps = 30.0
    viewer.scene.add(spheres)
    with TemporaryDirectory() as temp:
        mp4_path = os.path.join(temp, 'video.mp4')
        gif_path = os.path.join(temp, 'video.gif')
        frame_path = os.path.join(temp, 'frames')

        viewer.save_video(frame_dir=frame_path, video_dir=mp4_path, output_fps=30)
        assert os.path.exists(os.path.join(temp, 'video_0.mp4')), "MP4 file not found"
        frames = os.path.join(frame_path, "0000")
        assert os.path.exists(frames) and len(os.listdir(frames)) == sphere_positions.shape[0]

        viewer.save_video(frame_dir=frame_path, video_dir=gif_path, output_fps=30)
        assert os.path.exists(os.path.join(temp, 'video_0.gif')), "GIF file not found"
        frames = os.path.join(frame_path, "0001")
        assert os.path.exists(frames) and len(os.listdir(frames)) == sphere_positions.shape[0]

        viewer.save_video(video_dir=mp4_path)
        assert os.path.exists(os.path.join(temp, 'video_1.mp4')), "MP4 file not found"
        frames = os.path.join(frame_path, "0002")
        assert not os.path.exists(frames)


@reference()
@requires_smpl
def test_normals(viewer: Viewer):
    smpl_transparent = SMPLSequence.t_pose(SMPLLayer(model_type='smpl', gender='male', device=C.device), name='SMPL',
                                         position=np.array((-1, 0.0, 0.0)))
    smpl_opaque = SMPLSequence.t_pose(SMPLLayer(model_type='smpl', gender='male', device=C.device), name='SMPL',
                                         position=np.array((1, 0.0, 0.0)))


    smpl_transparent.mesh_seq.norm_coloring = True
    smpl_transparent.mesh_seq.color = smpl_transparent.mesh_seq.color[:3] + (0.5,)

    smpl_opaque.mesh_seq.norm_coloring = True
    viewer.scene.add(smpl_transparent, smpl_opaque)


def add_cube(viewer: Viewer, pos):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, 'cube.obj'))
    cube_mesh = Meshes(cube.vertices, cube.faces, name='Cube', position=pos, flat_shading=True)
    viewer.scene.add(cube_mesh)


@reference()
def test_weak_perspective_camera(viewer: Viewer):
    add_cube(viewer, np.array((0, 0, -5)))
    camera = WeakPerspectiveCamera(np.array([0.36266993, 0.62252431]), np.array([0.0100468, 0.36682156]), 1920, 1080, viewer=viewer)
    viewer.scene.origin.enabled = False
    viewer.set_temp_camera(camera)


@reference()
def test_opencv_camera(viewer: Viewer):
    add_cube(viewer, np.array((0, 0, -5)))
    extrinsics = np.array([
        [ 1.,          0.,          0.,         -0.09347329],
        [ 0.,         -1.,          0.,          0.08958537],
        [ 0.,          0.,         -1.,          3.0267725 ],
    ], np.float32)
    intrinsics = np.array([
        [1.10851252e+03, 0.00000000e+00, 6.40000000e+02],
        [0.00000000e+00, 1.10851252e+03, 3.60000000e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
    ], np.float32)
    camera = OpenCVCamera(intrinsics, extrinsics, 1280, 720, viewer=viewer)
    viewer.scene.origin.enabled = False
    viewer.set_temp_camera(camera)


@reference()
def test_vertex_face_colors(viewer: Viewer):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, "cube.obj"))

    face_colors = np.array([
        [1, 0, 0, 1], [1, 1, 0, 1],
        [0, 1, 0, 1], [1, 0, 1, 1],
        [0, 0, 1, 1], [0, 1, 1, 1],
        [1, 0, 0, 1], [1, 1, 0, 1],
        [0, 1, 0, 1], [1, 0, 1, 1],
        [0, 0, 1, 1], [0, 1, 1, 1],
    ], dtype=np.float32)

    per_face0 = Meshes(cube.vertices, cube.faces, name='Cube', face_colors=face_colors, position=[-3.0, 0.0, -3.0], flat_shading=True)
    per_face1 = Meshes(cube.vertices, cube.faces, name='Cube', position=[-3.0, 0.0, 3.0], flat_shading=True)
    per_face1.face_colors = face_colors
    per_face2 = Meshes(cube.vertices, cube.faces, name='Cube', face_colors=face_colors, position=[-3.0, 0.0, 0.0], flat_shading=True)
    per_face2.face_colors = face_colors

    vertex_colors = np.array([
        [1, 0, 0, 1], [1, 1, 0, 1],
        [0, 1, 0, 1], [1, 0, 1, 1],
        [0, 0, 1, 1], [0, 1, 1, 1],
        [0, 0, 0, 1], [1, 1, 1, 1],
    ], dtype=np.float32)

    per_vertex0 = Meshes(cube.vertices, cube.faces, vertex_colors=vertex_colors, name='Cube', position=[3.0, 0.0, -3.0], flat_shading=True)
    per_vertex1 = Meshes(cube.vertices, cube.faces, name='Cube', position=[3.0, 0.0, 0.0], flat_shading=True)
    per_vertex1.vertex_colors = vertex_colors
    per_vertex2 = Meshes(cube.vertices, cube.faces, vertex_colors=vertex_colors, name='Cube', position=[3.0, 0.0, 3.0], flat_shading=True)
    per_vertex2.vertex_colors = vertex_colors

    color = color=(0.8, 0.5, 0.1, 1)
    uniform1 = Meshes(cube.vertices, cube.faces, name='Cube', position=[0.0, 0.0, -3.0], color=color, flat_shading=True)
    uniform2 = Meshes(cube.vertices, cube.faces, name='Cube', position=[0.0, 0.0,  0.0], color=color, flat_shading=True)
    uniform2.color = color
    uniform3 = Meshes(cube.vertices, cube.faces, name='Cube', position=[0.0, 0.0,  3.0], flat_shading=True)
    uniform3.color = color

    viewer.scene.camera.dolly_zoom(-100.0)
    viewer.scene.camera.rotate_azimuth_elevation(250, 250)
    viewer.scene.floor.enabled = False
    viewer.shadows_enabled = False
    viewer.scene.add(per_vertex0, per_vertex1, per_vertex2, per_face0, per_face1, per_face2, uniform1, uniform2, uniform3)