import os

import numpy as np
import trimesh
from utils import RESOURCE_DIR, reference, requires_smpl, viewer

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy as aa2rot
from aitviewer.viewer import Viewer


@reference()
def test_renderables(viewer: Viewer):
    grid_xz = np.mgrid[-1.5:1.5:0.3, -1.5:1.5:0.3]
    n_lines = grid_xz.shape[1] * grid_xz.shape[2]

    xz_coords = np.reshape(grid_xz, (2, -1)).T
    line_starts = np.concatenate([xz_coords[:, 0:1], np.zeros((n_lines, 1)), xz_coords[:, 1:2]], axis=-1)
    line_ends = line_starts.copy()
    line_ends[:, 1] = 1.0
    line_strip = np.zeros((2 * n_lines, 3))
    line_strip[::2] = line_starts
    line_strip[1::2] = line_ends
    line_renderable = Lines(line_strip, r_base=0.03, mode="lines")

    line_dirs = line_ends - line_starts
    sphere_positions = line_ends + 0.1 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
    spheres = Spheres(sphere_positions, radius=0.05, color=(1.0, 0.0, 1.0, 1.0))

    rb_positions = line_ends + 0.4 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
    angles = np.arange(0.0, 2 * np.pi, step=2 * np.pi / n_lines)[:, None]
    axes = np.zeros((n_lines, 3))
    axes[:, 2] = 1.0
    rb_orientations = aa2rot(angles * axes)
    rbs = RigidBodies(rb_positions, rb_orientations)

    viewer.scene.add(line_renderable, spheres, rbs)
    viewer.scene.camera.position = np.array([2.0, 2.0, 2.0])
    viewer.scene.camera.target = np.array([0.0, 0.5, 0.0])


@reference()
def test_obj(viewer: Viewer):
    cube = trimesh.load(os.path.join(RESOURCE_DIR, "cube.obj"), process=False)
    cube_mesh = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[-7.0, 0.0, 0.0],
        flat_shading=True,
    )

    planet = trimesh.load(os.path.join(RESOURCE_DIR, "planet/planet.obj"), process=False)
    texture_image = os.path.join(RESOURCE_DIR, "planet/mars.png")
    planet_mesh = Meshes(
        planet.vertices,
        planet.faces,
        planet.vertex_normals,
        uv_coords=planet.visual.uv,
        path_to_texture=texture_image,
        position=[7.0, 0.0, 0.0],
    )

    drill = trimesh.load(os.path.join(RESOURCE_DIR, "drill/drill.obj"), process=False)
    texture_image = os.path.join(RESOURCE_DIR, "drill/drill_uv.png")
    drill_mesh = Meshes(
        drill.vertices,
        drill.faces,
        drill.vertex_normals,
        uv_coords=drill.visual.uv,
        path_to_texture=texture_image,
        scale=50.0,
        color=(1, 1, 1, 0.5),
    )

    viewer.scene.camera.position = (0, 5, 15)
    viewer.scene.camera.target = (0.15, 0.1, 0.33)
    viewer.auto_set_camera_target = False
    viewer.scene.add(planet_mesh, drill_mesh, cube_mesh)


@reference()
@requires_smpl
def test_smplh(viewer: Viewer):
    smplh_male = SMPLSequence.t_pose(
        SMPLLayer(model_type="smplh", gender="male", device=C.device),
        name="SMPL",
        position=np.array((-1, 0.0, 0.0)),
    )
    smplh_female = SMPLSequence.t_pose(
        SMPLLayer(model_type="smplh", gender="female", device=C.device),
        name="SMPL",
        position=np.array((1, 0.0, 0.0)),
    )
    viewer.scene.camera.position = np.array([0.0, 0.5, 3.5])
    viewer.scene.add(smplh_male, smplh_female)


@reference()
@requires_smpl
def test_smpl(viewer: Viewer):
    smpl_male = SMPLSequence.t_pose(
        SMPLLayer(model_type="smpl", gender="male", device=C.device),
        name="SMPL",
        position=np.array((-1.5, 0, 0)),
    )
    smpl_neutral = SMPLSequence.t_pose(
        SMPLLayer(model_type="smpl", gender="neutral", device=C.device),
        name="SMPL",
        position=np.array((0, 0, 0)),
    )
    smpl_female = SMPLSequence.t_pose(
        SMPLLayer(model_type="smpl", gender="female", device=C.device),
        name="SMPL",
        position=np.array((1.5, 0, 0)),
    )
    viewer.scene.camera.position = np.array([0.0, 0.5, 3.5])
    viewer.scene.add(smpl_male, smpl_female, smpl_neutral)


@reference()
@requires_smpl
def test_smplx(viewer: Viewer):
    smplx_male = SMPLSequence.t_pose(
        SMPLLayer(model_type="smplx", gender="male", device=C.device),
        name="SMPL",
        position=np.array((-1.5, 0, 0)),
    )
    smplx_neutral = SMPLSequence.t_pose(
        SMPLLayer(model_type="smplx", gender="neutral", device=C.device),
        name="SMPL",
        position=np.array((0, 0, 0)),
    )
    smplx_female = SMPLSequence.t_pose(
        SMPLLayer(model_type="smplx", gender="female", device=C.device),
        name="SMPL",
        position=np.array((1.5, 0, 0)),
    )
    viewer.scene.camera.position = np.array([0.0, 0.5, 3.5])
    viewer.scene.add(smplx_male, smplx_female, smplx_neutral)


@reference(count=3)
@requires_smpl
def test_amass(viewer: Viewer):
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    seq_amass = SMPLSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, "ACCAD/Female1Running_c3d/C2 - Run to stand_poses.npz"),
        fps_out=60.0,
        name="AMASS Running",
        color=c,
        show_joint_angles=True,
        log=False,
    )
    viewer.scene.camera.position = np.array([-3.3, 1.4, 0.2])
    viewer.scene.camera.target = np.array([-2.8, 1.0, -1.6])
    viewer.scene.add(seq_amass)
