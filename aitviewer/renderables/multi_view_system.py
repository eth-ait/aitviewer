from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.billboard import Billboard

import numpy as np
import os
import re


class MultiViewSystem(Node):
    """A multi view camera system which can be used to visualize cameras and images"""

    def __init__(self, camera_info_path, camera_images_path, cols, rows, viewer, **kwargs):
        """
        Load a MultiViewSystem from a file with camera information for M cameras 
        and the path to a directory with M subdirectories, containing N images each.

        :param camera_info_path: path to a camera info .npz file with the following file entries
            'ids':        Array of camera ids (M)
            'extrinsics': Array of camera extrinsics in the format used by OpenCV (M, 3, 4)
            'intrinsics': Array of camera intrinsics in the format used by OpenCV (M, 3, 3)
        :param camera_images_path: path to a directory with M subdirectories, one for each camera
            with the camera id as name. Each directory contains N images for the respective camera
            with a filename ending with the frame number.
            Example:
            camera_images_path/1/image001.png
            camera_images_path/1/image002.png
            camera_images_path/2/image001.png
            camera_images_path/2/image002.png
        :param cols: width  of the image in pixels, matching the size of the image expected by the intrinsics matrix
        :param rows: height of the image in pixels, matching the size of the image expected by the intrinsics matrix
        :param viewer: the viewer, used for changing the view to one of the cameras'
        """
        # Load camera information
        camera_info = np.load(open(camera_info_path, 'rb'))

        # Compute max number of images per camera and use it as number of frames for this node
        camera_n_frames = []
        for id in camera_info['ids']:
            path = os.path.join(camera_images_path, str(id))
            if os.path.isdir(path):
                camera_n_frames.append(len(os.listdir(path)))

        n_frames = max(camera_n_frames) if camera_n_frames else 1

        # Initialize node
        super(MultiViewSystem, self).__init__(n_frames=n_frames, **kwargs)

        # Vertices and faces for drawing cameras
        vertices = np.array([
            [ 0,  0, 0],
            [-1, -1, 1],
            [-1,  1, 1],
            [ 1, -1, 1],
            [ 1,  1, 1],
            
            [ 0.5,  1.1, 1],
            [-0.5,  1.1, 1],
            [   0,    2, 1],
        ], dtype=np.float32)
        vertices[:, 0] *= 0.17
        vertices[:, 1] *= 0.1
        vertices[:, 2]  *= 0.5
        vertices *= 0.3

        faces = np.array([
            [ 0, 1, 2],
            [ 0, 2, 4],
            [ 0, 4, 3],
            [ 0, 3, 1],
            [ 1, 3, 2],
            [ 4, 2, 3],
            [ 5, 6, 7],
            [ 5, 7, 6],
        ])

        self.ACTIVE_CAMERA_COLOR = (0.6, 0.1, 0.1, 1)
        self.INACTIVE_CAMERA_COLOR = (0.5, 0.5, 0.5, 1)

        # Compute position and orientation of each camera
        positions = []
        self.cameras = []
        for i in range(len(camera_info['ids'])):
            extrinsics = camera_info['extrinsics'][i]
            
            rot = np.linalg.inv(extrinsics[:, 0:3])
            pos = rot @ (-extrinsics[:, 3])
            rot[:,:2] = -rot[:,:2]

            pos -= rot[:, 2] * 0.15

            camera_mesh = Meshes(vertices, faces, position=pos, rotation = rot, cast_shadow=False)
            camera_mesh.color = self.INACTIVE_CAMERA_COLOR
            self.add(camera_mesh, show_in_hierarchy=False)
            self.cameras.append(camera_mesh)

            positions.append(pos)

        # Compute maximum distance from a camera to the center of all cameras and use it
        # to compute the distance at which to show the billboards
        positions = np.array(positions)
        camera_center = np.mean(positions, 0)
        max_dist = np.max(np.apply_along_axis(lambda x: np.linalg.norm(x - camera_center), 1, positions))
        self.billboard_distance = max_dist * 2

        self.viewer = viewer
        self.camera_info = camera_info
        self.camera_images_path = camera_images_path
        self.cols = cols
        self.rows = rows

        self.active_camera_index = None
        self.show_billboard = False
        self.billboard = None
        self.show_frustum = False
        self.frustum = None

        self.set_active_camera(0)

    def set_active_camera(self, index):
        if self.active_camera_index is not None:
            self.cameras[self.active_camera_index].color = self.INACTIVE_CAMERA_COLOR
        
        self.active_camera_index = index
        self.cameras[index].color = self.ACTIVE_CAMERA_COLOR


    def get_active_camera(self):
        idx = self.active_camera_index
        K = self.camera_info['intrinsics'][idx]
        Rt = self.camera_info['extrinsics'][idx]
        return OpenCVCamera(K, Rt, self.cols, self.rows)

    def change_view_to_active_camera(self):
        self.viewer.set_temp_camera(self.get_active_camera())

    def update_frustum(self):
        if self.frustum:
            self.remove(self.frustum)
            self.frustum = None

        if self.show_frustum:
            camera = self.get_active_camera()
            V, P = camera.compute_opengl_view_projection(self.cols, self.rows)
            ndc_from_world = P @ V
            world_from_ndc = np.linalg.inv(ndc_from_world)

            def transform(x):
                v = world_from_ndc @ np.append(x, 1.0)
                return v[:3] / v[3]

            # Comput z coordinate of a point at the given distance
            world_p = camera.position + camera.forward * self.billboard_distance
            ndc_p = (ndc_from_world @ np.concatenate([world_p, np.array([1])]))

            # Compute z after perspective division
            z = ndc_p[2] / ndc_p[3]

            lines = np.array([
                [-1, -1, -1], [-1,  1, -1],
                [-1, -1,  z], [-1,  1,  z],
                [ 1, -1, -1], [ 1,  1, -1],
                [ 1, -1,  z], [ 1,  1,  z],
                
                [-1, -1, -1], [-1, -1, z],
                [-1,  1, -1], [-1,  1, z],
                [ 1, -1, -1], [ 1, -1, z],
                [ 1,  1, -1], [ 1,  1, z],
                
                [-1, -1, -1], [ 1, -1, -1],
                [-1, -1,  z], [ 1, -1,  z],
                [-1,  1, -1], [ 1,  1, -1],
                [-1,  1,  z], [ 1,  1,  z],
            ])

            lines = np.apply_along_axis(transform, 1, lines)

            self.frustum = Lines(lines, position=self.position, r_base=0.005, mode='lines', cast_shadow=False)
            self.add(self.frustum, show_in_hierarchy = False)

    def update_billboard(self):
        # Release the old billboard
        if self.billboard:
            self.remove(self.billboard)
            self.billboard = None

        if self.show_billboard:
            # Look for images for the active camera
            camera_path = os.path.join(self.camera_images_path, str(self.camera_info['ids'][self.active_camera_index]))
            if not os.path.isdir(camera_path):
                print(f"Camera images not found at {camera_path}")
                return

            # Sort images by the frame number in the filename
            files = os.listdir(camera_path)
            regex = re.compile(r"(\d*)$")

            def sort_key(x):
                name = os.path.splitext(x)[0]
                return int(regex.search(name).group(0))

            paths = [os.path.join(camera_path, f) for f in sorted(files, key=sort_key)]
            
            #Create a new billboard for the currently active camera
            self.billboard = Billboard.from_opencv_camera_and_distance(self.get_active_camera(), self.billboard_distance, self.cols, self.rows, paths)
            self.billboard.current_frame_id = self.current_frame_id
            self.add(self.billboard, show_in_hierarchy = False)

    def gui(self, imgui):
        u_selected, active_index = imgui.combo("ID", self.active_camera_index, [str(id) for id in self.camera_info['ids'].tolist()])
        if u_selected:
            self.set_active_camera(active_index)

        if imgui.button("View from camera"):
            self.change_view_to_active_camera()

            # Also disable the frustum since it's going to block the view
            self.show_frustum = False
            self.update_frustum()
            
        u_billboard, self.show_billboard = imgui.checkbox("Show billboard", self.show_billboard)    
        if u_billboard or u_selected:
            self.update_billboard()

        u_frustum, self.show_frustum = imgui.checkbox("Show frustum", self.show_frustum)    
        if u_frustum or u_selected:
            self.update_frustum()
        