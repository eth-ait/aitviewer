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

from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node
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
            'ids':        A np array of camera ids (M)
            'intrinsics': A np array of camera intrinsics in the format used by OpenCV (M, 3, 3)
            'extrinsics': A np array of camera extrinsics in the format used by OpenCV (M, 3, 4)
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

        # Compute position and orientation of each camera
        positions = []
        self.cameras = []
        for i in range(len(camera_info['ids'])): 
            intrinsics = camera_info['intrinsics'][i]
            extrinsics = camera_info['extrinsics'][i]
            dist_coeffs = camera_info['dist_coeffs'][i]

            camera = OpenCVCamera(intrinsics, extrinsics, cols, rows, dist_coeffs=dist_coeffs, viewer=viewer, is_selectable=False)
            self.add(camera, show_in_hierarchy=False)
            self.cameras.append(camera)

            positions.append(camera.position)

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
        self.show_cameras = True

        self.set_active_camera(0)

    def set_active_camera(self, index):
        if self.active_camera_index is not None:
            camera = self.get_active_camera()
            camera.active = False
            camera.hide_frustum()
        
        self.active_camera_index = index
        self.cameras[index].active = True

    def get_active_camera(self):
        return self.cameras[self.active_camera_index]

    def view_from_active_camera(self):
        self.get_active_camera().view_from_camera()

    def update_frustum(self):
        camera = self.get_active_camera()
        if self.show_frustum:
            camera.show_frustum(self.cols, self.rows, self.billboard_distance)
        else:
            camera.hide_frustum()

    def update_billboard(self):
        # Release the old billboard
        if self.billboard:
            self.remove(self.billboard)
            self.billboard = None

        if self.show_billboard:
            # Look for images for the active camera
            camera_path = os.path.join(self.camera_images_path, str(self.camera_info['ids'][self.active_camera_index]))
            if not os.path.isdir(camera_path):
                print(f"Camera directory not found at {camera_path}")
                return

            # Sort images by the frame number in the filename
            files = os.listdir(camera_path)
            regex = re.compile(r"(\d*)$")

            def sort_key(x):
                name = os.path.splitext(x)[0]
                return int(regex.search(name).group(0))

            paths = [os.path.join(camera_path, f) for f in sorted(files, key=sort_key)]
            if not paths:
                print(f"Camera images not found at {camera_path}")
                return

            #Create a new billboard for the currently active camera
            self.billboard = Billboard.from_camera_and_distance(self.get_active_camera(), self.billboard_distance, self.cols, self.rows, paths)

            #Set the current frame index if we have an image for it
            if self.current_frame_id < len(paths):
                self.billboard.current_frame_id = self.current_frame_id
            
            self.add(self.billboard, show_in_hierarchy=False)
    
    def update_cameras(self):
        for c in self.cameras:
            c.enabled = self.show_cameras

    def gui(self, imgui):
        u_selected, active_index = imgui.combo("ID", self.active_camera_index, [str(id) for id in self.camera_info['ids'].tolist()])
        if u_selected:
            self.set_active_camera(active_index)
            self.view_from_active_camera()

        if imgui.button("View from camera"):
            self.view_from_active_camera()

            # Also disable the frustum since it's going to block the view
            self.show_frustum = False
            self.update_frustum()
            
        u_billboard, self.show_billboard = imgui.checkbox("Show billboard", self.show_billboard)    
        if u_billboard or u_selected:
            self.update_billboard()

        u_frustum, self.show_frustum = imgui.checkbox("Show frustum", self.show_frustum)    
        if u_frustum or u_selected:
            self.update_frustum()
        
        u_cameras, self.show_cameras = imgui.checkbox("Show cameras", self.show_cameras)
        if u_cameras:
            self.update_cameras()
    
    def gui_context_menu(self, imgui):
        imgui.text(f"Camera {self.camera_info['ids'][self.selected_camera_index]}")

        imgui.separator()
        _, s = imgui.menu_item(f"Activate camera", selected=self.active_camera_index == self.selected_camera_index)
        if s and self.active_camera_index != self.selected_camera_index:
            self.set_active_camera(self.selected_camera_index)
            self.update_billboard()
            self.update_frustum()
                
        _, v = imgui.menu_item(f"View from camera")
        if v:
            self.set_active_camera(self.selected_camera_index)
            self.update_billboard()
            self.update_frustum()
            self.view_from_active_camera()
        
        imgui.spacing()
        imgui.spacing()
        imgui.text(f"{self.name}")
        imgui.separator()
        
        u_billboard, self.show_billboard = imgui.checkbox("Show billboard", self.show_billboard)    
        if u_billboard:
            self.update_billboard()

        u_frustum, self.show_frustum = imgui.checkbox("Show frustum", self.show_frustum)    
        if u_frustum:
            self.update_frustum()
        
        u_cameras, self.show_cameras = imgui.checkbox("Show cameras", self.show_cameras)
        if u_cameras:
            self.update_cameras()
            
        
    def on_selection(self, node, tri_id):
        for idx, c in enumerate(self.cameras):
            if node in c.nodes:
                self.selected_camera_index = idx
                break
    
    # Disable outline rendering for this node and its children
    def render_outline(self, ctx, camera, prog):
        pass