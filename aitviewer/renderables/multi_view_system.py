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
import os
import re
from collections import OrderedDict

import numpy as np

from aitviewer.renderables.billboard import Billboard
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node


class MultiViewSystem(Node):
    """A multi view camera system which can be used to visualize cameras and images"""

    def __init__(
        self,
        camera_info_path,
        camera_images_path,
        cols,
        rows,
        viewer,
        start_frame=0,
        **kwargs,
    ):
        """
        Load a MultiViewSystem from a file with camera information for M cameras
        and the path to a directory with M subdirectories, containing N images each.

        :param camera_info_path: path to a camera info .npz file with the following file entries
            'ids':         A np array of camera ids (M)
            'intrinsics':  A np array of camera intrinsics in the format used by OpenCV (M, 3, 3)
            'extrinsics':  A np array of camera extrinsics in the format used by OpenCV (M, 3, 4)
            'dist_coeffs': A np array of camera distortion coefficients in the format used by OpenCV (M, 5)
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
        :param viewer: the viewer, used for changing the view to one of the cameras.
        :param start_frame: crops the sequence of all cameras to start with the frame that has this ID.
        """
        # Load camera information.
        camera_info = np.load(camera_info_path)

        # Compute max number of images per camera and use it as number of frames for this node.
        camera_n_frames = []
        for id in camera_info["ids"]:
            path = os.path.join(camera_images_path, str(id))
            if os.path.isdir(path):
                n_frames = len(os.listdir(path)) - start_frame
                assert n_frames > 0, f"Camera {id} has no images after frame {start_frame}"
                camera_n_frames.append(n_frames)

        n_frames = max(camera_n_frames) if camera_n_frames else 1

        # Initialize node.
        super(MultiViewSystem, self).__init__(n_frames=n_frames, **kwargs)

        # Compute position and orientation of each camera.
        positions = []
        self.cameras = []
        for i in range(len(camera_info["ids"])):
            intrinsics = camera_info["intrinsics"][i]
            extrinsics = camera_info["extrinsics"][i]
            dist_coeffs = camera_info["dist_coeffs"][i]

            camera = OpenCVCamera(
                intrinsics,
                extrinsics,
                cols,
                rows,
                dist_coeffs=dist_coeffs,
                viewer=viewer,
                is_selectable=False,
            )
            self.add(camera, show_in_hierarchy=False)
            self.cameras.append(camera)

            positions.append(camera.position)

        # Compute maximum distance from a camera to the center of all cameras and use it
        # to compute the distance at which to show the billboards.
        positions = np.array(positions)
        camera_center = np.mean(positions, 0)
        max_dist = np.max(np.apply_along_axis(lambda x: np.linalg.norm(x - camera_center), 1, positions))
        self.billboard_distance = max_dist * 2
        self.camera_positions = positions

        self.viewer = viewer
        self.camera_info = camera_info
        self.camera_images_path = camera_images_path
        self.cols = cols
        self.rows = rows
        self.start_frame = start_frame

        # Maps from active camera index to its billboard or None if billboards are disabled.
        self.active_cameras = OrderedDict()

        self._billboards_enabled = False
        self._frustums_enabled = False
        self._cameras_enabled = True

        self.selected_camera_index = None

    def _create_billboard_for_camera(self, camera_index):
        """Helper function to create a billboard for the camera at the given index."""
        # Look for images for the given camera.
        camera_path = os.path.join(self.camera_images_path, str(self.camera_info["ids"][camera_index]))
        if not os.path.isdir(camera_path):
            print(f"Camera directory not found at {camera_path}")
            return

        # Sort images by the frame number in the filename.
        files = [x for x in os.listdir(camera_path) if x.endswith(".jpg")]
        regex = re.compile(r"(\d*)$")

        def sort_key(x):
            name = os.path.splitext(x)[0]
            return int(regex.search(name).group(0))

        paths = [os.path.join(camera_path, f) for f in sorted(files, key=sort_key)]
        if not paths:
            print(f"Camera images not found at {camera_path}")
            return

        paths = paths[self.start_frame :]
        # Create a new billboard for the currently active camera.
        billboard = Billboard.from_camera_and_distance(
            self.cameras[camera_index],
            self.billboard_distance,
            self.cols,
            self.rows,
            paths,
        )

        # Set the current frame index if we have an image for it.
        if self.current_frame_id < len(paths):
            billboard.current_frame_id = self.current_frame_id

        return billboard

    def activate_camera(self, index):
        """Activates the camera at the given index, showing its frustum and billboard if enabled."""
        if index not in self.active_cameras:
            # Change camera color.
            camera = self.cameras[index]
            camera.active = True
            # Create a billboard if billboards are enabled.
            billboard = None
            if self._billboards_enabled:
                billboard = self._create_billboard_for_camera(index)
                self.add(billboard, show_in_hierarchy=False)
            self.active_cameras[index] = billboard
            # Show the camera furstum if frustums are enabled.
            if self._frustums_enabled:
                camera.show_frustum(self.cols, self.rows, self.billboard_distance)

    def deactivate_camera(self, index):
        """Deactivates the camera at the given index, hiding its frustum and billboard if enabled."""
        if index in self.active_cameras:
            # Change camera color.
            camera = self.cameras[index]
            camera.active = False
            # Hide frustum.
            camera.hide_frustum()
            # Remove billboard if it exists.
            billboard = self.active_cameras[index]
            if billboard:
                self.remove(billboard)
            # Remove camera from active cameras
            del self.active_cameras[index]

    def view_from_camera(self, index, viewport):
        """
        View from the camera with the given index deactivating all other cameras,
        this also enables the billboard and disables the frustums.
        """
        self.billboards_enabled = True
        self.frustums_enabled = False
        active_cameras = list(self.active_cameras.keys())
        for i in active_cameras:
            if i != index:
                self.deactivate_camera(i)
        self.activate_camera(index)
        self.cameras[index].view_from_camera(viewport)

    @property
    def frustums_enabled(self):
        """Returns True if the frustums are enabled and False otherwise."""
        return self._frustums_enabled

    @frustums_enabled.setter
    def frustums_enabled(self, enabled):
        """Setting this to True shows the frustums of active cameras."""
        if enabled == self._frustums_enabled:
            return
        self._frustums_enabled = enabled

        # Update all frustums of all active cameras.
        for i in self.active_cameras.keys():
            if enabled:
                self.cameras[i].show_frustum(self.cols, self.rows, self.billboard_distance)
            else:
                self.cameras[i].hide_frustum()

    @property
    def billboards_enabled(self):
        """Returns True if the billboards are enabled and False otherwise."""
        return self._billboards_enabled

    @billboards_enabled.setter
    def billboards_enabled(self, enabled):
        """Setting this to True shows the billobards of active cameras."""
        if enabled == self._billboards_enabled:
            return
        self._billboards_enabled = enabled

        # Update billboards  of all active cameras.
        for index, billboard in self.active_cameras.items():
            if enabled:
                billboard = self._create_billboard_for_camera(index)
                self.add(billboard, show_in_hierarchy=False)
                self.active_cameras[index] = billboard
            else:
                # Billboard can be None here if the camera images were not found.
                if billboard is not None:
                    self.remove(billboard)
                self.active_cameras[index] = None

    @property
    def cameras_enabled(self):
        """Returns True if the cameras are enabled and False otherwise."""
        return self._cameras_enabled

    @cameras_enabled.setter
    def cameras_enabled(self, enabled):
        """Setting this to True shows the cameras as meshes in the scene."""
        if enabled == self._cameras_enabled:
            return
        self._cameras_enabled = enabled

        # Update all cameras.
        for c in self.cameras:
            c.enabled = enabled

    @property
    def bounds(self):
        return self.get_bounds(self.camera_positions)

    @property
    def current_bounds(self):
        return self.bounds

    def _gui_checkboxes(self, imgui):
        _, self.cameras_enabled = imgui.checkbox("Show cameras", self.cameras_enabled)
        _, self.billboards_enabled = imgui.checkbox("Show billboard", self.billboards_enabled)
        _, self.frustums_enabled = imgui.checkbox("Show frustum", self.frustums_enabled)

    def gui(self, imgui):
        self._gui_checkboxes(imgui)
        imgui.spacing()

        active_index = -1
        if len(self.active_cameras) > 0:
            active_index = next(reversed(self.active_cameras.keys()))
        u_selected, active_index = imgui.combo("ID", active_index, [str(id) for id in self.camera_info["ids"].tolist()])
        if u_selected:
            self.view_from_camera(active_index, self.viewer.viewports[0])

        imgui.text("Active cameras:")
        active_cameras = list(self.active_cameras.keys())
        for i in active_cameras:
            if imgui.button(f"{self.camera_info['ids'][i]}", width=50):
                self.view_from_camera(i, self.viewer.viewports[0])
            imgui.same_line(spacing=10)
            if imgui.button(f"x##{i}"):
                self.deactivate_camera(i)

    def gui_context_menu(self, imgui, x: int, y: int):
        if self.selected_camera_index is None:
            return

        imgui.text(f"Camera {self.camera_info['ids'][self.selected_camera_index]}")
        imgui.separator()

        u, s = imgui.menu_item(
            f"Activate camera",
            selected=self.selected_camera_index in self.active_cameras,
        )
        if u:
            if s:
                self.activate_camera(self.selected_camera_index)
            else:
                self.deactivate_camera(self.selected_camera_index)

        _, v = imgui.menu_item(f"View from camera")
        if v:
            self.view_from_camera(self.selected_camera_index, self.viewer.get_viewport_at_position(x, y))

        imgui.spacing()
        imgui.spacing()
        imgui.text(f"{self.name}")
        imgui.separator()

        self._gui_checkboxes(imgui)

    def on_selection(self, node, instance_id, tri_id):
        # Find which camera is selected.
        for idx, c in enumerate(self.cameras):
            # A camera is selected if the selected node is one of its children.
            if node in c.nodes:
                self.selected_camera_index = idx
                break

    def render_outline(self, *args, **kwargs):
        # Render outline of the currently selected camera.
        if self.selected_camera_index is not None:
            self.cameras[self.selected_camera_index].render_outline(*args, **kwargs)
