"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import os
import shutil
import numpy as np
import tqdm
import trimesh
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.markers import Markers
from aitviewer.scene.node import Node
from aitviewer.utils.colors import vertex_colors_from_weights
from aitviewer.utils import to_numpy as c2c

import nimblephysics as nimble
import pickle as pkl

def load_osim(osim_path, geometry_path=C.osim_geometry):
    """Load an osim file"""
       
    assert os.path.exists(osim_path), f'Could not find osim file {osim_path}'
    osim_path = os.path.abspath(osim_path)
    
    # Check that there is a Geometry folder at the same level as the osim file
    file_geometry_path = os.path.join(os.path.dirname(osim_path), 'Geometry')
    
    import ipdb; ipdb.set_trace()
    if not os.path.exists(file_geometry_path):
        print(f'WARNING: No Geometry folder found at {file_geometry_path}, using {geometry_path} instead')
        # Create a copy of the osim file at the same level as the geometry folder
        tmp_osim_file = os.path.join(geometry_path, '..', 'tmp.osim')
        if os.path.exists(tmp_osim_file):
            #remove the old file
            os.remove(tmp_osim_file)
        shutil.copyfile(osim_path, tmp_osim_file)
        print(f'Copied {osim_path} to {tmp_osim_file}')
        osim_path = tmp_osim_file 
    
    osim : nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
    assert osim is not None, "Could not load osim file: {}".format(osim_path)
    return osim

class OSIMSequence(Node):
    """
    Represents a temporal sequence of OSSO poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(self,
                 osim,
                 motion,
                 color_markers_per_part = False,
                 color_markers_per_index = False, # Overrides color_markers_per_part
                 color_skeleton_per_part = False,
                 osim_path = None,
                 fps = None,
                 fps_in = None,
                 is_rigged = False,
                 show_joint_angles = False,
                 viewer = True,
                 **kwargs):
        """
        Initializer.
        :param osim_path: A osim model
        :param mot: A motion array 
        :osim_path: Path the osim model was loaded from (optional)
        :param kwargs: Remaining arguments for rendering.
        """
        self.osim_path = osim_path
        self.osim = osim
        self.motion = motion

        assert self.osim_path, "No osim path given"

        self.fps = fps
        self.fps_in = fps_in

        self._show_joint_angles = show_joint_angles
        self._is_rigged = is_rigged or show_joint_angles

        assert len(motion.shape) == 2
        super(OSIMSequence, self).__init__(n_frames=motion.shape[0], **kwargs)

        self._render_kwargs = kwargs

        # The node names of the skeleton model, the associated mesh and the template indices
        self.node_names = [n.getName() for n in osim.skeleton.getBodyNodes()]
        
        self.meshes_dict = {}
        self.indices_dict = {}
        self.generate_meshes_dict() # Populate self.meshes_dict and self.indices_dict
        self.create_template()

        # model markers
        markers_labels = [ml for ml in self.osim.markersMap.keys()]
        markers_labels.sort()
        self.markers_labels = markers_labels

        # Nodes
        self.vertices, self.faces, self.marker_trajectory, self.joints, self.joints_ori = self.fk()
        
        # TODO: fix that. This triggers a segfault at destruction so I hardcode it
        # self.joints_labels = [J.getName() for J in self.osim.skeleton.getJoints()]
        # self.joints_labels = ['ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r', 'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back', 'neck', 'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r', 'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l']
    
        if viewer == False:
            return

        if self._show_joint_angles:
            global_oris = self.joints_ori
            self.rbs = RigidBodies(self.joints, global_oris, length=0.01, name='Joint Angles')
            self.rbs.position = self.position
            self.rbs.rotation = self.rotation
            self._add_node(self.rbs)

        # Add meshes
        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Mesh'
        kwargs['color'] = kwargs.get('color', (160 / 255, 160 / 255, 160 / 255, 1.0))
        if color_skeleton_per_part:
            kwargs['vertex_colors'] = self.per_part_bone_colors()
        self.mesh_seq = Meshes(self.vertices, self.faces, **kwargs)
        self.mesh_seq.position = self.position
        self.mesh_seq.rotation = self.rotation
        self._add_node(self.mesh_seq)

        # Add markers
        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Markers'
        kwargs['color'] = kwargs.get('color', (0 / 255, 0 / 255, 255 / 255, 1.0))
        if color_markers_per_part:
            markers_color = self.per_part_marker_colors()
            kwargs['colors'] = markers_color
        if color_markers_per_index:
            marker_index_colors = vertex_colors_from_weights(weights=range(len(self.marker_trajectory[0])), scale_to_range_1=True, alpha=1)[np.newaxis, :, :]
            marker_index_colors = list(marker_index_colors)
            markers_color = marker_index_colors
            import ipdb; ipdb.set_trace()
            kwargs['colors'] = marker_index_colors
        self.markers_seq = Markers(points=self.marker_trajectory, markers_labels=self.markers_labels, 
                                point_size=10.0, **kwargs)
        self.markers_seq.position = self.position
        self.markers_seq.rotation = self.rotation
        self._add_node(self.markers_seq)
        
    def color_by_vertex_id(self):
        """
        Color the mesh by vertex index.
        """
        self.mesh_seq.color_by_vertex_index()

    def per_part_bone_colors(self):
        """ Color the mesh with one color per node. """
        vertex_colors = np.ones((self.n_frames, self.template.vertices.shape[0], 4))
        color_palette = vertex_colors_from_weights(np.arange(len(self.node_names)), shuffle=True)
        for i, node_name in enumerate(self.node_names):
            id_start, id_end = self.indices_dict[node_name]
            vertex_colors[:, id_start :id_end, 0:3] = color_palette[i, :]
        return vertex_colors

    def per_part_marker_colors(self):

        colors = vertex_colors_from_weights(np.arange(len(self.node_names)), alpha=1, shuffle=True)
        
        # Try to load a saved rigging file
        rigging_file = None
        if self.osim_path is not None:
            #try to load a rigging file
            rigging_file = self.osim_path.replace('.osim', f'_rigging.pkl')
            
        if not rigging_file is None and os.path.exists(rigging_file):
            print(f'Loading rigging file from {rigging_file}')
            rigging = pkl.load(open(rigging_file, 'rb'))
            marker_colors = colors[rigging['per_marker_rigging']]

        else:
            print(f'No rigging file {rigging_file} found. Fetching rigging for coloring.')
            colors = vertex_colors_from_weights(np.arange(len(self.node_names)), alpha=1, shuffle=True)

            markers_rigging = 1 * -np.ones(self.marker_trajectory.shape[1])
            marker_colors = np.ones((self.marker_trajectory.shape[1], 4))

            for mi, ml in (pbar := tqdm.tqdm(enumerate(self.markers_labels))):
                pbar.set_description("Computing the per marker rigging ")
                bone = self.osim.markersMap[ml][0].getName()
                bone_index = self.node_names.index(bone)
                markers_rigging[mi] = bone_index
                color = colors[bone_index]
                marker_colors[mi] = color
            # print(marker_colors)

        return marker_colors


    def generate_meshes_dict(self):
        """ Output a dictionary giving for each bone, the attached mesh"""

        current_index = 0
        self.indices_dict = {}
        self.meshes_dict = {}

        node_names = self.node_names
        for node_name in node_names:
            mesh_list = []
            body_node = self.osim.skeleton.getBodyNode(node_name)
            # print(f' Loading meshes for node: {node_name}')
            num_shape_nodes = body_node.getNumShapeNodes()
            if num_shape_nodes == 0:
                print(f'WARNING:\tNo shape nodes listed for  {node_name}')
            for shape_node_i in range(num_shape_nodes):
                shape_node = body_node.getShapeNode(shape_node_i)
                submesh_path = shape_node.getShape().getMeshPath()
                # Get the scaling for this meshes
                scale = shape_node.getShape().getScale()
                offset = shape_node.getRelativeTranslation()
                # Load the mesh
                try:
                    submesh = trimesh.load_mesh(submesh_path, process=False)
                    # print(f'Loaded mesh {submesh_path}')
                except Exception as e:
                    print(e)
                    print(f'WARNING:\tCould not load mesh {submesh_path}')
                    submesh = None
                    continue
                
                if submesh is not None:
                    trimesh.repair.fix_normals(submesh)
                    trimesh.repair.fix_inversion(submesh)
                    trimesh.repair.fix_winding(submesh)

                    # import pyvista
                    # submesh_poly = pyvista.read(submesh_path)
                    # faces_as_array = submesh_poly.faces.reshape((submesh_poly.n_faces, 4))[:, 1:] 
                    # submesh = trimesh.Trimesh(submesh_poly.points, faces_as_array) 

                    # Scale the bone to match .osim subject scaling
                    submesh.vertices[:] = submesh.vertices * scale
                    submesh.vertices[:] += offset
                    # print(f'submesh_path: {submesh_path}, Nb vertices: {submesh.vertices.shape[0]}')
                    mesh_list.append(submesh)

            # Concatenate meshes
            if mesh_list:
                node_mesh = trimesh.util.concatenate(mesh_list)
                self.indices_dict[node_name] = (current_index, current_index + node_mesh.vertices.shape[0])
                current_index += node_mesh.vertices.shape[0]
            else:
                node_mesh = None
                print("\t WARNING: No submesh for node:", node_name)
                self.indices_dict[node_name] = (current_index, current_index )
            
            # Add to the dictionary
            self.meshes_dict[node_name] = node_mesh
        print(self.meshes_dict)


    def create_template(self):

        part_meshes = []
        for node_name in self.node_names:
            mesh = self.meshes_dict[node_name]
            # assert mesh, "No mesh for node: {}".format(node_name)
            if mesh is None:
                print( "WARNING: No mesh for node: {}".format(node_name))
            if mesh:
                part_meshes.append(mesh)
        # part_meshes = [m for m in part_meshes if m]
        template = trimesh.util.concatenate(part_meshes)
        # import ipdb; ipdb.set_trace()
        
        template.remove_degenerate_faces()
        self.template = template

        #save mesh
        # # import ipdb; ipdb.set_trace()
        # self.template.export('template.obj')
        # print(f'Saved template to template.obj')

        # from psbody.mesh import Mesh
        # m = Mesh(filename='template.obj')
        # m.set_vertex_colors_from_weights(np.arange(m.v.shape[0]))
        # m.show()


    @classmethod
    def a_pose(cls, osim_path = None, **kwargs):
        """Creates a OSIM sequence whose single frame is a OSIM mesh in rest pose."""
        # Load osim file
        if osim_path is None:
            osim : nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
            osim_path = "RajagopalHumanBodyModel.osim" # This is not a real path, but it is needed to instantiate the sequence object
        else:
            osim = load_osim(osim_path)
            
        assert osim is not None, "Could not load osim file: {}".format(osim_path)
        motion = osim.skeleton.getPositions()[np.newaxis,:]

        return cls(osim, motion,
                    osim_path = osim_path,
                    **kwargs)
        
    @classmethod
    def zero_pose(cls, osim_path = None, **kwargs):
        """Creates a OSIM sequence whose single frame is a OSIM mesh in rest pose."""
        # Load osim file
        if osim_path is None:
            osim : nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
            osim_path = "RajagopalHumanBodyModel.osim" # This is not a real path, but it is needed to instantiate the sequence object
        else:
            osim = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
            
        assert osim is not None, "Could not load osim file: {}".format(osim_path)

        # motion = np.zeros((1, len(osim.skeleton.getBodyNodes())))
        motion = osim.skeleton.getPositions()[np.newaxis,:]
        motion = np.zeros_like(motion)
        # import ipdb; ipdb.set_trace()

        return cls(osim, motion,
                    osim_path = osim_path,
                    **kwargs)


    @classmethod
    def from_ab_folder(cls, ab_folder, trial, start_frame=None, end_frame=None, fps_out=None, **kwargs):   
        """
        Load an osim sequence from a folder returned by AddBiomechanics
        ab_folder: the folder returned by AddBiomechanics, ex: '/home/kellerm/Data/AddBiomechanics/CMU/01/smpl_head_manual'
        trial: Trial name
        start_frame: the first frame to load
        end_frame: the last frame to load
        fps_out: the output fps
        """
        
        if ab_folder[-1] != '/':
            ab_folder += '/'

        mot_file = ab_folder + f"IK/{trial}_ik.mot"
        osim_path = ab_folder + 'Models/optimized_scale_and_markers.osim'

        
        return OSIMSequence.from_files(osim_path=osim_path, mot_file=mot_file, start_frame=start_frame, end_frame=end_frame, fps_out=fps_out, **kwargs)



    @classmethod
    def from_files(cls, osim_path, mot_file, start_frame=None, end_frame=None, fps_out: int=None, ignore_fps=False, **kwargs):
        """Creates a OSIM sequence from addbiomechanics return data
        osim_path: .osim file path
        mot_file : .mot file path
        start_frame: first frame to use in the sequence
        end_frame: last frame to use in the sequence
        fps_out: frames per second of the output sequence
        """

        # Load osim file
        osim = load_osim(osim_path)

        # Load the .mot file
        mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
                    osim.skeleton, mot_file)

        motion = np.array(mot.poses.T)    

        # Crop and sample
        sf = start_frame or 0
        ef = end_frame or motion.shape[0]
        motion = motion[sf:ef]

        # estimate fps_in
        ts = np.array(mot.timestamps)   
        fps_estimated = 1/np.mean(ts[1:] - ts[:-1])
        fps_in = int(round(fps_estimated))
        
        if not ignore_fps:
            assert abs(1 - fps_estimated/fps_in) < 1e-5 , f"FPS estimation might be bad, {fps_estimated} rounded to {fps_in}, check."

            if fps_out is not None:
                assert fps_in%fps_out == 0, 'fps_out must be a interger divisor of fps_in'
                mask = np.arange(0, motion.shape[0], fps_in//fps_out)
                # motion = resample_positions(motion, fps_in, fps_out) #TODO: restore this 
                motion = motion[mask]
                
            del mot

        return cls(osim, motion, osim_path=osim_path, fps=fps_out, fps_in=fps_in, **kwargs)


    def fk(self):
        """Get vertices from the poses."""
        # Forward kinematics https://github.com/nimblephysics/nimblephysics/search?q=setPositions

        verts = np.zeros((self.n_frames, self.template.vertices.shape[0], self.template.vertices.shape[1]))
        markers = np.zeros((self.n_frames, len(self.markers_labels), 3))

        joints = np.zeros([self.n_frames, len(self.meshes_dict), 3])
        joints_ori = np.zeros([self.n_frames, len(self.meshes_dict), 3, 3])

        prev_verts = verts[0]
        prev_pose = self.motion[0, :]
        
        for frame_id in (pbar := tqdm.tqdm(range(self.n_frames))):
            pbar.set_description("Generating osim skeleton meshes ")

            pose = self.motion[frame_id, :]
            # If the pose did not change, use the previous frame verts
            if np.all(pose == prev_pose) and frame_id != 0:
                verts[frame_id] = prev_verts
                continue

            # Pose osim
            self.osim.skeleton.setPositions(self.motion[frame_id, :])

            # Since python 3.6, dicts have a fixed order so the order of this list should be marching labels
            markers[frame_id, :, :] = np.vstack(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).values())
            #Sanity check for previous comment
            assert list(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).keys()) == self.markers_labels, "Marker labels are not in the same order"

            for ni, node_name in enumerate(self.node_names):
                if ('thorax' in node_name) or ('lumbar' in node_name):
                    # We do not display the spine as the riggidly rigged mesh can't represent the constant curvature of the spine
                    continue
                mesh = self.meshes_dict[node_name]
                if mesh is not None:

                    part_verts = mesh.vertices

                    # pose part
                    transfo = self.osim.skeleton.getBodyNode(node_name).getWorldTransform()
                    
                    # Add a row of homogenous coordinates 
                    part_verts = np.concatenate([part_verts, np.ones((mesh.vertices.shape[0], 1))], axis=1)
                    part_verts = np.matmul(part_verts, transfo.matrix().T)[:,0:3]
                        
                    # Update the part in the full mesh       
                    id_start, id_end = self.indices_dict[node_name]
                    verts[frame_id, id_start :id_end, :] = part_verts

                    # Update joint                    
                    joints[frame_id, ni, :] = transfo.translation()
                    joints_ori[frame_id, ni, :, :] = transfo.rotation()
            

            prev_verts = verts[frame_id]
            prev_pose = pose

            
        faces = self.template.faces

        return c2c(verts), c2c(faces), markers, joints, joints_ori

    def redraw(self):
        self.vertices, self.faces, self.marker_trajectory, self.joints, self.joints_ori = self.fk()
        if self._is_rigged:
            self.skeleton_seq.joint_positions = self.joints
        self.mesh_seq.vertices = self.vertices
        self.marker_seq = self.marker_trajectory
        super().redraw()

        