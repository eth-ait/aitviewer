
# Code Developed by:
# Marilyn Keller, marilyn.keller@tuebingen.mpg.de
# Do not share or distribute without permission of the author

import numpy as np
import os
import pickle as pkl

import tqdm
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.node import Node
import nimblephysics as nimble
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.utils import mocap
from aitviewer.utils.mocap import clean_CMU_mocap_labels


class Markers(Node):
    """
    Draw a point clouds man!
    """

    def __init__(self, points, markers_labels, name="Mocap data", colors=None, lengths=None, point_size=5.0, radius = 0.0075,  color=(0.0, 0.0, 1.0, 1.0), as_spheres=True, **kwargs):
        """
        A sequence of point clouds. Each point cloud can have a varying number of points.
        Internally represented as a list of arrays.
        :param points: Sequence of points (F, P, 3)
        :param colors: Sequence of Colors (F, C, 4)
        :param lengths: Length mask for each frame of points denoting the usable part of the array
        :param point_size: Initial point size
        """
        # self.points = points
        super(Markers, self).__init__(name, n_frames=points.shape[0], color=color, **kwargs)

        # Check that the marker labels are sorted
        # markers_labels_copy = markers_labels.copy()
        # markers_labels_copy.sort()
        # assert markers_labels == markers_labels_copy

    
        self.markers_labels = markers_labels
        self.marker_trajectory = points # FxMx3
        self.color = color

    
        if self.marker_trajectory.shape[1]>200:
            as_spheres = False
            print(f"Too many markers ({self.marker_trajectory.shape[1]}). Switching to pointcloud.")


        #todo fix color bug
        for mi, marker_name in enumerate(self.markers_labels):

            if colors is not None:
                color = tuple(colors[mi])

            if as_spheres:
                markers_seq = Spheres(self.marker_trajectory[:,mi,:][:,np.newaxis,:], color=color, radius = radius, name=marker_name, **kwargs)
            else:
                markers_seq = PointClouds(self.marker_trajectory[:,mi,:][:,np.newaxis,:], name=marker_name, point_size=point_size, color=color, **kwargs)
            markers_seq.position = [0,0,0]
            markers_seq.rotation = self.rotation
            markers_seq.enabled = False
            self._add_node(markers_seq)


    @classmethod
    def from_c3d(cls, c3d_path, start_frame=None, end_frame=None, fps_out=None, colors=None, lengths=None, 
                 nb_markers_expected=None, point_size=5.0, color=(0.0, 0.0, 1.0, 1.0), y_up=True, **kwargs):
        """Load a sequence from an npz file. The filename becomes the name of the sequence"""

        # Load the marker trajectories
        try:
            c3dFile: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(c3d_path)
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

        if nb_markers_expected is not None :
            assert len(markers_labels) == nb_markers_expected, "Expected {} markers, found {}".format(nb_markers_expected, len(markers_labels))
        print(f"Found {nb_markers} markers: {markers_labels}")

        # List of per frame pc array 
        markers_array = np.zeros((len(markers_data_list), nb_markers, 3)) # FxMx3
        for frame_id, marker_data in (pbar := tqdm.tqdm(enumerate(markers_data_list))):
            pbar.set_description("Generating markers point clouds ")
            for marker_id, marker_name in enumerate(markers_labels):
                if marker_name in marker_data:        
                    marker_pos = marker_data[marker_name]
                    if np.any(np.abs(marker_pos)>10e2):
                        print("Warning: marker {} is too far away on frame {}, will be displayed in (0,0,0)".format(marker_name, frame_id))
                        marker_pos = np.nan * np.zeros((3))
                else:
                    marker_pos = np.nan * np.zeros((3))
                markers_array[frame_id, marker_id, :] = marker_pos

        if y_up:
            markers_array[:, :, [0,1,2]] = markers_array[:, :, [0,2,1]] # Swap y and z
            markers_array[:, :, 2] = -markers_array[:, :, 2] # Flip z
   
        # print(markers_array)  
        name = "Mocap " + os.path.splitext(os.path.basename(c3d_path))[0]

        # Crop frames and resample
        sf = start_frame or 0
        ef = end_frame or markers_array.shape[0]
        markers_array = markers_array[sf:ef]
   
        fps_in = c3dFile.framesPerSecond
        if fps_out is not None and fps_in != fps_out:
            assert fps_in%fps_out == 0, 'fps_out must be a interger divisor of fps_in'
            mask = np.arange(0, markers_array.shape[0], fps_in//fps_out)       
            # markers_array = resample_positions(markers_array, fps_in, fps_out) # This uses splines and don't deal with NaN
            markers_array = markers_array[mask]

        return cls(
            markers_labels = markers_labels,
            points = markers_array,
            name = name,
            colors=colors, 
            lengths=lengths, 
            point_size=point_size, 
            color=color, **kwargs,
        )

    @classmethod
    def from_synthetic(cls, synth_mocap_path, start_frame=None, end_frame=None, fps_out=None, colors=None, lengths=None, nb_markers_expected=None, point_size=5.0, color=(0.0, 0.0, 1.0, 1.0), **kwargs):
        """Load a sequence from an npz file. The filename becomes the name of the sequence"""

        assert os.path.exists(synth_mocap_path), f"File {synth_mocap_path} does not exist"
        # Load the marker trajectories   
        synthetic_markers = pkl.load(open(synth_mocap_path, 'rb'))
        
        fps_in = int(synthetic_markers.fps)
        assert 1-int(synthetic_markers.fps)/fps_in < 1e-3, "fps must be an integer"
        markers_labels = synthetic_markers.marker_names
        name = 'Synthetic markers'
    
        markers_array = synthetic_markers.marker_trajectory

        if fps_out is not None and abs(1 - fps_in/fps_out) > 1e-4 :
        
            assert fps_in%fps_out == 0, f'fps_out must be a interger divisor of fps_in, but got fps_in={fps_in} fps_out={fps_out}'
            mask = np.arange(0, markers_array.shape[0], int(fps_in//fps_out))
         
            # markers_array = resample_positions(markers_array, fps_in, fps_out) # This uses splines and don't deal with NaN
            markers_array = markers_array[mask]

        return cls(
            markers_labels = markers_labels,
            points = markers_array,
            name = name,
            colors=colors, 
            lengths=lengths, 
            point_size=point_size, 
            color=color, **kwargs,
        )
        
        
    @classmethod
    def from_SSM_pkl(cls, ssm_pkl_path, fps_out=None, colors=None, lengths=None, point_size=5.0, color=(0.0, 0.0, 1.0, 1.0), **kwargs):
        """Load a sequence from an npz file. The filename becomes the name of the sequence"""

        # Load the marker trajectories
        markers_data = pkl.load(open(ssm_pkl_path, 'rb'), encoding='latin1') #dict_keys(['labels', 'required_parameters', 'markers'])

        fps_in = int(markers_data['required_parameters']['frame_rate'])
        fps_in = 60 #For the SSM dataset, the frame rate specified in the pkl file appears wrong, I assume it is 60 fps
        # assert abs(1-fps_in/markers_data['required_parameters']['frame_rate'])<1e-3, 'Frame rate is not an integer'
     
        markers_labels = [label.decode('utf-8') for label in markers_data['labels']]
        name = 'SSM markers'
    
        markers_array = markers_data['markers']
        
        # rotate the mocap data to align them with amass
        markers_array[:, :, [0,1,2]] = markers_array[:, :, [2,1,0]] # Swap x and z
        markers_array[:, :, 0] = -markers_array[:, :, 0] # Flip x

        if fps_out is not None and fps_in != fps_out:
            assert fps_in%fps_out == 0, 'fps_out must be a interger divisor of fps_in'
            mask = np.arange(0, markers_array.shape[0], int(fps_in//fps_out))
         
            # markers_array = resample_positions(markers_array, fps_in, fps_out) # This uses splines and don't deal with NaN
            markers_array = markers_array[mask]
            
        print(f'fps_in={fps_in} fps_out={fps_out} markers_array.shape={markers_array.shape}')

        return cls(
            markers_labels = markers_labels,
            points = markers_array,
            name = name,
            colors=colors, 
            lengths=lengths, 
            point_size=point_size, 
            color=color, **kwargs
        )

    @classmethod
    def from_file(cls, mocap_file, **kwargs):
        
        if mocap_file.endswith('.c3d'):
            return cls.from_c3d(mocap_file, **kwargs)
        elif mocap_file.endswith('.npz'):
            return cls.from_synthetic(mocap_file, **kwargs)
        elif mocap_file.endswith('.pkl'):
            return cls.from_SSM_pkl(mocap_file, **kwargs)
        else:
            raise ValueError(f'Unknown mocap file format: {mocap_file}, must be .c3d, .npz or .pkl')
        

