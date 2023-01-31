---
title: Visualizing Pose Estimators
layout: default
nav_order: 3
parent: Parametric Human Models
---
# Visualizing Pose Estimators
A common use case is to display the result of monocular RGB-based pose estimators. It is straight-forward to load the SMPL outputs from these models. If in addition you know the camera model, it is also easy to overlay the SMPL outputs with the input RGB image.

For example, in the [`load_ROMP.py`](https://github.com/eth-ait/aitviewer/blob/main/examples/load_ROMP.py) example, we first load the outputs that we previously stored to disk:
```python
results = np.load("resources/romp/romp_output.npz", allow_pickle=True)['results'][()]
smpl_layer = SMPLLayer(model_type='smpl', gender='male', device=C.device)
romp_smpl = SMPLSequence(poses_body=results['body_pose'],
                         smpl_layer=smpl_layer,
                         poses_root=results['global_orient'],
                         betas=results['smpl_betas'],
                         color=(0.0, 106 / 255, 139 / 255, 1.0),
                         name='ROMP Estimate',
                         rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi))
```

Then we prepare the camera, taking care that we get the conventions correct:
```python
# When using pyrender with ROMP, an FOV of 60 degrees is used.
# We mimic this here so that we get the same visualization as ROMP.
fov = 60
f = max(cols, rows)/2. * 1./np.tan(np.radians(fov/2))
cam_intrinsics = np.array([[f, 0., cols / 2], [0., f, rows / 2], [0., 0., 1.]])

# The camera extrinsics are assumed to identity rotation and the translation is estimated by ROMP.
cam_extrinsics = np.eye(4)
cam_extrinsics[:3, 3] = results['cam_trans'][0]

# The OpenCVCamera class expects extrinsics with Y pointing down, so we flip both Y and Z axis to keep a
# positive determinant.
cam_extrinsics[1:3, :3] *= -1.0

# Create an OpenCV camera.
cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], cols, rows, viewer=viewer)
```

With the cameras set up, we just need to load the input images as a Billboard and add everything to the viewer.
```python
pc = Billboard.from_camera_and_distance(cameras, 4.0, cols, rows, [img_path])
viewer.scene.add(pc, romp_smpl, cameras)
```
