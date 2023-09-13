# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.renderables.supr import SUPRLayer, SUPRSequence
from aitviewer.viewer import Viewer

# Instantiate a SUPR layer. This requires that the respective repo has been installed via
# pip install git+https://github.com/ahmedosman/SUPR.git and that the model files are available on the path
# specified in `C.supr_models`.
#
# The directory structure should be:
#
# - models
#   |- supr_female.npy
#   |- supr_female_constrained.npy
#   |- supr_male.npy
#   |- supr_male_constrained.npy
#   |- supr_neutral.npy
#   |- supr_neutral_constrained.npy
model = SUPRLayer(constrained=False)

# Create a male SUPR T Pose.
template = SUPRSequence.t_pose(model, color=(0.62, 0.62, 0.62, 0.8))

v = Viewer()
v.scene.add(template)
v.run()
