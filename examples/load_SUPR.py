"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

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
