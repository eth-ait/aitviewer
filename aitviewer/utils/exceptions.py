"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev

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


class ExceptionModule(object):
    """
    Source: Trimesh package @ https://github.com/mikedh/trimesh/blob/master/trimesh/exceptions.py

    Create a dummy module which will raise an exception when attributes
    are accessed.
    For soft dependencies we want to survive failing to import but
    we would like to raise an appropriate error when the functionality is
    actually requested so the user gets an easily debuggable message.
    """

    def __init__(self, exc):
        self.exc = exc

    def __getattribute__(self, *args, **kwargs):
        # if it's asking for our class type return None
        # this allows isinstance() checks to not re-raise
        if args[0] == "__class__":
            return None.__class__
        # otherwise raise our original exception
        raise super(ExceptionModule, self).__getattribute__("exc")
