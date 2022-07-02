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

def requires_ctx(func):
    def _decorator(self, *args, **kwargs):
        if self.ctx is None:
            raise ValueError(
                'This function needs access to an OpenGL context - did you forget to call make_renderable()?')
        else:
            return func(self, *args, **kwargs)
    return _decorator


def default_to_current_frame(func):
    def _decorator(self, *args, **kwargs):
        if len(args) == 0:
            kwargs['frame_id'] = kwargs.get('frame_id', self._current_frame_id)
        return func(self, *args, **kwargs)
    return _decorator


def hooked(func):
    """Decarotor to call the super method of a function before calling the function itself."""
    def _decorator(self, *args, **kwargs):
        super_obj = super(type(self), self)
        super_fn = getattr(super_obj, func.__name__)
        super_fn(*args, **kwargs)
        return func(self, *args, **kwargs)
    return _decorator
