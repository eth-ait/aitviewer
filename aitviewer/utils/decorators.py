# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos


def requires_ctx(func):
    def _decorator(self, *args, **kwargs):
        if self.ctx is None:
            raise ValueError(
                "This function needs access to an OpenGL context - did you forget to call make_renderable()?"
            )
        else:
            return func(self, *args, **kwargs)

    return _decorator


def default_to_current_frame(func):
    def _decorator(self, *args, **kwargs):
        if len(args) == 0:
            kwargs["frame_id"] = kwargs.get("frame_id", self._current_frame_id)
        return func(self, *args, **kwargs)

    return _decorator


class hooked:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        func = self.fn

        def _decorator(self, *args, **kwargs):
            super_obj = super(owner, self)
            super_fn = getattr(super_obj, func.__name__)
            super_fn(*args, **kwargs)
            return func(self, *args, **kwargs)

        setattr(owner, name, _decorator)

    def __call__(self):
        assert (
            False
        ), "@hooked decorator object should never be called directly. This can happen if you apply this decorator to a function that is not a method."
