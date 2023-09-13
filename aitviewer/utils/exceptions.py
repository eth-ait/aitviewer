# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos


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
