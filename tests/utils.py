import functools
import pytest
import os
import numpy as np
from PIL import Image

from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C

# Test image size.
SIZE = (480, 270)

# Paths used for testing.
TEST_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
RESOURCE_DIR = os.path.join(TEST_DIR, "..", "examples", "resources")
REFERENCE_DIR = os.path.join(TEST_DIR, "reference")
FAILURE_DIR = os.path.join(TEST_DIR, "failure")
SMPL_PRESENT = os.path.exists(C.smplx_models)

# Dictionary that maps functions to its list of paths of reference images.
ref_funcs = {}

# Headless viewer used for tests.
headless = HeadlessRenderer(size=SIZE)


def generate_images(viewer, count):
    """Generator that yields 'count' sequential rendered images from the viewer."""
    viewer._init_scene()
    viewer.run_animations = False
    # Render count frames, advancing the current frame at each iteration.
    for _ in range(count):
        viewer.render(0, 0, export=True)
        yield viewer.get_current_frame_as_image()
        viewer.scene.next_frame()


@pytest.fixture
def viewer(refs):
    """Fixture that resets the viewer, yields it and then compares generated images with a set of reference images."""
    headless.reset()
    yield headless
    for ref, img in zip(refs, generate_images(headless, len(refs))):
        # Load and check the refernce image.
        ref_img = Image.open(ref)
        if np.all(np.asarray(img) != np.asarray(ref_img)):
            os.makedirs(FAILURE_DIR, exist_ok=True)

            # Store the wrong result for debugging.
            wrong = os.path.join(FAILURE_DIR, os.path.basename(ref))
            img.save(wrong)

            assert False, f"Image does not match reference {ref}, the wrong image has been saved to {wrong}"


def requires_smpl(func):
    """Decorator for tests that require the SMPL dataset, if the dataset is not found the test will be skipped."""
    @pytest.mark.skipif(not SMPL_PRESENT, reason="SMPL dataset directory not found")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def noreference(func):
    """Decorator for tests that use the viewer but do not require reference images."""
    @pytest.mark.parametrize('refs', [[]])
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def reference(name=None, count=1):
    """
    Returns a decorator for comparing images rendered by the viewer with a reference image.
    :param name: file name of the reference image without extension, if None the part after 'test_' of the function name is used.
    :param count: the number of frames to be rendered while testing, if count > 1 a suffix of '_i' is appended to the i-th frame's filename.
    """
    def decorator(func):
        nonlocal name, count

        # If a name is not given we use the name of the function trimming the starting "test_" part if present.
        if not name:
            name = func.__name__
            if name.startswith("test_"):
                name = name[5:]
            assert name, "Invalid test name"

        # If this references more than one image create a list of paths by appending the frame index.
        if count > 1:
            print_name = f"{name}_[0-{count - 1}]"
            refs = [os.path.join(REFERENCE_DIR, f"{name}_{i}.png") for i in range(count)]
        else:
            print_name = name
            refs = [os.path.join(REFERENCE_DIR, f"{name}.png")]

        # Keep track of this test function and its references, this is used to generate reference images for all tests.
        ref_funcs[func] = refs

        @pytest.mark.skipif(any([not os.path.exists(p) for p in refs]), reason=f"{print_name}.png reference(s) not found")
        @pytest.mark.parametrize('refs', [refs])
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator