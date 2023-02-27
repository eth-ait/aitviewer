import importlib
import os
import sys

import utils

if __name__ == "__main__":
    # Ensure that the directory that we will render reference images to exists.
    os.makedirs(utils.REFERENCE_DIR, exist_ok=True)

    # Get the headless viewer.
    viewer = utils.headless

    # Find test files.
    if len(sys.argv) == 1:
        # If no argument is specified import all python files that start with 'test' in test dir.
        files = [f for f in os.listdir(utils.TEST_DIR) if f.startswith("test")]
    else:
        # Otherwise treat each argument as an input module, only modules from the directory of this file
        # are supported.
        files = [os.path.basename(f) for f in sys.argv[1:]]

    # Import all files, at import time each function decorated with '@reference' will be registered
    # in the utils.ref_funcs dictionary.
    for f in files:
        mod = importlib.import_module(os.path.splitext(f)[0])

    # Iterate through all registered functions and create an image for each.
    for func, refs in utils.ref_funcs.items():
        print(f"{func.__name__}")

        # Reset the viewer.
        utils.initialize_viewer(viewer)

        # Call the test function to initialize the scene.
        func(viewer)

        # Render and save images.
        for ref, img in zip(refs, utils.generate_images(viewer, len(refs))):
            img.save(ref)
