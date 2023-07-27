import os
import shutil

import PyInstaller.__main__

OUTPUT_DIR = "installer"
BUILD_PATH = os.path.join(OUTPUT_DIR, "build")
DIST_PATH = os.path.join(OUTPUT_DIR, "dist")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# fmt: off
PyInstaller.__main__.run([
    "AITviewer.py",
    "--noconfirm",
    "--windowed",
    # "--exclude", "PyQt5",
    "--exclude", "matplotlib",
    "--exclude", "pandas",
    "--exclude", "cv2",
    "--exclude", "open3d",
    "--hidden-import", "moderngl_window.loaders.program.separate",
    "--hidden-import", "moderngl_window.loaders.program.single",
    "--hidden-import", "moderngl_window.loaders.program",
    "--hidden-import", "moderngl_window.context.pyglet",
    "--hidden-import", "glcontext",
    "--hidden-import", "encodings",
    "--workpath", BUILD_PATH,
    "--distpath", DIST_PATH,
    "--specpath", OUTPUT_DIR,
])
# fmt: on

root = os.path.join(DIST_PATH, "AITviewer")

os.makedirs(os.path.join(root, "aitviewer"), exist_ok=True)
shutil.copytree(
    os.path.join("aitviewer", "resources"), os.path.join(root, "aitviewer", "resources"), dirs_exist_ok=True
)
shutil.copytree(os.path.join("aitviewer", "shaders"), os.path.join(root, "aitviewer", "shaders"), dirs_exist_ok=True)
shutil.copy(os.path.join("aitviewer", "aitvconfig.yaml"), os.path.join(root, "aitviewer"))

# moderngl.window requires a scene folder to exist.
os.makedirs(os.path.join(root, "moderngl_window", "scene", "programs"), exist_ok=True)
