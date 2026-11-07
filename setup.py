"""Bundle the released per-item and derived data into the wheel.

``data_release/`` stays the single copy in the repository. It is copied to
``argus_uq/_data/`` at build time so a wheel works without the repository.
Raw closed-source records and the master long CSVs are not bundled.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

BUNDLED_DIRS = ("per_item_full", "derived", "subsets")
BUNDLED_FILES = ("manifest.json", "panel_drops.json")


class BuildPyWithData(build_py):
    def run(self):
        super().run()
        src = Path(__file__).resolve().parent / "data_release"
        dst = Path(self.build_lib) / "argus_uq" / "_data"
        for name in BUNDLED_DIRS:
            shutil.copytree(src / name, dst / name, dirs_exist_ok=True)
        for name in BUNDLED_FILES:
            shutil.copy2(src / name, dst / name)


setup(cmdclass={"build_py": BuildPyWithData})
