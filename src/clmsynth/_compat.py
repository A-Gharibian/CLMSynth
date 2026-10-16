# _compat.py
"""Optional dependencies. Each is imported where it is used."""

import importlib
from types import ModuleType

# Import name -> the package as users know it, and how to install it.
INSTALL: dict[str, tuple[str, str]] = {
    "yaml": ("PyYAML", 'pip install "clmsynth[yaml]"'),
    # Not on PyPI, so it cannot be an extra.
    "mdcgenpy": ("mdcgenpy", "pip install git+https://github.com/CN-TU/mdcgenpy"),
}


def import_optional_dependency(name: str, purpose: str) -> ModuleType:
    """`name`, imported. When its package is not installed, the ImportError says
    that `purpose` needs it and how to install it. Any other failure, an installed
    package that cannot import one of its own dependencies among them, is raised
    as it is: "not installed" would send the user to the wrong fix."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        top = name.split(".")[0]
        if (e.name or "").split(".")[0] != top:
            raise
        package, install = INSTALL[top]
        raise ImportError(f"{purpose} needs {package}, which is not installed: {install}") from None
