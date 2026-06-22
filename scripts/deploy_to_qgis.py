"""Dev helper: install the planesight/ plugin into the local QGIS plugins folder.

Symlinks (or copies, with --copy) the planesight package from this repo into the
QGIS3 default-profile plugins directory, so QGIS loads your working tree directly.
Cross-platform best-effort. Restart QGIS or use the Plugin Reloader afterwards.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

PLUGIN_NAME = "planesight"


def qgis_plugins_dir() -> Path:
    """Return the QGIS3 default-profile Python plugins directory for this OS."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ["APPDATA"]) / "QGIS" / "QGIS3"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "QGIS" / "QGIS3"
    else:
        base = Path.home() / ".local" / "share" / "QGIS" / "QGIS3"
    return base / "profiles" / "default" / "python" / "plugins"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy instead of symlink (use on Windows without admin/dev mode)",
    )
    args = parser.parse_args()

    src = Path(__file__).resolve().parent.parent / PLUGIN_NAME
    if not src.is_dir():
        print(f"[ERROR] plugin source not found: {src}")
        return 1

    dest_dir = qgis_plugins_dir()
    dest = dest_dir / PLUGIN_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)

    if args.copy:
        shutil.copytree(src, dest)
        print(f"[OK] copied {src} -> {dest}")
    else:
        try:
            dest.symlink_to(src, target_is_directory=True)
            print(f"[OK] symlinked {dest} -> {src}")
        except OSError as exc:
            print(f"[ERROR] symlink failed ({exc}); retry with --copy")
            return 1

    print("Restart QGIS or use the Plugin Reloader to load PlaneSight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
