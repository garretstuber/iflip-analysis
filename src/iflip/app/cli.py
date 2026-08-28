"""Console entry point — launches the Streamlit app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent / "streamlit_app.py"
    if not app.is_file():
        sys.stderr.write(f"streamlit app not yet implemented at {app}\n")
        sys.exit(1)
    subprocess.run(["streamlit", "run", str(app), *sys.argv[1:]], check=False)
