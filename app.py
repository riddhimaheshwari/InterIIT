import sys
from pathlib import Path

# Ensure project root is in sys.path for local module resolution
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Run the Streamlit dashboard application
import runpy
dashboard_path = project_root / "dashboard" / "app.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
