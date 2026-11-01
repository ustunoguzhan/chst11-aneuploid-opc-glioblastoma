import os

_HERE = os.path.dirname(os.path.abspath(__file__))

PACKAGE_DATA = os.path.join(_HERE, "data")
DATA_DIR = os.environ.get("GBM_DATA", os.path.join(_HERE, "data"))
RESULTS_DIR = os.environ.get("GBM_RESULTS", os.path.join(_HERE, "results"))
CGGA_DIR = os.environ.get("GBM_CGGA", os.path.join(DATA_DIR, "cgga"))
EXT_DIR = os.environ.get("GBM_EXT", os.path.join(RESULTS_DIR, "..", "ext"))

os.makedirs(RESULTS_DIR, exist_ok=True)
