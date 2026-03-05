import pathlib
import pyproj

# Set project root path
# PROJECT_PATH = r'JTEPA-2026-bus-network-topology-ridership-pattern'
PROJECT_PATH = pathlib.Path(__file__).resolve().parent.parent
PROJECT_PATH = pathlib.Path(PROJECT_PATH)

# Projected CRS for Singapore, EPSG:3414 / SVY21
PROJECTED_CRS = pyproj.CRS('EPSG:3414')
