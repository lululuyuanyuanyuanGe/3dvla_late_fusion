import sys
from pathlib import Path

root = Path(__file__).resolve()
for p in [root] + list(root.parents):
    candidate = p / "SpatialVLA_llava3d"
    if candidate.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from model.processing_spatialvla_dev import SpatialVLAProcessor
