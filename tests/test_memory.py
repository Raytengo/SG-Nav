import sys
import types
from unittest.mock import MagicMock

# ── stub out every heavy import so we can import scenegraph without GPU ──
mocks_to_stub = [
    'torch', 'cv2', 'numpy', 'omegaconf', 'supervision',
    'ollama', 'PIL', 'PIL.Image', 'sklearn', 'sklearn.cluster',
    'segment_anything', 'GroundingDINO',
    'GroundingDINO.groundingdino', 'GroundingDINO.groundingdino.datasets',
    'GroundingDINO.groundingdino.datasets.transforms',
    'utils', 'utils.utils_scenegraph', 'utils.utils_scenegraph.mapping',
    'utils.utils_scenegraph.slam_classes', 'utils.utils_scenegraph.utils',
    'utils.utils_scenegraph.grounded_sam_demo',
]

for mod_name in mocks_to_stub:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import importlib
scenegraph = importlib.import_module('scenegraph')
RoomNode = scenegraph.RoomNode


def test_room_node_has_memory_and_status():
    """Test that RoomNode has memory and status fields."""
    node = RoomNode('kitchen')
    assert hasattr(node, 'memory')
    assert node.memory == []
    assert hasattr(node, 'status')
    assert node.status == 'unvisited'
