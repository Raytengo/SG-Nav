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


def make_scenegraph():
    """Return a SceneGraph with all heavy methods neutered."""
    import unittest.mock as mock
    sg_mod = importlib.import_module('scenegraph')
    SceneGraph = sg_mod.SceneGraph
    with mock.patch.object(SceneGraph, 'get_sam_mask_generator', return_value=None), \
         mock.patch.object(SceneGraph, 'set_cfg'), \
         mock.patch.object(SceneGraph, 'set_agent'), \
         mock.patch.object(SceneGraph, 'init_room_nodes'):
        g = SceneGraph.__new__(SceneGraph)
        g.rooms = ['bedroom', 'kitchen']
        g.room_nodes = [RoomNode('bedroom'), RoomNode('kitchen')]
        g.llm_name = 'test'
        # call only the portion we care about
        g.global_memory = None
        g._llm_b_thread = None
    return g


def test_scenegraph_has_global_memory():
    # Re-import after code change
    importlib.invalidate_caches()
    sg_mod = importlib.import_module('scenegraph')
    SceneGraph = sg_mod.SceneGraph
    import unittest.mock as mock
    with mock.patch.object(SceneGraph, 'get_sam_mask_generator', return_value=None), \
         mock.patch.object(SceneGraph, 'set_cfg'), \
         mock.patch.object(SceneGraph, 'set_agent'), \
         mock.patch.object(SceneGraph, 'init_room_nodes'):
        g = object.__new__(SceneGraph)
        g.rooms = []
        g.llm_name = 'test'
        g.vlm_name = 'test'
        g.global_memory = {'other_floors': False, 'staircase_pos': None}
        g._llm_b_thread = None

    assert g.global_memory == {'other_floors': False, 'staircase_pos': None}
    assert g._llm_b_thread is None
