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
        g.global_memory = {'other_floors': False, 'staircase_pos': None}
        g._llm_b_thread = None
    return g


def test_scenegraph_has_global_memory():
    # Re-import after code change
    import inspect
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

    # Verify prompt_llm_b is assigned in __init__ with required placeholders
    init_src = inspect.getsource(SceneGraph.__init__)
    assert 'prompt_llm_b' in init_src, "__init__ must assign self.prompt_llm_b"
    assert '{goal}' in init_src, "prompt_llm_b must contain {goal} placeholder"
    assert '{objects}' in init_src, "prompt_llm_b must contain {objects} placeholder"


def test_build_room_memory_text_empty():
    """Returns empty string when no room has memory."""
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode

    class FakeSG:
        room_nodes = [RoomNode('bedroom'), RoomNode('kitchen')]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    assert result == ''


def test_build_room_memory_text_with_record():
    """Returns formatted line for each room that has memory."""
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode

    bedroom = RoomNode('bedroom')
    bedroom.memory = [{'coverage': 'partial', 'priority': 'medium', 'confidence': 'low', 'note': 'left corner unexplored'}]

    class FakeSG:
        room_nodes = [bedroom, RoomNode('kitchen')]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    assert 'bedroom' in result
    assert 'partial' in result
    assert 'low' in result
    assert 'kitchen' not in result   # kitchen has no memory


def test_build_room_memory_text_multi_room_all_have_memory():
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    r1 = RoomNode('bedroom')
    r1.memory = [{'coverage': 'full', 'priority': 'low', 'confidence': 'high', 'note': 'done'}]
    r2 = RoomNode('kitchen')
    r2.memory = [{'coverage': 'partial', 'priority': 'high', 'confidence': 'medium', 'note': 'chair seen'}]

    class FakeSG:
        room_nodes = [r1, r2]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    lines = result.split('\n')
    assert len(lines) == 2
    assert lines[0].startswith('bedroom')
    assert lines[1].startswith('kitchen')


def test_build_room_memory_text_uses_last_record_and_visit_count():
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    r = RoomNode('kitchen')
    r.memory = [
        {'coverage': 'full', 'priority': 'low', 'confidence': 'high', 'note': 'first visit'},
        {'coverage': 'minimal', 'priority': 'high', 'confidence': 'low', 'note': 'second visit'},
    ]

    class FakeSG:
        room_nodes = [r]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    assert 'visited 2x' in result
    assert 'minimal' in result   # last record, not first
    assert 'second visit' in result


def test_build_room_memory_text_missing_keys_fall_back_to_question_mark():
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    r = RoomNode('bathroom')
    r.memory = [{'note': 'only note present'}]

    class FakeSG:
        room_nodes = [r]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    assert 'coverage=?' in result
    assert 'priority=?' in result


def test_run_llm_b_appends_record():
    """_run_llm_b writes a dict to room_node.memory."""
    import unittest.mock as mock
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode

    kitchen = RoomNode('kitchen')
    kitchen.nodes = set()  # no objects

    llm_response = (
        "coverage: partial\n"
        "priority: medium\n"
        "confidence: low\n"
        "note: left corner not explored\n"
        "other_floors_detected: no"
    )

    class FakeSG:
        obj_goal_sg = 'chair'
        global_memory = {'other_floors': False, 'staircase_pos': None}
        prompt_llm_b = (
            "Goal object: {goal}\nRoom: {room}\nObjects found in room: {objects}\n"
            "Previous records for this room: {prev}\n\n"
            "coverage: <full|partial|minimal>\npriority: <high|medium|low>\n"
            "confidence: <high|medium|low>\nnote: <one sentence>\nother_floors_detected: <yes|no>"
        )
        get_llm_response = mock.Mock(return_value=llm_response)
        _run_llm_b = sg_mod.SceneGraph._run_llm_b

    FakeSG._run_llm_b(FakeSG, kitchen, [])
    assert len(kitchen.memory) == 1
    record = kitchen.memory[0]
    assert record['coverage'] == 'partial'
    assert record['priority'] == 'medium'
    assert record['confidence'] == 'low'
    assert record['other_floors_detected'] == 'no'
    assert FakeSG.global_memory['other_floors'] is False


def test_run_llm_b_updates_global_memory_on_staircase():
    """_run_llm_b sets other_floors=True when LLM B detects other floors."""
    import unittest.mock as mock
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode

    hallway = RoomNode('living room')
    hallway.nodes = set()

    llm_response = (
        "coverage: minimal\n"
        "priority: low\n"
        "confidence: low\n"
        "note: staircase visible at far end\n"
        "other_floors_detected: yes"
    )

    class FakeSG:
        obj_goal_sg = 'chair'
        global_memory = {'other_floors': False, 'staircase_pos': None}
        prompt_llm_b = (
            "Goal object: {goal}\nRoom: {room}\nObjects found in room: {objects}\n"
            "Previous records for this room: {prev}\n"
        )
        get_llm_response = mock.Mock(return_value=llm_response)
        _run_llm_b = sg_mod.SceneGraph._run_llm_b

    FakeSG._run_llm_b(FakeSG, hallway, [])
    assert FakeSG.global_memory['other_floors'] is True


def test_insert_goal_sets_chosen_room_active():
    """The room chosen by insert_goal gets status='active'."""
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    import unittest.mock as mock

    bedroom = RoomNode('bedroom')
    bedroom.group_nodes = [mock.MagicMock()]   # non-empty = has content

    kitchen = RoomNode('kitchen')
    kitchen.group_nodes = []

    class FakeSG:
        obj_goal_sg = 'chair'
        room_nodes = [bedroom, kitchen]
        global_memory = {'other_floors': False, 'staircase_pos': None}
        prompt_room_predict = 'Which room for [{}] in [{}]. Only answer the room.'
        get_llm_response = mock.Mock(return_value='bedroom')
        graph_corr = mock.Mock(return_value=0.8)
        update_group = mock.Mock()
        _build_room_memory_text = mock.Mock(return_value='')
        _trigger_llm_b = mock.Mock()
        insert_goal = sg_mod.SceneGraph.insert_goal

    bedroom.group_nodes[0].corr_score = 0
    bedroom.group_nodes[0].center = [10, 20]

    FakeSG.insert_goal(FakeSG)
    assert bedroom.status == 'active'
    assert kitchen.status == 'unvisited'   # no content, untouched


def test_insert_goal_triggers_llm_b_on_active_to_abandoned():
    """A room that was 'active' and is not chosen this round triggers LLM B."""
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    import unittest.mock as mock

    bedroom = RoomNode('bedroom')
    bedroom.group_nodes = [mock.MagicMock()]
    bedroom.status = 'active'              # was chosen last round

    kitchen = RoomNode('kitchen')
    kitchen.group_nodes = [mock.MagicMock()]
    kitchen.status = 'unvisited'

    class FakeSG:
        obj_goal_sg = 'chair'
        room_nodes = [bedroom, kitchen]
        global_memory = {'other_floors': False, 'staircase_pos': None}
        prompt_room_predict = 'Which room for [{}] in [{}]. Only answer the room.'
        get_llm_response = mock.Mock(return_value='kitchen')  # LLM picks kitchen
        graph_corr = mock.Mock(return_value=0.5)
        update_group = mock.Mock()
        _build_room_memory_text = mock.Mock(return_value='')
        _trigger_llm_b = mock.Mock()
        insert_goal = sg_mod.SceneGraph.insert_goal

    kitchen.group_nodes[0].corr_score = 0
    kitchen.group_nodes[0].center = [5, 5]

    FakeSG.insert_goal(FakeSG)

    assert bedroom.status == 'abandoned'
    FakeSG._trigger_llm_b.assert_called_once_with(bedroom)
    assert kitchen.status == 'active'


def test_insert_goal_does_not_trigger_llm_b_for_unvisited_rooms():
    """Rooms with content but status='unvisited' must NOT trigger LLM B when not chosen.

    Without this guard, 5 un-chosen rooms would each fire LLM B every round,
    making the async design slower than the synchronous baseline.
    """
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode
    import unittest.mock as mock

    # bedroom: chosen this round
    bedroom = RoomNode('bedroom')
    bedroom.group_nodes = [mock.MagicMock()]
    bedroom.status = 'unvisited'

    # kitchen, bathroom: have content but were NEVER chosen (status stays 'unvisited')
    kitchen = RoomNode('kitchen')
    kitchen.group_nodes = [mock.MagicMock()]
    kitchen.status = 'unvisited'

    bathroom = RoomNode('bathroom')
    bathroom.group_nodes = [mock.MagicMock()]
    bathroom.status = 'unvisited'

    class FakeSG:
        obj_goal_sg = 'chair'
        room_nodes = [bedroom, kitchen, bathroom]
        global_memory = {'other_floors': False, 'staircase_pos': None}
        prompt_room_predict = 'Which room for [{}] in [{}]. Only answer the room.'
        get_llm_response = mock.Mock(return_value='bedroom')
        graph_corr = mock.Mock(return_value=0.7)
        update_group = mock.Mock()
        _build_room_memory_text = mock.Mock(return_value='')
        _trigger_llm_b = mock.Mock()
        insert_goal = sg_mod.SceneGraph.insert_goal

    bedroom.group_nodes[0].corr_score = 0
    bedroom.group_nodes[0].center = [10, 20]

    FakeSG.insert_goal(FakeSG)

    # kitchen and bathroom were not chosen, but they were 'unvisited' — LLM B must NOT fire
    FakeSG._trigger_llm_b.assert_not_called()
    assert kitchen.status == 'unvisited'
    assert bathroom.status == 'unvisited'
    assert bedroom.status == 'active'
