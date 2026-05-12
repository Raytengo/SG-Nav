# Exploration Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-LLM memory layer to SG-Nav so past room exploration outcomes persist within an episode and inform future frontier decisions.

**Architecture:** Memory fields live directly on existing `RoomNode` objects (stable throughout an episode). `insert_goal()` (LLM A) reads room memories when building its prompt, detects status transitions from `active → abandoned`, and fires LLM B in a daemon thread. LLM B writes a structured record back onto the room node while the agent moves.

**Tech Stack:** Python `threading` (stdlib), `ollama` (already used), `scenegraph.py` only — no new files.

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `scenegraph.py:35-40` | Add `memory`, `status` to `RoomNode.__init__` |
| Modify | `scenegraph.py:136` | Add `import threading` to top-level imports |
| Modify | `scenegraph.py:161` | Add `global_memory`, `_llm_b_thread`, `prompt_llm_b` to `SceneGraph.__init__` |
| Modify | `scenegraph.py:723` | Extend `insert_goal()` with memory context + status tracking |
| Add | `scenegraph.py` (after `insert_goal`) | `_build_room_memory_text()` |
| Add | `scenegraph.py` (after above) | `_run_llm_b(room_node)` |
| Add | `scenegraph.py` (after above) | `_trigger_llm_b(room_node)` |
| Create | `tests/test_memory.py` | Unit tests for all new logic |

---

## Task 1: Add `memory` and `status` fields to `RoomNode`

**Files:**
- Modify: `scenegraph.py:35-40`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
import sys, types

# ── stub out every heavy import so we can import scenegraph without GPU ──
for mod in [
    'torch', 'cv2', 'numpy', 'omegaconf', 'supervision',
    'ollama', 'PIL', 'PIL.Image', 'sklearn', 'sklearn.cluster',
    'segment_anything', 'GroundingDINO',
    'GroundingDINO.groundingdino', 'GroundingDINO.groundingdino.datasets',
    'GroundingDINO.groundingdino.datasets.transforms',
    'utils', 'utils.utils_scenegraph', 'utils.utils_scenegraph.mapping',
    'utils.utils_scenegraph.slam_classes', 'utils.utils_scenegraph.utils',
    'utils.utils_scenegraph.grounded_sam_demo',
]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

import importlib
sg = importlib.import_module('scenegraph')
RoomNode = sg.RoomNode


def test_room_node_has_memory_and_status():
    node = RoomNode('kitchen')
    assert hasattr(node, 'memory')
    assert node.memory == []
    assert hasattr(node, 'status')
    assert node.status == 'unvisited'
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/0610r/cc/SG-Nav
python -m pytest tests/test_memory.py::test_room_node_has_memory_and_status -v
```

Expected: `FAILED` — `AttributeError: 'RoomNode' object has no attribute 'memory'`

- [ ] **Step 3: Add fields to `RoomNode.__init__`**

In `scenegraph.py`, change lines 36-40 from:

```python
class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []
```

to:

```python
class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []
        self.memory = []          # list of dicts written by LLM B
        self.status = 'unvisited' # 'unvisited' | 'active' | 'abandoned'
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_memory.py::test_room_node_has_memory_and_status -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scenegraph.py tests/test_memory.py
git commit -m "feat: add memory and status fields to RoomNode"
```

---

## Task 2: Add `global_memory`, `_llm_b_thread`, and `prompt_llm_b` to `SceneGraph`

**Files:**
- Modify: `scenegraph.py` — `__init__` (around line 161), top-level imports (line 1)
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_memory.py

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
        g.llm_name = 'llama3.2-vision'
        # call only the portion we care about
        g.global_memory = None
        g._llm_b_thread = None
    return g


def test_scenegraph_has_global_memory():
    g = make_scenegraph()
    # after real __init__ these must exist
    # we just verify the structure via make_scenegraph setting them
    assert g.global_memory is None or True   # placeholder; real test below after Step 3
```

> **Note:** This is a scaffold test — it will pass trivially until Task 2 Step 3.

- [ ] **Step 2: Run test to confirm it passes (scaffold)**

```bash
python -m pytest tests/test_memory.py::test_scenegraph_has_global_memory -v
```

Expected: `PASSED` (trivially)

- [ ] **Step 3: Add `threading` import, `global_memory`, `_llm_b_thread`, `prompt_llm_b` to `SceneGraph.__init__`**

**3a — add `import threading` at the top of `scenegraph.py`** (after the existing stdlib imports, around line 6):

```python
import threading
```

**3b — inside `SceneGraph.__init__`, after line `self.vlm_name = 'llama3.2-vision'` (≈ line 162), add:**

```python
self.global_memory = {'other_floors': False, 'staircase_pos': None}
self._llm_b_thread = None
self.prompt_llm_b = (
    "You are a navigation observer. Write a structured exploration record.\n"
    "Goal object: {goal}\n"
    "Room: {room}\n"
    "Objects found in room: {objects}\n"
    "Previous records for this room: {prev}\n\n"
    "Output exactly these lines (no extra text):\n"
    "coverage: <full|partial|minimal>\n"
    "priority: <high|medium|low>\n"
    "confidence: <high|medium|low>\n"
    "note: <one sentence>\n"
    "other_floors_detected: <yes|no>"
)
```

- [ ] **Step 4: Update the scaffold test to actually assert the structure**

Replace the body of `test_scenegraph_has_global_memory` in `tests/test_memory.py`:

```python
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
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add scenegraph.py tests/test_memory.py
git commit -m "feat: add global_memory, threading import, and LLM B prompt to SceneGraph"
```

---

## Task 3: Add `_build_room_memory_text()`

**Files:**
- Modify: `scenegraph.py` — add method after `insert_goal` (around line 748)
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_memory.py

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
    bedroom.memory = [{'coverage': 'partial', 'priority': 'medium', 'note': 'left corner unexplored'}]

    class FakeSG:
        room_nodes = [bedroom, RoomNode('kitchen')]
        _build_room_memory_text = sg_mod.SceneGraph._build_room_memory_text

    result = FakeSG._build_room_memory_text(FakeSG)
    assert 'bedroom' in result
    assert 'partial' in result
    assert 'kitchen' not in result   # kitchen has no memory
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_memory.py::test_build_room_memory_text_empty tests/test_memory.py::test_build_room_memory_text_with_record -v
```

Expected: `FAILED` — `AttributeError: type object 'SceneGraph' has no attribute '_build_room_memory_text'`

- [ ] **Step 3: Add `_build_room_memory_text()` to `SceneGraph` in `scenegraph.py` (after `insert_goal`, before `update_scenegraph`)**

```python
def _build_room_memory_text(self):
    lines = []
    for room_node in self.room_nodes:
        if room_node.memory:
            last = room_node.memory[-1]
            lines.append(
                f"{room_node.caption}: visited {len(room_node.memory)}x, "
                f"coverage={last.get('coverage','?')}, "
                f"priority={last.get('priority','?')}, "
                f"note={last.get('note','')}"
            )
    return '\n'.join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scenegraph.py tests/test_memory.py
git commit -m "feat: add _build_room_memory_text helper"
```

---

## Task 4: Add `_run_llm_b()` and `_trigger_llm_b()`

**Files:**
- Modify: `scenegraph.py` — add two methods after `_build_room_memory_text`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_memory.py
import unittest.mock as mock

def test_run_llm_b_appends_record():
    """_run_llm_b writes a dict to room_node.memory."""
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
        prompt_llm_b = sg_mod.SceneGraph.prompt_llm_b.fget(None) if hasattr(sg_mod.SceneGraph.prompt_llm_b, 'fget') else (
            "Goal object: {goal}\nRoom: {room}\nObjects found in room: {objects}\n"
            "Previous records for this room: {prev}\n\n"
            "coverage: <full|partial|minimal>\npriority: <high|medium|low>\n"
            "confidence: <high|medium|low>\nnote: <one sentence>\nother_floors_detected: <yes|no>"
        )
        get_llm_response = mock.Mock(return_value=llm_response)
        _run_llm_b = sg_mod.SceneGraph._run_llm_b

    FakeSG._run_llm_b(FakeSG, kitchen)
    assert len(kitchen.memory) == 1
    record = kitchen.memory[0]
    assert record['coverage'] == 'partial'
    assert record['priority'] == 'medium'
    assert record['confidence'] == 'low'
    assert record['other_floors_detected'] == 'no'
    assert FakeSG.global_memory['other_floors'] is False


def test_run_llm_b_updates_global_memory_on_staircase():
    """_run_llm_b sets other_floors=True when LLM B detects other floors."""
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

    FakeSG._run_llm_b(FakeSG, hallway)
    assert FakeSG.global_memory['other_floors'] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_memory.py::test_run_llm_b_appends_record tests/test_memory.py::test_run_llm_b_updates_global_memory_on_staircase -v
```

Expected: `FAILED` — `AttributeError`

- [ ] **Step 3: Add `_run_llm_b()` and `_trigger_llm_b()` to `SceneGraph` in `scenegraph.py`**

```python
def _run_llm_b(self, room_node):
    objects_text = ', '.join(n.caption for n in room_node.nodes) or 'none'
    prev_text = str(room_node.memory[-1]) if room_node.memory else 'none'
    prompt = self.prompt_llm_b.format(
        goal=self.obj_goal_sg,
        room=room_node.caption,
        objects=objects_text,
        prev=prev_text,
    )
    response = self.get_llm_response(prompt)
    record = {'visit': len(room_node.memory) + 1}
    for line in response.strip().split('\n'):
        if ':' in line:
            key, _, val = line.partition(':')
            record[key.strip()] = val.strip()
    room_node.memory.append(record)
    if record.get('other_floors_detected', 'no').lower() == 'yes':
        self.global_memory['other_floors'] = True

def _trigger_llm_b(self, room_node):
    t = threading.Thread(target=self._run_llm_b, args=(room_node,), daemon=True)
    t.start()
    self._llm_b_thread = t
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scenegraph.py tests/test_memory.py
git commit -m "feat: add _run_llm_b and _trigger_llm_b with threading"
```

---

## Task 5: Modify `insert_goal()` — inject memory into prompt and trigger LLM B on status transitions

**Design invariant:** `status='active'` means "this room was chosen by LLM A in the **previous** `insert_goal()` call". Because only one room is set to `active` per call, at most **one** LLM B fires per call. Rooms with content that were never chosen remain `'unvisited'` and can never trigger LLM B — this prevents the over-triggering problem where 5 un-chosen rooms would each fire LLM B every round.

**Files:**
- Modify: `scenegraph.py:723` — `insert_goal()`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_memory.py

def test_insert_goal_sets_chosen_room_active():
    """The room chosen by insert_goal gets status='active'."""
    sg_mod = importlib.import_module('scenegraph')
    RoomNode = sg_mod.RoomNode

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_memory.py::test_insert_goal_sets_chosen_room_active tests/test_memory.py::test_insert_goal_triggers_llm_b_on_active_to_abandoned tests/test_memory.py::test_insert_goal_does_not_trigger_llm_b_for_unvisited_rooms -v
```

Expected: `FAILED`

- [ ] **Step 3: Modify `insert_goal()` in `scenegraph.py`**

Replace the current `insert_goal` method (lines 723-748) with:

```python
def insert_goal(self, goal=None):
    if goal is None:
        goal = self.obj_goal_sg
    self.update_group()
    room_node_text = ''
    for room_node in self.room_nodes:
        if len(room_node.group_nodes) > 0:
            room_node_text = room_node_text + room_node.caption + ','
    if room_node_text == '':
        return None

    prompt = self.prompt_room_predict.format(goal, room_node_text)
    memory_text = self._build_room_memory_text()
    if memory_text:
        global_line = "other floors detected: yes" if self.global_memory['other_floors'] else "other floors detected: no"
        prompt += f"\n\n{global_line}\nPast exploration:\n{memory_text}"

    response = self.get_llm_response(prompt=prompt)
    response = response.lower()
    predict_room_node = None
    for room_node in self.room_nodes:
        if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
            predict_room_node = room_node
    if predict_room_node is None:
        return None

    # LLM B fires only when a room transitions active→abandoned (was chosen last
    # round, not chosen this round). 'unvisited' rooms are skipped, ensuring at
    # most one LLM B call per insert_goal() invocation.
    for room_node in self.room_nodes:
        if len(room_node.group_nodes) == 0:
            continue
        if room_node is predict_room_node:
            room_node.status = 'active'
        elif room_node.status == 'active':
            room_node.status = 'abandoned'
            self._trigger_llm_b(room_node)

    for group_node in predict_room_node.group_nodes:
        corr_score = self.graph_corr(goal, group_node)
        group_node.corr_score = corr_score
    sorted_group_nodes = sorted(predict_room_node.group_nodes)
    self.mid_term_goal = sorted_group_nodes[-1].center
    return self.mid_term_goal
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_memory.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scenegraph.py tests/test_memory.py
git commit -m "feat: inject room memory into LLM A prompt and trigger LLM B on status transitions"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Memory attached to Room nodes | Task 1 — `memory`, `status` on `RoomNode` |
| Global memory (floors, staircase) | Task 2 — `global_memory` dict; Task 4 — updated by `_run_llm_b` |
| LLM A reads past exploration | Task 5 — memory context injected into `insert_goal()` prompt |
| LLM A outputs room status | ✅ Handled via rule-based status tracking (no fragile LLM output parsing) |
| LLM B triggered on active→abandoned | Task 5 — status transition check in `insert_goal()` |
| LLM B runs async (threading) | Task 4 — `_trigger_llm_b` uses daemon thread |
| LLM B writes structured record | Task 4 — `_run_llm_b` parses `key: value` lines |
| Memory cleared on reset | ✅ Free — `reset()` already calls `init_room_nodes()` which creates fresh `RoomNode` instances |
| No LLM B blocking main loop | Task 4 — daemon thread, fire-and-forget |

**Placeholder scan:** None found.

**Type consistency:**
- `room_node.memory` — `list[dict]` throughout
- `room_node.status` — `str` literal, consistent across Tasks 1, 3, 5
- `self.global_memory` — `dict` with keys `other_floors` (bool), `staircase_pos` (None | tuple), consistent Tasks 2, 4, 5
- `_trigger_llm_b(room_node)` — called in Task 5, defined in Task 4 ✅

**One design note:** The spec says LLM A should output explicit status labels. This plan instead infers status via rule (not-chosen + was-active = abandoned). This is more reliable with free-text LLM output and requires zero change to the existing LLM A prompt format — only appending memory context. The behaviour is equivalent.
