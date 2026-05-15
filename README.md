# From 2D Priors to 3D Cognition with Memory: Reproducing and Enhancing Zero-Shot Object Navigation

**AIAA 3201 Introduction to Computer Vision, Spring 2026**  
Yenchi Tseng · Yongqi Zhang

---

We follow a three-stage curriculum to study zero-shot object-goal navigation (ZSON):

1. **Reproduce** two 2D baselines — CLIP-based semantic mapping (ZSON) and vision-language frontier evaluation (VLFM).
2. **Reproduce** the LLM-driven 3D scene graph planner SG-Nav on Matterport3D.
3. **Propose** an *Exploration Memory* enhancement for SG-Nav that improves SR by **+3.33 pp** without adding any model parameters.

The Exploration Memory augments each room node in the scene graph with a persistent status field and a list of structured visit records. A background *Memory Writer* (LLM-B) asynchronously records exploration outcomes after the agent leaves a room; a *Room Selector* (LLM-A) reads these records to avoid redundant re-exploration.


## Results

| Method | Dataset | SR (%) | SPL (%) |
|---|---|---|---|
| ZSON (paper) | MP3D | 4.80 | 15.30 |
| ZSON (ours) | MP3D | 4.30 | 14.00 |
| VLFM (paper) | HM3D | 52.50 | 30.40 |
| VLFM (ours) | HM3D | 52.30 | 30.24 |
| SG-Nav (paper) | MP3D | 40.00 | 16.00 |
| SG-Nav (ours, val_mini) | MP3D | 36.67 | 11.33 |
| **SG-Nav + Memory (ours)** | MP3D | **40.00** | **12.47** |

Evaluation on MP3D val_mini (30 episodes, 1 scene). All LLM calls use `llama3.2-vision` via Ollama.


## Method

### Exploration Memory Architecture

![pipeline](./assets/pipeline.png)

LLM-A plans with room-level memories, while LLM-B asynchronously summarizes explored rooms.

**Room node extensions**

Each `RoomNode` gains:
- `memory` — a list of structured dicts, one per visit (written by LLM-B)
- `status` — one of `unvisited | active | abandoned`

A `SceneGraph`-level dict stores episode-wide cues: `other_floors`, `staircase_pos`.

**Status state machine and LLM-B trigger**

```
unvisited ──→ active ──→ abandoned
                 ↑_____________|
           (re-chosen next step)
```

After `insert_goal()` identifies the chosen room:
1. The chosen room becomes `active`.
2. Any room that was `active` last round becomes `abandoned` and fires `_trigger_llm_b()`.

Only `active → abandoned` transitions trigger LLM-B — **at most one call per planning step**.

**Asynchronous Memory Writer (LLM-B)**

LLM-B runs in a Python daemon thread so the main navigation loop is never blocked. Its prompt includes the goal category, detected objects in the room, and the previous memory record. It outputs exactly five structured lines:

```
coverage: <full|partial|minimal>
priority: <high|medium|low>
confidence: <high|medium|low>
note: <one sentence>
other_floors_detected: <yes|no>
```

The parsed record is appended to `room_node.memory`. If `other_floors_detected: yes`, a global `other_floors` flag is set for the rest of the episode.

**Memory-augmented Room Selector (LLM-A)**

Before querying LLM-A, the most recent record of every visited room is serialised and injected into the prompt:

```
bedroom: visited 2x, coverage=partial, priority=medium, note=left corner unexplored
```

This gives LLM-A persistent context to avoid cycling back to low-priority rooms.


## Installation

**Step 1 — Dataset**

Download the [Matterport3D scene dataset](https://niessner.github.io/Matterport/) and [object-goal navigation episodes](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) from [here](https://cloud.tsinghua.edu.cn/f/03e0ca1430a344efa72b/?dl=1).

Set `SCENES_DIR` and `DATA_PATH` in `configs/challenge_objectnav2021.local.rgbd.yaml`.

```
MatterPort3D/
├── mp3d/
│   ├── 2azQ1b91cZZ/
│   │   └── 2azQ1b91cZZ.glb
│   └── ...
└── objectnav/
    └── mp3d/
        └── v1/
            └── val/
                ├── content/
                │   ├── 2azQ1b91cZZ.json.gz
                │   └── ...
                └── val.json.gz
```

**Step 2 — Environment**

```bash
conda create -n SG_Nav python==3.9
```

**Step 3 — Simulator**

```bash
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
pip install -e habitat-lab
HABITAT_SIM_PATH=$(pip show habitat_sim | grep 'Location:' | awk '{print $2}')
cp tools/agent.py ${HABITAT_SIM_PATH}/habitat_sim/agent/
```

**Step 4 — Packages**

```bash
conda install -c pytorch faiss-gpu=1.8.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Install Grounded SAM:
```bash
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
wget -O data/models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O data/models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Install GLIP:
```bash
cd GLIP
python setup.py build develop --user
mkdir MODEL && cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
cd ../../
```

Install Ollama and pull the LLM used by both LLM-A and LLM-B:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision
```

**Step 5 — Run**

```bash
python SG_Nav.py --visualize
```
