# dt-surgery

Real-to-sim for in-vivo rabbit appendectomy: turn the endoscope videos of
[surgicalrobotics/appendectomy-invivo](https://huggingface.co/datasets/surgicalrobotics/appendectomy-invivo)
into a Gaussian-splatting deformable-tissue simulation for robot policy learning.

## Stage 0: data checks

```bash
python3 -m venv .venv && .venv/bin/pip install h5py numpy opencv-python-headless huggingface_hub matplotlib
# metadata (labels/session/camera timestamps) for all 868 episodes
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('surgicalrobotics/appendectomy-invivo', repo_type='dataset', local_dir='data/appendectomy-invivo', allow_patterns=['*.json','*.md','*/camera.h5'])"
.venv/bin/python -I scripts/meta_summary.py data/appendectomy-invivo data/meta_index.json
scripts/fetch_heads.sh data/appendectomy-invivo $PWD/data/heads 4000000        # first 4 MB of each video
.venv/bin/python -I scripts/extract_first.py data/heads data/first_frames 0
.venv/bin/python -I scripts/session_drift.py data/meta_index.json data/first_frames data/sessions.json
scripts/fetch_full_sample.sh data/appendectomy-invivo $PWD/data/full            # first+last episode per batch
.venv/bin/python -I scripts/camera_motion_flow.py data/full data/camera_motion_flow.json 3
.venv/bin/python -I scripts/camera_motion_summary.py data/camera_motion_flow.json data/sessions.json data/meta_index.json data/camera_motion_summary.json
```

Findings (see `data/episode_index.csv`):
- `batch_XXXX` folders are upload chunks, not camera positions.
- 4 sessions separated by >25 min gaps: 14:06–14:45 (92), 15:54–17:13 (180), 18:09–19:38 (376), 20:10–21:04 (220).
- Monocular camera, roughly fixed per session; within-episode camera shift (p95, px @1920, lower bound)
  median 23 / 64 / 24 / 14 px for sessions 0–3. Session 3 is the most stable.
