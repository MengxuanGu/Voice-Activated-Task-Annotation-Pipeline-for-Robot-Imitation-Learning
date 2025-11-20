# robosuite_style_recorder.py
# Standalone HDF5 trajectory recorder with a schema similar to
# robosuite's DataCollectionWrapper aggregation. It stores raw actions
# (MuJoCo ctrl targets) and raw states (qpos + qvel) without any normalization.

import os
import h5py
import json
import time
import datetime
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class HDF5Recorder:
    out_dir: str = "./trajectory_storage"
    env_name: str = "CustomMuJoCo"
    env_info: Optional[dict] = None

    # Recording semantics configured by the caller BEFORE start_episode()
    state_after_action: bool = False          # True if state is logged AFTER applying action (ctrl then mj_step)
    action_names: Optional[List[str]] = None   # Full actuator name list (order defines action channel semantics)
    control_hz: Optional[float] = None         # Control/recording frequency (e.g., 200.0)

    # Internal runtime state
    _episode_idx: int = 0
    _file: Optional[h5py.File] = None
    _grp_data: Optional[h5py.Group] = None
    _states_buf: list[np.ndarray] = field(default_factory=list)
    _actions_buf: list[np.ndarray] = field(default_factory=list)
    _times_buf: list[float] = field(default_factory=list)   # relative timestamps within an episode
    _model_xml_str: Optional[str] = None
    _episode_t0: float = 0.0
    _nq: Optional[int] = None
    _nv: Optional[int] = None

    def _ensure_file(self):
        """Create the HDF5 file and top-level /data group once per process."""
        os.makedirs(self.out_dir, exist_ok=True)
        if self._file is None:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(self.out_dir, f"{ts}.hdf5")
            self._file = h5py.File(path, "w")
            self._grp_data = self._file.create_group("data")

            # Top-level metadata (human-readable)
            now = datetime.datetime.now()
            self._grp_data.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
            self._grp_data.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
            self._grp_data.attrs["env"] = self.env_name
            if self.env_info is not None:
                self._grp_data.attrs["env_info"] = json.dumps(self.env_info)

            # Generic env args field (kept for compatibility with robosuite-style readers)
            env_args = {
                "type": "custom",
                "env_name": self.env_name,
                "env_kwargs": {},
                "observation_modalities": ["state"],
                "action_modalities": ["actions"],
                "env_info": self.env_info or {},
            }
            self._grp_data.attrs["env_args"] = json.dumps(env_args)

    # ---------- Episode lifecycle ----------
    def start_episode(self, model_xml_str: Optional[str] = None, nq=None, nv=None):
        """Begin a new episode; clear buffers and set model/state metadata."""
        self._ensure_file()
        self._episode_idx += 1
        self._states_buf.clear()
        self._actions_buf.clear()
        self._times_buf.clear()

        self._model_xml_str = model_xml_str
        self._nq = nq
        self._nv = nv
        self._episode_t0 = time.perf_counter()

    def record_step(self, state_flat: np.ndarray, action_vec: np.ndarray):
        """
        Append one step of data.
        - state_flat: raw [qpos, qvel] flattened (float64)
        - action_vec: raw MuJoCo ctrl targets (float64), ordered by actuator
        No normalization, no reordering, no scaling.
        """
        self._states_buf.append(np.asarray(state_flat, dtype=np.float64))
        self._actions_buf.append(np.asarray(action_vec, dtype=np.float64))
        self._times_buf.append(time.perf_counter() - self._episode_t0)

    def end_episode(self, success: bool = True):
        """Finalize the current episode and write datasets/groups to HDF5."""
        if len(self._states_buf) == 0 or len(self._actions_buf) == 0:
            return

        # Align lengths across buffers
        n = min(len(self._states_buf), len(self._actions_buf), len(self._times_buf))
        states = np.stack(self._states_buf[:n], axis=0)     # (T, nq+nv)
        actions = np.stack(self._actions_buf[:n], axis=0)   # (T, nu) -- actuator-ordered ctrl
        times = np.asarray(self._times_buf[:n], dtype=np.float64)

        if success:
            ep = self._grp_data.create_group(f"demo_{self._episode_idx}")

            # Episode-level attributes
            if self._model_xml_str is not None:
                ep.attrs["model_file"] = self._model_xml_str
            if self._nq is not None:
                ep.attrs["nq"] = int(self._nq)
            if self._nv is not None:
                ep.attrs["nv"] = int(self._nv)

            # Datasets
            ds_states = ep.create_dataset("states", data=states)   # (T, nq+nv)
            ds_actions = ep.create_dataset("actions", data=actions)  # (T, nu)
            ep.create_dataset("time", data=times)                  # (T,)

            # Observation dict compatible with robomimic readers (hard link, no extra storage)
            obs = ep.create_group("obs")
            obs["state"] = ds_states

            # Normalized episode metadata
            ep.attrs["num_samples"] = int(ds_states.shape[0])
            ep.attrs["success"] = True
            ep.attrs["state_after_action"] = bool(self.state_after_action)
            ep.attrs["state_layout"] = json.dumps(["qpos", "qvel"])

            # Action semantics (critical for correct playback)
            # - type: we store MuJoCo data.ctrl (position targets in your setup)
            # - order: 'actuator' (strict actuator order in the compiled MJCF)
            # - names: actuator names provided by the caller (must match order)
            actions_meta = {
                "type": "mujoco_ctrl",
                "order": "actuator",
                "dim": int(ds_actions.shape[1]),
                "names": list(self.action_names or []),
            }
            ep.attrs["actions_meta"] = json.dumps(actions_meta)

            # Number of physics steps per action in your loop (you do 1 mj_step per control)
            ep.attrs["steps_per_action"] = 1

            # Control frequency (optional but useful for synchronized playback)
            if self.control_hz is not None:
                ep.attrs["control_hz"] = float(self.control_hz)

        # Clear buffers for the next episode
        self._states_buf.clear()
        self._actions_buf.clear()
        self._times_buf.clear()
        self._model_xml_str = None
        self._nq = None
        self._nv = None

    def close(self):
        """Flush and close the HDF5 file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._grp_data = None


# ---------- State flattening: store raw qpos + qvel ----------
def flatten_mujoco_state(data) -> np.ndarray:
    """Return a float64 vector [qpos, qvel] without any normalization or reordering."""
    return np.concatenate(
        [np.asarray(data.qpos, dtype=np.float64),
         np.asarray(data.qvel, dtype=np.float64)],
        axis=0
    )
