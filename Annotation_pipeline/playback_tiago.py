from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import os
import time
import h5py
import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import json


@dataclass
class PlaybackTiago:
    h5_path: str
    demo: Optional[str] = None           # e.g., "demo_1"; None = first available demo
    mode: str = "action"                 # "action" or "state"
    speed: float = 1.0                   # playback speed when 'time' exists
    control_hz: float = 200.0            # fallback tick rate when 'time' is absent
    loop: bool = False                   # whether to loop the playback
    start_time: Optional[float] = None   # playback window: absolute seconds in demo 'time'
    end_time: Optional[float] = None
    start_idx: Optional[int] = None      # playback window: index start (inclusive)
    end_idx: Optional[int] = None        # playback window: index end (exclusive)
    xml_root_dir: Optional[str] = None   # root dir to resolve <include> asset paths

    # ---------- debug options ----------
    debug: bool = True                                 # master switch for debug logging
    debug_names: List[str] = field(default_factory=lambda: [
        # pick names that matter most to you:
        "gripper_left_left_finger_joint_position",
        "gripper_left_right_finger_joint_position",
        "gripper_right_left_finger_joint_position",
        "gripper_right_right_finger_joint_position",
        "torso_lift_joint_position",
    ])
    debug_first_n: int = 5                             # print for first N frames
    debug_pause: bool = False                          # pause each debug frame
    debug_pdb: bool = False                            # call breakpoint() at first debug frame

    # Runtime
    _model: Optional[mujoco.MjModel] = None
    _data: Optional[mujoco.MjData] = None
    _state_after_action: bool = False
    _steps_per_action: int = 1
    _control_hz: float = 200.0
    _names: Optional[List[str]] = None                 # recorded action_names

    # ----------------- HDF5 loading -----------------
    def _select_demo_key(self, g: h5py.Group) -> str:
        demos = sorted([k for k in g.keys() if k.startswith("demo_")])
        if not demos:
            raise RuntimeError("No demo_* group found in HDF5.")
        if self.demo is None:
            return demos[0]
        if self.demo not in g:
            raise RuntimeError(f"Requested demo='{self.demo}' not found. Candidates: {demos}")
        return self.demo

    def _load_demo_meta(self) -> Tuple[str, str, dict]:
        with h5py.File(self.h5_path, "r") as f:
            g = f["data"]
            demo_key = self._select_demo_key(g)
            ep = g[demo_key]

            # Model XML (expanded at recording time)
            if "model_file" in ep.attrs:
                model_xml_str = ep.attrs["model_file"]
            elif "model_file" in ep:
                raw = ep["model_file"][()]
                model_xml_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            else:
                raise RuntimeError("model_file (expanded XML) is missing; cannot rebuild MuJoCo scene.")

            info = {
                "demo_key": demo_key,
                "has_actions": "actions" in ep,
                "has_states": "states" in ep,
                "has_time": "time" in ep,
                "state_after_action": bool(ep.attrs.get("state_after_action", False)),
                "steps_per_action": int(ep.attrs.get("steps_per_action", 1)),
                "actions_meta": json.loads(ep.attrs.get("actions_meta", "{}")) if "actions_meta" in ep.attrs else {},
                "control_hz": float(ep.attrs.get("control_hz", self.control_hz)),
            }
            return demo_key, model_xml_str, info

    def _load_arrays(self, demo_key: str):
        init = {}
        with h5py.File(self.h5_path, "r") as f:
            ep = f["data"][demo_key]
            actions = ep["actions"][:] if "actions" in ep else None
            states  = ep["states"][:]  if "states"  in ep else None
            times   = ep["time"][:]    if "time"    in ep else None
            if "qpos0" in ep and "qvel0" in ep:
                init["qpos0"] = ep["qpos0"][:]
                init["qvel0"] = ep["qvel0"][:]
            if "mocap_pos0" in ep:  init["mocap_pos0"]  = ep["mocap_pos0"][:]
            if "mocap_quat0" in ep: init["mocap_quat0"] = ep["mocap_quat0"][:]

        # Compute the window [i0, i1)
        def _len(x): return x.shape[0] if x is not None else 0
        L = max(_len(actions), _len(states), _len(times))
        i0, i1 = 0, L

        # Prefer time window if provided and 'times' exists
        if (times is not None) and (self.start_time is not None or self.end_time is not None):
            t0 = self.start_time if self.start_time is not None else float(times[0])
            t1 = self.end_time   if self.end_time   is not None else float(times[-1])
            idx = np.flatnonzero((times >= t0) & (times <= t1))
            if idx.size == 0:
                raise RuntimeError("No samples in the requested time window.")
            i0, i1 = int(idx[0]), int(idx[-1] + 1)
        else:
            # Otherwise use index window if provided
            if self.start_idx is not None:
                i0 = max(0, int(self.start_idx))
            if self.end_idx is not None:
                i1 = min(L, int(self.end_idx))
            i0 = min(i0, i1)

        # Slice arrays
        if actions is not None: actions = actions[i0:i1]
        if states  is not None: states  = states[i0:i1]
        if times   is not None: times   = times[i0:i1]

        return actions, states, times, init

    # ----------------- XML build (with include resolution) -----------------
    def _build_model(self, model_xml_str: str):
        tmp_dir = self.xml_root_dir or tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".xml",
                                         dir=tmp_dir, delete=False) as fp:
            fp.write(model_xml_str)
            tmp_xml_path = fp.name

        old_cwd = os.getcwd()
        try:
            if self.xml_root_dir:
                os.chdir(self.xml_root_dir)
            self._model = mujoco.MjModel.from_xml_path(tmp_xml_path)
            self._data  = mujoco.MjData(self._model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build MuJoCo model from temp XML at {tmp_xml_path}. "
                f"Hint: set xml_root_dir to the project/model root that contains 'assets'. "
                f"Original error: {e}"
            ) from e
        finally:
            if self.xml_root_dir:
                os.chdir(old_cwd)
            try:
                os.remove(tmp_xml_path)
            except Exception:
                pass

    # ----------------- Helpers: name/id mapping & pretty dump -----------------
    def _verify_names_exact(self, names: List[str]):
        """Ensure recorded action_names == model actuator order 0..nu-1 (strict equality)."""
        nu = self._model.nu
        model_names = [mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(nu)]
        if len(names) != nu:
            raise RuntimeError(f"Action dim mismatch: recorded={len(names)} vs model.nu={nu}")
        if list(names) != model_names:
            diff = "\n".join(f"{i:02d}: rec={names[i]} | mdl={model_names[i]}" for i in range(nu))
            raise RuntimeError("Recorded action_names != model actuator order. Refusing to play.\n" + diff)

    def _dump_mapping_line(self, i: int) -> str:
        """Return a one-line mapping info for actuator index i."""
        m = self._model
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        aid = i
        jid = int(m.actuator_trnid[aid][0])
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr = int(m.jnt_qposadr[jid])
        dadr = int(m.jnt_dofadr[jid])
        lo, hi = (m.actuator_ctrlrange[aid] if m.actuator_ctrlrange.size else (np.nan, np.nan))
        frange = (m.actuator_forcerange[aid] if m.actuator_forcerange.size else (np.nan, np.nan))
        kp = float(m.actuator_gainprm[aid][0])
        kv = float(m.actuator_dyntype[aid]) if hasattr(m, "actuator_dyntype") else np.nan
        gear = float(m.actuator_gear[aid][0])
        return (f"[MAP] {i:02d} act='{aname}' -> joint='{jname}' (jid={jid}, qpos_adr={qadr}, dof_adr={dadr}) "
                f"ctrlrange=[{lo:.4g},{hi:.4g}] forcerange=[{frange[0]:.4g},{frange[1]:.4g}] gear={gear:.3g} kp~{kp}")


    def _debug_header_once(self):
        if not self.debug:
            return
        print("\n===== DEBUG MAPPING (first relevant channels) =====")
        # print all for completeness (or limit to first 30 if too many)
        nu = self._model.nu
        for i in range(nu):
            print(self._dump_mapping_line(i))
        print("===== END DEBUG MAPPING =====\n")

    def _debug_probe_frame(self, frame_idx: int, action_vec: np.ndarray):
        """Print detailed info for debug_names at one frame: before write, after write, after step."""
        if not self.debug:
            return

        m, d = self._model, self._data
        nu = m.nu
        names = self._names or [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(nu)]
        name2idx = {nm: i for i, nm in enumerate(names)}
        sel = [i for i, nm in enumerate(names) if nm in set(self.debug_names)]

        print(f"\n--- DEBUG FRAME {frame_idx} ---")
        # BEFORE write
        for i in sel:
            aname = names[i]
            aid = i
            jid = int(m.actuator_trnid[aid][0]); jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            qadr = int(m.jnt_qposadr[jid])
            lo, hi = (m.actuator_ctrlrange[aid] if m.actuator_ctrlrange.size else (np.nan, np.nan))
            print(f"[BEFORE] {aname:<45} aid={aid:02d} j={jname:<35} qpos={d.qpos[qadr]: .6f} "
                  f"ctrl_cur={d.ctrl[aid]: .6f} action={action_vec[i]: .6f} range=[{lo:.4g},{hi:.4g}]")

    # ----------------- Playback primitives -----------------
    def _reconstruct_state(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nq, nv = self._model.nq, self._model.nv
        if s.shape[0] != (nq + nv):
            raise ValueError(f"State length mismatch: {s.shape[0]} != nq+nv={nq+nv}")
        qpos = s[:nq]
        qvel = s[nq:nq+nv]
        return qpos, qvel

    def _step_action(self, action: np.ndarray, substeps: int = 1, dbg_frame_idx: Optional[int] = None):
        """
        Apply one action (assumes recorded order == model actuator order), with detailed debug:
        - print selected channels BEFORE write, AFTER write, AFTER step
        - optionally pause or drop into pdb at first debug frame
        """
        a = np.asarray(action, dtype=np.float64)
        nu = self._model.nu
        n = min(nu, a.shape[0])

        # if self.debug and (dbg_frame_idx is not None) and (dbg_frame_idx < self.debug_first_n):
        #     self._debug_probe_frame(dbg_frame_idx, a)

        # No clipping: write exactly what was recorded
        n = min(self._model.nu, a.shape[0])
        if n > 0:
            self._data.ctrl[:n] = a[:n]


        # # AFTER write (check overwrites before stepping)
        # if self.debug and (dbg_frame_idx is not None) and (dbg_frame_idx < self.debug_first_n):
        #     m, d = self._model, self._data
        #     names = self._names or [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(n)]
        #     sel = [i for i, nm in enumerate(names) if nm in set(self.debug_names)]
        #     for i in sel:
        #         aname = names[i]
        #         aid = i
        #         jid = int(m.actuator_trnid[aid][0]); jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        #         qadr = int(m.jnt_qposadr[jid])
        #         print(f"[AFTER WRITE] {aname:<45} ctrl={d.ctrl[aid]: .6f} qpos={d.qpos[qadr]: .6f}")
        #     if self.debug_pdb and dbg_frame_idx == 0:
        #         breakpoint()  # drop into pdb here if needed
        #     if self.debug_pause:
        #         input("  (press Enter to step physics) ")

        # Step physics
        for _ in range(max(1, int(substeps))):
            mujoco.mj_step(self._model, self._data)

        # # AFTER step
        # if self.debug and (dbg_frame_idx is not None) and (dbg_frame_idx < self.debug_first_n):
        #     m, d = self._model, self._data
        #     names = self._names or [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(n)]
        #     sel = [i for i, nm in enumerate(names) if nm in set(self.debug_names)]
        #     for i in sel:
        #         aname = names[i]
        #         aid = i
        #         jid = int(m.actuator_trnid[aid][0]); jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        #         qadr = int(m.jnt_qposadr[jid])
        #         print(f"[AFTER STEP ] {aname:<45} ctrl={d.ctrl[aid]: .6f} qpos={d.qpos[qadr]: .6f}")

        #     if self.debug_pause:
        #         input("  (press Enter for next frame) ")

        # if dbg_frame_idx is not None and (dbg_frame_idx == 1 or dbg_frame_idx % 200 == 0):
        #     m, d = self._model, self._data
        #     # 打全量 ctrl（向量）
        #     #print(f"[CTRL] frame={dbg_frame_idx} ctrl_vector={d.ctrl[:].copy()}")

        #     # 如果想带名字逐项打印（更可读）
        #     torso_index = 3
        #     names = self._names or [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        #     torso_name = names[torso_index]
        #     torso_ctrl_value = d.ctrl[torso_index]
            
        #     # 格式化输出，保持和你原来的风格一致
        #     print(f"[CTRL] frame={dbg_frame_idx:>4}  {torso_index:02d}  {torso_name:<45}  {torso_ctrl_value: .6f}")
        # 条件：打印前 self.debug_first_n 帧 (你的设置是5)，并且之后每 200 帧打印一次
        # ==================== 新的、包含夹爪的调试代码块 ====================
        if dbg_frame_idx is not None and (dbg_frame_idx < self.debug_first_n or dbg_frame_idx % 200 == 0):
            m, d = self._model, self._data
            
            # --- 1. Get and print Torso's information ---
            try:
                torso_actuator_index = 3
                torso_ctrl = d.ctrl[torso_actuator_index]
                torso_joint_id = m.actuator_trnid[torso_actuator_index][0]
                torso_qpos_address = m.jnt_qposadr[torso_joint_id]
                torso_qpos = d.qpos[torso_qpos_address]
                print(f"[TORSO DEBUG] frame={dbg_frame_idx:>4} | ctrl = {torso_ctrl: .6f} | qpos = {torso_qpos: .6f}")
            except Exception as e:
                print(f"[TORSO DEBUG] frame={dbg_frame_idx:>4} | Error getting Torso info: {e}")

            # --- 2. Get and print information about the left gripper. ---
            try:
                # Actuator indices for the two fingers of the left gripper (based on your previous MAP output).
                lg_indices = [13, 14] 
                lg_names = ["L_Grip_R_Finger", "L_Grip_L_Finger"] # 简写名称

                # Get Ctrl value
                lg_ctrl = d.ctrl[lg_indices]
                
                # Get the qpos value (using the correct lookup method).
                lg_qpos = []
                for idx in lg_indices:
                    lg_jnt_id = m.actuator_trnid[idx][0]
                    lg_qpos_adr = m.jnt_qposadr[lg_jnt_id]
                    lg_qpos.append(d.qpos[lg_qpos_adr])
                
                print(f"[L_GRIP DEBUG] frame={dbg_frame_idx:>4} | ctrl = [{lg_ctrl[0]:.4f}, {lg_ctrl[1]:.4f}] | qpos = [{lg_qpos[0]:.4f}, {lg_qpos[1]:.4f}]")
            except Exception as e:
                print(f"[L_GRIP DEBUG] frame={dbg_frame_idx:>4} | Error getting Left Gripper info: {e}")

            # --- 3. Get and print information about the right gripper. ---
            try:
                # The actuator indices of the two fingers of the right gripper (based on your previous MAP output).
                rg_indices = [22, 23] 
                rg_names = ["R_Grip_R_Finger", "R_Grip_L_Finger"] # Abbreviated name

                # Get Ctrl value
                rg_ctrl = d.ctrl[rg_indices]
                
                # Get the qpos value (using the correct lookup method).
                rg_qpos = []
                for idx in rg_indices:
                    rg_jnt_id = m.actuator_trnid[idx][0]
                    rg_qpos_adr = m.jnt_qposadr[rg_jnt_id]
                    rg_qpos.append(d.qpos[rg_qpos_adr])

                print(f"[R_GRIP DEBUG] frame={dbg_frame_idx:>4} | ctrl = [{rg_ctrl[0]:.4f}, {rg_ctrl[1]:.4f}] | qpos = [{rg_qpos[0]:.4f}, {rg_qpos[1]:.4f}]")
            except Exception as e:
                print(f"[R_GRIP DEBUG] frame={dbg_frame_idx:>4} | Error getting Right Gripper info: {e}")
        # =====================================================================#

    def _step_state(self, state_vec: np.ndarray):
        qpos, qvel = self._reconstruct_state(state_vec)
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)

    def _play_once(self, actions: Optional[np.ndarray], states: Optional[np.ndarray],
                   times: Optional[np.ndarray], used_q0: bool) -> None:
        # Determine which index we STEP first
        start_i = 0 if used_q0 else (1 if (self.mode == "action" and self._state_after_action) else 0)

        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            if times is not None and len(times) > 1:
                # Use 'times' only for pacing; physics stepping per frame = steps_per_action
                t0 = times[0]
                wall0 = time.perf_counter()
                T = len(times)
                for i in range(start_i, T):
                    target = wall0 + (times[i] - t0) / max(1e-6, self.speed)
                    while True:
                        now = time.perf_counter()
                        if now >= target:
                            break
                        time.sleep(min(0.001, target - now))

                    if self.mode == "action":
                        if actions is None:
                            raise RuntimeError("mode='action' but 'actions' dataset is missing.")
                        self._step_action(actions[i], substeps=self._steps_per_action, dbg_frame_idx=i)
                    else:
                        if states is None:
                            raise RuntimeError("mode='state' but 'states' dataset is missing.")
                        self._step_state(states[i])

                    viewer.sync()
            else:
                # No timestamps: use control_hz for pacing; physics stepping remains steps_per_action
                hz = getattr(self, "_control_hz", self.control_hz)
                dt_sleep = 1.0 / max(1e-6, hz * self.speed)
                length = len(actions) if (self.mode == "action" and actions is not None) else len(states or [])
                for i in range(start_i, length):
                    if self.mode == "action":
                        self._step_action(actions[i], substeps=self._steps_per_action, dbg_frame_idx=i)
                    else:
                        self._step_state(states[i])
                    viewer.sync()
                    time.sleep(dt_sleep)

    # ----------------- Main entry -----------------
    def run(self):
        # Load meta/arrays and rebuild model
        demo_key, model_xml_str, info = self._load_demo_meta()
        actions, states, times, init = self._load_arrays(demo_key)
        self._build_model(model_xml_str)

        self._state_after_action = bool(info.get("state_after_action", False))
        self._steps_per_action   = int(info.get("steps_per_action", 1))
        self._control_hz         = float(info.get("control_hz", self.control_hz))
        actions_meta = info.get("actions_meta", {})

        # --- Strong verification: recorded action_names == model actuator order ---
        names = list(actions_meta.get("names", []))
        if not names:
            raise RuntimeError("HDF5 missing actions_meta.names; cannot verify channel order.")
        self._verify_names_exact(names)
        self._names = names

        # (2) Restore initial state: prefer explicit q0; otherwise use states[0]
        used_q0 = False
        if "qpos0" in init and "qvel0" in init:
            self._data.qpos[:] = init["qpos0"]
            self._data.qvel[:] = init["qvel0"]
            used_q0 = True
        elif states is not None and len(states) > 0:
            nq, nv = self._model.nq, self._model.nv
            self._data.qpos[:] = states[0][:nq]
            self._data.qvel[:] = states[0][nq:nq+nv]
        mujoco.mj_forward(self._model, self._data)

        # Debug mapping once
        if self.debug:
            self._debug_header_once()

        # (3) Preload ctrl to match the state we will start stepping from.
        if self.mode == "action" and actions is not None and len(actions) > 0:
            start_i = 0 if used_q0 else (1 if (self._state_after_action and len(actions) > 1) else 0)
            preload_idx = start_i - 1 if start_i > 0 else 0
            a0 = actions[preload_idx].astype(np.float64, copy=True)

            nu = self._model.nu
            if a0.shape[0] != nu:
                raise RuntimeError(f"Action length ({a0.shape[0]}) != model.nu ({nu}); expected full vector.")

            # No clipping
            # if self._model.actuator_ctrlrange.size:
            #     lo = self._model.actuator_ctrlrange[:, 0]
            #     hi = self._model.actuator_ctrlrange[:, 1]
            #     a0 = np.clip(a0, lo, hi)

            self._data.ctrl[:] = a0
            mujoco.mj_forward(self._model, self._data)

            if self.debug:
                print(f"[PRELOAD] using action index {preload_idx}")
                # quick preview for debug channels
                m, d = self._model, self._data
                for nm in self.debug_names:
                    try:
                        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
                        jid = int(m.actuator_trnid[aid][0])
                        qadr = int(m.jnt_qposadr[jid])
                        print(f"[PRELOAD] {nm:<45} ctrl={d.ctrl[aid]: .6f} qpos={d.qpos[qadr]: .6f}")
                    except Exception as e:
                        print(f"[PRELOAD] {nm}: not found ({e})")

        # (4) Playback (with optional loop)
        while True:
            self._play_once(actions, states, times, used_q0)
            if not self.loop:
                break


def _resolve_default_h5() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5 = os.path.join(base_dir, "trajectory_storage", "2025-11-14_16-49-34.hdf5")
    return h5


if __name__ == "__main__":
    fixed_h5 = _resolve_default_h5()
    if not os.path.isfile(fixed_h5):
        raise FileNotFoundError(
            f"Playback file not found: {fixed_h5}\n"
            f"Please place an HDF5 under ./trajectory_storage/ or update the path."
        )

    player = PlaybackTiago(
        h5_path=fixed_h5,
        mode="action",
        loop=False,
        xml_root_dir=r"d:\\Guided_research\\Voice_Activated_Task_Annotation_Pipeline\\models\\pal_tiago_dual",

        # ---- debug switches ----
        debug=True,
        debug_pause=False,   # Changing to True allows for manual execution frame by frame.
        debug_pdb=False,     # Setting it to True will cause the first frame to enter the PDB at the point where "Ctrl was written but no stepping" is executed.
        debug_first_n=5,     # Detailed printing of the first 5 frames
        debug_names=[
            "gripper_left_left_finger_joint_position",
            "gripper_left_right_finger_joint_position",
            "gripper_right_left_finger_joint_position",
            "gripper_right_right_finger_joint_position",
            "torso_lift_joint_position",
        ],
    )
    player.run()
