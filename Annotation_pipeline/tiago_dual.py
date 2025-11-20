from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib import TeleopMocap

from robot_recorder_tiago import HDF5Recorder, flatten_mujoco_state

import math  # geometry helpers
import tempfile

_HERE = Path(__file__).parent
_XML = _HERE / "models" / "pal_tiago_dual" / "tiago_scene.xml"

# ----------------------------- Velocity limits by logical joint -----------------------------
ARM_JOINT_NAMES = [
    "1_joint",
    "2_joint",
    "3_joint",
    "4_joint",
    "5_joint",
    "6_joint",
    "7_joint",
]

# Velocity amplitude limits (units depend on joint type)
CONTROLLED_JOINTS_AND_LIMITS = [
    *[
        (f"arm_left_{n}", l)
        for n, l in zip(ARM_JOINT_NAMES, [1.95, 1.95, 2.35, 2.35, 1.95, 1.76, 1.76])
    ],
    *[
        (f"arm_right_{n}", l)
        for n, l in zip(ARM_JOINT_NAMES, [1.95, 1.95, 2.35, 2.35, 1.95, 1.76, 1.76])
    ],
    ("torso_lift_joint", 0.07),
    ("base_x", 0.5),
    ("base_y", 0.5),
    ("base_th", 0.5),
]
CONTROLLED_JOINT_NAMES = [name for name, _ in CONTROLLED_JOINTS_AND_LIMITS]
VEL_LIMITS = dict(CONTROLLED_JOINTS_AND_LIMITS)

# --------------------------- Single source of truth: 24 recorded actuators ------------------
# Order here defines the 24-dim action order for BOTH recording and playback.
CONTROLLED_ACTUATOR_NAMES = [
    # Omni base (3)
    "base_x_position",
    "base_y_position",
    "base_th_position",

    # Torso (1)
    "torso_lift_joint_position",

    # Head (2)
    "head_1_joint_position",
    "head_2_joint_position",

    # Left arm (7)
    "arm_left_1_joint_position",
    "arm_left_2_joint_position",
    "arm_left_3_joint_position",
    "arm_left_4_joint_position",
    "arm_left_5_joint_position",
    "arm_left_6_joint_position",
    "arm_left_7_joint_position",

    # Left gripper (2)
    "gripper_left_right_finger_joint_position",
    "gripper_left_left_finger_joint_position",

    # Right arm (7)
    "arm_right_1_joint_position",
    "arm_right_2_joint_position",
    "arm_right_3_joint_position",
    "arm_right_4_joint_position",
    "arm_right_5_joint_position",
    "arm_right_6_joint_position",
    "arm_right_7_joint_position",

    # Right gripper (2)
    "gripper_right_right_finger_joint_position",
    "gripper_right_left_finger_joint_position",
]

# --------------------------------- Geometry/randomization helpers ---------------------------

def _geom_top_z(model, data, geom_name: str) -> float:
    """Return the top-surface z of a box-like/plane geometry."""
    gid = model.geom(geom_name).id
    center_z = float(data.geom_xpos[gid][2])
    half_h = float(model.geom_size[gid][2])  # for box/plane: 3rd size is half height
    return center_z + half_h

def _half_height_of_geom(model, geom_name: str) -> float:
    """Return half height for sphere/box; fallback otherwise."""
    gid = model.geom(geom_name).id
    gtype = int(model.geom_type[gid])
    sz = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(sz[0])  # radius
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(sz[2])  # half height
    else:
        return float(max(sz))  # fallback

def _sample_xy_in_site(model, data, site_name: str, margin: float = 0.01):
    """Uniformly sample (x, y) inside the projection of a box-type site."""
    sid = model.site(site_name).id
    pos = data.site_xpos[sid].copy()
    size = model.site_size[sid].copy()  # box: half-length/half-width/half-height
    x = np.random.uniform(pos[0] - size[0] + margin, pos[0] + size[0] - margin)
    y = np.random.uniform(pos[1] - size[1] + margin, pos[1] + size[1] - margin)
    return float(x), float(y)

def _set_body_freejoint_pose(model, data, body_name: str, pos_xyz, quat_wxyz=(1,0,0,0)):
    """Write a freejoint body's pose into qpos: (xyz + quat wxyz)."""
    bid = model.body(body_name).id
    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]
    assert jnum > 0, f"Body '{body_name}' has no joint; expected a freejoint."
    jid = jadr
    assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE, f"Joint on '{body_name}' is not freejoint."
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr:qadr+7] = np.array([pos_xyz[0], pos_xyz[1], pos_xyz[2], *quat_wxyz], dtype=float)

def _xy_half_extents(model, geom_name: str):
    """Return (hx, hy) half extents on XY for box/sphere; best-effort fallback otherwise."""
    gid = model.geom(geom_name).id
    gtype = int(model.geom_type[gid])
    sz = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(sz[0]), float(sz[1])
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(sz[0])
        return r, r
    else:
        return float(sz[0]), float(sz[1])

def randomize_pick_and_place_objects(
    model, data,
    region_site: str = "spawn_region",
    tabletop_geom: str = "tabletop",
    box_body: str = "pickup_box",
    box_geom: str = "box_bbox",
    ball_body: str = "pickup_ball",
    ball_geom: str = "ball_geom",
    min_xy_gap: float = 0.01,
    settle_steps: int = 10,
):
    """
    Randomly place a box and a ball within `region_site` (box-type), ensuring non-overlapping XY AABBs.
    Z is placed at tabletop top + half-height + eps to avoid penetration.
    """
    eps = 1e-3
    table_top = _geom_top_z(model, data, tabletop_geom)

    hx_box, hy_box = _xy_half_extents(model, box_geom)
    hx_ball, hy_ball = _xy_half_extents(model, ball_geom)

    for _ in range(500):
        x1, y1 = _sample_xy_in_site(model, data, region_site, margin=0.015)
        x2, y2 = _sample_xy_in_site(model, data, region_site, margin=0.015)
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        overlap_x = dx < (hx_box + hx_ball + min_xy_gap)
        overlap_y = dy < (hy_box + hy_ball + min_xy_gap)
        if not (overlap_x and overlap_y):
            break
    else:
        x2, y2 = x1 + (hx_box + hx_ball + min_xy_gap), y1 + (hy_box + hy_ball + min_xy_gap)

    h_box = _half_height_of_geom(model, box_geom)
    z_box = table_top + h_box + eps
    _set_body_freejoint_pose(model, data, box_body, (x1, y1, z_box), (1,0,0,0))

    h_ball = _half_height_of_geom(model, ball_geom)
    z_ball = table_top + h_ball + eps
    _set_body_freejoint_pose(model, data, ball_body, (x2, y2, z_ball), (1,0,0,0))

    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

# -------------------------------------- Teleop session --------------------------------------

class TiagoTeleopSession:
    def __init__(self, recorder: HDF5Recorder, stop_event=None):
        self.recorder = recorder
        self.stop_event = stop_event

        # Recording semantics / metadata (keep consistent with playback)
        self.recorder.state_after_action = False  # we log state AFTER applying the action
        self.recorder.control_hz = 200.0         # must match the main loop RateLimiter

    def run(self):
        model = mujoco.MjModel.from_xml_path(str(_XML))
        data = mujoco.MjData(model)

        #print("\n" + "="*60)
        #print("--- 正在检查关键 Actuator 的运行时 KP 增益 ---")
        
        # # You can add any executor names you are interested in to this list.
        # actuators_to_check = [
        #     "torso_lift_joint_position",
        #     "gripper_left_right_finger_joint_position",
        #     "gripper_right_right_finger_joint_position",
        # ]

        # for name in actuators_to_check:
        #     try:
        #         # 1. 通过名称获取执行器的ID
        #         actuator_id = model.actuator(name).id
                
        #         # 2. 从模型中访问该ID对应的 gainprm 数组
        #         # kp 值是 gainprm 的第一个元素
        #         kp_value = model.actuator_gainprm[actuator_id][0]
                
        #         print(f" 执行器 '{name}' (ID: {actuator_id}): KP = {kp_value}")

        #     except KeyError:
        #         print(f" 警告: 在模型中找不到名为 '{name}' 的执行器。")
        
        # print("="*60 + "\n")

        nq, nv, nu = model.nq, model.nv, model.nu

        # Derive ids from the single source of truth (the 24 actuator names above)
        def _act_id_or_fail(name: str) -> int:
            try:
                return model.actuator(name).id
            except Exception:
                raise RuntimeError(
                    f"[Missing actuator] '{name}' does not exist in the model.\n"
                    f"Fix CONTROLLED_ACTUATOR_NAMES to match your XML."
                )

        controlled_actuator_ids = np.array([_act_id_or_fail(n) for n in CONTROLLED_ACTUATOR_NAMES], dtype=int)
        controlled_joint_ids = np.array([model.actuator_trnid[a][0] for a in controlled_actuator_ids], dtype=int)

        # Record exactly these 24 channels, in this exact order
        self.recorder.action_names = CONTROLLED_ACTUATOR_NAMES

        # IK tasks and limits
        configuration = mink.Configuration(model)
        tasks = [
            base_task := mink.FrameTask(
                frame_name="base_link",
                frame_type="body",
                position_cost=1.0,
                orientation_cost=1.0,
            ),
            l_ee_task := mink.RelativeFrameTask(
                frame_name="left_gripper",
                frame_type="site",
                root_name="base_link",
                root_type="body",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            r_ee_task := mink.RelativeFrameTask(
                frame_name="right_gripper",
                frame_type="site",
                root_name="base_link",
                root_type="body",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            posture_task := mink.PostureTask(model, cost=1e-1),
        ]

        collision_pairs = [
            (["finger_left_1", "finger_left_2"], ["tabletop"]),
            # Add right fingers here if your geom names exist:
            (["finger_right_1", "finger_right_2"], ["tabletop"]),
        ]

        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.15,
        )

        limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocities=VEL_LIMITS),
            collision_avoidance_limit,
        ]

        # Mocap indices for teleoperation
        base_mid = model.body("base_target").mocapid[0]
        l_mid = model.body("left_gripper_target").mocapid[0]
        r_mid = model.body("right_gripper_target").mocapid[0]

        # Gripper actuators (these 4 are also included in CONTROLLED_ACTUATOR_NAMES)
        gripper_actuators = {
            "left": np.array([
                model.actuator("gripper_left_right_finger_joint_position").id,
                model.actuator("gripper_left_left_finger_joint_position").id,
            ], dtype=int),
            "right": np.array([
                model.actuator("gripper_right_right_finger_joint_position").id,
                model.actuator("gripper_right_left_finger_joint_position").id,
            ], dtype=int),
        }
        all_gripper_ids = np.concatenate([gripper_actuators["left"], gripper_actuators["right"]])

        # Keyboard teleoperation (mocap + grippers)
        key_callback = TeleopMocap(
            data,
            mocap_indices={"body": base_mid, "arm_left": l_mid, "arm_right": r_mid},
            gripper_actuators=gripper_actuators,
            gripper_open_width=0.035,  # typical range ~ [0, 0.045]
            gripper_close_width=0.0,
        )

        # Save the fully-expanded XML string for deterministic playback
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
        mujoco.mj_saveLastXML(tmp.name, model)
        with open(tmp.name, "r", encoding="utf-8") as f:
            model_xml_str = f.read()

        # Start the episode recording
        rate = RateLimiter(frequency=200.0, warn=False)
        self.recorder.start_episode(model_xml_str=model_xml_str, nq=nq, nv=nv)

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize to neutral keyframe and set initial task targets
            mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
            # print("\n--- 检查点 A (刚重置完Keyframe) ---")
            # print("Torso (qpos[3]) =", data.qpos[3])
            
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)
            base_task.set_target_from_configuration(configuration)

            # Randomize initial object poses
            randomize_pick_and_place_objects(model, data)
            randomize_pick_and_place_objects(model, data)
            # print("\n--- 检查点 B (刚随机化完物体) ---")
            # print("Torso (qpos[3]) =", data.qpos[3])

            # Align mocap targets to current end-effector sites (avoid jumps)
            mink.move_mocap_to_frame(model, data, "left_gripper_target", "left_gripper", "site")
            mink.move_mocap_to_frame(model, data, "right_gripper_target", "right_gripper", "site")

            # Sanity checks and helpful prints
            # print(f"[INFO] model.nu = {nu}")
            # print(f"[INFO] action dim = {len(CONTROLLED_ACTUATOR_NAMES)} (expected 24)")
            assert len(CONTROLLED_ACTUATOR_NAMES) == 24
            assert len(controlled_actuator_ids) == 24
            assert len(controlled_joint_ids) == 24
            # Print first few (name -> actuator id -> joint id) for debugging
            # for i, name in enumerate(CONTROLLED_ACTUATOR_NAMES[:25]):
            #     print(f"[MAP] {i:02d} {name} -> act_id {controlled_actuator_ids[i]} -> joint_id {controlled_joint_ids[i]}")

            # ------------------------------ Main teleop / recording loop ------------------------------
            while viewer.is_running() and not (self.stop_event and self.stop_event.is_set()):
                # Constrain base mocap to planar motion and update task targets from mocap
                base_pose = data.mocap_pos[base_mid].copy()
                base_pose[2] = 0.0
                data.mocap_pos[base_mid] = base_pose
                base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
                l_ee_task.set_target(mink.SE3.from_mocap_id(data, l_mid))
                r_ee_task.set_target(mink.SE3.from_mocap_id(data, r_mid))

                # Read keyboard input (moves mocaps + writes gripper commands into data.ctrl)
                key_callback.auto_key_move()

                # Solve IK in configuration space with limits and integrate
                vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", limits=limits)
                configuration.integrate_inplace(vel, rate.dt)

                # Build the control vector we apply (and record)
                # (a) Write joint-position targets (from configuration.q) into our 24 controlled actuators
                target_ctrl = np.zeros(nu, dtype=float)
                q_target_24 = configuration.q[controlled_joint_ids]
                target_ctrl[controlled_actuator_ids] = q_target_24

                # (b) Overwrite gripper channels with TeleopMocap commands to ensure grasp is recorded
                target_ctrl[all_gripper_ids] = data.ctrl[all_gripper_ids].copy()

                # (c) Apply to simulation (PD will act on these target positions)
                data.ctrl[:] = target_ctrl

                # Step physics; since state_after_action=False we record AFTER stepping
                mujoco.mj_step(model, data)

                # Record one step:
                #   - action_vec: EXACT 24-dim control we applied on controlled_actuator_ids
                #   - state_flat: flattened MuJoCo state after action
                action_vec = target_ctrl[controlled_actuator_ids].copy()
                
                # print("\n--- 检查点 C (主循环第一帧，录制前) ---")
                # print("qpos[3] =", data.qpos)

                state_flat = flatten_mujoco_state(data)
                self.recorder.record_step(state_flat=state_flat, action_vec=action_vec)

                viewer.sync()
                rate.sleep()

        # Finish the episode
        self.recorder.end_episode(success=True)
        self.recorder.close()
