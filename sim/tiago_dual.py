from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from mink.contrib import TeleopMocap

_HERE = Path(__file__).parent
_XML = _HERE / "models" / "pal_tiago_dual" / "tiago_scene.xml"

# Combined joint names and velocity limits for all controlled joints
ARM_JOINT_NAMES = [
    "1_joint",
    "2_joint",
    "3_joint",
    "4_joint",
    "5_joint",
    "6_joint",
    "7_joint",
]

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

MAX_ITERS = 10  # max iterations to solve IK per step

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    dof_ids = np.array([model.joint(name).id for name in CONTROLLED_JOINT_NAMES])
    actuator_ids = np.array(
        [model.actuator(f"{name}_position").id for name in CONTROLLED_JOINT_NAMES]
    )

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
        (["gripper_left_right_finger_collision"], ["table_collision"]),
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

    key_callback = TeleopMocap(data)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        base_task.set_target_from_configuration(configuration)

        mink.move_mocap_to_frame(
            model, data, "left_gripper_target", "left_gripper", "site"
        )
        mink.move_mocap_to_frame(
            model, data, "right_gripper_target", "right_gripper", "site"
        )

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            base_task.set_target(mink.SE3.from_mocap_name(model, data, "base_target"))
            l_ee_task.set_target(
                mink.SE3.from_mocap_name(model, data, "left_gripper_target")
            )
            r_ee_task.set_target(
                mink.SE3.from_mocap_name(model, data, "right_gripper_target")
            )

            key_callback.auto_key_move()

            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                "daqp",
                limits=limits,
            )
            configuration.integrate_inplace(vel, rate.dt)

            # data.qpos[:] = np.array(configuration.q, dtype=float) # Set configuration directly
            data.ctrl[actuator_ids] = configuration.q[
                dof_ids
            ]  # Set configuration with actuator dynamics
            mujoco.mj_step(model, data)

            # for visualization of the fromto sensors
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            viewer.sync()
            rate.sleep()
