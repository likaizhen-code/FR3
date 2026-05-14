import time
import mujoco
import mujoco.viewer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


ASSETS_PATH = os.path.join(os.path.dirname(__file__),"..", "assets")
END_EFF_FRAME_ID = 19


class FR3Sim:
    def __init__(self, interface_type="torque", render=True, dt=0.001, xml_path=None):
        assert interface_type in ["torque"], "The interface should be torque"
        self.interface_type = interface_type
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(
                os.path.join(ASSETS_PATH, "fr3_on_table.xml")
            )
        self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        _render_dt = 1 / 60
        self.render_ds_ratio = max(1, _render_dt // dt)
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        else:
            self.render = False
        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = dt
        self.step_counter = 0
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.q0 = np.array(
            [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
        )
        self.reset()
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv
        self.jacp = np.zeros((3, self.nv))
        self.jacr = np.zeros((3, self.nv))
        self.M = np.zeros((self.nv, self.nv))
        self.latest_command_stamp = time.time()
        self.actuator_tau = np.zeros(7)
        self.tau_ff = np.zeros(7)
        self.dq_des = np.zeros(7)

        self.ee_body_name = "hand"
        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ee_body_name,
        )

    def reset(self):
        self.data.qpos[:7] = self.q0
        self.data.qvel[:7] = np.zeros(7)
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def get_state(self):
        return self.data.qpos[:7], self.data.qvel[:7]

    def get_joint_acceleration(self):
        return self.data.qacc[:7]

    def step(self, finger_pos=None):
        tau = self.tau_ff
        self.actuator_tau = tau
        if finger_pos is not None:
            tau = np.append(tau, finger_pos)
            self.data.ctrl[:8] = tau.squeeze()
        else:
            self.data.ctrl[:7] = tau.squeeze()
        self.step_counter += 1
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def send_joint_torque(self, torques, finger_pos=None):
        self.tau_ff = torques
        self.latest_command_stamp = time.time()
        self.step(finger_pos)

    def get_gravity(self, q):

        q_backup = self.data.qpos[:7].copy()
        v_backup = self.data.qvel[:7].copy()

        self.data.qpos[:7] = q
        self.data.qvel[:7] = 0

        mujoco.mj_forward(self.model, self.data)

        mujoco.mj_rne(
            self.model,
            self.data,
            0,
            self.data.qfrc_bias
        )

        g = self.data.qfrc_bias[:7].copy()

        self.data.qpos[:7] = q_backup
        self.data.qvel[:7] = v_backup

        mujoco.mj_forward(self.model, self.data)

        return g

    def forward_kinematics(self, q, update=True):
        """
        Compute the forward kinematics for the end-effector frame.
        Args:
            q (np.ndarray): Joint positions (size: [model.nq] or [only joint DOFs])
        Returns:
            T_S_F (pinocchio.SE3): Transformation matrix of the end-effector frame.
        """
        q = np.append(q, [0.0, 0.0])
        T_S_F = self.pin_model.framePlacement(
            q, END_EFF_FRAME_ID, update_kinematics=update
        )
        return np.array(T_S_F)


    def get_jacobian(self, q):

        q_backup = self.data.qpos[:7].copy()

        self.data.qpos[:7] = q

        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacp,
            jacr,
            self.ee_body_id,
        )

        J = np.zeros((6, 7))

        J[0:3, :] = jacp[:, :7]
        J[3:6, :] = jacr[:, :7]

        self.data.qpos[:7] = q_backup
        mujoco.mj_forward(self.model, self.data)

        return J

    def get_pose(self, q):

        q_backup = self.data.qpos[:7].copy()

        self.data.qpos[:7] = q

        mujoco.mj_forward(self.model, self.data)

        pos = self.data.xpos[self.ee_body_id].copy()

        rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos

        self.data.qpos[:7] = q_backup
        mujoco.mj_forward(self.model, self.data)

        return T

    def get_ee_force_torque(self):
        force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
        torque_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque"
        )
        force = self.data.sensordata[force_id : force_id + 3]
        torque = self.data.sensordata[torque_id : torque_id + 3]
        return force.copy(), torque.copy()


    def get_sensor_force(self, sensor_name="left_sensor"):
        sensor_id = mujoco.mj_name2id(
        self.model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        sensor_name,)
        f = self.data.sensordata[sensor_id:sensor_id+3].copy()
        return f
    
    

    def get_mj_pose(self, body_name):
        """
        Get pose of a body in MuJoCo world frame.
        Args:
            body_name (str): body name in XML
        Returns:
            dict containing:
                position      : (3,)
                quaternion    : (4,) [x,y,z,w]
                rotation      : (3,3)
                transform     : (4,4)
        """
        # body id
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        # position
        pos = self.data.xpos[body_id].copy()
        # rotation matrix
        rot = self.data.xmat[body_id].reshape(3, 3).copy()
        # quaternion
        quat = R.from_matrix(rot).as_quat()
        # homogeneous transform
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return {
            "position": pos,
            "quaternion": quat,
            "rotation": rot,
            "transform": T,
        }
    

    def close(self):
        if self.render:
            self.viewer.close()
        # self.ros_node.destroy_node()
        # rclpy.shutdown()


