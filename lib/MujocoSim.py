import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
import os
import pinocchio as pin
from pinocchio import RobotWrapper
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
        urdf = os.path.join(ASSETS_PATH, "fr3.urdf")
        meshes_dir = os.path.join(ASSETS_PATH, "meshes")
        self.pin_model = RobotWrapper.BuildFromURDF(urdf, meshes_dir)

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

    def joint_state_callback(self, msg):
        """
        Callback to handle incoming joint states.
        """
        positions = list(msg.position[:7])
        velocities = list(msg.velocity[:7])
        efforts = list(msg.effort[:7])
        names = list(msg.name[:7])  # Ensure we only take the first 7 names
        # Swap the 2nd and 3rd elements (index 1 and 2) of the arrays
        positions[1], positions[2] = positions[2], positions[1]
        velocities[1], velocities[2] = velocities[2], velocities[1]
        efforts[1], efforts[2] = efforts[2], efforts[1]
        names[1], names[2] = names[2], names[1]
        self.latest_joint_states = {
            "names": names,
            "positions": positions,
            "velocities": velocities,
            "efforts": efforts,
        }

    def get_latest_joint_states(self):
        """
        Returns the most recent joint states received from the /joint_states topic.
        Returns:
            dict: {'names', 'positions', 'velocities', 'efforts'}
        """
        return self.latest_joint_states

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

    def send_joint_torque(self, torques, finger_pos=None):
        self.tau_ff = torques
        self.latest_command_stamp = time.time()
        self.step(finger_pos)

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

    def get_gravity(self, q):
        g = self.pin_model.gravity(np.append(q, [0.0, 0.0]))
        return g[:7]

    def get_dynamics(self, q, v):
        """
        Compute the joint-space inertia matrix M(q) and the nonlinear effects h(q, v) = C(q, v)*v + g(q).
        Args:
            q (np.ndarray): Joint positions (size: [model.nq] or [only joint DOFs])
            v (np.ndarray): Joint velocities (size: [model.nv] or [only joint DOFs])
        Returns:
            M (np.ndarray): Mass matrix (nv x nv)
            h (np.ndarray): Nonlinear effects (nv,)
        """
        q = np.append(q, [0.0, 0.0])  # Append zeros for the end-effector
        v = np.append(v, [0.0, 0.0])  # Append zeros for the end-effector
        M = self.pin_model.mass(q)
        h = self.pin_model.nle(q, v)
        return M, h

    def compute_tau(self, q, v, ddq_des):
        """
        Compute the total torque command using inverse dynamics:
        tau = M(q) * ddq_des + h(q, v)
        Args:
            q (np.ndarray): Joint positions
            v (np.ndarray): Joint velocities
            ddq_des (np.ndarray): Desired joint accelerations
        Returns:
            tau (np.ndarray): Joint torques
        """
        ddq_des = np.append(ddq_des, [0.0, 0.0])  # Append zeros for the end-effector
        M, h = self.get_dynamics(q, v)
        tau = M @ ddq_des + h
        return tau[:7]

    def get_jacobian(self, q):
        J_temp = self.pin_model.computeFrameJacobian(
            np.append(q, [0.0, 0.0]), END_EFF_FRAME_ID
        )
        J = np.zeros([6, 7])
        J[3:6, :] = J_temp[0:3, :7]
        J[0:3, :] = J_temp[3:6, :7]
        return J[:, :7]

    def get_pose(self, q):
        T_S_F = self.pin_model.framePlacement(
            np.append(q, [0.0, 0.0]), END_EFF_FRAME_ID  #FR3在Pinocchio模型中有9个DOF，末端执行器的位姿需要在原来的7维关节空间基础上补齐2维
        )
        return T_S_F.homogeneous  #T为pinocchio.SE3类型，homogeneous属性返回4x4的齐次变换矩阵

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
    
    
    def close(self):
        if self.render:
            self.viewer.close()
        # self.ros_node.destroy_node()
        # rclpy.shutdown()


