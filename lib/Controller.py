import numpy as np
from scipy.spatial.transform import Rotation as R

class Position_Controller:
    def __init__(self, robot):

        self.robot = robot

        # ===== 控制增益 =====
        self.Kp_pos = np.array([1500, 1500, 1500])
        self.Kd_pos = np.array([300, 300, 300])

        self.Kp_ori = np.array([80, 80, 80])
        self.Kd_ori = np.array([20, 20, 20])

        # ===== 目标 =====
        self.x_des = np.zeros(3)
        self.dx_des = np.zeros(3)

        self.R_des = np.eye(3)
        self.omega_des = np.zeros(3)

    # =========================
    # 设置目标
    # =========================
    def set_target(self, position, rotation_matrix):
        self.x_des = position
        self.R_des = rotation_matrix

    # =========================
    # 单步控制
    # =========================
    def step(self):

        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        x = T[:3, 3]
        R_ee = T[:3, :3]

        # ===== Jacobian =====
        J = self.robot.get_jacobian(q)

        # ===== 速度 =====
        dx_full = J @ dq
        v = dx_full[:3]
        omega = dx_full[3:]

        # ===== 位置误差 =====
        pos_error = self.x_des - x
        vel_error = self.dx_des - v

        # ===== 姿态误差（SO3）=====
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], self.R_des[:, 0]) +
            np.cross(R_ee[:, 1], self.R_des[:, 1]) +
            np.cross(R_ee[:, 2], self.R_des[:, 2])
        )

        omega_error = self.omega_des - omega

        # ===== wrench =====
        force = self.Kp_pos * pos_error + self.Kd_pos * vel_error
        moment = self.Kp_ori * ori_error + self.Kd_ori * omega_error

        wrench = np.concatenate([force, moment])

        # ===== torque =====
        tau = J.T @ wrench + self.robot.get_gravity(q)

        return tau


class Admittance_Controller:
    def __init__(self, robot):
        self.robot = robot

        # =============================
        # Position Admittance parameters
        # =============================
        self.M_adm = np.diag([2.0, 2.0, 2.0])
        self.D_adm = np.diag([8.0, 8.0, 8.0])
        self.K_adm = np.diag([10.0, 10.0, 10.0])

        # Position tracking gains
        self.Kp = np.array([1.0, 1.0, 1.0])
        self.Kd = np.array([0.1, 0.1, 0.1])

        # =============================
        # Orientation control gains
        # =============================
        self.Kp_ori = np.array([3.0, 3.0, 3.0])
        self.Kd_ori = np.array([0.2, 0.2, 0.2])

        self.dt = 0.001

        # Initial pose
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        self.x_des = T[:3, 3].copy()
        self.dx_des = np.zeros(3)

        # 加入姿态目标
        self.R_des = T[:3, :3].copy()
        self.omega_des = np.zeros(3)

        self.ee_body_id = self.robot.model.body(b"hand").id

    # =============================
    # Reset target
    # =============================
    def reset_target(self):
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        self.x_des = T[:3, 3].copy()
        self.dx_des = np.zeros(3)

        self.R_des = T[:3, :3].copy()
        self.omega_des = np.zeros(3)

    # =============================
    # Set desired pose (position + orientation)
    # =============================
    def set_target(self, position, R_target=None):
        self.x_des = position.copy()
        self.dx_des = np.zeros(3)

        if R_target is not None:
            self.R_des = R_target.copy()
            self.omega_des = np.zeros(3)

    # =============================
    # Main step
    # =============================
    def step(self):

        # ===== Robot state =====
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        x = T[:3, 3]
        R_ee = T[:3, :3]

        # ===== Jacobian =====
        J = self.robot.get_jacobian(q)

        dx_full = J @ dq
        dx = dx_full[:3]
        omega = dx_full[3:]

        # ===== External force =====
        f_ext = self.robot.get_sensor_force()
        # ==================================
        # Position Admittance dynamics
        # Mx¨ + Dx˙ + K(x_des-x) = f_ext
        # ==================================
        ddx_des = np.linalg.inv(self.M_adm) @ (
            f_ext
            - self.D_adm @ self.dx_des
            - self.K_adm @ (self.x_des - x)
        )

        self.dx_des += ddx_des * self.dt
        self.x_des += self.dx_des * self.dt

        # ===== Position tracking =====
        pos_error = self.x_des - x
        vel_error = self.dx_des - dx

        desired_force = self.Kp * pos_error + self.Kd * vel_error
        # ==================================
        # Orientation constraint
        # ==================================
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], self.R_des[:, 0]) +
            np.cross(R_ee[:, 1], self.R_des[:, 1]) +
            np.cross(R_ee[:, 2], self.R_des[:, 2])
        )

        omega_error = self.omega_des - omega

        desired_moment = (
            self.Kp_ori * ori_error +
            self.Kd_ori * omega_error
        )
        desired_force[2] = -0.5  # 竖直方向不施加力，完全由重力约束
        # ===== Full wrench =====
        wrench = np.concatenate([desired_force, desired_moment])

        # ===== Joint torque =====
        tau = J.T @ wrench + self.robot.get_gravity(q)

        return tau

class Impedance_Controller:
    def __init__(self, robot):

        self.robot = robot

        # =========================
        # 位置阻抗参数
        # =========================
        self.Kp_pos = np.array([150, 150, 150])
        self.Kd_pos = np.array([300, 300, 300])

        # =========================
        # 姿态阻抗参数
        # =========================
        self.Kp_ori = np.array([80, 80, 80])
        self.Kd_ori = np.array([20, 20, 20])

        # =========================
        # 目标状态
        # =========================
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        self.x_des = T[:3, 3].copy()
        self.dx_des = np.zeros(3)

        self.R_des = T[:3, :3].copy()
        self.omega_des = np.zeros(3)

    # =========================
    # 设置目标位姿
    # =========================
    def set_target(self, position, rotation_matrix):
        self.x_des = position.copy()
        self.R_des = rotation_matrix.copy()

    # =========================
    # 重置目标为当前位置
    # =========================
    def reset_target(self):
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        self.x_des = T[:3, 3].copy()
        self.dx_des = np.zeros(3)

        self.R_des = T[:3, :3].copy()
        self.omega_des = np.zeros(3)

    # =========================
    # 单步控制
    # =========================
    def step(self):

        # ===== 当前状态 =====
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)

        x = T[:3, 3]
        R_ee = T[:3, :3]

        # ===== Jacobian =====
        J = self.robot.get_jacobian(q)


        dx_full = J @ dq
        v = dx_full[:3]
        omega = dx_full[3:]

        # =========================
        # 位置误差
        # =========================
        pos_error = self.x_des - x
        vel_error = self.dx_des - v

        force = self.Kp_pos * pos_error + self.Kd_pos * vel_error

        # =========================
        # 姿态误差 (SO3)
        # =========================
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], self.R_des[:, 0]) +
            np.cross(R_ee[:, 1], self.R_des[:, 1]) +
            np.cross(R_ee[:, 2], self.R_des[:, 2])
        )

        omega_error = self.omega_des - omega

        moment = self.Kp_ori * ori_error + self.Kd_ori * omega_error

        # =========================
        # 合成 wrench
        # =========================
        wrench = np.concatenate([force, moment])

        # =========================
        # 关节力矩
        # =========================
        tau = J.T @ wrench + self.robot.get_gravity(q)

        return tau