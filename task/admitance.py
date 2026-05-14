'''可用于拖动示教'''
import numpy as np
from lib.MujocoSim import FR3Sim
from lib.Controller import Admittance_Controller

# =========================
# 创建机器人
# =========================
my_robot = FR3Sim(xml_path="./assets/fr3_on_table.xml")

# =========================
# 继承并创建控制器
# =========================
class adm_controller(Admittance_Controller):
    def __init__(self, robot):
        super().__init__(robot)
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

        # ===== 外部作用在质心上的力，只算线性部分=====
        f_ext = self.robot.data.xfrc_applied[self.ee_body_id][:3]

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

        # ===== Full wrench =====
        wrench = np.concatenate([desired_force, desired_moment])

        # ===== Joint torque =====
        tau = J.T @ wrench + self.robot.get_gravity(q)

        return tau


controller = adm_controller(my_robot)
# =========================
# 控制循环
# =========================

while True:

    # 计算控制力矩
    tau = controller.step()

    # 发送到机器人
    my_robot.send_joint_torque(tau, 10)   # gripper open

