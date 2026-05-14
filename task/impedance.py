import numpy as np
from lib.MujocoSim import FR3Sim
from lib.Controller import Impedance_Controller
from scipy.spatial.transform import Rotation as R

# =========================
# 创建机器人
# =========================
robot = FR3Sim()

# =========================
# 创建阻抗控制器
# =========================
controller = Impedance_Controller(robot)

# =========================
# 设置目标位置（可选）
# =========================
controller.set_target(
    position=np.array([0.5, -0.5, 1.5]),
    rotation_matrix=R.from_euler('xyz', [0, np.pi,np.pi/2]).as_matrix()
)

# =========================
# 控制循环
# =========================

while True:

    # 计算关节力矩
    tau = controller.step()

    # 发送力矩控制
    robot.send_joint_torque(tau, 0)   # gripper open

