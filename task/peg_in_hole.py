"""位控+导纳控制示例：FR3机械臂的插拔任务
   若统一成小阻抗，靠近阶段慢且精度不够，若是大阻抗不满足柔顺插入要求
   插入阶段只调节位置，保持姿态
   导纳控制中z轴受重力影响"""
import numpy as np
from lib.MujocoSim import FR3Sim
from scipy.spatial.transform import Rotation as R
from lib.Controller import Position_Controller,Admittance_Controller
from lib.Clock import Clock

my_robot = FR3Sim(xml_path="assets/fr3_peg_in_hole.xml")
pos_controller = Position_Controller(my_robot)
adm_controller = Admittance_Controller(my_robot)
dt = 0.001
Clock = Clock(dt)
# =========================
# 物体初始位置/放置位置
# =========================
obj_pos = my_robot.get_mj_pose("object")["position"]
place_pos = my_robot.get_mj_pose("box2")["position"]
R_des = R.from_euler('xyz', [0, 3.14, 1.57]).as_matrix()

# =========================
# 抓取路径
# =========================
pre_grasp = obj_pos + np.array([0, 0, 0.3])
grasp = obj_pos + np.array([0, 0, 0.15])
lift = obj_pos + np.array([0, 0, 0.3])

# =========================
# 放置路径
# =========================
place_pre = place_pos + np.array([0, 0, 0.3])
place_down = place_pos + np.array([0, 0, 0.25])
place_lift = place_pos + np.array([0, 0, 0.4])

def move_to(target, steps=2000, gripper=10):
    pos_controller.set_target(target, R_des)
    for _ in range(steps):
        tau = pos_controller.step()
        my_robot.send_joint_torque(tau, gripper)
        Clock.wait()




# =========================
# 抓取流程
# =========================

# 1️⃣ 打开夹爪 + 到上方
move_to(pre_grasp, steps=1000, gripper=10)

# 2️⃣ 慢慢下降
move_to(grasp, steps=1000, gripper=10)

# 3️⃣ 夹紧
for _ in range(1000):
    tau = pos_controller.step()
    my_robot.send_joint_torque(tau, -20)  # 关闭夹爪
    Clock.wait()

# 4️⃣ 抬起
move_to(lift, steps=500, gripper=-20)

# =========================
# 放置流程
# =========================

# 5️⃣ 移动到放置上方
move_to(place_pre, steps=1000, gripper=-20)


move_to(place_down, steps=1000, gripper=-20)

# 6️⃣ 导纳控制插入
for _ in range(8000):
    tau = adm_controller.step()
    my_robot.send_joint_torque(tau, -20)
    Clock.wait()




# 7️⃣ 松开抬起撤离
for _ in range(2000):
    tau = adm_controller.step()
    my_robot.send_joint_torque(tau, 5)  # 打开夹爪
    Clock.wait()
move_to(place_lift, steps=3000, gripper=5)

