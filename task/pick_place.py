import numpy as np
from lib.MujocoSim import FR3Sim
from scipy.spatial.transform import Rotation as R
from lib.Controller import Position_Controller
from lib.Clock import Clock

my_robot = FR3Sim(xml_path="./assets/fr3_pick_place.xml")
controller = Position_Controller(my_robot)
Clock = Clock(0.001)
R_des = R.from_euler('xyz', [3.14, 0, 0]).as_matrix()

# =========================
# 物体初始位置（自己根据XML改）
# =========================
obj_pos = np.array([0.45, 0.1, 0.1])
# =========================
# 抓取路径
# =========================
pre_grasp = obj_pos + np.array([0, 0, 0.15])
grasp = obj_pos + np.array([0, 0, 0.08])
lift = obj_pos + np.array([0, 0, 0.2])
# =========================
# 放置位置
# =========================
place_pos = np.array([0.45, -0.25, 0.25])

place_pre = place_pos + np.array([0, 0, 0.15])
place_down = place_pos + np.array([0, 0, 0.08])
place_lift = place_pos + np.array([0, 0, 0.2])
# =========================
# 移动
# =========================
def move_to(target, steps=2000, gripper=10):
    controller.set_target(target, R_des)
    for _ in range(steps):
        tau = controller.step()
        my_robot.send_joint_torque(tau, gripper)
        Clock.wait()

# =========================
# 抓取流程
# =========================

# 1️⃣ 打开夹爪 + 到上方
move_to(pre_grasp, steps=3000, gripper=10)

# 2️⃣ 慢慢下降
move_to(grasp, steps=3000, gripper=10)

# 3️⃣ 夹紧
for _ in range(2000):
    tau = controller.step()
    my_robot.send_joint_torque(tau, -10)  # 关闭夹爪
    Clock.wait()

# 4️⃣ 抬起
move_to(lift, steps=3000, gripper=-10)

# =========================
# 放置流程
# =========================

# 5️⃣ 移动到放置上方
move_to(place_pre, steps=3000, gripper=-10)

# 6️⃣ 下降
move_to(place_down, steps=3000, gripper=-10)

# 7️⃣ 松开
for _ in range(1500):
    tau = controller.step()
    my_robot.send_joint_torque(tau, 10)  # 打开夹爪
    Clock.wait()

# 8️⃣ 抬起撤离
move_to(place_lift, steps=3000, gripper=10)