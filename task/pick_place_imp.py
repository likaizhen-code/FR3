'''阻抗控制完成抓取放置
   根据夹爪接触力判断是否抓取'''
import numpy as np
from lib.MujocoSim import FR3Sim
from scipy.spatial.transform import Rotation as R
from lib.Controller import Impedance_Controller
from lib.Clock import Clock

my_robot = FR3Sim(xml_path="./assets/fr3_pick_place.xml")
controller = Impedance_Controller(my_robot)
Clock = Clock(0.001)
R_des = R.from_euler('xyz', [3.14, 0, 3.14]).as_matrix()

# =========================
# 位置定义
# =========================
obj_pos = my_robot.get_mj_pose("object")["position"]
place_pos = my_robot.get_mj_pose("box1")["position"]

pre_grasp = obj_pos + np.array([0, 0, 0.3])
grasp    = obj_pos + np.array([0, 0, 0.15])
lift     = obj_pos + np.array([0, 0, 0.3])

place_pre  = place_pos + np.array([0, 0, 0.3])
place_down = place_pos + np.array([0, 0, 0.28])
place_lift = place_pos + np.array([0, 0, 0.4])

def move_to(target, steps=2000, gripper=10):
    controller.set_target(target, R_des)
    for _ in range(steps):
        tau = controller.step()
        my_robot.send_joint_torque(tau, gripper)
        Clock.wait()

def smart_grasp(force_threshold=-5):
    """
    自动判断是否夹到物体：
    - 夹爪闭合
    - 检测末端受力 > 阈值 → 判定抓到了
    - 没抓到会一直等
    """
    print("正在抓取物体...")
    # 持续闭合夹爪 + 检测力
    while True:
        tau = controller.step()
        my_robot.send_joint_torque(tau, -10)  # 闭合夹爪
        Clock.wait()

        # 获取当前末端受力
        f = my_robot.get_sensor_force()
        force_norm = np.linalg.norm(f)

        # 力小于阈值 
        if force_norm < force_threshold:
            break



# 1. 移动到物体上方
move_to(pre_grasp, steps=2000, gripper=10)

# 2. 下降到抓取高度
move_to(grasp, steps=2000, gripper=10)

# 3. 【智能抓取】抓到才继续
smart_grasp(force_threshold=5)

# 4. 抬起
move_to(lift, steps=2000, gripper=-10)

# 5. 移动到放置区上方
move_to(place_pre, steps=2000, gripper=-10)

# 6. 下降
move_to(place_down, steps=2000, gripper=-10)

# 7. 松开夹爪
for _ in range(1500):
    tau = controller.step()
    my_robot.send_joint_torque(tau, 10)
    Clock.wait()

# 8. 抬起撤离
move_to(place_lift, steps=2000, gripper=10)

print("任务完成！")