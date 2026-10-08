"""LQR 与 MPC 任务空间轨迹跟踪对比示例

默认模式:打开查看器 + Clock 实时节奏, 先 LQR 后 MPC 各演示一圈;
--smoke 模式:render=False 无头运行、不等实时, 两个控制器各完整跟踪一圈,
打印一行 SMOKE PASS/FAIL 结果, 并保存对比图 out/tracking_comparison.png。

参考轨迹:运行时由 get_pose(q0) 得到初始末端位置,
在其附近生成水平面内的圆(半径 0.1m、周期 8s), 保持在桌面之上、工作空间内。
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.MujocoSim import FR3Sim
from lib.Controller import LQR_Controller, MPC_Controller
from lib.Clock import Clock

# =========================
# 参数
# =========================
DT = 0.001          # 仿真步长(1kHz)
RADIUS = 0.1        # 圆轨迹半径(m)
PERIOD = 8.0        # 圆轨迹周期(s)
TAIL_STEPS = 1000   # 一圈结束后定点停稳的尾段时间
OUT_PATH = os.path.join("out", "tracking_comparison.png")

# smoke 通过阈值
RMS_LIMIT_MM = 30.0
MAX_LIMIT_MM = 100.0


# =========================
# 参考轨迹:绕初始末端位置的水平圆(保持 z 不变)
# =========================
def circle_trajectory(p0, radius=RADIUS, period=PERIOD, dt=DT):
    n_steps = int(round(period / dt))
    times = np.arange(n_steps + 1) * dt
    ang = 2.0 * np.pi * times / period

    positions = np.column_stack([
        p0[0] + radius * (np.cos(ang) - 1.0),
        p0[1] + radius * np.sin(ang),
        np.full_like(times, p0[2]),
    ])
    return times, positions


# =========================
# 单个控制器完整跟踪一圈并记录
# =========================
def run_tracking(controller, robot, times, positions, n_steps, clock=None):
    controller.set_trajectory(times, positions)
    ee_traj = np.zeros((n_steps, 3))
    err_mm = np.zeros(n_steps)
    tau_peak = 0.0

    for i in range(n_steps):

        # 计算关节力矩并发送
        tau = controller.step()
        robot.send_joint_torque(tau, 0)   # gripper open

        # 记录末端位置与跟踪误差(参考点取同拍采样)
        q, dq = robot.get_state()
        p_ee = robot.get_pose(q)[:3, 3]
        if not np.all(np.isfinite(p_ee)) or not np.all(np.isfinite(tau)):
            return None   # 数值发散

        idx = min(i + 1, len(positions) - 1)
        ee_traj[i] = p_ee
        err_mm[i] = np.linalg.norm(p_ee - positions[idx]) * 1000.0
        tau_peak = max(tau_peak, float(np.max(np.abs(tau))))

        if clock is not None:
            clock.wait()

    return {"ee": ee_traj, "err_mm": err_mm, "tau_peak": tau_peak}


# =========================
# 保存对比图(参考轨迹 + 两条实际末端轨迹)
# =========================
def save_comparison_plot(ref, lqr_ee, mpc_ee, out_path=OUT_PATH):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 5.5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", linewidth=1.5, label="参考轨迹")
    ax3d.plot(lqr_ee[:, 0], lqr_ee[:, 1], lqr_ee[:, 2], "b-", linewidth=1.0, label="LQR 实际")
    ax3d.plot(mpc_ee[:, 0], mpc_ee[:, 1], mpc_ee[:, 2], "r-", linewidth=1.0, label="MPC 实际")
    ax3d.set_title("末端轨迹跟踪对比 (3D)")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend()

    axxy = fig.add_subplot(1, 2, 2)
    axxy.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.5, label="参考轨迹")
    axxy.plot(lqr_ee[:, 0], lqr_ee[:, 1], "b-", linewidth=1.0, label="LQR 实际")
    axxy.plot(mpc_ee[:, 0], mpc_ee[:, 1], "r-", linewidth=1.0, label="MPC 实际")
    axxy.set_title("俯视图 (XY)")
    axxy.set_xlabel("x [m]")
    axxy.set_ylabel("y [m]")
    axxy.axis("equal")
    axxy.grid(True)
    axxy.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser(description="LQR / MPC 任务空间轨迹跟踪对比")
    parser.add_argument("--smoke", action="store_true", help="无头快速自检模式")
    args = parser.parse_args()

    robot = FR3Sim(render=not args.smoke)
    clock = Clock(DT) if not args.smoke else None
    n_steps = int(round(PERIOD / DT)) + TAIL_STEPS

    # ===== 参考轨迹:初始末端位置附近的水平圆 =====
    p0 = robot.get_pose(robot.q0)[:3, 3]
    times, positions = circle_trajectory(p0)

    # ===== 先 LQR 后 MPC, 各跟踪一圈 =====
    results = {}
    for name, cls in [("lqr", LQR_Controller), ("mpc", MPC_Controller)]:
        if name == "mpc":
            robot.reset()   # 第二轮开始前重置回 q0

        controller = cls(robot)
        res = run_tracking(controller, robot, times, positions, n_steps, clock=clock)
        if res is None:
            if args.smoke:
                print(f"SMOKE FAIL: {name.upper()} 跟踪过程数值发散")
            robot.close()
            sys.exit(1)
        results[name] = res

    robot.close()

    # ===== 统计 =====
    stats = {}
    for name, res in results.items():
        stats[f"{name}_rms_mm"] = float(np.sqrt(np.mean(res["err_mm"] ** 2)))
        stats[f"{name}_max_mm"] = float(np.max(res["err_mm"]))
        stats[f"{name}_tau_peak"] = float(res["tau_peak"])

    # ===== 保存对比图 =====
    save_comparison_plot(positions, results["lqr"]["ee"], results["mpc"]["ee"])

    if args.smoke:
        fail_reasons = []
        for key, val in stats.items():
            if not np.isfinite(val):
                fail_reasons.append(f"{key} 非有限值")
        for name in ["lqr", "mpc"]:
            if stats[f"{name}_rms_mm"] > RMS_LIMIT_MM:
                fail_reasons.append(f"{name} RMS 误差 {stats[f'{name}_rms_mm']:.2f}mm 超过 {RMS_LIMIT_MM}mm")
            if stats[f"{name}_max_mm"] > MAX_LIMIT_MM:
                fail_reasons.append(f"{name} 最大误差 {stats[f'{name}_max_mm']:.2f}mm 超过 {MAX_LIMIT_MM}mm")

        if fail_reasons:
            print(f"SMOKE FAIL: {'; '.join(fail_reasons)}")
            sys.exit(1)

        print(
            "SMOKE PASS "
            f"lqr_rms_mm={stats['lqr_rms_mm']:.2f} "
            f"mpc_rms_mm={stats['mpc_rms_mm']:.2f} "
            f"lqr_max_mm={stats['lqr_max_mm']:.2f} "
            f"mpc_max_mm={stats['mpc_max_mm']:.2f} "
            f"lqr_tau_peak={stats['lqr_tau_peak']:.2f} "
            f"mpc_tau_peak={stats['mpc_tau_peak']:.2f}"
        )
        sys.exit(0)

    # ===== 默认模式:打印摘要 =====
    print(f"轨迹起点 p0 = {np.round(p0, 4).tolist()}, 半径 {RADIUS}m, 周期 {PERIOD}s")
    for name in ["lqr", "mpc"]:
        print(
            f"{name.upper()}: RMS {stats[f'{name}_rms_mm']:.2f}mm, "
            f"最大 {stats[f'{name}_max_mm']:.2f}mm, 力矩峰值 {stats[f'{name}_tau_peak']:.2f}Nm"
        )
    print(f"对比图已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
