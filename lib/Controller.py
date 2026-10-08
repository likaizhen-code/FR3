import numpy as np
import mujoco
import osqp
import scipy.sparse as sparse
from scipy.linalg import solve_continuous_are
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
        self.Kp_pos = np.array([1000, 1000, 1000])
        self.Kd_pos = np.array([200, 200, 200])

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


class LQR_Controller:
    """
    任务空间轨迹跟踪 LQR 控制器

    末端位置绕参考轨迹按双积分器线性化 (A=[[0,I],[0,0]], B=[[0],[I]]),
    由连续时间代数 Riccati 方程求反馈增益 K;
    u = K·x 为期望加速度, 经任务空间惯性 M_x=(J M⁻¹ Jᵀ)⁻¹(阻尼伪逆)
    映射为期望力; 姿态保持初始, 采用与上面控制器相同的 SO3 叉积误差。
    """

    def __init__(self, robot):

        self.robot = robot

        # =========================
        # 双积分器线性化模型
        # x = [e_p; e_v] (6), u = 期望加速度 (3)
        # =========================
        self.A = np.zeros((6, 6))
        self.A[:3, 3:] = np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[3:, :] = np.eye(3)

        # ===== LQR 权重 =====
        self.Q_lqr = np.diag([3.6e5, 3.6e5, 3.6e5, 1300.0, 1300.0, 1300.0])
        self.R_lqr = np.eye(3)

        # ===== 增益初值(ARE 求解失败时的后备 PD)=====
        self.K = np.hstack([600.0 * np.eye(3), 50.0 * np.eye(3)])

        # ===== ARE 重算周期:仿真 1kHz,约 100Hz 重算 =====
        self.dt = 0.001
        self.are_interval = 100
        self._are_tick = self.are_interval

        # =========================
        # 姿态 PD 参数(同 Impedance_Controller)
        # =========================
        self.Kp_ori = np.array([80, 80, 80])
        self.Kd_ori = np.array([20, 20, 20])

        # ===== 阻尼伪逆系数 =====
        self.damping = 1e-6

        # =========================
        # 关节力矩限幅(从模型读取, 无限时用 FR3 默认值)
        # =========================
        forcerange = np.array(self.robot.model.actuator_forcerange[:7])
        if np.all(forcerange[:, 1] > 0):
            self.tau_max = forcerange[:, 1].astype(float)
        else:
            self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        # ===== 参考轨迹 =====
        self.traj_times = np.array([0.0])
        self.traj_pos = np.zeros((1, 3))

        # ===== 初始姿态目标 / 内部时钟 =====
        self.reset()

    # =========================
    # 设置末端位置参考轨迹(姿态保持初始)
    # =========================
    def set_trajectory(self, times, positions):
        self.traj_times = np.asarray(times, dtype=float).ravel()
        self.traj_pos = np.asarray(positions, dtype=float).reshape(-1, 3)

    # =========================
    # 重置内部时钟与姿态目标(配合 robot.reset() 使用)
    # =========================
    def reset(self):
        self.t = 0.0
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)
        self.R_des = T[:3, :3].copy()

    # =========================
    # 按时间线性插值参考点(返回位置与速度)
    # =========================
    def _interpolate_reference(self, t):
        times = self.traj_times
        pos = self.traj_pos
        n = len(times)

        if t <= times[0]:
            return pos[0].copy(), np.zeros(3)
        if t >= times[-1]:
            return pos[-1].copy(), np.zeros(3)

        i = min(max(np.searchsorted(times, t, side="right") - 1, 0), n - 2)
        dt_seg = times[i + 1] - times[i]
        if dt_seg <= 0:
            return pos[i].copy(), np.zeros(3)

        alpha = (t - times[i]) / dt_seg
        p_ref = pos[i] + alpha * (pos[i + 1] - pos[i])
        v_ref = (pos[i + 1] - pos[i]) / dt_seg
        return p_ref, v_ref

    # =========================
    # 由 ARE 求 LQR 增益(失败时保留上一拍增益)
    # =========================
    def _solve_gain(self):
        try:
            P = solve_continuous_are(self.A, self.B, self.Q_lqr, self.R_lqr)
            K_new = np.linalg.solve(self.R_lqr, self.B.T @ P)
            if np.all(np.isfinite(K_new)):
                self.K = K_new
        except np.linalg.LinAlgError:
            pass

    # =========================
    # 关节空间惯量阵(仅机械臂 7 个关节)
    # =========================
    def _mass_matrix(self, q):
        robot = self.robot

        q_backup = robot.data.qpos[:7].copy()
        v_backup = robot.data.qvel[:7].copy()

        robot.data.qpos[:7] = q
        robot.data.qvel[:7] = 0

        mujoco.mj_forward(robot.model, robot.data)

        M = np.zeros((robot.model.nv, robot.model.nv))
        mujoco.mj_fullM(robot.model, M, robot.data.qM)

        robot.data.qpos[:7] = q_backup
        robot.data.qvel[:7] = v_backup
        mujoco.mj_forward(robot.model, robot.data)

        return M[:7, :7]

    # =========================
    # 任务空间惯性 M_x = (J M⁻¹ Jᵀ)⁻¹(阻尼伪逆, 防奇异)
    # =========================
    def _task_inertia(self, q, J):
        M = self._mass_matrix(q)
        Jp = J[:3, :]

        Minv_Jt = np.linalg.solve(M + 1e-9 * np.eye(7), Jp.T)
        M_x = np.linalg.solve(Jp @ Minv_Jt + self.damping * np.eye(3), np.eye(3))

        return M_x

    # =========================
    # 姿态误差力矩(SO3 叉积误差, 同上面控制器)
    # =========================
    def _orientation_moment(self, R_ee, omega):
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], self.R_des[:, 0]) +
            np.cross(R_ee[:, 1], self.R_des[:, 1]) +
            np.cross(R_ee[:, 2], self.R_des[:, 2])
        )

        omega_error = -omega

        moment = self.Kp_ori * ori_error + self.Kd_ori * omega_error

        return moment

    # =========================
    # 期望加速度 -> 期望力 -> 关节力矩(重力补偿 + 限幅)
    # =========================
    def _acceleration_to_torque(self, q, J, R_ee, omega, a_des):
        M_x = self._task_inertia(q, J)

        force = M_x @ a_des
        moment = self._orientation_moment(R_ee, omega)

        wrench = np.concatenate([force, moment])

        tau = J.T @ wrench + self.robot.get_gravity(q)
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        return tau

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

        # ===== 参考点插值 =====
        p_ref, v_ref = self._interpolate_reference(self.t)

        # ===== 任务空间误差状态 x = [e_p; e_v] =====
        e_p = p_ref - x
        e_v = v_ref - v
        x_err = np.concatenate([e_p, e_v])

        # ===== LQR 增益(约 100Hz 重算) =====
        self._are_tick += 1
        if self._are_tick >= self.are_interval:
            self._solve_gain()
            self._are_tick = 0

        # ===== 期望加速度 u = K·x =====
        a_des = self.K @ x_err

        # ===== 力矩映射 =====
        tau = self._acceleration_to_torque(q, J, R_ee, omega, a_des)

        self.t += self.dt

        return tau


class MPC_Controller:
    """
    任务空间轨迹跟踪 MPC 控制器

    线性(时变)双积分器预测模型, 时域 N=20、决策频率 100Hz
    (每 10 个仿真步重解一次 QP, 期间保持首步控制);
    位置/速度参考跟踪代价 + 控制量代价, OSQP 求解(P 用 csc 上三角稀疏);
    约束:末端速度与加速度上下限; 求解失败时回退到上一拍解;
    首步期望加速度的力矩映射与 LQR_Controller 相同。
    """

    def __init__(self, robot, N=20, mpc_dt=0.01):

        self.robot = robot

        # =========================
        # 预测参数
        # =========================
        self.N = N
        self.mpc_dt = mpc_dt
        self.dt = 0.001
        self.solve_interval = max(1, int(round(mpc_dt / self.dt)))

        # ===== 代价权重 =====
        self.w_pos = 2.0e5
        self.w_vel = 2.0e3
        self.w_u = 1.0
        self.w_term = 10.0

        # ===== 末端速度/加速度上下限 =====
        self.v_max = 0.5
        self.a_max = 2.0

        # =========================
        # 姿态 PD 参数(同 Impedance_Controller)
        # =========================
        self.Kp_ori = np.array([80, 80, 80])
        self.Kd_ori = np.array([20, 20, 20])

        # ===== 阻尼伪逆系数 =====
        self.damping = 1e-6

        # ===== 关节力矩限幅 =====
        forcerange = np.array(self.robot.model.actuator_forcerange[:7])
        if np.all(forcerange[:, 1] > 0):
            self.tau_max = forcerange[:, 1].astype(float)
        else:
            self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        # =========================
        # 参考轨迹 / 求解状态
        # =========================
        self.traj_times = np.array([0.0])
        self.traj_pos = np.zeros((1, 3))

        self._U_last = np.zeros(3 * N)     # 上一拍解(回退用)
        self._a_des = np.zeros(3)          # 保持的首步期望加速度
        self._solve_tick = self.solve_interval
        self._solver = None

        self.reset()

        # ===== 预测矩阵与 QP 稀疏结构(常数, 只建一次) =====
        self._build_prediction()
        self._build_qp()

    # =========================
    # 设置末端位置参考轨迹(姿态保持初始)
    # =========================
    def set_trajectory(self, times, positions):
        self.traj_times = np.asarray(times, dtype=float).ravel()
        self.traj_pos = np.asarray(positions, dtype=float).reshape(-1, 3)

    # =========================
    # 重置内部时钟、姿态目标与求解状态
    # =========================
    def reset(self):
        self.t = 0.0
        self._a_des = np.zeros(3)
        self._U_last = np.zeros(3 * self.N)
        self._solve_tick = self.solve_interval
        q, dq = self.robot.get_state()
        T = self.robot.get_pose(q)
        self.R_des = T[:3, :3].copy()

    # =========================
    # 按时间线性插值参考点(返回位置与速度)
    # =========================
    def _interpolate_reference(self, t):
        times = self.traj_times
        pos = self.traj_pos
        n = len(times)

        if t <= times[0]:
            return pos[0].copy(), np.zeros(3)
        if t >= times[-1]:
            return pos[-1].copy(), np.zeros(3)

        i = min(max(np.searchsorted(times, t, side="right") - 1, 0), n - 2)
        dt_seg = times[i + 1] - times[i]
        if dt_seg <= 0:
            return pos[i].copy(), np.zeros(3)

        alpha = (t - times[i]) / dt_seg
        p_ref = pos[i] + alpha * (pos[i + 1] - pos[i])
        v_ref = (pos[i + 1] - pos[i]) / dt_seg
        return p_ref, v_ref

    # =========================
    # 离散双积分器预测矩阵
    # x_{k+1} = Ad x_k + Bd u_k, X = Phi x0 + Gam U
    # =========================
    def _build_prediction(self):
        h = self.mpc_dt
        N = self.N

        Ad = np.eye(6)
        Ad[:3, 3:] = h * np.eye(3)
        Bd = np.zeros((6, 3))
        Bd[:3, :] = 0.5 * h * h * np.eye(3)
        Bd[3:, :] = h * np.eye(3)

        # A 的幂次(Ad^0 ... Ad^N)
        powers = [np.eye(6)]
        for _ in range(N):
            powers.append(powers[-1] @ Ad)

        Phi = np.zeros((6 * N, 6))
        Gam = np.zeros((6 * N, 3 * N))
        for k in range(1, N + 1):
            Phi[6 * (k - 1):6 * k, :] = powers[k]
            for j in range(k):
                Gam[6 * (k - 1):6 * k, 3 * j:3 * (j + 1)] = powers[k - 1 - j] @ Bd

        # 位置/速度选择矩阵
        S_p = np.zeros((3 * N, 6 * N))
        S_v = np.zeros((3 * N, 6 * N))
        for k in range(N):
            S_p[3 * k:3 * k + 3, 6 * k:6 * k + 3] = np.eye(3)
            S_v[3 * k:3 * k + 3, 6 * k + 3:6 * k + 6] = np.eye(3)

        # U -> 预测位置/速度, x0 -> 预测位置/速度
        self.C_p = S_p @ Gam
        self.C_v = S_v @ Gam
        self.D_p = S_p @ Phi
        self.D_v = S_v @ Phi

        # 位置/速度权重(末态加权)
        w_p_vec = np.full(3 * N, self.w_pos)
        w_p_vec[-3:] *= self.w_term
        w_v_vec = np.full(3 * N, self.w_vel)
        w_v_vec[-3:] *= self.w_term
        self.w_p_vec = w_p_vec
        self.w_v_vec = w_v_vec

    # =========================
    # OSQP 问题结构(P 用 csc 上三角稀疏)
    # =========================
    def _build_qp(self):
        n_u = 3 * self.N

        P_dense = (
            2.0 * (self.C_p.T * self.w_p_vec) @ self.C_p +
            2.0 * (self.C_v.T * self.w_v_vec) @ self.C_v +
            2.0 * self.w_u * np.eye(n_u)
        )
        self.P = sparse.triu(sparse.csc_matrix(P_dense), format="csc")

        # 约束:加速度上下限(单位阵) + 末端速度上下限(C_v)
        A_ineq = sparse.vstack(
            [
                sparse.identity(n_u, format="csc"),
                sparse.csc_matrix(self.C_v),
            ],
            format="csc",
        )
        self.A_ineq = A_ineq

        q0 = np.zeros(n_u)
        l0 = np.concatenate([-self.a_max * np.ones(n_u), -self.v_max * np.ones(n_u)])
        u0 = -l0

        self._solver = osqp.OSQP()
        self._solver.setup(P=self.P, q=q0, A=A_ineq, l=l0, u=u0, verbose=False)

    # =========================
    # 求解 QP; 失败时回退到上一拍解
    # =========================
    def _solve_mpc(self, x0):
        N = self.N
        n_u = 3 * N

        # ===== 时变参考序列(x_1 ... x_N 对应的时刻) =====
        p_ref_seq = np.zeros(n_u)
        v_ref_seq = np.zeros(n_u)
        for k in range(1, N + 1):
            p_k, v_k = self._interpolate_reference(self.t + k * self.mpc_dt)
            p_ref_seq[3 * (k - 1):3 * k] = p_k
            v_ref_seq[3 * (k - 1):3 * k] = v_k

        # ===== 代价一次项 =====
        d_p = self.D_p @ x0 - p_ref_seq
        d_v = self.D_v @ x0 - v_ref_seq
        q_vec = 2.0 * (
            self.C_p.T @ (self.w_p_vec * d_p) +
            self.C_v.T @ (self.w_v_vec * d_v)
        )

        # ===== 约束上下限 =====
        l = np.concatenate(
            [-self.a_max * np.ones(n_u), -self.v_max * np.ones(n_u) - d_v]
        )
        u = np.concatenate(
            [self.a_max * np.ones(n_u), self.v_max * np.ones(n_u) - d_v]
        )

        # ===== 求解(失败回退上一拍解) =====
        U = None
        try:
            self._solver.update(q=q_vec, l=l, u=u)
            res = self._solver.solve()
            if res.x is not None and str(res.info.status).lower().startswith("solved"):
                if np.all(np.isfinite(res.x)):
                    U = np.clip(res.x, -self.a_max, self.a_max)
        except Exception:
            U = None

        if U is None:
            U = self._U_last
        else:
            # 平移一步作为下一拍的回退解
            self._U_last = np.concatenate([U[3:], U[-3:]])

        return U[:3]

    # =========================
    # 关节空间惯量阵(仅机械臂 7 个关节)
    # =========================
    def _mass_matrix(self, q):
        robot = self.robot

        q_backup = robot.data.qpos[:7].copy()
        v_backup = robot.data.qvel[:7].copy()

        robot.data.qpos[:7] = q
        robot.data.qvel[:7] = 0

        mujoco.mj_forward(robot.model, robot.data)

        M = np.zeros((robot.model.nv, robot.model.nv))
        mujoco.mj_fullM(robot.model, M, robot.data.qM)

        robot.data.qpos[:7] = q_backup
        robot.data.qvel[:7] = v_backup
        mujoco.mj_forward(robot.model, robot.data)

        return M[:7, :7]

    # =========================
    # 任务空间惯性 M_x = (J M⁻¹ Jᵀ)⁻¹(阻尼伪逆, 防奇异)
    # =========================
    def _task_inertia(self, q, J):
        M = self._mass_matrix(q)
        Jp = J[:3, :]

        Minv_Jt = np.linalg.solve(M + 1e-9 * np.eye(7), Jp.T)
        M_x = np.linalg.solve(Jp @ Minv_Jt + self.damping * np.eye(3), np.eye(3))

        return M_x

    # =========================
    # 姿态误差力矩(SO3 叉积误差, 同上面控制器)
    # =========================
    def _orientation_moment(self, R_ee, omega):
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], self.R_des[:, 0]) +
            np.cross(R_ee[:, 1], self.R_des[:, 1]) +
            np.cross(R_ee[:, 2], self.R_des[:, 2])
        )

        omega_error = -omega

        moment = self.Kp_ori * ori_error + self.Kd_ori * omega_error

        return moment

    # =========================
    # 期望加速度 -> 期望力 -> 关节力矩(重力补偿 + 限幅)
    # =========================
    def _acceleration_to_torque(self, q, J, R_ee, omega, a_des):
        M_x = self._task_inertia(q, J)

        force = M_x @ a_des
        moment = self._orientation_moment(R_ee, omega)

        wrench = np.concatenate([force, moment])

        tau = J.T @ wrench + self.robot.get_gravity(q)
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        return tau

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

        # ===== 每 10 个仿真步重解一次 QP, 期间保持首步控制 =====
        self._solve_tick += 1
        if self._solve_tick >= self.solve_interval:
            x0 = np.concatenate([x, v])
            self._a_des = self._solve_mpc(x0)
            self._solve_tick = 0

        # ===== 力矩映射 =====
        tau = self._acceleration_to_torque(q, J, R_ee, omega, self._a_des)

        self.t += self.dt

        return tau