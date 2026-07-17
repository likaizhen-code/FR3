# FR3 MuJoCo Simulation Infrastructure

基于 MuJoCo 的 Franka Research 3 (FR3) 机械臂仿真与控制基础设施。

---

## 🚀 Quick Start

### 环境准备与依赖安装

建议使用 Conda 创建独立的 Python 3.10 环境：

```bash
# 1. 创建 Python 3.10 虚拟环境
conda create -n fr3 python=3.10 -y

# 2. 激活虚拟环境
conda activate fr3

# 3. 安装项目依赖项
pip install -r requirements.txt

# 4. 以可编辑模式安装当前包
pip install -e .

FR3/
├── assets/                 # 存放环境场景与机械臂关节 MJCF/URDF 描述文件
├── lib/                    # 底层核心库 (控制器与仿真引擎封装)
│   ├── Controlled.py       # 控制器定义 (如力矩/阻抗/位姿控制等)
│   └── MujocoSim.py        # MuJoCo 动力学仿真计算、状态更新与传感器数据获取
├── task/                   # 具体执行任务与策略训练逻辑
├── requirements.txt        # 项目依赖列表
└── README.md               # 项目说明文档
