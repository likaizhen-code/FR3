# Quick Start
 1. conda create -n fr3 python=3.10 -y
 2. conda activate fr3
 3. pip install -r requirements.txt
 4. pip install -e .

# Structure
-FR3
    --assets
        '''存放环境与机械臂关节描述文件'''
    --task
        '''具体执行任务'''
    --lib
        ---Controlled.py
            '''控制器定义'''
        ---MujocoSim.py
            '''动力学计算及数据获取'''
    --Readme.md
    --requirements.txt