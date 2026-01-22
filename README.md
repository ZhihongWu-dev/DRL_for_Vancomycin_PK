# DRL-for-Vancomycin-PK

This project uses DRL methods to determine the best Vancomycin dosing for ICU patients based on MIMIC demo data.

## Structure

```text
DRL_for_Vancomycin_PK/
├── data_processing/   # 数据清洗、预处理脚本 (Python)
├── algorithms/        # 核心算法模型、训练脚本 (PyTorch)
│   └── iql/            # IQL离线RL实现与训练工具
├── configs/           # 训练配置文件
├── original_data/     # 原始数据
├── intermediate_data/ # 中间数据
├── tools/             # 实验辅助脚本
├── pytest.ini         # 测试配置
└── README.md          # 项目总览

