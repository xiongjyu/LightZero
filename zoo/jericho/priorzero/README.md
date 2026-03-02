# PriorZero 训练指南

## 🚀 训练步骤

### 1. 进入工作目录
首先，切换到 PriorZero 的项目根目录：
`cd LightZero/zoo/jericho/priorzero`

### 2. 配置环境参数
在启动训练前，根据你的硬件资源（如 GPU 数量、内存大小）和实验需求，修改配置文件：
* **文件路径**: `src/priorzero_config.py`

### 3. 启动分布式训练 (DDP)
确认配置无误后，执行预置的任务脚本启动多卡并行的分布式数据并行 (DDP) 训练：

```bash
bash scripts/run_priorzero_ddp.sh