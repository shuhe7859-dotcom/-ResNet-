"""
配置文件 —— 所有路径与超参数集中管理

说明：路径全部基于本文件所在目录自动推导，项目克隆到任意路径都能直接运行，
无需手动修改下面的常量；如有特殊需求，可通过环境变量覆盖（见各处注释）。
"""

import os

# ==================== 项目根目录 ====================
# 默认为本文件所在目录，可用环境变量 GARBAGE_ROOT 覆盖
ROOT_DIR = os.environ.get("GARBAGE_ROOT") or os.path.dirname(os.path.abspath(__file__))

# ==================== 数据集路径 ====================
# 目录结构（按文件夹名自动映射类别）：
#   数据集/cardboard/ 数据集/glass/ 数据集/metal/ 数据集/paper/ 数据集/plastic/
# 代码运行时会自动扫描该目录下的子目录名生成类别列表，
# 因此新增或删减类别只需调整数据集目录，无需改动代码。
# 可用环境变量 GARBAGE_DATA_ROOT 指向其他位置的数据集。
DATA_ROOT = os.environ.get("GARBAGE_DATA_ROOT") or os.path.join(ROOT_DIR, "数据集")

# ==================== 模型保存路径 ====================
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "best_model.pdparams")

# ==================== 日志与可视化路径 ====================
TRAIN_LOG_PATH = os.path.join(ROOT_DIR, "train_log.txt")
VDL_LOG_DIR = os.path.join(ROOT_DIR, "vdl_logs")

# ==================== 训练超参数 ====================
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8           # 训练集比例，其余为验证集
VAL_SPLIT = 0.2
RANDOM_SEED = 42
INPUT_SIZE = 224             # ResNet 标准输入尺寸

# ==================== DeepSeek API 配置 ====================
# 出于安全考虑，API Key 不写入代码仓库，请通过环境变量提供：
#   Windows PowerShell:  $env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"
#   Linux / macOS:       export DEEPSEEK_API_KEY="sk-xxxxxxxx"
# 未配置时程序会自动降级，仅使用 ResNet 的分类结果（见 llm_fusion.py）。
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
