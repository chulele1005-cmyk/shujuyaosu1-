# 区域健康风险预测与智能诊疗决策

> Health Risk Prediction & Intelligent Clinical Decision Support

## 项目简介

本项目构建了一套完整的健康风险预测与智能诊疗决策系统，涵盖多源医疗数据获取、清洗融合、隐私保护、风险预测建模到可视化决策支持的全流程。

## 项目结构

```
health_risk_project/
├── data/
│   ├── raw/                  # 原始下载数据 (44个文件, ~1.5GB)
│   ├── processed/            # 清洗后数据 (.parquet)
│   ├── fused/                # 融合特征矩阵
│   ├── privacy_protected/    # 隐私保护后数据
│   └── sample/               # 演示用样本数据
├── src/
│   ├── data_acquisition.py     # 步骤3.1 — 多源数据获取
│   ├── data_cleaning.py        # 步骤3.2 — 数据清洗与标准化
│   ├── data_fusion.py          # 步骤3.3 — 多模态数据融合
│   ├── privacy_protection.py   # 步骤3.4 — 隐私安全保护
│   ├── chronic_disease_model.py    # 步骤4.1 — 慢性病风险模型 (待开发)
│   ├── health_trend_forecast.py    # 步骤4.2 — 趋势预测 (待开发)
│   ├── medical_resource_forecast.py # 步骤4.3 — 资源需求预测 (待开发)
│   ├── dashboard_app.py        # 步骤5.1 — 可视化面板 (待开发)
│   └── report_generator.py     # 步骤5.2 — 报告生成 (待开发)
├── outputs/
│   ├── models/               # 训练好的模型 (.pkl)
│   ├── figures/              # 图表截图 (.png)
│   └── report/               # PDF 报告
├── requirements.txt          # Python 依赖清单
├── run_all.py                # 一键运行入口
└── README.md                 # 本文件
```

## 环境要求

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows 10/11 或 Ubuntu 20.04+ |
| Python 版本 | Python 3.9 及以上 |
| 包管理器 | pip 23.0+ 或 conda |
| GPU (可选) | NVIDIA CUDA 11.8+ (用于深度学习加速) |
| IDE | VS Code / PyCharm / JupyterLab |

## 快速开始

### 1. 创建虚拟环境 (推荐)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行完整流水线

```bash
# 运行全部任务一 (数据获取→清洗→融合→隐私保护)
python run_all.py

# 跳过数据下载 (数据已存在时)
python run_all.py --skip-download

# 仅运行任务一
python run_all.py --task 1
```

### 4. 分步运行

```bash
# 步骤 3.1: 数据获取
python src/data_acquisition.py

# 步骤 3.2: 数据清洗
python src/data_cleaning.py

# 步骤 3.3: 数据融合
python src/data_fusion.py

# 步骤 3.4: 隐私保护
python src/privacy_protection.py
```

## 数据来源

本项目使用以下国际公开脱敏医疗数据平台的数据：

| 序号 | 数据集 | 来源 | 用途 |
|------|--------|------|------|
| 1 | Pima Indians Diabetes | Kaggle | 糖尿病风险预测 |
| 2 | Heart Disease UCI | Kaggle | 心血管风险评估 |
| 3 | Cardiovascular Disease | Kaggle | 心血管风险评估 |
| 4 | Breast Cancer Wisconsin | Kaggle/UCI | 乳腺肿瘤分类 |
| 5 | Stroke Prediction | Kaggle | 中风风险预测 |
| 6 | NHANES | CDC 官网 | 慢性病风险因素 |
| 7 | BRFSS (Behavioral Risk Factors) | CDC 官网 | 人群健康行为趋势 |
| 8 | Diabetes 130-US Hospitals | UCI | 糖尿病管理预测 |
| 9 | Cervical Cancer Risk Factors | UCI | 宫颈癌风险因素 |
| 10 | WHO GHO (Global Health Observatory) | WHO 官网 | 全球卫生统计 |
| 11 | NCD-RisC (NCD Risk Factor Collaboration) | NCD-RisC | 全球慢病风险因素 |
| 12 | COVID-19 India | Kaggle | 疫情时序分析 |
| 13 | Healthcare Operations | Kaggle | 医疗运营资源配置 |
| 14 | Medical Insurance | Kaggle | 医疗费用预测 |
| 15 | Chronic Disease | Kaggle | 慢性病特征分析 |

## 隐私保护措施

- ✅ SHA-256 哈希脱敏: 所有标识列已不可逆哈希处理
- ✅ ε-差分隐私: 统计发布满足 ε=1.0 差分隐私保护
- ✅ K-匿名性: K ≥ 5, 确保个体不可区分
- ✅ 全程使用脱敏公开数据, 未涉及真实患者隐私信息

## 合规声明

本项目严格按照以下规范执行:

- 《中华人民共和国数据安全法》
- 《中华人民共和国个人信息保护法》
- 医疗数据伦理规范
- 项目预测模型输出为科研辅助参考, 不构成临床诊断

## 许可证

本项目仅用于科研与教育目的。
