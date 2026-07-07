# House Prices：变量选择与工程的艺术

基于 Kaggle House Prices 竞赛，对 79 个特征进行系统性的缺失值语义分析、编码对比实验、特征筛选和模型集成。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 主要发现

Ames 市的房价由几个关键维度驱动：

- **面积是最强的单一信号** — GrLivArea（地上居住面积）在所有模型里都是第一重要的特征，但它的边际贡献有递减趋势：超过 3000 sqft 之后每平方英尺的溢价开始下降
- **品质比年份更重要** — OverallQual 的影响力稳定排在 Top 3，远超 YearBuilt。一个精心维护的老房子可以比疏于打理的新房子卖得更贵
- **地段存在明显的溢价分层** — Neighborhood 的 target encoding 揭示出高达 2-3 倍的价格差异，NoRidge（北岭）和 StoneBr（石溪）对均价有 50%+ 的溢价
- **新的比旧的好，但翻新也有用** — HouseAge 每增加 1 年，房价约下降 0.5-1%；但 WasRemodeled=1 的房子能回收一部分贬损

一个具体数字：Ordinal 编码 + LightGBM 默认参数的 baseline RMSLE 是 0.13 左右，经过 Optuna 调参 + Stacking 集成后可以压到 0.12 以内。

---

## 分析流程

```
第 1 章 · 问题定义    →  RMSLE 解读 + Ames 城市背景 + 特征全景
第 2 章 · EDA         →  SalePrice 分布 + 数值特征关联 + NA 语义分类表（19 个特征）
第 3 章 · 特征工程     →  缺失值填充 + 10 个衍生特征 + 编码对比实验（Ridge+LGB 双裁判）
                         + MI/RFECV/Lasso 三法特征筛选 + 双 Pipeline
第 4 章 · Baseline     →  7 模型基线（线性×3 + 树×3 + CatBoost 原生）
第 5 章 · 深度调参     →  Optuna 贝叶斯优化（XGBoost / LightGBM / CatBoost）
第 6 章 · 集成         →  Stacking vs 加权平均 vs 最优单模型
第 7 章 · 特征重要性   →  Gain-based + Permutation + Lasso 系数 + PDP
第 8 章 · 方法论复盘   →  特征工程经验总结
```

---

## 技术栈

Python、pandas、numpy、scikit-learn、XGBoost、LightGBM、CatBoost、Optuna、category_encoders、matplotlib、seaborn、scipy

---

## Kaggle 提交结果

| 指标 | 结果 |
|------|------|
| RMSLE | **0.11913** |
| 模型 | Stacking（XGB + LGB + CatBoost → Ridge） |
| CV 策略 | 5-Fold |
| 特征数 | 80+（含衍生特征和编码特征） |

---

## 复现

```bash
# 安装依赖
pip install -r requirements.txt

# 打开 Notebook
jupyter notebook house-prices-analysis.ipynb

# 或者直接运行脚本
python run.py
```

---

## 文件结构

```
house-prices-analysis/
├── README.md
├── house-prices-analysis.ipynb    # 完整分析过程（8 章）
├── run.py                         # 独立可复现脚本
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv                  # 训练集（1,460 条）
│   ├── test.csv                   # 测试集（1,459 条）
│   └── sample_submission.csv
└── output/
    ├── submission.csv
    ├── kaggle-score.png
    └── figures/
```

---

LFH24 · 2026.07
