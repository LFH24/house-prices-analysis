# House Prices — 设计规格书（v2 · 已修订）

## 元信息

- **项目**：House Prices: Advanced Regression Techniques
- **类型**：Kaggle 竞赛 · 回归
- **评估指标**：RMSE(log) — 本质等同于 RMSLE
- **设计日期**：2026-07-06（修订 2026-07-06）
- **叙事主线**：变量选择与工程的艺术（79 个特征的系统筛选与编码方法论）
- **优化深度**：深度版（目标 top 20%，score < 0.12）

## 项目定位

与前两个项目的技能互补：

| 维度 | Titanic | Bike Sharing | House Prices |
|------|---------|-------------|-------------|
| 任务类型 | 二分类 | 时间序列回归 | 横截面回归 |
| 特征类型 | 少量混合 | 时间+天气数值 | **大规模类别+数值混合** |
| 核心挑战 | 缺失值填补 | 时间特征工程 | **类别编码 + NA语义 + 特征筛选** |
| 特征数 | ~10 | ~26 | **79 + 衍生** |

这是三个项目中特征最复杂、类别最多、NA 语义最丰富的一个。

## 数据概况

- **训练集**：1,460 rows × 79 features + SalePrice (target)
- **测试集**：1,459 rows × 79 features
- **特征构成**：43 个类别特征 + 36 个数值特征
- **缺失值**：训练集 19 个特征有 NA，但 NA 的语义因特征而异

## 项目方案：混合策略（方案 C）

NA 语义分析 → 编码策略分模型对比实验 → MI + RFECV（并集）+ Lasso sanity check → Optuna 调参 → Stacking 集成。领域判断 + 统计验证互相校验，每一步都有 A/B 对比。

## 章节设计

### 第 1 章 · 问题定义

- RMSLE 指标解读（对数变换 → 百分比误差，与 Bike Sharing 联动）
- Ames 城市背景：爱荷华州立大学所在地，房价受学术人口驱动
- 目标 score < 0.12（top 20% 分界线）
- 79 个特征的全景概览

### 第 2 章 · EDA + 缺失值语义分析

**2.1 目标变量**
- SalePrice 分布（右偏）→ log 变换正态性
- **异常值处理策略**：删除 GrLivArea > 4000 且 SalePrice < 300000 的训练样本（社区公认的离群点，面积巨大但售价极低，约 2-3 个）。另检查超高售价的 leverage point 是否需要剔除。异常值仅在训练集上删除，测试集保留。

**2.2 数值特征**
- 与 SalePrice 的 Pearson/Spearman 相关系数排名（top 15 / bottom 15）
- 面积类特征共线性扫描（1stFlrSF + 2ndFlrSF = GrLivArea；TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF）
- YearBuilt/YearRemodAdd 的时间趋势 + 二战后的建造潮

**2.3 类别特征 + NA 语义分类（本章重点）**
- 逐个有 NA 的类别特征分析 NA 含义，形成 **完整 NA 语义分类表**：

| 特征 | NA 语义 | 填充值 |
|------|---------|--------|
| Alley | 无巷子通道 | "No_Alley" |
| BsmtQual | 无地下室 | "No_Basement" |
| BsmtCond | 无地下室 | "No_Basement" |
| BsmtExposure | 无地下室 | "No_Basement" |
| BsmtFinType1 | 无地下室 | "No_Basement" |
| BsmtFinType2 | 无地下室 | "No_Basement" |
| FireplaceQu | 无壁炉 | "No_Fireplace" |
| GarageType | 无车库 | "No_Garage" |
| GarageFinish | 无车库 | "No_Garage" |
| GarageQual | 无车库 | "No_Garage" |
| GarageCond | 无车库 | "No_Garage" |
| PoolQC | 无泳池 | "No_Pool" |
| Fence | 无围栏 | "No_Fence" |
| MiscFeature | 无杂项设施 | "No_Misc" |
| MasVnrType | 多数为无贴面（MasVnrArea=0），少量可能真缺失 | 先填 "None"，再按 MasVnrArea 验证 |
| MasVnrArea | 同 MasVnrType — 无贴面则面积为 0 | 填 0 |
| LotFrontage | 真缺失（因地段而异） | 按 Neighborhood 中位数填充 |
| GarageYrBlt | 无车库 → 车的建造年份不适用 | 填 0，配合 has_garage flag |
| Electrical | 真缺失（仅 1 条） | 填众数 |

- 类别特征的基数分布（高基数需特别处理：Neighborhood 25 类、Exterior1st 15 类）

### 第 3 章 · 特征工程（核心章节）

**3.1 缺失值填充 + 异常值处理**
- 按 2.3 的 NA 语义表执行差异化填充（类别填字符串标记、数值填 0 或按组中位数）
- 构造 `has_garage`、`has_basement`、`has_fireplace`、`has_pool`、`has_fence` 等 binary flag
- **异常值处理**：训练集删除 GrLivArea > 4000 且 SalePrice < 300000 的约 2-3 个样本

**3.2 新特征构造**

基础衍生：
- `TotalBath` = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
- `TotalSF` = GrLivArea + TotalBsmtSF + WoodDeckSF + OpenPorchSF — **总可用面积**
- `HouseAge` = YrSold − YearBuilt
- `RemodAge` = YrSold − YearRemodAdd
- `PorchSF` = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
- `OverallScore` = OverallQual × OverallCond（综合品质评分）

新增高 ROI 特征（审核补充）：
- `WasRemodeled` = (YearBuilt ≠ YearRemodAdd).astype(int) — 是否翻新过
- `GarageAge` = YrSold − GarageYrBlt — 车库年龄。**NA 处理**：无车库时 GarageYrBlt 先在缺失值填充阶段填 0，但计算 GarageAge 后会得到一个无意义的大值（如 2008）。对此类样本（has_garage=0），强制将 GarageAge 填 0，模型通过 has_garage flag 学到"无车库时 GarageAge 不重要"。
- `LogTotalSF` = np.log1p(TotalSF) — 面积的 log 变换，更接近正态。**与 TotalSF 的关系**：线性模型用 LogTotalSF **替代** TotalSF（log 变换后更接近正态满足线性假设），树模型**两者都保留**让模型自动选择分裂特征。
- `Qual_x_SF` = OverallQual × GrLivArea — 品质与面积的交互

⚠️ **共线性处理**：`TotalSF` 与 `GrLivArea`、`TotalBsmtSF` 高度共线。线性模型使用 `TotalSF` **替代**其组成部分，树模型可同时保留两者（树模型对共线不敏感，但分裂效率受影响）。LogTotalSF 遵循同样策略：线性模型用 LogTotalSF 替代 TotalSF，树模型两者共存。在编码对比实验中验证哪种方式更好。

**3.3 类别编码对比实验（修订：分模型裁判）**

编码方案的偏好取决于模型家族——线性模型和树模型对同一种编码的响应完全不同：

| 编码 | 线性模型 (Ridge/Lasso) | 树模型 (XGB/LGB) | CatBoost |
|------|------------------------|-------------------|----------|
| One-Hot | ✅ 需要（去假序） | ⚠️ 增维稀释分裂效率 | ❌ 不需要 |
| Ordinal | ❌ 引入不存在的序 | ✅ 足够好 | ❌ 不需要 |
| Target Encoding | ✅ 效果好 | ✅ 效果好 | ❌ 不需要（自有原生处理） |

**实验设计**：对比 4 组编码策略，用 Ridge + LightGBM 分别做裁判（双裁判），5-fold KFold：

- **A — Ordinal 编码**：质量类有序（Ex=5, Gd=4, TA=3, Fa=2, Po=1），名义类标签编码
- **B — One-Hot 编码**：全 one-hot + 高频过滤（< 1% 合并为 Other）
- **C — Target Encoding**：使用 `category_encoders.TargetEncoder`（内置 OOF + smoothing），smoothing 参数用默认 Bayesian smoothing：`global_mean × smoothing + category_mean × n / (n + smoothing)`
- **D — 混合（推荐）**：质量类 ordinal + 名义类 target encoding

**CatBoost 单独路径**：CatBoost 直接传原始类别列（`cat_features` 参数），不做任何手动编码，充分利用其原生类别处理能力。

**产出**：
- "线性模型用 X 编码，树模型用 Y 编码，CatBoost 用原生"
- 最终选择按 CV score 说话，不预设结论

**3.4 特征筛选（修订：MI + RFECV 并集，Lasso 做 sanity check）**

- **MI（互信息）**：捕获非线性关系，排名 → 取 top N
- **RFECV（RF estimator）**：用 RF 特征重要性做递归消除
- **Lasso L1 正则**：观察零系数特征 → 如果 MI/RFECV 排名靠前但 Lasso 系数为零，说明可能是非线性特征，保留
- **最终策略**：**MI 和 RFECV 取并集，Lasso 做 sanity check**（不取交集——太激进会丢非线性特征）。训练集仅 1460 条，特征宁多勿少，树模型自带特征选择能力。
- **CatBoost 不参与 3.4 的特征筛选流程**：CatBoost 走原生路径，输入全量特征（含原始类别列），靠内部 L2 正则化 + depth-based regularization 自动处理。调参后 CatBoost 的 feature importance 可事后用来交叉验证其他模型的筛选结果是否一致。

**3.5 数据预处理管道（修订：双 Pipeline）**

两条独立的 sklearn Pipeline，因为线性模型和树模型的预处理需求不同：

```
linear_pipeline = Pipeline([
    ('imputer', ...),        # NA 填充
    ('encoder', ...),        # 类别编码（One-Hot 或 Target Encoding）
    ('poly', ...),           # 可选：多项式交互
    ('scaler', StandardScaler()),  # ✅ 线性模型必须标准化
    ('selector', ...),       # 特征筛选
])

tree_pipeline = Pipeline([
    ('imputer', ...),        # NA 填充
    ('encoder', ...),        # 类别编码（Ordinal 或 Target Encoding）
    ('selector', ...),       # 特征筛选（树模型不需要标准化）
])
```

CatBoost 使用独立处理路径（直接传 cat_features）。

### 第 4 章 · Baseline 模型

- **线性基准**：Ridge / Lasso / ElasticNet（log-target，5-fold KFold，使用 linear_pipeline）
- **树模型基准**：RF / XGBoost / LightGBM（默认参数，使用 tree_pipeline）
- **CatBoost 基准**：默认参数 + cat_features 原生模式
- 产出：7 个模型的 baseline score 对比表（包含编码方案列）
- 判断 feature set → model family 的最佳匹配

### 第 5 章 · 深度调参

**工具**：Optuna（贝叶斯优化），200-300 trials/model，5-fold KFold，scoring='neg_root_mean_squared_error'

**搜索空间（审核补充）**：

XGBoost：
```python
{
    'n_estimators': IntDistribution(200, 1000),
    'max_depth': IntDistribution(3, 8),
    'learning_rate': FloatDistribution(0.01, 0.2, log=True),
    'subsample': FloatDistribution(0.6, 1.0),
    'colsample_bytree': FloatDistribution(0.5, 1.0),
    'reg_alpha': FloatDistribution(1e-3, 10, log=True),
    'reg_lambda': FloatDistribution(1e-3, 10, log=True),
    'min_child_weight': IntDistribution(1, 20),
}
```

LightGBM：
```python
{
    'n_estimators': IntDistribution(200, 1000),
    'max_depth': IntDistribution(3, 10),
    'num_leaves': IntDistribution(20, 150),
    'learning_rate': FloatDistribution(0.01, 0.2, log=True),
    'subsample': FloatDistribution(0.6, 1.0),
    'colsample_bytree': FloatDistribution(0.5, 1.0),
    'reg_alpha': FloatDistribution(1e-3, 10, log=True),
    'reg_lambda': FloatDistribution(1e-3, 10, log=True),
    'min_child_samples': IntDistribution(5, 60),
}
```

CatBoost：
```python
{
    'iterations': IntDistribution(300, 1000),
    'depth': IntDistribution(4, 10),
    'learning_rate': FloatDistribution(0.01, 0.2, log=True),
    'l2_leaf_reg': FloatDistribution(0.1, 10, log=True),
    'border_count': IntDistribution(32, 255),
}
```

**防过拟合**：监控 train vs CV score 差距；Optuna 的 pruning 机制自动停止劣化 trial。

### 第 6 章 · 集成

- StackingRegressor：基学习器 = 调参后的 XGBoost + LightGBM + CatBoost
- 元学习器 = Ridge
- 对比：简单加权平均 vs Stacking vs 最优单模型
- 选最优策略 → 全量训练 → 生成 submission

### 第 7 章 · 特征重要性与房价洞察

- Feature importance（XGBoost/LightGBM gain-based）
- Permutation importance（模型无关验证，更可靠）
- Lasso 系数解读（对数空间 → 实际含义：exp(coef) − 1 ≈ 特征每增加 1 单位的价格变动百分比）
- PDP：GrLivArea、YearBuilt、OverallQual 的部分依赖图，观察边际递减拐点
- 核心结论：影响 Ames 房价最大的 5 个因素 + 反直觉发现

## 技术栈

Python、pandas、numpy、scikit-learn、XGBoost、LightGBM、CatBoost、Optuna、category_encoders、matplotlib、seaborn、scipy

## 不使用 SHAP

原因：当前 numpy 2.5.1 + numba 0.66.0 版本冲突导致 SHAP 不可用。替代方案：feature_importances_ + permutation importance + PDP 组合。

## 修订记录

| 版本 | 日期 | 修订内容 |
|------|------|----------|
| v1 | 2026-07-06 | 初稿，7 章 + 方案 C |
| v2 | 2026-07-06 | 审核修订：编码裁判用 Ridge+LGB 双裁判 / CatBoost 原生路径 / 异常值处理策略 / 特征筛选改为并集 / TE 用 category_encoders / 双 Pipeline / 补全 NA 语义表 + 衍生特征 + Optuna 搜索空间 |

## 文件结构

```
house-prices-analysis/
├── README.md
├── house-prices-analysis.ipynb
├── run.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv
│   └── test.csv
├── docs/
│   └── specs/
│       └── 2026-07-06-house-prices-design.md
└── output/
    ├── submission.csv
    ├── kaggle-score.png
    └── figures/
```

## 与 Bike Sharing 的差异总结

| 维度 | Bike Sharing | House Prices |
|------|-------------|-------------|
| CV 策略 | TimeSeriesSplit | standard KFold |
| 主要缺失值类型 | 传感器故障（0 值） | 结构性缺失（NA=不存在） |
| 特征工程重心 | 时间循环编码 + 交互 | 类别编码三方案对比 + NA 语义表 |
| 编码策略 | 不需要（全数值） | Ordinal / One-Hot / Target Encoding 分模型 |
| 新模型 | — | CatBoost（原生类别处理） |
| 调参工具 | RandomizedSearchCV | Optuna（贝叶斯优化 + pruning） |
| 可解释性 | — | Permutation importance + PDP + Lasso 系数 |
| Pipeline | 单 pipeline | 双 pipeline（linear / tree 分流） |
