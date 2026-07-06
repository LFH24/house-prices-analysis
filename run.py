"""House Prices — 独立可复现脚本，生成 submission.csv"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== 加载数据 ====================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

print(f'训练集: {train.shape}, 测试集: {test.shape}')

# ==================== 异常值处理 ====================
anomaly_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
train = train[~anomaly_mask].reset_index(drop=True)
print(f'删除异常值: {anomaly_mask.sum()} 条')

# ==================== 合并 + 缺失值填充 ====================
y = np.log1p(train['SalePrice'])
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0, ignore_index=True)

# 结构性缺失 → 字符串标记
na_fill_str = {
    'Alley': 'No_Alley', 'BsmtQual': 'No_Basement', 'BsmtCond': 'No_Basement',
    'BsmtExposure': 'No_Basement', 'BsmtFinType1': 'No_Basement', 'BsmtFinType2': 'No_Basement',
    'FireplaceQu': 'No_Fireplace', 'GarageType': 'No_Garage', 'GarageFinish': 'No_Garage',
    'GarageQual': 'No_Garage', 'GarageCond': 'No_Garage', 'PoolQC': 'No_Pool',
    'Fence': 'No_Fence', 'MiscFeature': 'No_Misc',
}
for col, fill_val in na_fill_str.items():
    all_data[col] = all_data[col].fillna(fill_val)

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# Binary flags
all_data['has_garage'] = (all_data['GarageType'] != 'No_Garage').astype(int)
all_data['has_basement'] = (all_data['BsmtQual'] != 'No_Basement').astype(int)
all_data['has_fireplace'] = (all_data['FireplaceQu'] != 'No_Fireplace').astype(int)
all_data['has_pool'] = (all_data['PoolQC'] != 'No_Pool').astype(int)
all_data['has_fence'] = (all_data['Fence'] != 'No_Fence').astype(int)
all_data['has_alley'] = (all_data['Alley'] != 'No_Alley').astype(int)

# ==================== 特征工程 ====================
all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] +
                          all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['TotalSF'] = (all_data['GrLivArea'] + all_data['TotalBsmtSF'] +
                       all_data['WoodDeckSF'] + all_data['OpenPorchSF'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['PorchSF'] = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                       all_data['X3SsnPorch'] + all_data['ScreenPorch'])
all_data['OverallScore'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['WasRemodeled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)
all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']
all_data.loc[all_data['has_garage'] == 0, 'GarageAge'] = 0
all_data['LogTotalSF'] = np.log1p(all_data['TotalSF'])
all_data['Qual_x_SF'] = all_data['OverallQual'] * all_data['GrLivArea']

# ==================== Ordinal 编码 ====================
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0,
               'No_Basement': 0, 'No_Garage': 0, 'No_Fireplace': 0,
               'No_Pool': 0, 'No_Fence': 0, 'No_Misc': 0, 'No_Alley': 0}

ordinal_quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                         'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                         'GarageCond', 'PoolQC', 'Fence']

for col in ordinal_quality_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(quality_map).fillna(0).astype(int)

cat_cols_updated = all_data.select_dtypes(include=['object']).columns.tolist()
nominal_cols = [c for c in cat_cols_updated if c not in ordinal_quality_cols]
for col in nominal_cols:
    if col in all_data.columns:
        all_data[col] = pd.factorize(all_data[col])[0]

print(f'编码后特征数: {all_data.shape[1]}')

# ==================== 分离 train/test ====================
train_len = len(train)
X = all_data[:train_len].copy()
X_test = all_data[train_len:].copy()

# ==================== 特征筛选 ====================
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_selected = X.columns[mi_scores > 0.01].tolist()

rf_for_rfecv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfecv = RFECV(rf_for_rfecv, step=1, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
rfecv.fit(X, y)
rfecv_selected = X.columns[rfecv.support_].tolist()

final_features = list(set(mi_selected) | set(rfecv_selected))
print(f'MI selected: {len(mi_selected)}, RFECV selected: {len(rfecv_selected)}, Union: {len(final_features)}')

X_final = X[final_features].values
X_test_final = X_test[final_features].values

# ==================== CV ====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model, X_data, y_data):
    scores = cross_val_score(model, X_data, y_data, cv=kf,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -np.mean(scores)

# ==================== Optuna: XGBoost ====================
print('\n=== XGBoost Optuna ===')
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    }
    return rmse_cv(XGBRegressor(**params), X_final, y)

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=200, show_progress_bar=True)
print(f'XGBoost Best: {xgb_study.best_value:.5f}')

# ==================== Optuna: LightGBM ====================
print('\n=== LightGBM Optuna ===')
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
        'random_state': 42, 'verbose': -1, 'n_jobs': -1, 'force_col_wise': True,
    }
    return rmse_cv(LGBMRegressor(**params), X_final, y)

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=200, show_progress_bar=True)
print(f'LightGBM Best: {lgb_study.best_value:.5f}')

# ==================== Optuna: CatBoost (原生) ====================
print('\n=== CatBoost Optuna ===')
cb_cat_features = X.select_dtypes(include=['object']).columns.tolist() if X.select_dtypes(include=['object']).shape[1] > 0 else []

def cb_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42, 'verbose': 0, 'thread_count': -1,
    }
    if cb_cat_features:
        params['cat_features'] = cb_cat_features
    return rmse_cv(CatBoostRegressor(**params), X, y)

cb_study = optuna.create_study(direction='minimize')
cb_study.optimize(cb_objective, n_trials=150, show_progress_bar=True)
print(f'CatBoost Best: {cb_study.best_value:.5f}')

# ==================== 选择最优策略 ====================
xgb_best = XGBRegressor(**xgb_study.best_params, random_state=42, verbosity=0, n_jobs=-1)
lgb_best = LGBMRegressor(**lgb_study.best_params, random_state=42, verbose=-1, n_jobs=-1, force_col_wise=True)

all_strategies = {
    'Tuned XGBoost': xgb_study.best_value,
    'Tuned LightGBM': lgb_study.best_value,
    'Tuned CatBoost': cb_study.best_value,
}

best_name = min(all_strategies, key=all_strategies.get)
best_score = all_strategies[best_name]
print(f'\n=== 最优策略: {best_name} (RMSLE={best_score:.5f}) ===')

# ==================== 全量训练 + 生成提交 ====================
if 'XGBoost' in best_name:
    xgb_best.fit(X_final, y)
    test_pred_log = xgb_best.predict(X_test_final)
elif 'LightGBM' in best_name:
    lgb_best.fit(X_final, y)
    test_pred_log = lgb_best.predict(X_test_final)
elif 'CatBoost' in best_name:
    cb_best_params = cb_study.best_params.copy()
    cb_best_params.update({'random_seed': 42, 'verbose': 0, 'thread_count': -1})
    if cb_cat_features:
        cb_best_params['cat_features'] = cb_cat_features
    cb_best = CatBoostRegressor(**cb_best_params)
    cb_best.fit(X, y)
    test_pred_log = cb_best.predict(X_test)
else:
    xgb_best.fit(X_final, y)
    test_pred_log = xgb_best.predict(X_test_final)

test_pred = np.maximum(np.expm1(test_pred_log), 0)

os.makedirs('output', exist_ok=True)
submission = pd.DataFrame({'Id': test_id, 'SalePrice': test_pred})
submission.to_csv('output/submission.csv', index=False)

print(f'预测范围: ${test_pred.min():,.0f} ~ ${test_pred.max():,.0f}')
print(f'预测均值: ${test_pred.mean():,.0f}')
print(f'提交文件: output/submission.csv ({len(submission)} 条)')
print(f'CV RMSLE: {best_score:.5f}')
