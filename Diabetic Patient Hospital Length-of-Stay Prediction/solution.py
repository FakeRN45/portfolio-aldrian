import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    import os
    os.system('pip install catboost')
    import catboost as cb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

print("Loading dataset...")
train = pd.read_csv('dataset/public/train.csv')
test = pd.read_csv('dataset/public/test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['time_in_hospital'] = -1 
df = pd.concat([train, test], ignore_index=True)

print("Feature Engineering...")

df['race'] = df['race'].fillna('Unknown')
df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
df['A1Cresult'] = df['A1Cresult'].fillna('None')
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Unknown')

age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
           '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
           '[80-90)': 85, '[90-100)': 95}
df['age_num'] = df['age'].map(age_map).fillna(50) if df['age'].astype(str).str.contains(r'\[').any() else df['age']

df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df['inpatient_ratio'] = df['number_inpatient'] / (df['total_prior_visits'] + 1)
df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int) 
df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)

df['clinical_severity_index'] = df['num_lab_procedures'] + df['num_procedures'] + df['num_medications']
df['procedures_per_med'] = df['num_procedures'] / (df['num_medications'] + 1)
df['lab_per_med'] = df['num_lab_procedures'] / (df['num_medications'] + 1)

df['A1C_med_change'] = df['A1Cresult'].astype(str) + "_" + df['change'].astype(str)
df['A1C_diabetes_med'] = df['A1Cresult'].astype(str) + "_" + df['diabetesMed'].astype(str)
df['diag_combo'] = df['diag_1'].astype(str) + "_" + df['diag_2'].astype(str) + "_" + df['diag_3'].astype(str)
df['adm_type_source'] = df['admission_type_id'].astype(str) + "_" + df['admission_source_id'].astype(str)

meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

df['num_meds_taken'] = (df[meds] != 'No').sum(axis=1)
df['num_meds_changed'] = ((df[meds] == 'Up') | (df[meds] == 'Down')).sum(axis=1)

df = df.replace([np.inf, -np.inf], 0)

cat_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
            'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 
            'A1Cresult', 'change', 'diabetesMed', 'readmitted', 'A1C_med_change', 
            'A1C_diabetes_med', 'diag_combo', 'adm_type_source'] + meds

for col in cat_cols:
    df[col] = df[col].astype(str).replace(['nan', 'NaN', ''], 'Unknown')

train_df = df[df['is_train'] == 1].drop(['is_train'], axis=1)
test_df = df[df['is_train'] == 0].drop(['is_train', 'time_in_hospital'], axis=1)

X = train_df.drop(['id', 'time_in_hospital'], axis=1)
y = train_df['time_in_hospital']
X_test = test_df.drop(['id'], axis=1)

print("\nStarting 5-Fold Training...")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

lgb_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'learning_rate': 0.04, 'max_depth': 8, 'num_leaves': 127, 'min_child_samples': 40, 'subsample': 0.8, 'n_estimators': 2000}
xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.04, 'max_depth': 7, 'min_child_weight': 10, 'subsample': 0.8, 'tree_method': 'hist', 'n_estimators': 2000}
cb_params = {'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'learning_rate': 0.05, 'depth': 7, 'l2_leaf_reg': 5, 'random_seed': 42, 'verbose': False, 'iterations': 2000}

fold_loop = tqdm(kf.split(X, y), total=n_splits, desc="Training Folds")

for fold, (train_idx, val_idx) in enumerate(fold_loop):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    te = ce.TargetEncoder(cols=cat_cols, smoothing=15)
    
    X_tr_enc = te.fit_transform(X_tr, y_tr).replace([np.inf, -np.inf], 0).fillna(0)
    X_va_enc = te.transform(X_va).replace([np.inf, -np.inf], 0).fillna(0)
    X_test_enc = te.transform(X_test).replace([np.inf, -np.inf], 0).fillna(0)

    model_lgb = lgb.LGBMRegressor(**lgb_params, random_state=42 + fold)
    model_lgb.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])
    
    model_xgb = xgb.XGBRegressor(**xgb_params, random_state=42 + fold)
    model_xgb.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)

    model_cb = cb.CatBoostRegressor(**cb_params)
    model_cb.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], early_stopping_rounds=150, verbose=False)
    
    val_pred_lgb = model_lgb.predict(X_va_enc)
    val_pred_xgb = model_xgb.predict(X_va_enc)
    val_pred_cb = model_cb.predict(X_va_enc)
    
    fold_pred = (val_pred_lgb * 0.4) + (val_pred_cb * 0.4) + (val_pred_xgb * 0.2)
    oof_preds[val_idx] = fold_pred
    
    fold_rmse = rmse(y_va, fold_pred)
    fold_loop.set_postfix({'RMSE': f"{fold_rmse:.4f}"})
    
    lgb_preds += model_lgb.predict(X_test_enc) / n_splits
    xgb_preds += model_xgb.predict(X_test_enc) / n_splits
    cb_preds += model_cb.predict(X_test_enc) / n_splits

print("\n" + "="*40)
final_rmse = rmse(y, oof_preds)
final_mae = mean_absolute_error(y, oof_preds)
final_r2 = r2_score(y, oof_preds)

print("OVERALL EVALUATION (Out-Of-Fold):")
print(f"Ensemble RMSE  : {final_rmse:.4f}")
print(f"Average MAE    : {final_mae:.4f}")
print(f"R-Squared (R2) : {final_r2:.4f}")
print("="*40)

final_test_preds = (lgb_preds * 0.4) + (cb_preds * 0.4) + (xgb_preds * 0.2)
final_test_preds = np.clip(final_test_preds, 1.0, 14.0)

submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print("File 'submission.csv' saved!")