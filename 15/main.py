import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

print("Loading...")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Not found.")
    exit()

print("\nProcessing data...")
combined_df = pd.concat([train_df.drop('accident_risk', axis=1), test_df], ignore_index=True)

safe_num_lanes = combined_df['num_lanes'].replace(0, 1)
combined_df['speed_limit_per_lane'] = combined_df['speed_limit'] / safe_num_lanes
combined_df['curvature_x_speed'] = combined_df['curvature'] * combined_df['speed_limit']
combined_df['lanes_x_curvature'] = combined_df['num_lanes'] * combined_df['curvature']

bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

train_processed = combined_df.iloc[:len(train_df)]
test_processed = combined_df.iloc[len(train_df):]
X = train_processed.drop('id', axis=1)
y = train_df['accident_risk']
X_test = test_processed.drop('id', axis=1)

print("\nTraining the model...")

lgb_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
    'learning_rate': 0.005, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 40,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt', 'max_depth': 8,
}

N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_predictions = np.zeros(X_test.shape[0])
oof_rmse_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Melatih Fold {fold + 1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False)])

    val_preds = model.predict(X_val)
    fold_test_preds = model.predict(X_test)
    test_predictions += fold_test_preds / N_SPLITS
    
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    oof_rmse_scores.append(rmse)

mean_oof_rmse = np.mean(oof_rmse_scores)
print("\n-----------------------------------------------------")
print(f"Average Score: {mean_oof_rmse:.5f}")
print("-----------------------------------------------------")

try:
    print("\nSearching 'submission-1.csv' to merge...")
    old_submission_df = pd.read_csv('submission-1.csv')
    
    new_submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_predictions})
    
    blended_df = pd.merge(new_submission_df, old_submission_df, on='id', suffixes=('_new', '_old'))
    
    blended_df['accident_risk'] = (blended_df['accident_risk_new'] * 0.5) + (blended_df['accident_risk_old'] * 0.5)
    
    final_submission_df = blended_df[['id', 'accident_risk']]
    
    final_submission_df.to_csv('submission_blended.csv', index=False)
    print("\nFile 'submission_blended.csv' saved!")

except FileNotFoundError:
    print("\nFile 'submission-1.csv' not found.")
    print("Save the result of the new model only without blending.")
    submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_predictions})
    submission_df.to_csv('submission-2.csv', index=False)
    print("\nFile 'submission-2.csv' saved!")