import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

print("Reading...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

def feature_engineering(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HeartRatePerMin'] = df['Heart_Rate'] / df['Duration']
    df['TempHR'] = df['Body_Temp'] * df['Heart_Rate']
    df['Workload'] = df['Duration'] * df['Weight']
    df['Height_to_Weight'] = df['Height'] / df['Weight']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Temp_per_Height'] = df['Body_Temp'] / df['Height']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3]).astype(int)
    df['Duration_per_Weight'] = df['Duration'] / df['Weight']
    df['HR_per_Weight'] = df['Heart_Rate'] / df['Weight']
    return df

print("Feature engineering...")
train = feature_engineering(train)
test = feature_engineering(test)

X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])
X_test = test.drop(columns=['id'])

X.fillna(X.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_valid_cat = np.zeros(len(X))
preds_test_cat = np.zeros(len(X_test))
preds_valid_xgb = np.zeros(len(X))
preds_test_xgb = np.zeros(len(X_test))

print("Starting Ensemble...")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0
    )
    model_cat.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold))
    preds_valid_cat[valid_idx] = model_cat.predict(X_valid_fold)
    preds_test_cat += model_cat.predict(X_test) / kf.n_splits

    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        verbosity=0,
        eval_metric='rmse',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        verbose=False
    )
    preds_valid_xgb[valid_idx] = model_xgb.predict(X_valid_fold)
    preds_test_xgb += model_xgb.predict(X_test) / kf.n_splits

preds_valid_ensemble = (preds_valid_cat + preds_valid_xgb) / 2
preds_test_ensemble = (preds_test_cat + preds_test_xgb) / 2

rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(preds_valid_ensemble)))
print(f"\nEnsemble validation RMSLE score: {rmsle:.5f}")

submission['Calories'] = np.expm1(preds_test_ensemble)
submission.to_csv('submission.csv', index=False)
print("saved 'submission.csv'!")